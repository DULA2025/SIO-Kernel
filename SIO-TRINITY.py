"""
SIO-TRINITY v3.1: The Pan-Cancer Logic Gate
-------------------------------------------
A spectral folding engine for designing pH-gated chimeric trimers.
Targets: KRAS (Chain A), EGFR (Chain B), BRAF (Chain C).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.interpolate import CubicSpline
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import sys
import os
import logging
import argparse
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("SIO-TRINITY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MASTER SEQUENCES (v1 Candidate) ---
SEQ_A_KRAS = "KLVVVGAGGVGKSALTIQLIQNHFVDEYDPTQEDSYRKQVVIDGETCLLDILDTAGQEEY" # 60 residues
SEQ_B_EGFR = "PYVAIKELGEGAFGKVYKGIWIPEGEKVKIPVAIKELREA"                     # 40 residues
SEQ_C_BRAF = "VLEAVRLLFFVAPFGSGQLIDQDVRGEVEKPLKDVQRLSQ"                     # 40 residues
LINKER     = "GGGGSGGGGS"                                                  # 10 residues

# --- CHEMICAL PHYSICS KERNEL ---

KD_SCALE = {
    'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,
    'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5
}
PKA_VALS = {'D':3.9,'E':4.2,'H':6.0,'K':10.5,'R':12.5,'Y':10.0,'C':8.3}

def get_hydrophobicity_map(sequence, device):
    """ Generates H_ij matrix favoring hydrophobic burial (Zipper Effect). """
    n = len(sequence)
    vals = [KD_SCALE.get(aa, 0.0) for aa in sequence]
    v = torch.tensor(vals, dtype=torch.float32, device=device)
    v_i = v.unsqueeze(1).repeat(1, n)
    v_j = v.unsqueeze(0).repeat(n, 1)
    # Sigmoid boost for hydrophobic pairs (Sum > 4.0)
    return 1.0 + 0.5 * torch.tanh((v_i + v_j) / 4.0)

def get_electrostatic_map(sequence, device, pH):
    """ Generates Q_ij matrix based on pH-dependent protonation states. """
    n = len(sequence)
    q_list = []
    for aa in sequence:
        q = 0.0
        if aa in ['K','R']: q = 1.0/(1.0+10**(pH-PKA_VALS[aa]))
        elif aa=='H': q = 1.0/(1.0+10**(pH-6.0)) # Histidine Switch
        elif aa in ['D','E','Y','C']: q = -1.0/(1.0+10**(PKA_VALS[aa]-pH))
        q_list.append(q)
    q_t = torch.tensor(q_list, dtype=torch.float32, device=device)
    q_i = q_t.unsqueeze(1).repeat(1,n); q_j = q_t.unsqueeze(0).repeat(n,1)
    # Coulomb logic: Like charges repel (Low K), Opposites attract (High K)
    # Inverted tanh: +Product -> Repulsion -> <1.0
    return 1.0 - (0.5 * torch.tanh(q_i*q_j * 2.0))

def generate_spline_params(n, controls, device, pH):
    """ Generates Delta/Alpha vectors from genomic control points. """
    x = np.arange(n)
    nodes = np.linspace(0, n-1, len(controls['delta']))
    d_spl = CubicSpline(nodes, controls['delta'])
    a_spl = CubicSpline(nodes, controls['alpha'])
    d_seq = torch.tensor(d_spl(x), dtype=torch.float32, device=device)
    a_seq = torch.tensor(a_spl(x), dtype=torch.float32, device=device)
    
    # --- ACID TRIGGER ---
    # Global detection of Tumor Microenvironment
    acid = 1.0 / (1.0 + torch.exp(4.0 * (pH - 6.8)))
    if acid > 0.1:
        d_seq += (0.3 * acid) # Push towards Beta-Sheet (Aggregation)
        a_seq *= (1.0 - (0.5 * acid)) # Melt Tertiary Structure
        
    return torch.clamp(d_seq,0.1,0.7), torch.clamp(a_seq,0.01,0.95)

def build_block_kernel(seq, controls, theta, pH):
    """ Builds the geometry/chemistry kernel for a single chain block. """
    n = len(seq)
    H = get_hydrophobicity_map(seq, device)
    Q = get_electrostatic_map(seq, device, pH)
    d, a = generate_spline_params(n, controls, device, pH)
    
    idx = torch.arange(n, device=device)
    i, j = torch.meshgrid(idx, idx, indexing='ij')
    diff = torch.abs(i-j).float()
    
    d_avg = (d[i]+d[j])/2.0
    a_avg = (a[i]+a[j])/2.0
    
    # SIO Equation
    sig_g = torch.log(torch.tensor(n))/torch.sqrt(2.0*d_avg)
    sig_c = torch.log(torch.tensor(n))
    decay = (1.0-a_avg)*torch.exp(-diff**2/(2*sig_g**2)) + a_avg*(1.0/(1.0+(diff/sig_c)**2))
    
    K = decay * torch.cos(2*np.pi*d_avg*diff+theta) * (1.0+torch.sin(2*np.pi*diff/6.0)) * H * Q
    K.masked_fill_(diff==0, 1.0)
    return K

def get_trimer_embedding(full_seq, cuts, controls_list, theta, scaling, pH):
    """ 
    Constructs the 3x3 Block Matrix for the Trimer.
    [ K_AA  K_AB  K_AC ]
    [ K_BA  K_BB  K_BC ]
    [ K_CA  K_CB  K_CC ]
    """
    seqA, seqB, seqC = full_seq[0:cuts[0]], full_seq[cuts[0]:cuts[1]], full_seq[cuts[1]:]
    seqs = [seqA, seqB, seqC]
    
    # 1. Build Diagonal Blocks (Intra-Chain Physics)
    Ks = []
    for i in range(3):
        Ks.append(build_block_kernel(seqs[i], controls_list[i], theta, pH))
        
    # 2. Build Off-Diagonal Blocks (Inter-Chain Glue)
    # Physics: Repulsive at pH 7.4, Sticky at pH 6.5
    baseline = 0.05
    if pH < 6.8: baseline = 0.4 # The "Pact" Glue
    
    # We construct full maps then slice for cross-terms
    H_full = get_hydrophobicity_map(full_seq, device)
    Q_full = get_electrostatic_map(full_seq, device, pH)
    
    # Slice Helper
    def get_cross(idx1, idx2):
        s1, e1 = (0, cuts[0]) if idx1==0 else ((cuts[0], cuts[1]) if idx1==1 else (cuts[1], len(full_seq)))
        s2, e2 = (0, cuts[0]) if idx2==0 else ((cuts[0], cuts[1]) if idx2==1 else (cuts[1], len(full_seq)))
        H_c = H_full[s1:e1, s2:e2]
        Q_c = Q_full[s1:e1, s2:e2]
        return torch.full((e1-s1, e2-s2), baseline, device=device) * H_c * Q_c

    K_AB = get_cross(0, 1); K_AC = get_cross(0, 2); K_BC = get_cross(1, 2)
    K_BA = K_AB.T; K_CA = K_AC.T; K_CB = K_BC.T
    
    # 3. Assemble
    row1 = torch.cat([Ks[0], K_AB, K_AC], dim=1)
    row2 = torch.cat([K_BA, Ks[1], K_BC], dim=1)
    row3 = torch.cat([K_CA, K_CB, Ks[2]], dim=1)
    K_total = torch.cat([row1, row2, row3], dim=0)
    
    # 4. Embed
    eigvals, eigvecs = torch.linalg.eigh(K_total.double())
    eigvals, eigvecs = eigvals.float(), eigvecs.float()
    idx_sort = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx_sort]
    
    coords = eigvecs[:, :3] * torch.sqrt(eigvals[idx_sort][:3])
    
    # Physical Scaling
    if torch.norm(coords)>1e-6:
        bonds = torch.norm(coords[1:]-coords[:-1], dim=1)
        mean_b = torch.mean(bonds)
        if mean_b > 1e-9: coords *= (scaling / mean_b)
            
    return coords

# --- REFINEMENT ---

class DifferentiableRefiner(nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.coords = nn.Parameter(raw.clone().detach().requires_grad_(True))
    def forward(self): return self.coords

def loss_geometry(coords):
    # Bond integrity
    diffs = coords[1:] - coords[:-1]
    bond_loss = torch.mean((torch.norm(diffs, dim=1) - 3.8)**2)
    # Steric Clash
    pdist = torch.cdist(coords, coords)
    mask = torch.eye(len(coords), device=coords.device)
    for k in range(len(coords)-1): mask[k,k+1]=1; mask[k+1,k]=1
    clash_loss = torch.sum(torch.relu(4.0 - (pdist + mask*100)))
    return bond_loss*10.0 + clash_loss*0.5

def run_refinement(raw):
    refiner = DifferentiableRefiner(raw).to(device)
    opt = optim.Adam(refiner.parameters(), lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = loss_geometry(refiner())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        opt.step()
    return refiner.coords.detach().cpu().numpy()

# --- METRICS ---

def calc_lid_dist(coords, start, end): return np.linalg.norm(coords[start]-coords[end])
def calc_concavity(coords):
    com = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - com, axis=1)
    return ((dists[0]+dists[-1])/2.0) - np.mean(dists[len(coords)//3 : 2*len(coords)//3])
def calc_com_dist(c1, c2): return np.linalg.norm(np.mean(c1,axis=0) - np.mean(c2,axis=0))

# --- PDB EXPORT ---
def save_pdb(sequence, coords, filename):
    mol = Chem.MolFromSequence(sequence)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    conf = mol.GetConformer()
    # Align C-alphas
    ca_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetPDBResidueInfo().GetName().strip()=="CA"]
    for i, pos_idx in enumerate(ca_atoms):
        if i >= len(coords): break
        conf.SetAtomPosition(pos_idx, Point3D(float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
    Chem.PDBWriter(filename).write(mol)

# --- MAIN ENGINE ---

def main():
    parser = argparse.ArgumentParser(description="SIO-TRINITY v3.1")
    parser.add_argument("--ph", type=float, default=7.4, help="Environment pH")
    parser.add_argument("--mode", default="fold", choices=["fold", "unfold"])
    parser.add_argument("--output", default="sio_trinity_out.pdb")
    args = parser.parse_args()
    
    # 1. Assemble the Trinity
    full_seq = SEQ_A_KRAS + LINKER + SEQ_B_EGFR + LINKER + SEQ_C_BRAF
    lenA, lenB, lenC = len(SEQ_A_KRAS), len(SEQ_B_EGFR), len(SEQ_C_BRAF)
    lenL = len(LINKER)
    
    # Cut indices [End of A, End of B] (Including linkers in the blocks for simplicity, or separate?)
    # For physics blocks, let's group Linker 1 with A, Linker 2 with B.
    cuts = [lenA + lenL, lenA + lenL + lenB + lenL] 
    
    log.info(f"--- SIO-TRINITY v3.1 ---")
    log.info(f"Target: KRAS/EGFR/BRAF Chimeric Prion")
    log.info(f"Length: {len(full_seq)} residues")
    log.info(f"State:  {'BLOODSTREAM (Safe)' if args.ph > 7.0 else 'TUMOR (Lethal)'}")
    
    # 2. Initialize Genome (Standardized for v1 Candidate)
    # In a full run, we evolve these. Here we use the "Converged" params from the research session.
    # High Delta for Tumor (0.5), Low for Blood (0.25)
    
    # We use random initialization + the pH trigger handles the physics
    controls = [
        {'delta': np.random.uniform(0.2,0.3,8), 'alpha': np.random.uniform(0.4,0.6,8)}, # A
        {'delta': np.random.uniform(0.2,0.3,6), 'alpha': np.random.uniform(0.4,0.6,6)}, # B
        {'delta': np.random.uniform(0.2,0.3,6), 'alpha': np.random.uniform(0.4,0.6,6)}  # C
    ]
    theta = 0.0
    scaling = 3.8
    
    # 3. Run SIO Kernel
    log.info("Computing Spectral Manifold...")
    raw_coords = get_trimer_embedding(full_seq, cuts, controls, theta, scaling, args.ph)
    
    # 4. Refine Geometry
    log.info("Refining Physics (PyTorch)...")
    final_coords = run_refinement(raw_coords)
    
    # 5. Calculate Diagnostics
    # Slices
    cA = final_coords[0:lenA]
    cB = final_coords[cuts[0]:cuts[0]+lenB]
    cC = final_coords[cuts[1]:cuts[1]+lenC]
    
    lid_dist = calc_lid_dist(cA, 5, 20)      # KRAS Lid
    cup_conc = calc_concavity(cB)            # EGFR Cup
    wedge_dist = calc_lid_dist(cC, 10, 30)   # BRAF Wedge
    
    dist_AB = calc_com_dist(cA, cB)
    dist_BC = calc_com_dist(cB, cC)
    dist_AC = calc_com_dist(cA, cC)
    
    log.info(f"--- METRICS ---")
    log.info(f"KRAS Lid Openness: {lid_dist:.1f} A (Target: <12 Safe, >20 Lethal)")
    log.info(f"EGFR Cup Depth:    {cup_conc:.1f} A (Target: >5 Safe, <0 Lethal)")
    log.info(f"BRAF Wedge Dist:   {wedge_dist:.1f} A")
    log.info(f"Inter-Chain Dist:  {np.mean([dist_AB, dist_BC, dist_AC]):.1f} A (Target: >30 Safe, <10 Lethal)")
    
    # 6. Export
    save_pdb(full_seq, final_coords, args.output)
    log.info(f"Structure saved to {args.output}")

if __name__ == "__main__":
    main()
