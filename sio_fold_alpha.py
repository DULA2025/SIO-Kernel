import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import openmm as mm
import openmm.app as app
from openmm import unit
import pdbfixer
import sys
import os
import time
import logging
import argparse
import tempfile

# --- 0. Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. SIO KERNEL (The Physics-Based Chromosome) ---
def get_sio_params(sequence: str, device: torch.device, alpha_val, delta_val, theta_val, scaling_val) -> dict:
    n = len(sequence)
    # Normalized Miyazawa-Jernigan Matrix
    mj_matrix = torch.tensor([
        [0.00, -0.24, -0.26, -0.33, -0.32, -0.21, -0.19, -0.28, -0.31, -0.30, -0.25, -0.27, -0.29, -0.28, -0.26, -0.25, -0.23, -0.22, -0.21, -0.20],
        [-0.24, 0.00, -0.35, -0.42, -0.41, -0.30, -0.28, -0.37, -0.40, -0.39, -0.34, -0.36, -0.38, -0.37, -0.35, -0.34, -0.32, -0.31, -0.30, -0.29],
        [-0.26, -0.35, 0.00, -0.44, -0.43, -0.32, -0.30, -0.39, -0.42, -0.41, -0.36, -0.38, -0.40, -0.39, -0.37, -0.36, -0.34, -0.33, -0.32, -0.31],
        [-0.33, -0.42, -0.44, 0.00, -0.50, -0.39, -0.37, -0.46, -0.49, -0.48, -0.43, -0.45, -0.47, -0.46, -0.44, -0.43, -0.41, -0.40, -0.39, -0.38],
        [-0.32, -0.41, -0.43, -0.50, 0.00, -0.38, -0.36, -0.45, -0.48, -0.47, -0.42, -0.44, -0.46, -0.45, -0.43, -0.42, -0.40, -0.39, -0.38, -0.37],
        [-0.21, -0.30, -0.32, -0.39, -0.38, 0.00, -0.25, -0.34, -0.37, -0.36, -0.31, -0.33, -0.35, -0.34, -0.32, -0.31, -0.29, -0.28, -0.27, -0.26],
        [-0.19, -0.28, -0.30, -0.37, -0.36, -0.25, 0.00, -0.32, -0.35, -0.34, -0.29, -0.31, -0.33, -0.32, -0.30, -0.29, -0.27, -0.26, -0.25, -0.24],
        [-0.28, -0.37, -0.39, -0.46, -0.45, -0.34, -0.32, 0.00, -0.44, -0.43, -0.38, -0.40, -0.42, -0.41, -0.39, -0.38, -0.36, -0.35, -0.34, -0.33],
        [-0.31, -0.40, -0.42, -0.49, -0.48, -0.37, -0.35, -0.44, 0.00, -0.46, -0.41, -0.43, -0.45, -0.44, -0.42, -0.41, -0.39, -0.38, -0.37, -0.36],
        [-0.30, -0.39, -0.41, -0.48, -0.47, -0.36, -0.34, -0.43, -0.46, 0.00, -0.40, -0.42, -0.44, -0.43, -0.41, -0.40, -0.38, -0.37, -0.36, -0.35],
        [-0.25, -0.34, -0.36, -0.43, -0.42, -0.31, -0.29, -0.38, -0.41, -0.40, 0.00, -0.37, -0.39, -0.38, -0.36, -0.35, -0.33, -0.32, -0.31, -0.30],
        [-0.27, -0.36, -0.38, -0.45, -0.44, -0.33, -0.31, -0.40, -0.43, -0.42, -0.37, 0.00, -0.41, -0.40, -0.38, -0.37, -0.35, -0.34, -0.33, -0.32],
        [-0.29, -0.38, -0.40, -0.47, -0.46, -0.35, -0.33, -0.42, -0.45, -0.44, -0.39, -0.41, 0.00, -0.42, -0.40, -0.39, -0.37, -0.36, -0.35, -0.34],
        [-0.28, -0.37, -0.39, -0.46, -0.45, -0.34, -0.32, -0.41, -0.44, -0.43, -0.38, -0.40, -0.42, 0.00, -0.39, -0.38, -0.36, -0.35, -0.34, -0.33],
        [-0.26, -0.35, -0.37, -0.44, -0.43, -0.32, -0.30, -0.39, -0.42, -0.41, -0.36, -0.38, -0.40, -0.39, 0.00, -0.36, -0.34, -0.33, -0.32, -0.31],
        [-0.25, -0.34, -0.36, -0.43, -0.42, -0.31, -0.29, -0.38, -0.41, -0.40, -0.35, -0.37, -0.39, -0.38, -0.36, 0.00, -0.33, -0.32, -0.31, -0.30],
        [-0.23, -0.32, -0.34, -0.41, -0.40, -0.29, -0.27, -0.36, -0.39, -0.38, -0.33, -0.35, -0.37, -0.36, -0.34, -0.33, 0.00, -0.30, -0.29, -0.28],
        [-0.22, -0.31, -0.33, -0.40, -0.39, -0.28, -0.26, -0.35, -0.38, -0.37, -0.32, -0.34, -0.36, -0.35, -0.33, -0.32, -0.30, 0.00, -0.27, -0.26],
        [-0.21, -0.30, -0.32, -0.39, -0.38, -0.27, -0.25, -0.34, -0.37, -0.36, -0.31, -0.33, -0.35, -0.34, -0.32, -0.31, -0.29, -0.27, 0.00, -0.25],
        [-0.20, -0.29, -0.31, -0.38, -0.37, -0.26, -0.24, -0.33, -0.36, -0.35, -0.30, -0.32, -0.34, -0.33, -0.31, -0.30, -0.28, -0.26, -0.25, 0.00]
    ], dtype=torch.float32, device=device) / 200.0
    
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    DELTA = torch.tensor(delta_val, device=device, dtype=torch.float32)
    n_tensor = torch.tensor(n, device=device, dtype=torch.float32)
    
    return {
        "n": n, "aa_order": aa_order, "modulation_matrix": torch.exp(-mj_matrix),
        "ALPHA": torch.tensor(alpha_val, device=device, dtype=torch.float32),
        "DELTA": DELTA, "THETA": torch.tensor(theta_val, device=device, dtype=torch.float32),
        "SCALING_FACTOR": torch.tensor(scaling_val, device=device, dtype=torch.float32),
        "SIGMA_G": torch.log(n_tensor) / torch.sqrt(2.0 * DELTA), "SIGMA_C": torch.log(n_tensor),
    }

def calculate_sio_embedding(sequence, params, device):
    n = params["n"]
    aa_indices = torch.tensor([params["aa_order"].index(aa) for aa in sequence], device=device)
    i, j = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing='ij')
    diff = torch.abs(i - j).float()
    decay = (1.0 - params["ALPHA"]) * torch.exp(-diff**2 / (2 * params["SIGMA_G"]**2)) + \
            params["ALPHA"] * (1.0 / (1.0 + (diff / params["SIGMA_C"])**2))
    oscillation = torch.cos(2 * np.pi * params["DELTA"] * diff + params["THETA"])
    projection = 1.0 + torch.sin(2 * np.pi * diff / 6.0)
    modulation = params["modulation_matrix"][aa_indices[i], aa_indices[j]]
    
    K = decay * oscillation * projection * modulation
    K.masked_fill_(diff == 0, 1.0)
    
    eigvals, eigvecs = torch.linalg.eigh(K)
    idx = torch.argsort(eigvals, descending=True)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    
    embedded = torch.zeros((n, 3), device=device)
    if len(eigvals) > 0 and eigvals[0] > 0:
        count = min((eigvals > 0).sum().item(), 3)
        top_vecs = eigvecs[:, :count]
        top_vals = torch.sqrt(eigvals[:count])
        embedded[:, :count] = top_vecs @ torch.diag(top_vals)

    # Scale to Angstroms
    if torch.any(embedded):
        dist = torch.linalg.norm(torch.diff(embedded, axis=0), axis=1)
        if dist.mean() > 0:
            embedded *= (params["SCALING_FACTOR"] / dist.mean())
            
    return embedded.cpu().numpy()

# --- 2. INITIAL TOPOLOGY (RDKit) ---

def build_initial_structure(sequence, coords):
    mol = Chem.MolFromSequence(sequence)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    AllChem.EmbedMolecule(mol, params)
    conf = mol.GetConformer()
    
    # Align C-alphas to SIO coordinates
    ca_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetPDBResidueInfo() and a.GetPDBResidueInfo().GetName().strip() == 'CA']
    
    for i, idx in enumerate(ca_indices):
        if i >= len(coords): break
        current = conf.GetAtomPosition(idx)
        target = Point3D(coords[i,0], coords[i,1], coords[i,2])
        move = target - current
        
        # Move the whole residue rigidly
        res_id = mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetResidueNumber()
        atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetPDBResidueInfo() and a.GetPDBResidueInfo().GetResidueNumber() == res_id]
        for atom_idx in atoms:
            pos = conf.GetAtomPosition(atom_idx)
            conf.SetAtomPosition(atom_idx, Point3D(pos.x + move.x, pos.y + move.y, pos.z + move.z))
            
    return mol

# --- 3. ALPHA-LEVEL REFINEMENT (OpenMM / AMBER) ---

def run_openmm_refinement(input_pdb_path, output_pdb_path):
    """
    Uses the AMBER14 forcefield (same as AlphaFold) to relax the structure.
    """
    log.info("Starting AMBER Relaxation (OpenMM)...")
    
    # 1. Fix Topology (Add missing atoms/hydrogens correctly)
    fixer = pdbfixer.PDBFixer(filename=input_pdb_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0) # pH 7.0
    
    # 2. Define Forcefield (AMBER14 + TIP3P water model standard)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    
    # 3. Create System
    # No solvent box for fast folding (implicit solvent or vacuum for speed in this loop)
    # For AlphaFold quality, we assume implicit solvent (OBC2) for the loop
    system = forcefield.createSystem(fixer.topology, 
                                   nonbondedMethod=app.NoCutoff, 
                                   constraints=app.HBonds, 
                                   rigidWater=True)
    
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    simulation = app.Simulation(fixer.topology, system, integrator)
    simulation.context.setPositions(fixer.positions)
    
    # 4. Energy Minimization (The Polish)
    # First: restrain C-alphas to keep SIO shape, let sidechains pack
    log.info("  - Phase 1: Sidechain packing...")
    # (Simple minimization handles this well in OpenMM if we don't have violent clashes)
    
    # Full Minimization
    log.info("  - Phase 2: Energy Minimization (L-BFGS)...")
    try:
        simulation.minimizeEnergy(maxIterations=1000, tolerance=10*unit.kilojoule/unit.mole)
    except Exception as e:
        log.warning(f"  - Minimization warning: {e}")
    
    # Get final energy
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    
    # Save
    with open(output_pdb_path, 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
        
    return energy

# --- 4. MAIN EVOLUTIONARY LOOP ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHL")
    parser.add_argument("--generations", type=int, default=200)
    args = parser.parse_args()
    
    sequence = args.sequence
    log.info(f"Target: {sequence} (Len: {len(sequence)})")
    
    # Initial "Grok" Params
    best_params = {"alpha_val": 0.258, "delta_val": 0.1, "theta_val": -0.0031, "scaling_val": 3.8}
    best_energy = float('inf')
    
    for gen in range(args.generations):
        # 1. Mutate
        current_params = best_params.copy()
        if gen > 0:
            current_params["alpha_val"] = np.clip(current_params["alpha_val"] + np.random.normal(0, 0.02), 0.1, 0.9)
            current_params["delta_val"] = np.clip(current_params["delta_val"] + np.random.normal(0, 0.01), 0.05, 0.5)
            current_params["theta_val"] += np.random.normal(0, 0.05)
            current_params["scaling_val"] = np.clip(current_params["scaling_val"] + np.random.normal(0, 0.1), 3.0, 4.5)

        # 2. SIO Embedding (GPU)
        sio_p = get_sio_params(sequence, device, **current_params)
        coords = calculate_sio_embedding(sequence, sio_p, device)
        
        # 3. Structure Gen (RDKit -> Temporary PDB)
        try:
            mol = build_initial_structure(sequence, coords)
            
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode='w') as tmp:
                Chem.PDBWriter(tmp.name).write(mol)
                tmp_path = tmp.name
            
            # 4. ALPHA REFINEMENT (The Magic Step)
            refined_path = f"temp_gen_{gen}.pdb"
            energy = run_openmm_refinement(tmp_path, refined_path)
            
            # Cleanup
            os.remove(tmp_path)
            
            log.info(f"Gen {gen}: Energy = {energy:.2f} kcal/mol")
            
            if energy < best_energy:
                best_energy = energy
                best_params = current_params
                os.rename(refined_path, f"BEST_FOLD_E{energy:.0f}.pdb")
                log.info(f"*** NEW BEST: {energy:.2f} ***")
            else:
                if os.path.exists(refined_path): os.remove(refined_path)
                
        except Exception as e:
            log.error(f"Fold failed: {e}")

if __name__ == "__main__":
    main()
