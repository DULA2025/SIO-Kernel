"""
SIO-Fold v5.1: Myoglobin Fixed (Expansion Patch)
Status: SCALING FIX APPLIED
Date: December 2025
Author: DULA2025

Fixes:
- Increased Lattice Scaling (6.0 - 8.0) to prevent initial atomic overlap (NaNs).
- RDKit Builder made more permissive for large chains.
"""

import torch
import numpy as np
from scipy.interpolate import CubicSpline
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import openmm as mm
import openmm.app as app
from openmm import unit
import os
import logging
import argparse
import warnings
import sys

# --- 0. SETUP ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("SIO-Myo-Fix")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import pdbfixer
    HAS_PDBFIXER = True
    log.info(">> PDBFixer Detected: ENABLED")
except ImportError:
    HAS_PDBFIXER = False
    log.info(">> PDBFixer Not Found: Using OpenMM Modeller Fallback")

# Viscosity Field
MJ_TENSOR_RAW = torch.tensor([
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
], device=device)
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

# --- 1. SIO TOPOLOGY ENGINE ---
def get_mj_tensor(sequence):
    n = len(sequence)
    indices = torch.tensor([AA_ORDER.index(aa) for aa in sequence], device=device)
    grid_x, grid_y = torch.meshgrid(indices, indices, indexing='ij')
    M_vac = torch.exp(-MJ_TENSOR_RAW[grid_x, grid_y] / 200.0) 
    return M_vac

def solve_sio_flow(sequence, control_points, theta, scaling):
    n = len(sequence)
    x_nodes = np.linspace(0, n - 1, len(control_points['delta']))
    x_dense = np.arange(n)
    
    delta_field = CubicSpline(x_nodes, control_points['delta'])(x_dense)
    alpha_field = CubicSpline(x_nodes, control_points['alpha'])(x_dense)
    
    delta_t = torch.tensor(delta_field, device=device, dtype=torch.float32)
    alpha_t = torch.tensor(alpha_field, device=device, dtype=torch.float32)

    idx = torch.arange(n, device=device)
    i, j = torch.meshgrid(idx, idx, indexing='ij')
    d = torch.abs(i - j).float()
    
    delta_avg = (delta_t[i] + delta_t[j]) / 2.0
    alpha_avg = (alpha_t[i] + alpha_t[j]) / 2.0
    
    sigma_g = torch.log(torch.tensor(n)) / torch.sqrt(2.0 * delta_avg)
    sigma_c = torch.log(torch.tensor(n))
    
    decay = (1.0 - alpha_avg) * torch.exp(-d**2 / (2 * sigma_g**2)) + \
            alpha_avg * (1.0 / (1.0 + (d / sigma_c)**2))
            
    resonance = torch.cos(2 * np.pi * delta_avg * d + theta)
    lattice = 1.0 + torch.sin(2 * np.pi * d / 6.0)
    M_vac = get_mj_tensor(sequence)
    
    K = decay * resonance * lattice * M_vac
    K.fill_diagonal_(1.0)
    
    eigvals, eigvecs = torch.linalg.eigh(K.double())
    idx_sorted = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx_sorted].float()
    eigvals = eigvals[idx_sorted].float()
    
    valid = eigvals > 1e-4
    if valid.sum() < 3:
        coords = eigvecs[:, :3] @ torch.diag(torch.sqrt(torch.abs(eigvals[:3])))
    else:
        coords = eigvecs[:, valid][:, :3] @ torch.diag(torch.sqrt(eigvals[valid][:3]))
        
    coords_np = coords.cpu().numpy()
    if n > 1:
        bond_lengths = np.linalg.norm(np.diff(coords_np, axis=0), axis=1)
        avg_bond = np.mean(bond_lengths)
        if avg_bond > 1e-6:
            coords_np *= (scaling / avg_bond)
            
    return coords_np

# --- 2. RDKIT BUILDER ---
def build_atomic_model(sequence, coordinates, output_file):
    mol = Chem.MolFromSequence(sequence)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.maxIterations = 50
    res = AllChem.EmbedMolecule(mol, params)
    
    if res == -1:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(0,0,0))
        mol.AddConformer(conf, assignId=True)
    
    pdb_block = Chem.MolToPDBBlock(mol)
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    conf = mol.GetConformer()
    
    ca_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetPDBResidueInfo() and a.GetPDBResidueInfo().GetName().strip() == "CA"]
    limit = min(len(ca_atoms), len(coordinates))
    
    for k in range(limit):
        idx = ca_atoms[k]
        pos = conf.GetAtomPosition(idx)
        target = Point3D(float(coordinates[k][0]), float(coordinates[k][1]), float(coordinates[k][2]))
        diff = target - pos
        res_info = mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        res_id = res_info.GetResidueNumber()
        res_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetPDBResidueInfo() and a.GetPDBResidueInfo().GetResidueNumber() == res_id]
        
        for atom_id in res_atoms:
            curr = conf.GetAtomPosition(atom_id)
            conf.SetAtomPosition(atom_id, Point3D(curr.x + diff.x, curr.y + diff.y, curr.z + diff.z))
    
    mol_stripped = Chem.RemoveHs(mol, sanitize=False)
    Chem.PDBWriter(output_file).write(mol_stripped)

# --- 3. PHYSICS ENGINE ---
def run_vacuum_physics(pdb_path, output_path):
    try:
        forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
        
        if HAS_PDBFIXER:
            fixer = pdbfixer.PDBFixer(filename=pdb_path)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            topology = fixer.topology
            positions = fixer.positions
        else:
            pdb = app.PDBFile(pdb_path)
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.addHydrogens(forcefield, pH=7.0)
            topology = modeller.topology
            positions = modeller.positions

        system = forcefield.createSystem(topology, 
                                       nonbondedMethod=app.CutoffNonPeriodic,
                                       nonbondedCutoff=2.0*unit.nanometer,
                                       constraints=app.HBonds, 
                                       rigidWater=True)
        
        friction_std = 4.0/unit.picosecond
        integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, friction_std, 2.0*unit.femtoseconds)
        
        simulation = app.Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
        
        simulation.minimizeEnergy(maxIterations=1500)
        
        state = simulation.context.getState(getEnergy=True)
        energy_initial = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        
        n_res = len(list(topology.residues()))
        trap_threshold = -5.0 * n_res 
        energy_final = energy_initial
        
        if energy_initial > trap_threshold:
            integrator.setFriction(0.01/unit.picosecond)
            simulation.step(5000)
            integrator.setFriction(friction_std)
            simulation.minimizeEnergy(maxIterations=1500)
            state = simulation.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        
        with open(output_path, 'w') as f:
            app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
            
        return energy_final

    except Exception as e:
        return 99999.0 

# --- 4. EVOLUTION ---
class VortexState:
    def __init__(self, n_nodes):
        self.delta = np.random.uniform(0.2, 0.45, n_nodes) 
        self.alpha = np.random.uniform(0.2, 0.8, n_nodes) 
        self.theta = np.random.uniform(-np.pi, np.pi)     
        
        # --- THE FIX ---
        # Scaling increased to 6.0 - 8.0 Angstroms
        # This creates a "Fluffy" initial state that won't clash
        self.scaling = np.random.uniform(6.0, 8.0)        
        
        self.energy = float('inf')
        self.pdb = ""

    def mutate(self):
        if np.random.rand() < 0.5:
            idx = np.random.randint(len(self.delta))
            self.delta[idx] = np.clip(self.delta[idx] + np.random.normal(0, 0.05), 0.15, 0.5)
            self.alpha[idx] = np.clip(self.alpha[idx] + np.random.normal(0, 0.05), 0.1, 0.9)
        else:
            self.theta += np.random.normal(0, 0.1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDMASNYKELGFQG")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--nodes", type=int, default=24) 
    args = parser.parse_args()
    
    log.info(f"--- SIO-FOLD v5.1 (Expansion Fix) ---")
    log.info(f"Target Length: {len(args.sequence)} Residues")
    
    pop_size = 12
    population = [VortexState(args.nodes) for _ in range(pop_size)]
    best_state = None
    
    tmp_dir = "sio_workspace"
    os.makedirs(tmp_dir, exist_ok=True)
    
    for step in range(args.steps):
        for i, vortex in enumerate(population):
            if step > 0 and vortex != best_state:
                vortex.mutate()
            
            controls = {'delta': vortex.delta, 'alpha': vortex.alpha}
            coords = solve_sio_flow(args.sequence, controls, vortex.theta, vortex.scaling)
            
            raw_pdb = os.path.join(tmp_dir, f"step_{step}_vortex_{i}.pdb")
            final_pdb = os.path.join(tmp_dir, f"step_{step}_vortex_{i}_relaxed.pdb")
            
            try:
                build_atomic_model(args.sequence, coords, raw_pdb)
                energy = run_vacuum_physics(raw_pdb, final_pdb)
                
                if np.isnan(energy): energy = 99999.0
                
                vortex.energy = energy
                vortex.pdb = final_pdb
                
                if best_state is None or energy < best_state.energy:
                    best_state = vortex
                    milestone = f"SIO_MYO_E{energy:.0f}.pdb"
                    import shutil
                    shutil.copy(final_pdb, milestone)
                    log.info(f"STEP {step}: Globin Attractor (E={energy:.2f}) -> {milestone}")

            except Exception as e:
                pass
                
        population.sort(key=lambda x: x.energy)
        half = pop_size // 2
        for k in range(half, pop_size):
            parent = population[k-half]
            child = VortexState(args.nodes)
            child.delta = parent.delta.copy()
            child.alpha = parent.alpha.copy()
            child.theta = parent.theta
            child.scaling = parent.scaling
            population[k] = child

    log.info(f"--- SIMULATION COMPLETE ---")

if __name__ == "__main__":
    main()
