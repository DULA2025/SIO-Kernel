"""
SIO-Fold v6.0: Quaternary Assembly Engine
Status: MULTI-CHAIN / DOCKING ENABLED
Date: December 2025
Author: DULA2025

Updates:
- Supports multi-chain input (delimited by ':').
- "Staging Area" logic to spawn chains in close proximity.
- Physics Engine updated to dock separate molecules.
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
log = logging.getLogger("SIO-Quat")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PDBFixer Check
try:
    import pdbfixer
    HAS_PDBFIXER = True
    log.info(">> PDBFixer Detected: ENABLED")
except ImportError:
    HAS_PDBFIXER = False
    log.info(">> PDBFixer Not Found: Using OpenMM Modeller Fallback")

# Viscosity Field (Miyazawa-Jernigan)
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

# --- 2. MULTI-CHAIN BUILDER ---
def build_quaternary_system(sequences, coords_list, output_file):
    """
    Builds multiple chains and places them in a shared simulation box.
    """
    combined_modeller = None
    forcefield = app.ForceField('amber14-all.xml') # Temp load for addHydrogens check
    
    # Staging Offsets (to prevent overlap at spawn)
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([25.0, 0.0, 0.0]), # Chain B shifted X
        np.array([0.0, 25.0, 0.0]), # Chain C shifted Y
        np.array([0.0, 0.0, 25.0]), # Chain D shifted Z
        np.array([25.0, 25.0, 0.0]) # Chain E...
    ]

    temp_pdbs = []

    for i, (seq, coords) in enumerate(zip(sequences, coords_list)):
        # 1. Build Single Chain RDKit
        mol = Chem.MolFromSequence(seq)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.maxIterations = 50
        res = AllChem.EmbedMolecule(mol, params)
        
        if res == -1:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for a in range(mol.GetNumAtoms()): conf.SetAtomPosition(a, Point3D(0,0,0))
            mol.AddConformer(conf, assignId=True)
            
        pdb_block = Chem.MolToPDBBlock(mol)
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
        conf = mol.GetConformer()
        
        # 2. Align to SIO
        ca_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetPDBResidueInfo().GetName().strip() == "CA"]
        limit = min(len(ca_atoms), len(coords))
        
        offset_vec = offsets[i % len(offsets)]
        
        for k in range(limit):
            idx = ca_atoms[k]
            pos = conf.GetAtomPosition(idx)
            # Apply SIO Coords + Staging Offset
            target = np.array(coords[k]) + offset_vec
            target_pt = Point3D(target[0], target[1], target[2])
            
            diff = target_pt - pos
            res_id = mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetResidueNumber()
            res_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetPDBResidueInfo().GetResidueNumber() == res_id]
            
            for atom_id in res_atoms:
                curr = conf.GetAtomPosition(atom_id)
                conf.SetAtomPosition(atom_id, Point3D(curr.x + diff.x, curr.y + diff.y, curr.z + diff.z))
        
        # 3. Save Temp Skeleton
        mol_stripped = Chem.RemoveHs(mol, sanitize=False)
        temp_name = f"temp_chain_{i}.pdb"
        Chem.PDBWriter(temp_name).write(mol_stripped)
        temp_pdbs.append(temp_name)

    # 4. Merge into one Topology using Modeller
    for i, temp_pdb in enumerate(temp_pdbs):
        pdb = app.PDBFile(temp_pdb)
        if combined_modeller is None:
            combined_modeller = app.Modeller(pdb.topology, pdb.positions)
        else:
            combined_modeller.add(pdb.topology, pdb.positions)
        os.remove(temp_pdb)

    # 5. Fix Hydrogens Globally
    combined_modeller.addHydrogens(forcefield, pH=7.0)
    
    # 6. Save Combined Start
    with open(output_file, 'w') as f:
        app.PDBFile.writeFile(combined_modeller.topology, combined_modeller.positions, f)

# --- 3. PHYSICS ENGINE (Docking Enabled) ---
def run_vacuum_physics(pdb_path, output_path):
    try:
        pdb = app.PDBFile(pdb_path)
        forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
        
        system = forcefield.createSystem(pdb.topology, 
                                       nonbondedMethod=app.CutoffNonPeriodic,
                                       nonbondedCutoff=2.0*unit.nanometer,
                                       constraints=app.HBonds, 
                                       rigidWater=True)
        
        # Medium Friction to allow docking movement
        friction_std = 2.0/unit.picosecond
        integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, friction_std, 2.0*unit.femtoseconds)
        
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        
        # 1. Relaxation (Individual Folding)
        simulation.minimizeEnergy(maxIterations=500)
        
        # 2. Docking Phase (Long Dynamics)
        # We run steps to allow the chains to drift together via electrostatics
        simulation.step(5000) 
        
        # 3. Final Tightening
        simulation.minimizeEnergy(maxIterations=1000)
        
        state = simulation.context.getState(getEnergy=True)
        energy_final = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        
        with open(output_path, 'w') as f:
            app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
            
        return energy_final

    except Exception as e:
        log.warning(f"Physics Error: {e}")
        return 99999.0 

# --- 4. EVOLUTION ---
class VortexState:
    def __init__(self, n_nodes, n_chains):
        # We need independent parameters for EACH chain? 
        # Simplified: Shared physics, but solution differs by sequence
        self.delta = np.random.uniform(0.2, 0.45, n_nodes) 
        self.alpha = np.random.uniform(0.2, 0.8, n_nodes) 
        self.theta = np.random.uniform(-np.pi, np.pi)     
        self.scaling = np.random.uniform(4.0, 5.0)        
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
    # Accept multiple sequences separated by colon
    parser.add_argument("--sequences", type=str, required=True, help="SEQ_A:SEQ_B:SEQ_C")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=12)
    args = parser.parse_args()
    
    seq_list = args.sequences.split(":")
    
    log.info(f"--- SIO-FOLD v6.0 (Quaternary) ---")
    log.info(f"Number of Chains: {len(seq_list)}")
    for i, s in enumerate(seq_list):
        log.info(f"  Chain {i+1}: {len(s)} residues")
    
    pop_size = 8 # Lower pop for complex docking
    population = [VortexState(args.nodes, len(seq_list)) for _ in range(pop_size)]
    best_state = None
    
    tmp_dir = "sio_workspace_quat"
    os.makedirs(tmp_dir, exist_ok=True)
    
    for step in range(args.steps):
        for i, vortex in enumerate(population):
            if step > 0 and vortex != best_state:
                vortex.mutate()
            
            # Solve SIO for EACH chain
            controls = {'delta': vortex.delta, 'alpha': vortex.alpha}
            all_coords = []
            
            for seq in seq_list:
                # Assuming shared manifold physics for the complex (Simplified)
                # In v7, we could evolve per-chain parameters
                c = solve_sio_flow(seq, controls, vortex.theta, vortex.scaling)
                all_coords.append(c)
            
            raw_pdb = os.path.join(tmp_dir, f"step_{step}_vortex_{i}.pdb")
            final_pdb = os.path.join(tmp_dir, f"step_{step}_vortex_{i}_docked.pdb")
            
            try:
                build_quaternary_system(seq_list, all_coords, raw_pdb)
                energy = run_vacuum_physics(raw_pdb, final_pdb)
                
                if np.isnan(energy): energy = 99999.0
                
                vortex.energy = energy
                vortex.pdb = final_pdb
                
                if best_state is None or energy < best_state.energy:
                    best_state = vortex
                    milestone = f"SIO_QUAT_E{energy:.0f}.pdb"
                    import shutil
                    shutil.copy(final_pdb, milestone)
                    log.info(f"STEP {step}: Complex Attractor (E={energy:.2f}) -> {milestone}")

            except Exception as e:
                pass
                
        population.sort(key=lambda x: x.energy)
        half = pop_size // 2
        for k in range(half, pop_size):
            parent = population[k-half]
            child = VortexState(args.nodes, len(seq_list))
            child.delta = parent.delta.copy()
            child.alpha = parent.alpha.copy()
            child.theta = parent.theta
            child.scaling = parent.scaling
            population[k] = child

    log.info(f"--- ASSEMBLY COMPLETE ---")

if __name__ == "__main__":
    main()