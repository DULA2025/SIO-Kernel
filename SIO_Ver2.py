import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import time
import logging

# --- 0. Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Running SIO model on device: {device}")

# --- 1. SIO PARAMETER FUNCTION (The "Chromosome") ---

def get_sio_params(sequence: str, 
                     device: torch.device, 
                     # --- These are the "genes" AlphaEvolve will control ---
                     alpha_val: float = 0.2526, 
                     delta_val: float = 1.0 / 3.6,
                     theta_val: float = 1.0 / 6.0,
                     scaling_val: float = 3.8
                     # ----------------------------------------------------
                    ) -> dict:
    """
    Encapsulates all SIO kernel parameters.
    The arguments are now the "genome" to be optimized by AlphaEvolve.
    """
    n = len(sequence)

    # Miyazawa-Jernigan contact potentials (scaled and normalized)
    mj_matrix = torch.tensor([
        [0.00, -0.24, -0.26, -0.33, -0.32, -0.21, -0.19, -0.28, -0.31, -0.30, -0.25, -0.27, -0.29, -0.28, -0.26, -0.25, -0.23, -0.22, -0.21, -0.20],  # A
        [-0.24, 0.00, -0.35, -0.42, -0.41, -0.30, -0.28, -0.37, -0.40, -0.39, -0.34, -0.36, -0.38, -0.37, -0.35, -0.34, -0.32, -0.31, -0.30, -0.29],  # R
        [-0.26, -0.35, 0.00, -0.44, -0.43, -0.32, -0.30, -0.39, -0.42, -0.41, -0.36, -0.38, -0.40, -0.39, -0.37, -0.36, -0.34, -0.33, -0.32, -0.31],  # N
        [-0.33, -0.42, -0.44, 0.00, -0.50, -0.39, -0.37, -0.46, -0.49, -0.48, -0.43, -0.45, -0.47, -0.46, -0.44, -0.43, -0.41, -0.40, -0.39, -0.38],  # D
        [-0.32, -0.41, -0.43, -0.50, 0.00, -0.38, -0.36, -0.45, -0.48, -0.47, -0.42, -0.44, -0.46, -0.45, -0.43, -0.42, -0.40, -0.39, -0.38, -0.37],  # C
        [-0.21, -0.30, -0.32, -0.39, -0.38, 0.00, -0.25, -0.34, -0.37, -0.36, -0.31, -0.33, -0.35, -0.34, -0.32, -0.31, -0.29, -0.28, -0.27, -0.26],  # Q
        [-0.19, -0.28, -0.30, -0.37, -0.36, -0.25, 0.00, -0.32, -0.35, -0.34, -0.29, -0.31, -0.33, -0.32, -0.30, -0.29, -0.27, -0.26, -0.25, -0.24],  # E
        [-0.28, -0.37, -0.39, -0.46, -0.45, -0.34, -0.32, 0.00, -0.44, -0.43, -0.38, -0.40, -0.42, -0.41, -0.39, -0.38, -0.36, -0.35, -0.34, -0.33],  # G
        [-0.31, -0.40, -0.42, -0.49, -0.48, -0.37, -0.35, -0.44, 0.00, -0.46, -0.41, -0.43, -0.45, -0.44, -0.42, -0.41, -0.39, -0.38, -0.37, -0.36],  # H
        [-0.30, -0.39, -0.41, -0.48, -0.47, -0.36, -0.34, -0.43, -0.46, 0.00, -0.40, -0.42, -0.44, -0.43, -0.41, -0.40, -0.38, -0.37, -0.36, -0.35],  # I
        [-0.25, -0.34, -0.36, -0.43, -0.42, -0.31, -0.29, -0.38, -0.41, -0.40, 0.00, -0.37, -0.39, -0.38, -0.36, -0.35, -0.33, -0.32, -0.31, -0.30],  # L
        [-0.27, -0.36, -0.38, -0.45, -0.44, -0.33, -0.31, -0.40, -0.43, -0.42, -0.37, 0.00, -0.41, -0.40, -0.38, -0.37, -0.35, -0.34, -0.33, -0.32],  # K
        [-0.29, -0.38, -0.40, -0.47, -0.46, -0.35, -0.33, -0.42, -0.45, -0.44, -0.39, -0.41, 0.00, -0.42, -0.40, -0.39, -0.37, -0.36, -0.35, -0.34],  # M
        [-0.28, -0.37, -0.39, -0.46, -0.45, -0.34, -0.32, -0.41, -0.44, -0.43, -0.38, -0.40, -0.42, 0.00, -0.39, -0.38, -0.36, -0.35, -0.34, -0.33],  # F
        [-0.26, -0.35, -0.37, -0.44, -0.43, -0.32, -0.30, -0.39, -0.42, -0.41, -0.36, -0.38, -0.40, -0.39, 0.00, -0.36, -0.34, -0.33, -0.32, -0.31],  # P
        [-0.25, -0.34, -0.36, -0.43, -0.42, -0.31, -0.29, -0.38, -0.41, -0.40, -0.35, -0.37, -0.39, -0.38, -0.36, 0.00, -0.33, -0.32, -0.31, -0.30],  # S
        [-0.23, -0.32, -0.34, -0.41, -0.40, -0.29, -0.27, -0.36, -0.39, -0.38, -0.33, -0.35, -0.37, -0.36, -0.34, -0.33, 0.00, -0.30, -0.29, -0.28],  # T
        [-0.22, -0.31, -0.33, -0.40, -0.39, -0.28, -0.26, -0.35, -0.38, -0.37, -0.32, -0.34, -0.36, -0.35, -0.33, -0.32, -0.30, 0.00, -0.27, -0.26],  # W
        [-0.21, -0.30, -0.32, -0.39, -0.38, -0.27, -0.25, -0.34, -0.37, -0.36, -0.31, -0.33, -0.35, -0.34, -0.32, -0.31, -0.29, -0.27, 0.00, -0.25],  # Y
        [-0.20, -0.29, -0.31, -0.38, -0.37, -0.26, -0.24, -0.33, -0.36, -0.35, -0.30, -0.32, -0.34, -0.33, -0.31, -0.30, -0.28, -0.26, -0.25, 0.00]   # V
    ], dtype=torch.float32, device=device) / 200.0  # Scaling factor
    
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    
    # Use the injected, evolvable parameters
    # Convert to tensors on the correct device
    DELTA = torch.tensor(delta_val, device=device, dtype=torch.float32)
    n_tensor = torch.tensor(n, device=device, dtype=torch.float32)
    
    # Return a dictionary of all parameters
    return {
        "n": n,
        "aa_order": aa_order,
        "modulation_matrix": torch.exp(-mj_matrix),
        "ALPHA": torch.tensor(alpha_val, device=device, dtype=torch.float32),
        "DELTA": DELTA,
        "THETA": torch.tensor(theta_val, device=device, dtype=torch.float32),
        "SCALING_FACTOR": torch.tensor(scaling_val, device=device, dtype=torch.float32),
        "SIGMA_G": torch.log(n_tensor) / torch.sqrt(2.0 * DELTA),
        "SIGMA_C": torch.log(n_tensor),
    }

# --- 2. SIO EMBEDDING CALCULATION (GPU-ACCELERATED) ---

def calculate_sio_embedding(sequence: str, params: dict, device: torch.device) -> np.ndarray:
    """
    Calculates the 3D C-alpha embedding from the sequence and parameters.
    Runs all tensor operations on the specified device (e.g., 'cuda').
    """
    n = params["n"]
    log.info(f"Building {n}x{n} kernel on {device}...")
    t0 = time.time()
    
    # Vectorized kernel matrix construction on GPU
    aa_indices = torch.tensor([params["aa_order"].index(aa) for aa in sequence], device=device)
    i, j = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing='ij')
    diff = torch.abs(i - j).float()
    mask_diag = (diff == 0)

    decay_g = torch.exp(-diff**2 / (2 * params["SIGMA_G"]**2))
    decay_c = 1.0 / (1.0 + (diff / params["SIGMA_C"])**2)
    decay = (1.0 - params["ALPHA"]) * decay_g + params["ALPHA"] * decay_c

    oscillation = torch.cos(2 * np.pi * params["DELTA"] * diff + params["THETA"])
    projection = 1.0 + torch.sin(2 * np.pi * diff / 6.0)

    aa_i = aa_indices[i]
    aa_j = aa_indices[j]
    modulation = params["modulation_matrix"][aa_i, aa_j]

    K = decay * oscillation * projection * modulation
    K.masked_fill_(mask_diag, 1.0) # Set diagonal to 1.0
    log.info(f"Kernel built in {time.time() - t0:.4f} s")

    # Eigen-decomposition (the slowest step, now on GPU)
    t0 = time.time()
    eigvals, eigvecs = torch.linalg.eigh(K)
    log.info(f"Eigendecomposition (GPU) in {time.time() - t0:.4f} s")

    # Filter, sort, and select top 3 eigenvectors
    positive_idx = eigvals > 1e-9
    positive_eigvals = eigvals[positive_idx]
    pos_eigvecs = eigvecs[:, positive_idx]
    
    idx = positive_eigvals.argsort(descending=True)
    positive_eigvals = positive_eigvals[idx]
    pos_eigvecs = pos_eigvecs[:, idx]

    # Embed in 3D and scale to angstroms
    if len(positive_eigvals) >= 3:
        sqrt_d = torch.sqrt(positive_eigvals[:3])
        embedded = pos_eigvecs[:, :3] @ torch.diag(sqrt_d)
    else:
        log.warning("Fewer than 3 positive eigenvalues found. Result may be 2D or 1D.")
        embedded = torch.zeros((n, 3), device=device)
        if len(positive_eigvals) > 0:
            top_vecs = pos_eigvecs[:, :min(len(positive_eigvals), 3)]
            top_vals = torch.sqrt(positive_eigvals[:min(len(positive_eigvals), 3)])
            embedded[:, :top_vecs.shape[1]] = top_vecs @ torch.diag(top_vals)

    # Scale to typical C-alpha distance (from evolved param)
    if torch.any(embedded):
        distances = torch.linalg.norm(torch.diff(embedded, axis=0), axis=1)
        if distances.mean() > 0:
            scale_factor = params["SCALING_FACTOR"] / distances.mean()
            embedded *= scale_factor

    # Return as a NumPy array for RDKit (which runs on CPU)
    return embedded.cpu().numpy().astype(float)

# --- 3. RDKIT MOLECULE GENERATION AND REFINEMENT ---

def build_pdb_from_embedding(sequence: str, embedded_coords: np.ndarray) -> (Chem.Mol, float, np.ndarray):
    """
    Builds a full atomic model using RDKit, constrained to the SIO embedding.
    Returns the RDKit molecule, final energy, and final C-alpha positions.
    """
    log.info("Building RDKit molecule...")
    n = len(sequence)
    peptide = Chem.MolFromSequence(sequence)
    if peptide is None:
        raise ValueError("Failed to create molecule from sequence.")
    peptide = Chem.AddHs(peptide)

    # Generate an initial 3D conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42 # Keep seed constant for reproducibility
    params.useRandomCoords = True
    params.maxIterations = 500
    conf_id = AllChem.EmbedMolecule(peptide, params)
    if conf_id < 0:
        log.warning("RDKit EmbedMolecule failed, creating blank conformer.")
        conf = Chem.Conformer(peptide.GetNumAtoms())
        conf_id = peptide.AddConformer(conf, assignId=True)
    
    conf = peptide.GetConformer(conf_id)

    # Get Cα indices
    ca_indices = [a.GetIdx() for a in peptide.GetAtoms() if a.GetPDBResidueInfo() and a.GetPDBResidueInfo().GetName().strip() == 'CA']
    if len(ca_indices) != n:
        raise ValueError(f"CRITICAL ERROR: C-alpha count mismatch. Expected {n}, found {len(ca_indices)}.")

    # Rigidly translate each residue to align its CA with the SIO embedding position
    log.info("Translating residues to SIO C-alpha coordinates...")
    for i in range(n):
        res_num = i + 1
        res_atoms = [a.GetIdx() for a in peptide.GetAtoms() if a.GetPDBResidueInfo() and a.GetPDBResidueInfo().GetResidueNumber() == res_num]
        ca_idx = ca_indices[i]
        
        initial_ca_pos = conf.GetAtomPosition(ca_idx)
        new_ca_pos = Point3D(embedded_coords[i, 0], embedded_coords[i, 1], embedded_coords[i, 2])
        
        delta_x = new_ca_pos.x - initial_ca_pos.x
        delta_y = new_ca_pos.y - initial_ca_pos.y
        delta_z = new_ca_pos.z - initial_ca_pos.z
        
        for atom_idx in res_atoms:
            old_pos = conf.GetAtomPosition(atom_idx)
            new_pos = Point3D(old_pos.x + delta_x, old_pos.y + delta_y, old_pos.z + delta_z)
            conf.SetAtomPosition(atom_idx, new_pos)

    # --- 4. CONSTRAINED ENERGY MINIMIZATION (THE "EVALUATOR") ---
    log.info("Running constrained energy minimization (MMFF)...")
    t0 = time.time()
    props = AllChem.MMFFGetMoleculeProperties(peptide)
    ff = AllChem.MMFFGetMoleculeForceField(peptide, props, confId=conf_id)
    ff.Initialize()

    for idx in ca_indices:
        ff.MMFFAddPositionConstraint(idx, 1.0, 500.0) 

    ff.Minimize(maxIts=2000)
    final_energy = ff.CalcEnergy() # This is the "cost"
    log.info(f"Minimization complete in {time.time() - t0:.2f} s")

    final_ca_positions = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in ca_indices])
    
    return peptide, final_energy, final_ca_positions

# --- 5. VISUALIZATION AND SAVING (FOR BEST RESULT) ---

def visualize_and_save_best(sequence: str, initial_embedding: np.ndarray, final_positions: np.ndarray, final_energy: float, peptide_mol: Chem.Mol):
    """
    Saves the PDB file and generates the comparison plot for the BEST fold found.
    """
    # --- Save PDB ---
    pdb_filename = 'sio_BEST_FOLD.pdb'
    try:
        with Chem.PDBWriter(pdb_filename) as writer:
            writer.write(peptide_mol)
        log.info(f"*** Best PDB file saved as '{pdb_filename}' ***")
    except Exception as e:
        log.error(f"Failed to save PDB file: {e}")

    # --- Plotting ---
    log.info("Generating final visualization...")
    fig = plt.figure(figsize=(14, 6))
    
    colors = ['blue' if aa == 'A' else 'red' for aa in sequence]

    # Plot 1: The initial SIO embedding
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(initial_embedding[:, 0], initial_embedding[:, 1], initial_embedding[:, 2], '-k', linewidth=1, alpha=0.5)
    ax1.scatter(initial_embedding[:, 0], initial_embedding[:, 1], initial_embedding[:, 2], c=colors, s=100, edgecolors='black', zorder=5)
    ax1.set_title('Best SIO C-Alpha Embedding (GPU)')
    ax1.set_xlabel('X (Å)'); ax1.set_ylabel('Y (Å)'); ax1.set_zlabel('Z (Å)')

    # Plot 2: The final, energy-minimized Cα trace
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], '-k', linewidth=1, alpha=0.5)
    ax2.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], c=colors, s=100, edgecolors='black', zorder=5)
    title = f'Best Minimized Cα Trace (Energy: {final_energy:.2f} kcal/mol)'
    ax2.set_title(title)
    ax2.set_xlabel('X (Å)'); ax2.set_ylabel('Y (Å)'); ax2.set_zlabel('Z (Å)')

    plt.tight_layout()
    plot_filename = 'sio_BEST_visualization.png'
    plt.savefig(plot_filename)
    log.info(f"*** Best visualization saved as '{plot_filename}' ***")

# --- 6. The "Propose" Step: Evolutionary Mutation ---

def mutate_parameters(best_params: dict) -> dict:
    """
    This simulates the AlphaEvolve "propose" step [cite: 221]
    It takes the current best parameters and creates a new, 
    slightly mutated "genome" for the next generation.
    """
    # Create a copy to mutate
    new_params = best_params.copy()
    
    # Add small Gaussian noise to each parameter
    # np.clip ensures the values stay in a sensible range
    
    # Mutate ALPHA (e.g., std dev of 0.01)
    new_params["alpha_val"] = np.clip(
        new_params["alpha_val"] + np.random.normal(0.0, 0.01), 
        0.05, 0.95 # Keep between 5% and 95%
    )
    
    # Mutate DELTA (e.g., std dev of 0.01)
    new_params["delta_val"] = np.clip(
        new_params["delta_val"] + np.random.normal(0.0, 0.01),
        0.1, 0.5 # Keep periodicity reasonable
    )
    
    # Mutate THETA (e.g., std dev of 0.02)
    new_params["theta_val"] = new_params["theta_val"] + np.random.normal(0.0, 0.02)
    
    # Mutate SCALING_FACTOR (e.g., std dev of 0.05 Angstroms)
    new_params["scaling_val"] = np.clip(
        new_params["scaling_val"] + np.random.normal(0.0, 0.05),
        3.0, 4.5 # Keep C-alpha distance reasonable
    )
    
    return new_params


# --- Main execution ---
if __name__ == "__main__":
    
    sequence = "AGAGGAAGAGGAGAAGGAGA" # de novo example
    
    log.info(f"--- STARTING EVOLUTIONARY SEARCH for: {sequence} ---")

    # --- Evolutionary Loop Settings ---
    MAX_GENERATIONS = 1000       # Run 100 loops
    TARGET_FITNESS_SCORE = 0  # Stop if energy becomes negative
    
    # --- Initialization ---
    best_score = -float('inf')
    best_params = {
        "alpha_val": 0.2526,
        "delta_val": 1.0 / 3.6,
        "theta_val": 1.0 / 6.0,
        "scaling_val": 3.8
    }
    best_fold_data = {} # To store the PDB, coords, etc.
    
    log.info(f"Targeting Fitness > {TARGET_FITNESS_SCORE} or {MAX_GENERATIONS} Generations.")
    
    # --- The "Propose-Test-Refine" Loop ---
    for gen in range(MAX_GENERATIONS):
        
        log.info(f"--- Generation {gen + 1} / {MAX_GENERATIONS} ---")
        
        # 1. PROPOSE: Mutate the *best known* parameters
        if gen == 0:
            # Start with the baseline for the first run
            current_params = best_params
        else:
            current_params = mutate_parameters(best_params)
            
        log.info(f"Testing params: {current_params}")

        # 2. TEST: Run the full SIO pipeline
        try:
            sio_params = get_sio_params(sequence, device, **current_params)
            initial_ca_embedding = calculate_sio_embedding(sequence, sio_params, device)
            final_peptide, final_e, final_ca_coords = build_pdb_from_embedding(sequence, initial_ca_embedding)
            
            # 3. REFINE: Get the score and check if it's a new best
            current_fitness = -1 * final_e
            
            log.info(f"Generation {gen + 1} Fitness: {current_fitness:.2f} (Best: {best_score:.2f})")

            if current_fitness > best_score:
                log.info(f"*** NEW BEST SCORE FOUND! ***")
                best_score = current_fitness
                best_params = current_params
                # Save all the data associated with this best fold
                best_fold_data = {
                    "sequence": sequence,
                    "initial_embedding": initial_ca_embedding,
                    "final_positions": final_ca_coords,
                    "final_energy": final_e,
                    "peptide_mol": final_peptide
                }

            # Check for termination condition
            if best_score > TARGET_FITNESS_SCORE:
                log.info(f"*** TARGET FITNESS REACHED! Stopping search. ***")
                break
                
        except Exception as e:
            log.error(f"Generation {gen + 1} failed with error: {e}. Skipping.")

    # --- Search Complete ---
    log.info(f"--- EVOLUTIONARY SEARCH COMPLETE ---")
    log.info(f"Max generations reached.")
    log.info(f"Best Fitness Score: {best_score:.2f}")
    log.info(f"Best Energy ('Cost'): {best_fold_data.get('final_energy', 0):.2f} kcal/mol")
    log.info(f"Discovered Parameters: {best_params}")
    
    # 4. Save and visualize the single best result from the entire run
    if "peptide_mol" in best_fold_data:
        visualize_and_save_best(**best_fold_data)
    else:
        log.error("Search finished with no successful folds found.")
