import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

# --- 1. SIO PARAMETERS AND KERNEL DEFINITION ---

# Amino acid sequence (de novo example: AGAGGAAGAGGAGAAGGAGA)
sequence = "AGAGGAAGAGGAGAAGGAGA"
n = len(sequence)

# Miyazawa-Jernigan contact potentials (scaled and normalized)
mj_matrix = np.array([
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
]) / 200.0  # Scaling factor from MJ paper
modulation_matrix = np.exp(-mj_matrix)

# Amino acid order for MJ matrix
aa_order = "ARNDCQEGHILKMFPSTWYV"

# Kernel parameters
ALPHA = 0.2526
DELTA = 1/3.6  # Adjusted for alpha-helical periodicity
THETA = 1/6
SIGMA_G = np.log(n) / np.sqrt(2 * DELTA)
SIGMA_C = np.log(n)

# Hybrid kernel function with modulation
def hybrid_kernel(diff, aa_i, aa_j):
    if diff == 0:
        return 1.0
    decay_g = np.exp(-diff**2 / (2 * SIGMA_G**2))
    decay_c = 1.0 / (1.0 + (diff / SIGMA_C)**2)
    decay = (1.0 - ALPHA) * decay_g + ALPHA * decay_c
    oscillation = np.cos(2 * np.pi * DELTA * diff + THETA)
    projection = 1.0 + np.sin(2 * np.pi * diff / 6.0)
    i_idx = aa_order.index(aa_i)
    j_idx = aa_order.index(aa_j)
    modulation = modulation_matrix[i_idx, j_idx]
    return decay * oscillation * projection * modulation

# --- 2. SIO EMBEDDING CALCULATION ---

# Build kernel matrix and perform eigenvalue decomposition
K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        diff = abs(i - j)
        K[i, j] = hybrid_kernel(diff, sequence[i], sequence[j])

eigvals, eigvecs = np.linalg.eigh(K)
positive_idx = eigvals > 1e-9
positive_eigvals = eigvals[positive_idx]
pos_eigvecs = eigvecs[:, positive_idx]
idx = positive_eigvals.argsort()[::-1]
positive_eigvals = positive_eigvals[idx]
pos_eigvecs = pos_eigvecs[:, idx]

# Embed in 3D and scale to angstroms
if len(positive_eigvals) >= 3:
    sqrt_d = np.sqrt(positive_eigvals[:3])
    embedded = pos_eigvecs[:, :3] * sqrt_d[None, :]
else:
    embedded = np.zeros((n, 3))

if np.any(embedded):
    distances = np.linalg.norm(np.diff(embedded, axis=0), axis=1)
    if distances.mean() > 0:
        scale_factor = 3.8 / distances.mean()
        embedded *= scale_factor

# --- 3. RDKIT MOLECULE GENERATION AND REFINEMENT ---

# Build peptide and add hydrogens
peptide = Chem.MolFromSequence(sequence)
if peptide is None:
    raise ValueError("Failed to create molecule from sequence.")
peptide = Chem.AddHs(peptide)

# Generate an initial 3D conformer
params = AllChem.ETKDGv3()
params.randomSeed = 42
params.useRandomCoords = True
params.maxIterations = 500
conf_id = AllChem.EmbedMolecule(peptide, params)
if conf_id < 0:
    conf = Chem.Conformer(peptide.GetNumAtoms())
    conf_id = peptide.AddConformer(conf, assignId=True)

conf = peptide.GetConformer(conf_id)

# Get Cα indices
ca_indices = [a.GetIdx() for a in peptide.GetAtoms() if a.GetPDBResidueInfo() and a.GetPDBResidueInfo().GetName().strip() == 'CA']
if len(ca_indices) != n:
    raise ValueError(f"CRITICAL ERROR: C-alpha count mismatch. Expected {n}, found {len(ca_indices)}.")

# Rigidly translate each residue to align its CA with the SIO embedding position
for i in range(n):
    res_num = i + 1
    res_atoms = [a.GetIdx() for a in peptide.GetAtoms() if a.GetPDBResidueInfo() and a.GetPDBResidueInfo().GetResidueNumber() == res_num]
    ca_idx = ca_indices[i]
    initial_ca_pos = conf.GetAtomPosition(ca_idx)
    new_ca_pos = Point3D(embedded[i, 0], embedded[i, 1], embedded[i, 2])
    delta_x = new_ca_pos.x - initial_ca_pos.x
    delta_y = new_ca_pos.y - initial_ca_pos.y
    delta_z = new_ca_pos.z - initial_ca_pos.z
    for atom_idx in res_atoms:
        old_pos = conf.GetAtomPosition(atom_idx)
        new_pos = Point3D(old_pos.x + delta_x, old_pos.y + delta_y, old_pos.z + delta_z)
        conf.SetAtomPosition(atom_idx, new_pos)

# --- 4. CONSTRAINED ENERGY MINIMIZATION ---

# Minimize energy using MMFF with Cα atoms constrained
props = AllChem.MMFFGetMoleculeProperties(peptide)
ff = AllChem.MMFFGetMoleculeForceField(peptide, props, confId=conf_id)
ff.Initialize()

for idx in ca_indices:
    ff.MMFFAddPositionConstraint(idx, 1.0, 500.0)

ff.Minimize(maxIts=2000)
final_energy = ff.CalcEnergy()

# Compute RMSD between SIO embedding and final Cα positions
positions = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in ca_indices])
rmsd = np.sqrt(np.mean(np.sum((embedded - positions)**2, axis=1)))

print(f"Final MMFF energy after minimization: {final_energy:.2f} kcal/mol")
print(f"RMSD between embedding and final positions: {rmsd:.2f} Å")

# Save to PDB
try:
    with Chem.PDBWriter('sio_denovo_fold.pdb') as writer:
        writer.write(peptide)
    print("PDB file saved as 'sio_denovo_fold.pdb'.")
except Exception as e:
    print(f"Failed to save PDB file: {e}")

# --- 5. VISUALIZATION ---
fig = plt.figure(figsize=(14, 6))

# Plot 1: The initial SIO embedding (the "target" shape)
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], '-k', linewidth=1, alpha=0.5)
for i, pos in enumerate(embedded):
    color = 'blue' if sequence[i] == 'A' else 'red'  # A (hydrophobic) in blue, G (polar) in red
    ax1.scatter(pos[0], pos[1], pos[2], c=color, s=100, edgecolors='black', zorder=5)
ax1.set_title('Continuous 3D SIO Embedding')
ax1.set_xlabel('X (Å)'); ax1.set_ylabel('Y (Å)'); ax1.set_zlabel('Z (Å)')

# Plot 2: The final, energy-minimized Cα trace
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-k', linewidth=1, alpha=0.5)
for i, pos in enumerate(positions):
    color = 'blue' if sequence[i] == 'A' else 'red'
    ax2.scatter(pos[0], pos[1], pos[2], c=color, s=100, edgecolors='black', zorder=5)
title = f'Minimized Cα Trace (Energy: {final_energy:.2f} kcal/mol)'
ax2.set_title(title)
ax2.set_xlabel('X (Å)'); ax2.set_ylabel('Y (Å)'); ax2.set_zlabel('Z (Å)')

plt.tight_layout()
plt.savefig('sio_denovo_visualization.png')
print("Visualization saved as 'sio_denovo_visualization.png'.")
