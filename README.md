# Spectral Integral Operator for Protein Folding (SIO)

[<image-card alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" ></image-card>](https://opensource.org/licenses/MIT)
[<image-card alt="Python Version" src="https://img.shields.io/badge/python-3.8%2B-blue.svg" ></image-card>](https://www.python.org/downloads/)

## Overview

This repository implements a first-principles approach to protein folding using the **Spectral Integral Operator (SIO)**, grounded in the **Carbon-Prime Framework**. The SIO bridges number theory and biophysics by modeling amino acid sequences as a one-dimensional system and generating a continuous three-dimensional (3D) embedding of the protein backbone through spectral decomposition. This method complements data-driven tools like AlphaFold, particularly for de novo or novel proteins lacking evolutionary histories.

The project draws from the seminal paper: *[From Number Theory to Molecular Structure: A First-Principles Approach to Protein Folding using the Spectral Integral Operator](A First-Principles Approach to Protein Folding using the Spectral Integral Operator.pdf)* (Carbon-Prime Framework Initiative, October 15, 2025).

Key features include:
- A hybrid kernel incorporating Gaussian and Cauchy decay terms, oscillatory periodicity, mod 6 projection for backbone symmetry, and a 20x20 modulation matrix based on Miyazawa-Jernigan contact potentials.
- Integration with RDKit for all-atom structure generation and constrained energy minimization.
- Hybrid compatibility with AlphaFold for refinement.

### Protein Folding Hierarchy
<image-card alt="Protein Structure Hierarchy" src="protein_hierarchy.png" ></image-card>
*(Diagram illustrating primary to quaternary protein structures, as referenced in the methodology.)*

## Installation

1. Clone the repository:
