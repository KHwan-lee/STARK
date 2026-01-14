# STARK

Official PyTorch implementation of  
**STARK: Structure-Aware and Adaptive Representation Learning for Continual Knowledge Graph Embedding**  
(*The Web Conference 2026, WWW 2026*)


<img width="799" height="649" alt="overview" src="https://github.com/user-attachments/assets/850781c7-f1b0-4b16-86e6-3b9bd87a3e4a" />

STARK addresses **continual knowledge graph embedding**, where new entities and relations are incrementally introduced over time.
The framework adaptively allocates LoRA capacity based on structural importance and mitigates hubness via an adaptive TransE loss.

---

## Overview

Continual knowledge graphs evolve as new facts, entities, and relations arrive over time.
STARK is designed to efficiently adapt embeddings in this dynamic setting by jointly considering
**structural novelty** and **group-aware margin constraints**.

Key features:
- Structural Novelty Prioritization (SNP) for identifying important new entities
- Adaptive TransE Loss (ATL) to alleviate hubness in many-to-one relations
- Efficient incremental updates via LoRA-based parameter adaptation

---

## Requirements

Please refer to `requirements.txt` for the required dependencies.

---

## Datasets

We evaluate STARK on continual knowledge graph benchmarks including:

- FB-CKGE
- ENTITY / RELATION / FACT / HYBRID

## To Run the Code

To run STARK, simply execute the following command:

```bash
./main.sh

## Acknowledgement
Our codebase is built upon the implementation of [FastKGE](https://github.com/seukgcode/FastKGE). We thank the authors for their great work and for making the code publicly available.

