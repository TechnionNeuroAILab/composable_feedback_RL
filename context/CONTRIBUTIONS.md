# Context contributions

Reference papers and linked implementations in this repository.

## Papers in this folder

| File | Paper | Used for |
|------|-------|----------|
| `lsewd24.pdf` | B-feedback / composable feedback (LSEWD) | Core conf-gate and meta-gradient DQN work |
| `ncomms13276.pdf` | B-feedback matrix (Nature Communications) | Composable feedback experiments |
| `HOOF_paper.pdf` | Paul et al. (2019) — HOOF | DQN adaptation in `code/paper_hpo_dqn/hoof.py` |
| `hypercontroller_paper.pdf` | Gornet, Kantaros & Sinopoli (2025) — HyperController | DQN adaptation in `code/paper_hpo_dqn/hypercontroller.py` |
| `SEARL_paper.pdf` | Franke et al. (2021) — SEARL | DQN adaptation in `code/paper_hpo_dqn/searl.py` |

## Paper-inspired DQN HPO (Aug 2026)

Controlled DQN adaptations of HOOF, HyperController, and SEARL on a shared
branching/dueling DQN core. Compared on CartPole-v1 (100k env steps) and
HalfCheetah-v4 (1M env steps), seed 1.

- **Code:** `code/paper_hpo_dqn/`
- **Results:** `results/paper_hpo_dqn/` (see `SUMMARY.md`)
- **Tests:** `tests/test_paper_hpo_dqn.py`

Adaptation boundaries and attribution are documented in `code/paper_hpo_dqn/README.md`.
