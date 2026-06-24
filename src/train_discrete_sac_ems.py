"""Train a discrete SAC agent on the EMS candidate-action environment.

Run with:
    python train_discrete_sac_ems.py --device cuda
"""

from __future__ import annotations

import sys

from train_multi_algo_ems import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--algorithm", "sac", *sys.argv[1:]]
    main()
