"""python -m membrane_crystallization [--quick] [--jobs N]: print the measured tables."""

import argparse
import os

from .sweeps import report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="membrane-crystallization",
                                 description="SYNTHETIC streaming-PCA sweeps.")
    ap.add_argument("--quick", action="store_true", help="few seeds and generations (smoke run)")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    if a.jobs < 1:
        ap.error("--jobs must be >= 1")
    print("SYNTHETIC: classical streaming estimate of an observation covariance.\n")
    if a.quick:
        print(report(2, 3, 2, 2, 3, jobs=a.jobs))
    else:
        print(report(jobs=a.jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
