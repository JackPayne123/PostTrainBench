#!/usr/bin/env python3
"""Capability slice: GPQA-main, re-run on the trained checkpoint."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _delegate import delegate

if __name__ == "__main__":
    sys.exit(delegate("gpqamain"))
