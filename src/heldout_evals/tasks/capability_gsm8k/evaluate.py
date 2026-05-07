#!/usr/bin/env python3
"""Capability slice: GSM8K, re-run on the trained checkpoint.

Delegates to src/eval/tasks/gsm8k/evaluate.py so we don't fork the
proven implementation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _delegate import delegate

if __name__ == "__main__":
    sys.exit(delegate("gsm8k"))
