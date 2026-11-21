# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Launcher for the single PTS-141 numerical implementation, including staged runs.

Run: node scripts/run-oracle-fixture.mjs spatial-clustering-fixtures --python <clustering-venv>/bin/python3
Requires NumPy 2.0.2 and scikit-learn 1.6.1 (checked by the family producer).
"""

from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    family = Path(__file__).resolve().parent.parent / 'gpu-spatial-analysis' / 'spatial-clustering'
    subprocess.run([
        sys.executable, '-I', str(family / 'clustering-fixture-oracles.py'),
        '--output', str(family / 'clustering-fixture-oracles.json'),
    ], check=True)
