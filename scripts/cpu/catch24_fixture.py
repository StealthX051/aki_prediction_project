"""Synthetic Catch24 parity fixture; contains no patient data."""
import json
import platform
from importlib.metadata import version
from pathlib import Path
import sys

import pycatch22


def fixture_signal():
    state = 20261030
    signal = []
    for i in range(2048):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        signal.append(((state / 2**32) - 0.5) + (i % 31) / 31)
    return signal


def extract():
    result = pycatch22.catch22_all(fixture_signal(), catch24=True)
    return {
        'python': platform.python_version(), 'platform': platform.platform(),
        'pycatch22': version('pycatch22'),
        'fixture': 'LCG-20261030-2048-v1',
        'names': result['names'], 'values': result['values'],
    }


if __name__ == '__main__':
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(extract(), indent=2, allow_nan=False)+'\n')
    print('Wrote synthetic Catch24 reference:', destination)
