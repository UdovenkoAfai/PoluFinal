#!/usr/bin/env python3
"""Wrapper kept inside dataset/ to match the submission structure."""
from pathlib import Path
import runpy

project_root = Path(__file__).resolve().parents[2]
runpy.run_path(str(project_root / 'generator' / 'generate_dataset.py'), run_name='__main__')
