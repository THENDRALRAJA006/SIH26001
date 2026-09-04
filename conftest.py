"""
Root conftest.py — adds the project root to sys.path so that
'ml', 'backend', etc. are importable by all tests without installation.
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))
