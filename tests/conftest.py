# Adds repo root to sys.path so `src` is importable without installation.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
