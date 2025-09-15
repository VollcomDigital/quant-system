import os
import sys

# Ensure the repository root is importable as a package prefix (e.g., `src.*`)
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
