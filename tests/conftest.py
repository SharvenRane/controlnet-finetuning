import os
import sys

import torch

# Make the project root importable so `import src` works under pytest.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch.manual_seed(0)
