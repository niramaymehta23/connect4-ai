"""
network.py — Connect4 value network.

Architecture (value head only; policy head can be added later):
  Conv2d(3,  64, 3, padding=1) → BatchNorm2d(64) → ReLU
  Conv2d(64, 64, 3, padding=1) → BatchNorm2d(64) → ReLU
  Conv2d(64, 64, 3, padding=1) → BatchNorm2d(64) → ReLU
  Flatten
  Linear(64*6*7, 256) → ReLU
  Linear(256, 1)      → Tanh   (output ∈ (-1, 1); +1 = current player wins)

BatchNorm layers are added vs the PHASE_B.md spec for stability during
early training when the network has seen very few positions.

Device selection (MPS → CPU) is resolved at import time and printed once.
"""
import torch
import torch.nn as nn
from pathlib import Path
from ..game.constants import ROWS, COLS

# ---------------------------------------------------------------------------
# Device — resolved once at import, printed for visibility
# ---------------------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Connect4Net(nn.Module):
    """Value network for Connect 4.  Input: (B, 3, ROWS, COLS).  Output: (B, 1)."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(3,  64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * ROWS * COLS, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """x: (B, 3, ROWS, COLS) → (B, 1)"""
        return self.head(self.conv(x))

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path):
        """Save state dict to `path`.  Creates parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device):
        """Load state dict from `path`, return network in eval mode on `device`."""
        net = cls().to(device)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        net.eval()
        return net
