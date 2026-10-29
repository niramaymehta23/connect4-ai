"""
network.py — Connect4 value network.

Architecture (value head only; policy head can be added later):
  Conv2d(3,  64, 3, padding=1) → BatchNorm2d(64) → ReLU
  Conv2d(64, 64, 3, padding=1) → BatchNorm2d(64) → ReLU
  Conv2d(64, 64, 3, padding=1) → BatchNorm2d(64) → ReLU
  Flatten
  Linear(64*6*7, 256) → ReLU
  Linear(256, 1)      → Tanh   (output ∈ (-1, 1); +1 = current player wins)

Training uses `training_losses` (cross-entropy before the Tanh), not MSE.

BatchNorm layers are added vs the PHASE_B.md spec for stability during
early training when the network has seen very few positions.

Colour-blind input: channel 2 of the encoding says which colour is to move.
Nets that read it learned a per-colour value offset that tracked whichever
colour had been winning recent self-play games, so the value swung by up to
~1.0 between consecutive plies purely because the turn changed. New nets
replace that plane with a constant one (still marking the board edge under
zero padding); whose turn it is remains readable from the stone counts.
The choice is stored with the weights, and checkpoints saved before it
existed load as colour-aware so they behave exactly as they were trained.

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

    def __init__(self, colour_blind=True):
        super().__init__()
        self.register_buffer('colour_blind', torch.tensor(colour_blind))

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
        return torch.tanh(self._logits(x))

    def _logits(self, x):
        if self.colour_blind:
            x = torch.cat([x[:, :2], torch.ones_like(x[:, 2:])], dim=1)
        return self.head[:-1](self.conv(x))   # every head layer except the Tanh

    def training_losses(self, x, outcomes):
        """
        Return (objective, mse) for outcomes in [-1, 1]: optimise the first,
        log the second (comparable with older runs).

        tanh(z) = 2·sigmoid(2z) - 1, so the objective is cross-entropy on the
        win probability. MSE on the tanh output has a vanishing gradient once
        a prediction saturates at ±1, which left nets stuck predicting exactly
        +1 for one colour; this gradient stays large when confidently wrong.
        """
        logits = self._logits(x)
        objective = nn.functional.binary_cross_entropy_with_logits(2 * logits, (outcomes + 1) / 2)
        mse = nn.functional.mse_loss(torch.tanh(logits), outcomes)
        return objective, mse

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
        state = torch.load(path, map_location=device, weights_only=True)
        state.setdefault('colour_blind', torch.tensor(False))
        net = cls().to(device)
        net.load_state_dict(state)
        net.eval()
        return net
