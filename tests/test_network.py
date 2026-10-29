import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is optional for core tests')
class NetworkTests(unittest.TestCase):
    def _flip_turn_plane(self, x):
        flipped = x.clone()
        flipped[:, 2] = 1.0 - flipped[:, 2]
        return flipped

    def test_new_network_ignores_the_colour_plane(self):
        from src.nn.network import Connect4Net
        torch.manual_seed(0)
        net = Connect4Net().eval()
        x = (torch.rand(8, 3, 6, 7) < 0.3).float()
        with torch.no_grad():
            self.assertTrue(torch.equal(net(x), net(self._flip_turn_plane(x))))

    def test_colour_blind_flag_survives_save_and_load(self):
        from src.nn.network import Connect4Net
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'net.pt'
            Connect4Net().save(path)
            self.assertTrue(Connect4Net.load(path, 'cpu').colour_blind)

    def test_checkpoint_without_flag_loads_colour_aware(self):
        from src.nn.network import Connect4Net
        torch.manual_seed(0)
        legacy = Connect4Net(colour_blind=False)
        state = legacy.state_dict()
        del state['colour_blind']
        x = (torch.rand(8, 3, 6, 7) < 0.3).float()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.pt'
            torch.save(state, path)
            net = Connect4Net.load(path, 'cpu')
        self.assertFalse(net.colour_blind)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(x), legacy.eval()(x)))
            self.assertFalse(torch.equal(net(x), net(self._flip_turn_plane(x))))

    def test_training_loss_recovers_from_a_saturated_wrong_prediction(self):
        from src.nn.network import Connect4Net
        net = Connect4Net()
        with torch.no_grad():
            net.head[3].bias.fill_(50.0)          # output saturated at +1
        x = torch.zeros(4, 3, 6, 7)
        lost = -torch.ones(4, 1)
        objective, mse = net.training_losses(x, lost)
        objective.backward()
        self.assertAlmostEqual(mse.item(), 4.0, places=4)
        self.assertGreater(net.head[3].bias.grad.item(), 0.5)
        net.zero_grad()
        torch.nn.functional.mse_loss(net(x), lost).backward()
        self.assertLess(net.head[3].bias.grad.abs().item(), 1e-6)

    def test_training_mse_matches_forward_output(self):
        from src.nn.network import Connect4Net
        torch.manual_seed(0)
        net = Connect4Net().eval()
        x = (torch.rand(8, 3, 6, 7) < 0.3).float()
        outcomes = torch.tensor([[1.0], [-1.0], [0.0], [1.0]] * 2)
        _, mse = net.training_losses(x, outcomes)
        self.assertAlmostEqual(mse.item(), torch.nn.functional.mse_loss(net(x), outcomes).item(), places=6)
