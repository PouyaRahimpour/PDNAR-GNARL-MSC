import torch

from src.gnarl.policies.masked_policy import masked_categorical


def test_masked_distribution_never_assigns_mass_to_invalid_action():
    distribution = masked_categorical(torch.tensor([2.0, 10.0, 1.0]), torch.tensor([True, False, True]))
    assert distribution.probs[1].item() == 0.0
    assert all(int(distribution.sample()) != 1 for _ in range(100))
