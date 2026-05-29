import torch

from gdrl.envs import GDRLEnv
from gdrl.training import TinyJumpCNN, load_policy, save_policy


def test_tiny_jump_cnn_outputs_two_action_logits() -> None:
    env = GDRLEnv("levels/tiny_spikes.json")
    obs, _ = env.reset()
    model = TinyJumpCNN()

    logits = model(torch.as_tensor(obs).unsqueeze(0))

    assert tuple(logits.shape) == (1, 2)
    assert model.act(obs) in {0, 1}


def test_policy_save_load_round_trip(tmp_path) -> None:
    model = TinyJumpCNN()
    path = tmp_path / "policy.pt"

    save_policy(model, path)
    loaded = load_policy(path)

    x = torch.zeros(1, 1, 96, 96)
    assert torch.allclose(model(x), loaded(x))
