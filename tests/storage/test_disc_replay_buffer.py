import torch

from rsl_rl.storage.disc_replay_buffer import DiscReplayBuffer, store_disc_replay_from_rollout
from rsl_rl.storage.circular_buffer import CircularBuffer


def test_disc_replay_buffer_push_and_sample():
    buf = DiscReplayBuffer(max_size=5, device="cpu")
    samples = torch.arange(20, dtype=torch.float32).reshape(5, 2, 2)
    buf.push(samples)
    assert len(buf) == 5
    out = buf.sample(3)
    assert out.shape == (3, 2, 2)


def test_disc_replay_buffer_limits_push_when_full():
    buf = DiscReplayBuffer(max_size=4, device="cpu")
    for _ in range(3):
        buf.push(torch.ones(2, 1, 3))
    assert len(buf) == 4
    sampled = buf.sample(4)
    assert sampled.shape == (4, 1, 3)


def test_store_disc_replay_from_rollout_respects_replay_samples():
    disc_buf = CircularBuffer(max_len=3, batch_size=2, device="cpu")
    disc_buf.append(torch.zeros(2, 1, 4))
    disc_buf.append(torch.ones(2, 1, 4))
    replay = DiscReplayBuffer(max_size=4, device="cpu")

    store_disc_replay_from_rollout(
        disc_buf, replay, fetch_length=2, replay_samples=1000
    )
    assert len(replay) == 4

    pushed = store_disc_replay_from_rollout(
        disc_buf, replay, fetch_length=2, replay_samples=2
    )
    assert pushed == 2
