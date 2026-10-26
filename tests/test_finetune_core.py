import numpy as np
import torch

from otp import finetune as F
from otp.model import forecast, set_threads


def test_lora_is_identity_at_init_and_trains(tmp_path):
    set_threads(1)
    torch.manual_seed(0)
    fc = F.load_trainable("5M")
    x = torch.tensor(np.sin(np.arange(1024) / 7.0)[None].repeat(2, 0) + 3.0, dtype=torch.float32)
    with torch.no_grad():
        before = F.forecast_train(fc, x, 64)
    n = F.add_lora(fc.model, rank=4)
    assert 0 < n < sum(p.numel() for p in fc.model.parameters())
    with torch.no_grad():
        after = F.forecast_train(fc, x, 64)
    assert torch.allclose(before, after, atol=1e-5)  # B = 0 at init
    sampler = F.Sampler(factories=[], replay_frac=1.0, contexts=(256,), horizons=(32,))
    rec = F.train(fc, sampler, steps=3, batch_size=4, lr=1e-3, warmup=1, log_every=0)
    assert rec["trainable_params"] == n and np.isfinite(rec["final_loss"])
    with torch.no_grad():
        trained = F.forecast_train(fc, x, 64)
    F.export(fc, str(tmp_path / "ck"))
    q = forecast(str(tmp_path / "ck"), x.numpy(), 64, symmetric_averaging=False, clip_nonnegative=False)[:, 0]
    assert np.allclose(q, trained.numpy(), atol=1e-3 * (1 + np.abs(q).max()))
