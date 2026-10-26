"""In-prior control grid: random trend + season + AR noise (what Timestral was pretrained on)."""
from otp.generators.primitives import in_prior


def configs():
    return [(f"in_prior/p{p}", (lambda seed, p=p: in_prior(seed, period=p))) for p in (12, 24, 96, 168)]
