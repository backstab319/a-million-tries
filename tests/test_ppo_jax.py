"""The JAX network must compute exactly what the torch SnakeNet computes (checkpoints are shared)."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ailearns.agents.ppo import SnakeNet, obs_shape  # noqa: E402
from ailearns.agents.ppo_jax import forward, net_spec  # noqa: E402


@pytest.mark.parametrize("arch", ["mlp", "cnn", "cnn-deep"])
def test_forward_matches_torch(arch):
    torch.manual_seed(0)
    shape = obs_shape(10, "ego2")
    net = SnakeNet(shape, arch)
    with torch.no_grad():  # non-trivial weights in the heads too
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = np.random.default_rng(0).random((16, *shape), dtype=np.float32)
    with torch.no_grad():
        tl, tv = net(torch.from_numpy(x))
    params = {k: jnp.asarray(v.numpy()) for k, v in net.state_dict().items()}
    with jax.default_matmul_precision("float32"):
        jl, jv = forward(net_spec(net), params, jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(jl), tl.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jv), tv.numpy(), rtol=1e-4, atol=1e-5)
