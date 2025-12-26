import pytest
import torch
from torch.func import jvp

import tide


def test_maxwelltm_jvp_cuda_matches_finite_difference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MaxwellTM JVP test.")
    if not tide.backend_utils.is_backend_available():

        pytest.skip("C/CUDA backend library not available.")

    device = torch.device("cuda")
    dtype = torch.float32

    ny, nx = 20, 32
    nt = 40
    dx = 0.02
    dt = 4e-11
    pml_width = 6
    stencil = 4

    epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.ones_like(epsilon) * 1e-3
    mu = torch.ones_like(epsilon)

    source_locations = torch.tensor([[[ny // 2, nx // 4]]], dtype=torch.long, device=device)
    receiver_locations = torch.tensor(
        [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
    )

    freq = 200e6
    wavelet = tide.ricker(freq, nt, dt, peak_time=1.0 / freq, dtype=dtype, device=device)
    source_amplitude = wavelet.view(1, 1, nt)

    def forward_fn(eps):
        return tide.maxwelltm(
            eps,
            sigma,
            mu,
            grid_spacing=dx,
            dt=dt,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=pml_width,
            stencil=stencil,
            save_snapshots=False,
        )[-1]

    tangent = torch.zeros_like(epsilon)
    tangent[ny // 2, nx // 2] = 0.1

    output, jvp_output = jvp(forward_fn, (epsilon,), (tangent,))

    h = 5e-3
    output_pert = forward_fn(epsilon + h * tangent)
    fd_approx = (output_pert - output) / h

    rel_error = torch.linalg.norm(jvp_output - fd_approx) / torch.linalg.norm(fd_approx)
    assert rel_error < 1e-2
