import pytest
import torch

import tide


def test_rwii_runs_and_produces_finite_grads():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWII tests.")

    device = torch.device("cuda")
    dtype = torch.float32

    ny, nx = 32, 32
    eps_r = 10.0
    conductivity = 0.0

    epsilon = torch.full((ny, nx), eps_r, device=device, dtype=dtype, requires_grad=True)
    sigma = torch.full_like(epsilon, conductivity, requires_grad=True)
    mu = torch.ones_like(epsilon)

    freq0 = 9e8
    dt = 1e-11
    nt = 64

    wavelet = tide.ricker(freq0, nt, dt, peak_time=1.0 / freq0, dtype=dtype, device=device)
    n_shots = 2
    source_amplitude = wavelet.view(1, 1, nt).repeat(n_shots, 1, 1)

    src_y, src_x = ny // 2, nx // 2
    rec_y, rec_x = ny // 2, nx // 2 + 4
    source_location = torch.tensor(
        [[[src_y, src_x]], [[src_y, src_x]]],
        device=device,
        dtype=torch.int64,
    )
    receiver_location = torch.tensor(
        [[[rec_y, rec_x]], [[rec_y, rec_x]]],
        device=device,
        dtype=torch.int64,
    )

    *_, receivers = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.005,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=8,
        save_snapshots=None,
        model_gradient_sampling_interval=1,
        gradient_mode="rwii",
        storage_mode="device",
        alpha_rwii=3e-4,
    )

    loss = receivers.square().sum()
    loss.backward()
    torch.cuda.synchronize()

    assert torch.isfinite(epsilon.grad).all()
    assert torch.isfinite(sigma.grad).all()
    assert float(epsilon.grad.abs().sum().item()) > 0.0


def test_rwii_forward_matches_snapshot():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWII tests.")

    device = torch.device("cuda")
    dtype = torch.float32

    ny, nx = 32, 32
    eps_r = 10.0
    conductivity = 0.0

    epsilon = torch.full((ny, nx), eps_r, device=device, dtype=dtype)
    sigma = torch.full_like(epsilon, conductivity)
    mu = torch.ones_like(epsilon)

    freq0 = 9e8
    dt = 1e-11
    nt = 64

    wavelet = tide.ricker(freq0, nt, dt, peak_time=1.0 / freq0, dtype=dtype, device=device)
    n_shots = 2
    source_amplitude = wavelet.view(1, 1, nt).repeat(n_shots, 1, 1)

    src_y, src_x = ny // 2, nx // 2
    rec_y, rec_x = ny // 2, nx // 2 + 4
    source_location = torch.tensor(
        [[[src_y, src_x]], [[src_y, src_x]]],
        device=device,
        dtype=torch.int64,
    )
    receiver_location = torch.tensor(
        [[[rec_y, rec_x]], [[rec_y, rec_x]]],
        device=device,
        dtype=torch.int64,
    )

    *_, rec_snap = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.005,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=8,
        save_snapshots=False,
        gradient_mode="snapshot",
        storage_mode="none",
    )

    *_, rec_rwii = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.005,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=8,
        save_snapshots=False,
        gradient_mode="rwii",
        storage_mode="device",
        alpha_rwii=3e-4,
    )

    torch.testing.assert_close(rec_rwii, rec_snap, rtol=1e-5, atol=1e-6)

