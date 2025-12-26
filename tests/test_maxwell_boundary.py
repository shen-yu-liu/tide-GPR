import tempfile

import pytest
import torch

import tide


def _run_grad(
    gradient_mode: str,
    storage_mode: str,
    storage_path: str,
    *,
    storage_compression: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = torch.float32

    ny, nx = 32, 32
    eps_r = 10.0
    conductivity = 1e-3

    epsilon = torch.full(
        (ny, nx), eps_r, device=device, dtype=dtype, requires_grad=True
    )
    sigma = torch.full_like(epsilon, conductivity, requires_grad=True)
    mu = torch.ones_like(epsilon)

    freq0 = 9e8
    dt = 1e-11
    nt = 64

    wavelet = tide.ricker(
        freq0, nt, dt, peak_time=1.0 / freq0, dtype=dtype, device=device
    )
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
        gradient_mode=gradient_mode,
        storage_mode=storage_mode,
        storage_path=storage_path,
        storage_compression=storage_compression,
    )

    loss = receivers.square().sum()
    loss.backward()
    torch.cuda.synchronize()
    return (
        epsilon.grad.detach().cpu(),
        sigma.grad.detach().cpu(),
        receivers.detach().cpu(),
    )


def test_boundary_storage_modes_match():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for boundary storage tests.")

    with tempfile.TemporaryDirectory() as storage_path:
        eps_dev, sig_dev, rec_dev = _run_grad(
            "boundary", "device", storage_path, storage_compression=False
        )
        eps_cpu, sig_cpu, rec_cpu = _run_grad(
            "boundary", "cpu", storage_path, storage_compression=False
        )
        eps_disk, sig_disk, rec_disk = _run_grad(
            "boundary", "disk", storage_path, storage_compression=False
        )

    torch.testing.assert_close(rec_cpu, rec_dev, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rec_disk, rec_dev, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eps_cpu, eps_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eps_disk, eps_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sig_cpu, sig_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sig_disk, sig_dev, rtol=1e-4, atol=1e-5)


def test_boundary_storage_bf16_modes_match():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for boundary storage tests.")

    with tempfile.TemporaryDirectory() as storage_path:
        eps_dev, sig_dev, rec_dev = _run_grad(
            "boundary", "device", storage_path, storage_compression=True
        )
        eps_cpu, sig_cpu, rec_cpu = _run_grad(
            "boundary", "cpu", storage_path, storage_compression=True
        )
        eps_disk, sig_disk, rec_disk = _run_grad(
            "boundary", "disk", storage_path, storage_compression=True
        )

    torch.testing.assert_close(rec_cpu, rec_dev, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rec_disk, rec_dev, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eps_cpu, eps_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eps_disk, eps_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sig_cpu, sig_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sig_disk, sig_dev, rtol=1e-4, atol=1e-5)


def test_boundary_grad_matches_snapshot():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for boundary storage tests.")

    with tempfile.TemporaryDirectory() as storage_path:
        eps_snap, sig_snap, rec_snap = _run_grad(
            "snapshot", "device", storage_path, storage_compression=False
        )
        eps_bnd, sig_bnd, rec_bnd = _run_grad(
            "boundary", "device", storage_path, storage_compression=False
        )

    torch.testing.assert_close(rec_bnd, rec_snap, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eps_bnd, eps_snap, rtol=1e-4, atol=1e-2)
    torch.testing.assert_close(sig_bnd, sig_snap, rtol=1e-4, atol=2e-1)
