from typing import List, Tuple

import torch
from . import utils

def set_pml_profiles(
    pml_width: List[int],
    accuracy: int,
    fd_pad: List[int],
    dt: float,
    grid_spacing: List[float],
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    ny: int,
    nx: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Sets up PML profiles for a staggered grid.

    Args:
        pml_width: A list of integers specifying the width of the PML
            on each side (top, bottom, left, right).
        accuracy: The finite-difference accuracy order.
        fd_pad: A list of integers specifying the padding for finite-difference.
        dt: The time step.
        grid_spacing: A list of floats specifying the grid spacing in
            y and x directions.
        max_vel: The maximum velocity in the model.
        dtype: The data type of the tensors (e.g., torch.float32).
        device: The device on which the tensors will be (e.g., 'cuda', 'cpu').
        pml_freq: The PML frequency.
        ny: The number of grid points in the y direction.
        nx: The number of grid points in the x direction.

    Returns:
        A tuple containing:
        - List of torch.Tensors for a, b profiles: [ay, ayh, ax, axh, by, byh, bx, bxh]
        - List of torch.Tensors for k profiles: [ky, kyh, kx, kxh]

    """
    pml_start: List[float] = [
        fd_pad[0] + pml_width[0],
        ny - 1 - fd_pad[1] - pml_width[1],
        fd_pad[2] + pml_width[2],
        nx - 1 - fd_pad[3] - pml_width[3],
    ]
    max_pml = max(
        [
            pml_width[0] * grid_spacing[0],
            pml_width[1] * grid_spacing[0],
            pml_width[2] * grid_spacing[1],
            pml_width[3] * grid_spacing[1],
        ],
    )

    # Integer grid PML profiles
    ay, by, ky = utils.setup_pml(
        pml_width[:2],
        pml_start[:2],
        max_pml,
        dt,
        ny, 
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=grid_spacing[0],
    )
    ax, bx, kx = utils.setup_pml(
        pml_width[2:],
        pml_start[2:],
        max_pml,
        dt,
        nx,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=grid_spacing[1],
    )

    # Half grid PML profiles
    ayh, byh, kyh = utils.setup_pml_half(
        pml_width[:2],
        pml_start[:2],
        max_pml,
        dt,
        ny,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=grid_spacing[0],
    )
    axh, bxh, kxh = utils.setup_pml_half(
        pml_width[2:],
        pml_start[2:],
        max_pml,
        dt,
        nx,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=grid_spacing[1],
    )
    
    # Reshape for broadcasting: [batch, ny, nx]
    ay = ay[None, :, None]
    ayh = ayh[None, :, None]
    ax = ax[None, None, :]
    axh = axh[None, None, :]
    by = by[None, :, None]
    byh = byh[None, :, None]
    bx = bx[None, None, :]
    bxh = bxh[None, None, :]
    
    ky = ky[None, :, None]
    kyh = kyh[None, :, None]
    kx = kx[None, None, :]
    kxh = kxh[None, None, :]

    return [ay, ayh, ax, axh, by, byh, bx, bxh], [ky, kyh, kx, kxh]






def diffy1(a: torch.Tensor, stencil: int, rdy: torch.Tensor) -> torch.Tensor:
    """Calculates the first y derivative at integer grid points."""
    if stencil   == 2:
        return torch.nn.functional.pad(
            (a[..., 1:, :] - a[..., :-1, :]) * rdy, (0, 0, 1, 0)
        )
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1, :] - a[..., 1:-2, :])
                + -1 / 24 * (a[..., 3:, :] - a[..., :-3, :])
            )
            * rdy,
            (0, 0, 2, 1),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2, :] - a[..., 2:-3, :])
                + -25 / 384 * (a[..., 4:-1, :] - a[..., 1:-4, :])
                + 3 / 640 * (a[..., 5:, :] - a[..., :-5, :])
            )
            * rdy,
            (0, 0, 3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3, :] - a[..., 3:-4, :])
            + -245 / 3072 * (a[..., 5:-2, :] - a[..., 2:-5, :])
            + 49 / 5120 * (a[..., 6:-1, :] - a[..., 1:-6, :])
            + -5 / 7168 * (a[..., 7:, :] - a[..., :-7, :])
        )
        * rdy,
        (0, 0, 4, 3),
    )


def diffx1(a: torch.Tensor, stencil: int, rdx: torch.Tensor) -> torch.Tensor:
    """Calculates the first x derivative at integer grid points."""
    if stencil == 2:
        return torch.nn.functional.pad((a[..., 1:] - a[..., :-1]) * rdx, (1, 0))
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1] - a[..., 1:-2])
                + -1 / 24 * (a[..., 3:] - a[..., :-3])
            )
            * rdx,
            (2, 1),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2] - a[..., 2:-3])
                + -25 / 384 * (a[..., 4:-1] - a[..., 1:-4])
                + 3 / 640 * (a[..., 5:] - a[..., :-5])
            )
            * rdx,
            (3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3] - a[..., 3:-4])
            + -245 / 3072 * (a[..., 5:-2] - a[..., 2:-5])
            + 49 / 5120 * (a[..., 6:-1] - a[..., 1:-6])
            + -5 / 7168 * (a[..., 7:] - a[..., :-7])
        )
        * rdx,
        (4, 3),
    )


def diffyh1(a: torch.Tensor, stencil: int, rdy: torch.Tensor) -> torch.Tensor:
    """Calculates the first y derivative at half integer grid points."""
    if stencil == 2:
        return torch.nn.functional.pad(
            (a[..., 2:, :] - a[..., 1:-1, :]) * rdy, (0, 0, 1, 1)
        )
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1, :] - a[..., 2:-2, :])
                + -1 / 24 * (a[..., 4:, :] - a[..., 1:-3, :])
            )
            * rdy,
            (0, 0, 2, 2),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2, :] - a[..., 3:-3, :])
                + -25 / 384 * (a[..., 5:-1, :] - a[..., 2:-4, :])
                + 3 / 640 * (a[..., 6:, :] - a[..., 1:-5, :])
            )
            * rdy,
            (0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3, :] - a[..., 4:-4, :])
            + -245 / 3072 * (a[..., 6:-2, :] - a[..., 3:-5, :])
            + 49 / 5120 * (a[..., 7:-1, :] - a[..., 2:-6, :])
            + -5 / 7168 * (a[..., 8:, :] - a[..., 1:-7, :])
        )
        * rdy,
        (0, 0, 4, 4),
    )


def diffzh1(a: torch.Tensor, stencil: int, rdz: torch.Tensor) -> torch.Tensor:
    """Calculates the first z derivative at half integer grid points.

    For a tensor with shape [..., nz, ny, nx], the derivative is taken along
    the z dimension at half-grid locations.
    """
    if stencil == 2:
        return torch.nn.functional.pad(
            (a[..., 2:, :, :] - a[..., 1:-1, :, :]) * rdz, (0, 0, 0, 0, 1, 1)
        )
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1, :, :] - a[..., 2:-2, :, :])
                + -1 / 24 * (a[..., 4:, :, :] - a[..., 1:-3, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 2, 2),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2, :, :] - a[..., 3:-3, :, :])
                + -25 / 384 * (a[..., 5:-1, :, :] - a[..., 2:-4, :, :])
                + 3 / 640 * (a[..., 6:, :, :] - a[..., 1:-5, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3, :, :] - a[..., 4:-4, :, :])
            + -245 / 3072 * (a[..., 6:-2, :, :] - a[..., 3:-5, :, :])
            + 49 / 5120 * (a[..., 7:-1, :, :] - a[..., 2:-6, :, :])
            + -5 / 7168 * (a[..., 8:, :, :] - a[..., 1:-7, :, :])
        )
        * rdz,
        (0, 0, 0, 0, 4, 4),
    )


def diffxh1(a: torch.Tensor, stencil: int, rdx: torch.Tensor) -> torch.Tensor:
    """Calculates the first x derivative at half integer grid points."""
    if stencil == 2:
        return torch.nn.functional.pad((a[..., 2:] - a[..., 1:-1]) * rdx, (1, 1))
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1] - a[..., 2:-2])
                + -1 / 24 * (a[..., 4:] - a[..., 1:-3])
            )
            * rdx,
            (2, 2),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2] - a[..., 3:-3])
                + -25 / 384 * (a[..., 5:-1] - a[..., 2:-4])
                + 3 / 640 * (a[..., 6:] - a[..., 1:-5])
            )
            * rdx,
            (3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3] - a[..., 4:-4])
            + -245 / 3072 * (a[..., 6:-2] - a[..., 3:-5])
            + 49 / 5120 * (a[..., 7:-1] - a[..., 2:-6])
            + -5 / 7168 * (a[..., 8:] - a[..., 1:-7])
        )
        * rdx,
        (4, 4),
    )


