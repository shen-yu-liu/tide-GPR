import itertools
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch

from .common import (
    CallbackState,
    Callback,
    cfl_condition,
    upsample,
    downsample_and_movedim,
    validate_model_gradient_sampling_interval,
    validate_freq_taper_frac,
    validate_time_pad_frac,
)
from .storage import (
    TemporaryStorage,
    storage_mode_to_int,
    STORAGE_DEVICE,
    STORAGE_CPU,
    STORAGE_DISK,
    STORAGE_NONE,
)
from . import staggered
from . import backend_utils

# Physical constants
EP0 = 8.8541878128e-12   # vacuum permittivity (F/m)
MU0 = 1.2566370614359173e-06  # vacuum permeability (H/m)

_CTX_HANDLE_COUNTER = itertools.count()
_CTX_HANDLE_REGISTRY: dict[int, dict[str, Any]] = {}


def _register_ctx_handle(ctx_data: dict[str, Any]) -> torch.Tensor:
    handle = next(_CTX_HANDLE_COUNTER)
    _CTX_HANDLE_REGISTRY[handle] = ctx_data
    return torch.tensor(handle, dtype=torch.int64)


def _get_ctx_handle(handle: int) -> dict[str, Any]:
    try:
        return _CTX_HANDLE_REGISTRY[handle]
    except KeyError as exc:
        raise RuntimeError(f"Unknown context handle: {handle}") from exc


def _release_ctx_handle(handle: Optional[int]) -> None:
    if handle is None:
        return
    _CTX_HANDLE_REGISTRY.pop(handle, None)


def _compute_boundary_indices_flat(
    ny: int,
    nx: int,
    pml_y0: int,
    pml_x0: int,
    pml_y1: int,
    pml_x1: int,
    boundary_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Return flattened indices (y * nx + x) for a PML-side boundary ring.

    The ring is a union of strips adjacent to the physical domain inside the
    PML region. It is intended for boundary saving/injection and should have
    width >= FD_PAD (stencil/2).
    """
    if boundary_width <= 0:
        raise ValueError("boundary_width must be positive.")

    max_width = min(pml_y0, ny - pml_y1, pml_x0, nx - pml_x1)
    if boundary_width > max_width:
        raise ValueError(
            "boundary_width must fit within the padded grid; "
            f"got boundary_width={boundary_width}, max_width={max_width}."
        )

    # Construct indices in a cache-friendly order:
    # - top strip rows (contiguous in memory per row)
    # - bottom strip rows
    # - middle rows: left strip then right strip per row
    # This ordering avoids the mask+nonzero pattern which produces a more
    # irregular access pattern on GPU gather/scatter.
    x_all = torch.arange(nx, device="cpu", dtype=torch.int64)

    top_y = torch.arange(pml_y0 - boundary_width, pml_y0, device="cpu", dtype=torch.int64)
    bottom_y = torch.arange(pml_y1, pml_y1 + boundary_width, device="cpu", dtype=torch.int64)
    mid_y = torch.arange(pml_y0, pml_y1, device="cpu", dtype=torch.int64)
    x_left = torch.arange(pml_x0 - boundary_width, pml_x0, device="cpu", dtype=torch.int64)
    x_right = torch.arange(pml_x1, pml_x1 + boundary_width, device="cpu", dtype=torch.int64)

    top = (top_y[:, None] * nx + x_all[None, :]).reshape(-1)
    bottom = (bottom_y[:, None] * nx + x_all[None, :]).reshape(-1)

    mid_base = mid_y * nx
    mid_left = (mid_base[:, None] + x_left[None, :])
    mid_right = (mid_base[:, None] + x_right[None, :])
    mid = torch.cat((mid_left, mid_right), dim=1).reshape(-1)

    flat = torch.cat((top, bottom, mid), dim=0).to(torch.int64)
    return flat.to(device=device)


def prepare_parameters(
    epsilon_r: torch.Tensor, sigma: torch.Tensor, mu_r: torch.Tensor, dt: float
) -> tuple:
    """Prepare update coefficients for Maxwell equations.
    
    Args:
        epsilon_r: Relative permittivity.
        sigma: Conductivity (S/m).
        mu_r: Relative permeability.
        dt: Time step (s).
    
    Returns:
        Tuple of (Ca, Cb, Cq) coefficients.
    """
    # Convert to absolute values
    epsilon = epsilon_r * EP0
    mu = mu_r * MU0
    
    # Ca and Cb for E-field update: E^{n+1} = Ca*E^n + Cb*(curl H)
    # Ca = (1 - sigma*dt/(2*epsilon)) / (1 + sigma*dt/(2*epsilon))
    # Cb = dt/epsilon / (1 + sigma*dt/(2*epsilon))
    denom = 1.0 + sigma * dt / (2.0 * epsilon)
    ca = (1.0 - sigma * dt / (2.0 * epsilon)) / denom
    cb = (dt / epsilon) / denom
    
    # Cq for H-field update: H^{n+1/2} = H^{n-1/2} - Cq*(curl E)
    cq = dt / mu
    
    return ca, cb, cq


class MaxwellTM(torch.nn.Module):
    """2D TM mode Maxwell equations solver using FDTD method.
    
    This module solves the TM (Transverse Magnetic) mode Maxwell equations
    in 2D with fields (Ey, Hx, Hz) using the FDTD method with CPML absorbing
    boundary conditions.
    
    Args:
        epsilon: Relative permittivity tensor [ny, nx]. 
            For vacuum/air, use 1.0. For common materials:
            - Water: ~80
            - Glass: ~4-7
            - Soil (dry): ~3-5
            - Concrete: ~4-8
        sigma: Electrical conductivity tensor [ny, nx] in S/m.
            For lossless media, use 0.0.
        mu: Relative permeability tensor [ny, nx].
            For most non-magnetic materials, use 1.0.
        grid_spacing: Grid spacing in meters. Can be a single value (same for
            both directions) or a sequence [dy, dx].
        epsilon_requires_grad: Whether to compute gradients for permittivity.
        sigma_requires_grad: Whether to compute gradients for conductivity.
    
    Note:
        The input parameters are RELATIVE values (dimensionless). They will be
        multiplied internally by the vacuum permittivity (ε₀ = 8.854e-12 F/m)
        and vacuum permeability (μ₀ = 1.257e-6 H/m) respectively.
    """
    def __init__(
        self,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        mu: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        epsilon_requires_grad: Optional[bool] = None,
        sigma_requires_grad: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if epsilon_requires_grad is not None and not isinstance(epsilon_requires_grad, bool):
            raise TypeError(
                f"epsilon_requires_grad must be bool or None, "
                f"got {type(epsilon_requires_grad).__name__}",
            )
        if not isinstance(epsilon, torch.Tensor):
            raise TypeError(
                f"epsilon must be torch.Tensor, got {type(epsilon).__name__}",
            )
        if sigma_requires_grad is not None and not isinstance(sigma_requires_grad, bool):
            raise TypeError(
                f"sigma_requires_grad must be bool or None, "
                f"got {type(sigma_requires_grad).__name__}",
            )
        if not isinstance(sigma, torch.Tensor):
            raise TypeError(
                f"sigma must be torch.Tensor, got {type(sigma).__name__}",
            )
        if not isinstance(mu, torch.Tensor):
            raise TypeError(
                f"mu must be torch.Tensor, got {type(mu).__name__}",
            )

        # If requires_grad not specified, preserve the input tensor's setting
        if epsilon_requires_grad is None:
            epsilon_requires_grad = epsilon.requires_grad
        if sigma_requires_grad is None:
            sigma_requires_grad = sigma.requires_grad

        self.epsilon = torch.nn.Parameter(
            epsilon, requires_grad=epsilon_requires_grad
        )
        self.sigma = torch.nn.Parameter(sigma, requires_grad=sigma_requires_grad)
        self.register_buffer("mu", mu)  # In normal we don't optimize mu
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: float,
        source_amplitude: Optional[torch.Tensor],  # [shot,source,time]
        source_location: Optional[torch.Tensor],  # [shot,source,2]
        receiver_location: Optional[torch.Tensor],  # [shot,receiver,2]
        stencil: int = 2,
        pml_width: Union[int, Sequence[int]] = 20,
        max_vel: Optional[float] = None,
        Ey_0: Optional[torch.Tensor] = None,
        Hx_0: Optional[torch.Tensor] = None,
        Hz_0: Optional[torch.Tensor] = None,
        m_Ey_x: Optional[torch.Tensor] = None,
        m_Ey_z: Optional[torch.Tensor] = None,
        m_Hx_z: Optional[torch.Tensor] = None,
        m_Hz_x: Optional[torch.Tensor] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        save_snapshots: Optional[bool] = None,
        forward_callback: Optional[Callback] = None,
        backward_callback: Optional[Callback] = None,
        callback_frequency: int = 1,
        python_backend: Union[bool, str] = False,
        gradient_mode: str = "snapshot",
        storage_mode: str = "device",
        storage_path: str = ".",
        storage_compression: bool = False,
        storage_bytes_limit_device: Optional[int] = None,
        storage_bytes_limit_host: Optional[int] = None,
        storage_chunk_steps: int = 0,
        boundary_width: int = 0,
        alpha_rwii: float = 0.0,
    ):
        # Type assertions for buffer and parameter tensors
        assert isinstance(self.epsilon, torch.Tensor)
        assert isinstance(self.sigma, torch.Tensor)
        assert isinstance(self.mu, torch.Tensor)
        return maxwelltm(
            self.epsilon,
            self.sigma,
            self.mu,
            self.grid_spacing,
            dt,
            source_amplitude,
            source_location,
            receiver_location,
            stencil,
            pml_width,
            max_vel,
            Ey_0,
            Hx_0,
            Hz_0,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            nt,
            model_gradient_sampling_interval,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            save_snapshots,
            forward_callback,
            backward_callback,
            callback_frequency,
            python_backend,
            gradient_mode,
            storage_mode,
            storage_path,
            storage_compression,
            storage_bytes_limit_device,
            storage_bytes_limit_host,
            storage_chunk_steps,
            boundary_width,
            alpha_rwii,
        )


def maxwelltm(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int = 2,
    pml_width: Union[int, Sequence[int]] = 20,
    max_vel: Optional[float] = None,
    Ey_0: Optional[torch.Tensor] = None,
    Hx_0: Optional[torch.Tensor] = None,
    Hz_0: Optional[torch.Tensor] = None,
    m_Ey_x: Optional[torch.Tensor] = None,
    m_Ey_z: Optional[torch.Tensor] = None,
    m_Hx_z: Optional[torch.Tensor] = None,
    m_Hz_x: Optional[torch.Tensor] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    save_snapshots: Optional[bool] = None,
    forward_callback: Optional[Callback] = None,
    backward_callback: Optional[Callback] = None,
    callback_frequency: int = 1,
    python_backend: Union[bool, str] = False,
    gradient_mode: str = "snapshot",
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: bool = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    boundary_width: int = 0,
    alpha_rwii: float = 0.0,
):
    """2D TM mode Maxwell equations solver.
    
    This is the main entry point for Maxwell TM propagation. It automatically
    handles CFL condition checking and time step resampling when needed.
    
    If the user-provided time step (dt) is too large for numerical stability,
    the source signal will be upsampled internally and receiver data will be
    downsampled back to the original sampling rate.
    
    Args:
        epsilon: Relative permittivity tensor [ny, nx].
        sigma: Electrical conductivity tensor [ny, nx] in S/m.
        mu: Relative permeability tensor [ny, nx].
        grid_spacing: Grid spacing in meters. Single value or [dy, dx].
        dt: Time step in seconds.
        source_amplitude: Source waveform [n_shots, n_sources, nt].
        source_location: Source locations [n_shots, n_sources, 2].
        receiver_location: Receiver locations [n_shots, n_receivers, 2].
        stencil: FD stencil order (2, 4, 6, or 8).
        pml_width: PML width (single int or [top, bottom, left, right]).
        max_vel: Maximum wave velocity. If None, computed from model.
        Ey_0, Hx_0, Hz_0: Initial field values.
        m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x: Initial CPML memory variables.
        nt: Number of time steps (required if source_amplitude is None).
        model_gradient_sampling_interval: Interval for storing gradient snapshots.
            Values > 1 reduce memory usage during backpropagation.
        freq_taper_frac: Fraction of frequency spectrum to taper (0.0-1.0).
            Helps reduce ringing artifacts during resampling.
        time_pad_frac: Fraction for zero padding before FFT (0.0-1.0).
            Helps reduce wraparound artifacts during resampling.
        time_taper: Whether to apply Hann window (mainly for testing).
        save_snapshots: Whether to save wavefield snapshots for gradient computation.
            If None (default), snapshots are saved only when model parameters
            require gradients. Set to False to disable snapshot saving even
            when gradients are needed (useful for memory-constrained scenarios
            where you'll use randomized trace estimation or other gradient-free
            methods). Set to True to force snapshot saving even without gradients.
        forward_callback: Callback function called during forward propagation.
        backward_callback: Callback function called during backward (adjoint)
            propagation. Receives the same CallbackState as forward_callback,
            but with is_backward=True and gradients available.
        callback_frequency: How often to call the callback.
        python_backend: False for C/CUDA, True or 'eager'/'jit'/'compile' for Python.
        gradient_mode: Gradient computation mode:
            - "snapshot": store Ey/curl(H) snapshots for ASM (CPU/CUDA).
            - "boundary": store a boundary ring and reconstruct during backward (CUDA only).
            - "rwii": reduced-wavefield mode (CUDA only, currently sigma==0 only).
        storage_mode: Where to store intermediate snapshots for the ASM
            backward pass. One of "device", "cpu", "disk", "none", or "auto".
        storage_path: Base path for disk storage when storage_mode="disk".
        storage_compression: Whether to compress stored snapshots.
        storage_bytes_limit_device: Soft limit in bytes for device snapshot
            storage when storage_mode="auto".
        storage_bytes_limit_host: Soft limit in bytes for host snapshot
            storage when storage_mode="auto".
        storage_chunk_steps: Optional chunk size (in stored steps) for
            CPU/disk modes. Currently unused.
        boundary_width: Width of boundary storage region (stage 2 only).
        alpha_rwii: RWII scaling parameter (stage 2 only).
        
    Returns:
        Tuple of (Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x, receiver_amplitudes).
    """
    # Validate resampling parameters
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    freq_taper_frac = validate_freq_taper_frac(freq_taper_frac)
    time_pad_frac = validate_time_pad_frac(time_pad_frac)
    
    # Check inputs
    if source_location is not None and source_location.numel() > 0:
        if source_location[..., 0].max() >= epsilon.shape[-2]:
            raise RuntimeError(
                f"Source location dim 0 must be less than {epsilon.shape[-2]}"
            )
        if source_location[..., 1].max() >= epsilon.shape[-1]:
            raise RuntimeError(
                f"Source location dim 1 must be less than {epsilon.shape[-1]}"
            )

    if receiver_location is not None and receiver_location.numel() > 0:
        if receiver_location[..., 0].max() >= epsilon.shape[-2]:
            raise RuntimeError(
                f"Receiver location dim 0 must be less than {epsilon.shape[-2]}"
            )
        if receiver_location[..., 1].max() >= epsilon.shape[-1]:
            raise RuntimeError(
                f"Receiver location dim 1 must be less than {epsilon.shape[-1]}"
            )

    if not isinstance(callback_frequency, int):
        raise TypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise ValueError("callback_frequency must be positive.")

    device = epsilon.device

    # Normalize grid_spacing to list
    if isinstance(grid_spacing, (int, float)):
        grid_spacing_list = [float(grid_spacing), float(grid_spacing)]
    else:
        grid_spacing_list = list(grid_spacing)

    # Compute maximum velocity if not provided
    if max_vel is None:
        # For EM waves: v = 1 / sqrt(epsilon * mu)
        # For vacuum: v = 1 / sqrt(EP0 * MU0) = c0 ≈ 3e8 m/s
        max_vel_computed = float((1.0 / torch.sqrt(epsilon * mu)).max().item())
    else:
        max_vel_computed = max_vel
    
    # Check CFL condition and compute step_ratio
    inner_dt, step_ratio = cfl_condition(grid_spacing_list, dt, max_vel_computed)
    
    # Upsample source if needed for CFL
    source_amplitude_internal = source_amplitude
    if step_ratio > 1 and source_amplitude is not None and source_amplitude.numel() > 0:
        source_amplitude_internal = upsample(
            source_amplitude,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )
    
    # Compute internal number of time steps
    nt_internal = None
    if nt is not None:
        nt_internal = nt * step_ratio
    elif source_amplitude_internal is not None:
        nt_internal = source_amplitude_internal.shape[-1]
    
    # Call the propagation function with internal dt and upsampled source
    result = maxwell_func(
        python_backend,
        epsilon,
        sigma,
        mu,
        grid_spacing,
        inner_dt,  # Use internal time step for CFL compliance
        source_amplitude_internal,
        source_location,
        receiver_location,
        stencil,
        pml_width,
        max_vel_computed,  # Pass computed max_vel so it's not recomputed
        Ey_0,
        Hx_0,
        Hz_0,
        m_Ey_x,
        m_Ey_z,
        m_Hx_z,
        m_Hz_x,
        nt_internal,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        save_snapshots,
        forward_callback,
        backward_callback,
        callback_frequency,
        gradient_mode,
        storage_mode,
        storage_path,
        storage_compression,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        boundary_width,
        alpha_rwii,
    )
    
    # Unpack result
    Ey_out, Hx_out, Hz_out, m_Ey_x_out, m_Ey_z_out, m_Hx_z_out, m_Hz_x_out, receiver_amplitudes = result
    
    # Downsample receiver data if we upsampled
    if step_ratio > 1 and receiver_amplitudes.numel() > 0:
        receiver_amplitudes = downsample_and_movedim(
            receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )
        # Move time back to first dimension to match expected output format
        receiver_amplitudes = torch.movedim(receiver_amplitudes, -1, 0)
    
    return (
        Ey_out,
        Hx_out,
        Hz_out,
        m_Ey_x_out,
        m_Ey_z_out,
        m_Hx_z_out,
        m_Hz_x_out,
        receiver_amplitudes,
    )


_update_E_jit: Optional[Callable] = None
_update_E_compile: Optional[Callable] = None
_update_H_jit: Optional[Callable] = None
_update_H_compile: Optional[Callable] = None

# These will be set after the functions are defined
_update_E_opt: Optional[Callable] = None
_update_H_opt: Optional[Callable] = None


def maxwell_func(
    python_backend: Union[bool, str],
    *args,
) -> Tuple[
    torch.Tensor,  # Ey
    torch.Tensor,  # Hx
    torch.Tensor,  # Hz
    torch.Tensor,  # m_Ey_x
    torch.Tensor,  # m_Ey_z
    torch.Tensor,  # m_Hx_z
    torch.Tensor,  # m_Hz_x
    torch.Tensor,  # receiver_amplitudes
]:
    """Dispatch to Python or C/CUDA backend for Maxwell propagation."""
    global _update_E_jit, _update_E_compile, _update_E_opt
    global _update_H_jit, _update_H_compile, _update_H_opt

    # Check if we should use Python backend or C/CUDA backend
    use_python = python_backend
    if not use_python:
        # Try to use C/CUDA backend
        try:
            from . import backend_utils
            if not backend_utils.is_backend_available():
                import warnings
                warnings.warn(
                    "C/CUDA backend not available, falling back to Python backend. "
                    "To use the C/CUDA backend, compile the library first.",
                    RuntimeWarning,
                )
                use_python = True
        except ImportError:
            import warnings
            warnings.warn(
                "backend_utils not available, falling back to Python backend.",
                RuntimeWarning,
            )
            use_python = True

    if use_python:
        if python_backend is True or python_backend is False:
            mode = "eager"  # Default to eager
        elif isinstance(python_backend, str):
            mode = python_backend.lower()
        else:
            raise TypeError(
                f"python_backend must be bool or str, but got {type(python_backend)}"
            )

        if mode == "jit":
            if _update_E_jit is None:
                _update_E_jit = torch.jit.script(update_E)
            _update_E_opt = _update_E_jit
            if _update_H_jit is None:
                _update_H_jit = torch.jit.script(update_H)
            _update_H_opt = _update_H_jit
        elif mode == "compile":
            if _update_E_compile is None:
                _update_E_compile = torch.compile(update_E, fullgraph=True)
            _update_E_opt = _update_E_compile
            if _update_H_compile is None:
                _update_H_compile = torch.compile(update_H, fullgraph=True)
            _update_H_opt = _update_H_compile
        elif mode == "eager":
            _update_E_opt = update_E
            _update_H_opt = update_H
        else:
            raise ValueError(f"Unknown python_backend value {mode!r}.")

        return maxwell_python(*args)
    else:
        # Use C/CUDA backend
        return maxwell_c_cuda(*args)


def maxwell_python(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int,
    pml_width: Union[int, Sequence[int]],
    max_vel: Optional[float],
    Ey_0: Optional[torch.Tensor],
    Hx_0: Optional[torch.Tensor],
    Hz_0: Optional[torch.Tensor],
    m_Ey_x_0: Optional[torch.Tensor],
    m_Ey_z_0: Optional[torch.Tensor],
    m_Hx_z_0: Optional[torch.Tensor],
    m_Hz_x_0: Optional[torch.Tensor],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: Optional[bool],
    forward_callback: Optional[Callback],
    backward_callback: Optional[Callback],
    callback_frequency: int,
    gradient_mode: str = "snapshot",
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: bool = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    boundary_width: int = 0,
    alpha_rwii: float = 0.0,
):
    """Performs the forward propagation of the 2D TM Maxwell equations.

    This function implements the FDTD time-stepping loop for the TM mode
    (Ey, Hx, Hz) with CPML absorbing boundary conditions.
    
    - Models are padded by fd_pad + pml_width with replicate mode
    - Wavefields are padded by fd_pad only with zero padding
    - Output wavefields are cropped by fd_pad only (PML region is preserved)

    Args:
        epsilon: Permittivity model [ny, nx].
        sigma: Conductivity model [ny, nx].
        mu: Permeability model [ny, nx].
        grid_spacing: Grid spacing (dy, dx) or single value for both.
        dt: Time step.
        source_amplitude: Source amplitudes [n_shots, n_sources, nt].
        source_location: Source locations [n_shots, n_sources, 2].
        receiver_location: Receiver locations [n_shots, n_receivers, 2].
        stencil: Finite difference stencil order (2, 4, 6, or 8).
        pml_width: PML width on each side [top, bottom, left, right] or single value.
        max_vel: Maximum velocity for PML (if None, computed from model).
        Ey_0, Hx_0, Hz_0: Initial field values.
        m_Ey_x_0, m_Ey_z_0, m_Hx_z_0, m_Hz_x_0: Initial CPML memory variables.
        nt: Number of time steps (required if source_amplitude is None).
        model_gradient_sampling_interval: Interval for storing gradients.
        freq_taper_frac: Frequency taper fraction.
        time_pad_frac: Time padding fraction.
        time_taper: Whether to apply time taper.
        save_snapshots: Whether to save wavefield snapshots for backward pass.
            If None, determined by requires_grad on model parameters.
        forward_callback: Callback function called during propagation.
        callback_frequency: Frequency of callback calls.

    Returns:
        Tuple containing:
            - Ey: Final electric field [n_shots, ny + pml, nx + pml]
            - Hx, Hz: Final magnetic fields
            - m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x: Final CPML memory variables
            - receiver_amplitudes: Recorded data at receivers [nt, n_shots, n_receivers]
    """

    from .common import create_or_pad, zero_interior

    # These should be set by maxwell_func before calling this function
    assert _update_E_opt is not None, "_update_E_opt must be set by maxwell_func"
    assert _update_H_opt is not None, "_update_H_opt must be set by maxwell_func"

    # Validate inputs
    if epsilon.ndim != 2:
        raise RuntimeError("epsilon must be 2D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    device = epsilon.device
    dtype = epsilon.dtype
    model_ny, model_nx = epsilon.shape  # Original model dimensions

    gradient_mode_str = gradient_mode.lower()
    if gradient_mode_str != "snapshot":
        raise NotImplementedError(
            f"gradient_mode={gradient_mode!r} is not implemented yet; "
            "only 'snapshot' is supported."
        )

    storage_mode_str = storage_mode.lower()
    if storage_mode_str in {"cpu", "disk"}:
        raise ValueError(
            "python_backend does not support storage_mode='cpu' or 'disk'. "
            "Use the C/CUDA backend or storage_mode='device'/'none'."
        )
    if storage_compression:
        raise NotImplementedError(
            "storage_compression is not implemented yet; set storage_compression=False."
        )

    # Normalize grid_spacing to list
    if isinstance(grid_spacing, (int, float)):
        grid_spacing = [float(grid_spacing), float(grid_spacing)]
    else:
        grid_spacing = list(grid_spacing)
    dy, dx = grid_spacing

    # Normalize pml_width to list [top, bottom, left, right]
    if isinstance(pml_width, int):
        pml_width_list = [pml_width] * 4
    else:
        pml_width_list = list(pml_width)
        if len(pml_width_list) == 1:
            pml_width_list = pml_width_list * 4
        elif len(pml_width_list) == 2:
            pml_width_list = [pml_width_list[0], pml_width_list[0], pml_width_list[1], pml_width_list[1]]

    # Determine number of time steps
    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided")
        nt = source_amplitude.shape[-1]
    
    # Type cast to ensure nt is int for type checker
    nt_steps: int = int(nt)

    # Determine number of shots
    if source_amplitude is not None and source_amplitude.numel() > 0:
        n_shots = source_amplitude.shape[0]
    elif source_location is not None and source_location.numel() > 0:
        n_shots = source_location.shape[0]
    elif receiver_location is not None and receiver_location.numel() > 0:
        n_shots = receiver_location.shape[0]
    else:
        n_shots = 1

    # Compute maximum velocity for PML if not provided
    if max_vel is None:
        # For EM waves: v = 1 / sqrt(epsilon * mu)
        max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item())

    # Compute PML frequency (dominant frequency estimate)
    pml_freq = 0.5 / dt  # Nyquist as default

    # =========================================================================
    # Padding strategy:
    # - fd_pad: padding for finite difference stencil accuracy
    # - pml_width: padding for PML absorbing layers
    # - Total model padding = fd_pad + pml_width
    # - Wavefield padding = fd_pad only (wavefields include PML region)
    # =========================================================================
    
    # FD padding based on stencil: accuracy // 2
    fd_pad = stencil // 2
    # fd_pad_list: [y0, y1, x0, x1] - for 2D staggered grid, asymmetric because
    # staggered diff a[1:] - a[:-1] reduces array size by 1, so we need fd_pad-1 at end
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
    
    # Total padding for models = fd_pad + pml_width
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]
    
    # Calculate padded dimensions
    # Model is padded by total_pad on each side
    padded_ny = model_ny + total_pad[0] + total_pad[1]
    padded_nx = model_nx + total_pad[2] + total_pad[3]
    
    # Pad model tensors with replicate mode (extend boundary values)
    padded_size = (padded_ny, padded_nx)
    epsilon_padded = create_or_pad(epsilon, total_pad, device, dtype, padded_size, mode='replicate')
    sigma_padded = create_or_pad(sigma, total_pad, device, dtype, padded_size, mode='replicate')
    mu_padded = create_or_pad(mu, total_pad, device, dtype, padded_size, mode='replicate')

    # Prepare update coefficients using padded models
    ca, cb, cq = prepare_parameters(epsilon_padded, sigma_padded, mu_padded, dt)

    # Expand coefficients for batch dimension
    ca = ca[None, :, :]  # [1, padded_ny, padded_nx]
    cb = cb[None, :, :]
    cq = cq[None, :, :]

    # =========================================================================
    # Initialize wavefields
    # Wavefields are padded by fd_pad only (they include the PML region)
    # Size = [n_shots, model_ny + pml_width*2 + fd_pad*2, model_nx + ...]
    # Which equals [n_shots, padded_ny, padded_nx]
    # =========================================================================
    size_with_batch = (n_shots, padded_ny, padded_nx)
    
    # Helper function to initialize wavefields with fd_pad padding
    def init_wavefield(field_0: Optional[torch.Tensor]) -> torch.Tensor:
        """Initialize wavefield with fd_pad zero padding.
        
        Zero padding is used for wavefields because the fd_pad region should
        always be zero after output cropping and re-padding. The staggered grid
        operators only read from this region but don't need non-zero values there
        for correct propagation.
        """
        if field_0 is not None:
            # User provides [n_shots, ny, nx] or [ny, nx]
            if field_0.ndim == 2:
                field_0 = field_0[None, :, :].expand(n_shots, -1, -1)
            # Pad with asymmetric fd_pad_list for staggered grid (zero padding)
            return create_or_pad(field_0, fd_pad_list, device, dtype, size_with_batch, mode='constant')
        return torch.zeros(size_with_batch, device=device, dtype=dtype)

    Ey = init_wavefield(Ey_0)
    Hx = init_wavefield(Hx_0)
    Hz = init_wavefield(Hz_0)
    m_Ey_x = init_wavefield(m_Ey_x_0)
    m_Ey_z = init_wavefield(m_Ey_z_0)
    m_Hx_z = init_wavefield(m_Hx_z_0)
    m_Hz_x = init_wavefield(m_Hz_x_0)
    
    # Zero out interior of PML auxiliary variables (optimization)
    # PML memory variables should only be non-zero in PML regions.
    # This works correctly even with user-provided initial states because:
    # 1. The output preserves PML region (only fd_pad is cropped)
    # 2. zero_interior only zeros the interior, preserving PML boundary values
    # 3. Interior values are already zero in correctly propagated wavefields
    # Dimension mapping for zero_interior:
    # - m_Ey_x: x-direction auxiliary -> dim=1 (zero y-interior, keep x-boundaries)
    # - m_Ey_z: y/z-direction auxiliary -> dim=0 (zero x-interior, keep y-boundaries)
    # - m_Hx_z: y/z-direction auxiliary -> dim=0
    # - m_Hz_x: x-direction auxiliary -> dim=1
    pml_aux_dims = [1, 0, 0, 1]  # [m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x]
    for wf, dim in zip([m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x], pml_aux_dims):
        zero_interior(wf, fd_pad_list, pml_width_list, dim)

    # Set up PML profiles for the padded domain
    pml_profiles, kappa_profiles = staggered.set_pml_profiles(
        pml_width=pml_width_list,
        accuracy=stencil,
        fd_pad=fd_pad_list,
        dt=dt,
        grid_spacing=grid_spacing,
        max_vel=max_vel,
        dtype=dtype,
        device=device,
        pml_freq=pml_freq,
        ny=padded_ny,
        nx=padded_nx,
    )
    # pml_profiles = [ay, ayh, ax, axh, by, byh, bx, bxh]
    ay, ay_h, ax, ax_h, by, by_h, bx, bx_h = pml_profiles
    # kappa_profiles = [ky, kyh, kx, kxh]
    kappa_y, kappa_y_h, kappa_x, kappa_x_h = kappa_profiles

    # Reciprocal grid spacing
    rdy = torch.tensor(1.0 / dy, device=device, dtype=dtype)
    rdx = torch.tensor(1.0 / dx, device=device, dtype=dtype)
    dt_tensor = torch.tensor(dt, device=device, dtype=dtype)

    # =========================================================================
    # Prepare source and receiver indices
    # Original positions are in the un-padded model coordinate system.
    # We need to offset by total_pad (fd_pad + pml_width) to get padded coords.
    # =========================================================================
    flat_model_shape = padded_ny * padded_nx
    
    if source_location is not None and source_location.numel() > 0:
        # Adjust source positions by total padding offset
        source_y = source_location[..., 0] + total_pad[0]  # Add top offset
        source_x = source_location[..., 1] + total_pad[2]  # Add left offset
        sources_i = (source_y * padded_nx + source_x).long()  # [n_shots, n_sources]
        n_sources = source_location.shape[1]
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        # Adjust receiver positions by total padding offset
        receiver_y = receiver_location[..., 0] + total_pad[0]  # Add top offset
        receiver_x = receiver_location[..., 1] + total_pad[2]  # Add left offset
        receivers_i = (receiver_y * padded_nx + receiver_x).long()
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    # Initialize receiver amplitudes
    if n_receivers > 0:
        receiver_amplitudes = torch.zeros(nt_steps, n_shots, n_receivers, device=device, dtype=dtype)
    else:
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

    # Prepare callback data - models dict uses the padded models
    callback_models = {
        "epsilon": epsilon_padded,
        "sigma": sigma_padded,
        "mu": mu_padded,
        "ca": ca,
        "cb": cb,
        "cq": cq,
    }
    
    # Callback fd_pad is the actual fd_pad used for wavefields
    callback_fd_pad = fd_pad_list

    # Source injection coefficient: -cb * dt / (dx * dy)
    # Since our cb already contains dt/epsilon, we need: -cb / (dx * dy)
    # This normalizes the source by cell volume for correct amplitude
    source_coeff = -1.0 / (dx * dy)

    # Time stepping loop
    for step in range(nt_steps):
        # Callback at specified frequency
        if forward_callback is not None and step % callback_frequency == 0:
            callback_wavefields = {
                "Ey": Ey,
                "Hx": Hx,
                "Hz": Hz,
                "m_Ey_x": m_Ey_x,
                "m_Ey_z": m_Ey_z,
                "m_Hx_z": m_Hx_z,
                "m_Hz_x": m_Hz_x,
            }
            # Create CallbackState for standardized interface
            callback_state = CallbackState(
                dt=dt,
                step=step,
                nt=nt_steps,
                wavefields=callback_wavefields,
                models=callback_models,
                gradients=None,  # No gradients during forward pass
                fd_pad=callback_fd_pad,
                pml_width=pml_width_list,
                is_backward=False,
                grid_spacing=[dy, dx],
            )
            forward_callback(callback_state)

        # Update H fields: H^{n+1/2} = H^{n-1/2} + ...
        Hx, Hz, m_Ey_x, m_Ey_z = _update_H_opt(
            cq,
            Hx,
            Hz,
            Ey,
            m_Ey_x,
            m_Ey_z,
            kappa_y,
            kappa_y_h,
            kappa_x,
            kappa_x_h,
            ay,
            ay_h,
            ax,
            ax_h,
            by,
            by_h,
            bx,
            bx_h,
            rdy,
            rdx,
            dt_tensor,
            stencil,
        )

        # Update E field: E^{n+1} = E^n + ...
        Ey, m_Hx_z, m_Hz_x = _update_E_opt(
            ca,
            cb,
            Hx,
            Hz,
            Ey,
            m_Hx_z,
            m_Hz_x,
            kappa_y,
            kappa_y_h,
            kappa_x,
            kappa_x_h,
            ay,
            ay_h,
            ax,
            ax_h,
            by,
            by_h,
            bx,
            bx_h,
            rdy,
            rdx,
            dt_tensor,
            stencil,
        )

        # Inject source into Ey field (after E update, following reference implementation)
        # Source term: Ey += -cb * f * dt / (dx * dz) = -cb * f / (dx * dz) since cb contains dt
        if source_amplitude is not None and source_amplitude.numel() > 0 and n_sources > 0:
            # source_amplitude: [n_shots, n_sources, nt]
            src_amp = source_amplitude[:, :, step]  # [n_shots, n_sources]
            # Get cb at source locations for proper scaling
            cb_flat = cb.reshape(1, flat_model_shape).expand(n_shots, -1)
            cb_at_src = cb_flat.gather(1, sources_i)  # [n_shots, n_sources]
            # Apply source with coefficient: -cb * f / (dx * dy)
            scaled_src = cb_at_src * src_amp * source_coeff
            Ey = (
                Ey.reshape(n_shots, flat_model_shape)
                .scatter_add(1, sources_i, scaled_src)
                .reshape(size_with_batch)
            )

        # Record at receivers (after source injection)
        if n_receivers > 0:
            receiver_amplitudes[step] = (
                Ey.reshape(n_shots, flat_model_shape)
                .gather(1, receivers_i)
            )

    # =========================================================================
    # Output cropping:
    # Only remove fd_pad, keep the PML region in the output wavefields.
    # Output shape: [n_shots, model_ny + pml_width_y, model_nx + pml_width_x]
    # =========================================================================
    s = (
        slice(None),  # batch dimension
        slice(fd_pad_list[0], padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None),
        slice(fd_pad_list[2], padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None),
    )
    
    return (
        Ey[s],
        Hx[s],
        Hz[s],
        m_Ey_x[s],
        m_Ey_z[s],
        m_Hx_z[s],
        m_Hz_x[s],
        receiver_amplitudes,
    )


def update_E(
    ca: torch.Tensor,
    cb: torch.Tensor,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    Ey: torch.Tensor,
    m_Hx_z: torch.Tensor,
    m_Hz_x: torch.Tensor,
    kappa_y: torch.Tensor,
    kappa_y_h: torch.Tensor,
    kappa_x: torch.Tensor,
    kappa_x_h: torch.Tensor,
    ay: torch.Tensor,
    ay_h: torch.Tensor,
    ax: torch.Tensor,
    ax_h: torch.Tensor,
    by: torch.Tensor,
    by_h: torch.Tensor,
    bx: torch.Tensor,
    bx_h: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    dt: torch.Tensor,
    stencil: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Update electric field Ey with CPML absorbing boundary conditions.

    For TM mode, the update equation is:
        Ey^{n+1} = Ca * Ey^n + Cb * (dHz/dx - dHx/dz)

    With CPML, we split the derivatives and apply auxiliary variables:
        dHz/dx -> dHz/dx / kappa_x + m_Hz_x
        dHx/dz -> dHx/dz / kappa_y + m_Hx_z

    Args:
        ca, cb: Update coefficients from material parameters
        Hx, Hz: Magnetic field components
        Ey: Electric field component to update
        m_Hx_z, m_Hz_x: CPML auxiliary memory variables
        kappa_y, kappa_y_h: CPML kappa profiles in y direction
        kappa_x, kappa_x_h: CPML kappa profiles in x direction
        ay, ay_h, ax, ax_h: CPML a coefficients
        by, by_h, bx, bx_h: CPML b coefficients
        rdy, rdx: Reciprocal of grid spacing (1/dy, 1/dx)
        dt: Time step
        stencil: Finite difference stencil order (2, 4, 6, or 8)

    Returns:
        Updated Ey, m_Hx_z, m_Hz_x
    """


    # Compute spatial derivatives using staggered grid operators
    # dHz/dx at integer grid points (where Ey lives)
    dHz_dx = staggered.diffx1(Hz, stencil, rdx)
    # dHx/dz at integer grid points (where Ey lives)
    dHx_dz = staggered.diffy1(Hx, stencil, rdy)

    # Update CPML auxiliary variables using standard CPML recursion:
    # psi_new = b * psi_old + a * derivative
    # m_Hz_x stores the x-direction PML memory for Hz derivative
    m_Hz_x = bx * m_Hz_x + ax * dHz_dx
    # m_Hx_z stores the z-direction PML memory for Hx derivative
    m_Hx_z = by * m_Hx_z + ay * dHx_dz

    # Apply CPML correction to derivatives
    # In CPML: d/dx -> (1/kappa) * d/dx + m
    dHz_dx_pml = dHz_dx / kappa_x + m_Hz_x
    dHx_dz_pml = dHx_dz / kappa_y + m_Hx_z

    # Update Ey using the FDTD update equation
    # Ey^{n+1} = Ca * Ey^n + Cb * (dHz/dx - dHx/dz)
    Ey = ca * Ey + cb * (dHz_dx_pml - dHx_dz_pml)

    return Ey, m_Hx_z, m_Hz_x


def update_H(
    cq: torch.Tensor,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    Ey: torch.Tensor,
    m_Ey_x: torch.Tensor,
    m_Ey_z: torch.Tensor,
    kappa_y: torch.Tensor,
    kappa_y_h: torch.Tensor,
    kappa_x: torch.Tensor,
    kappa_x_h: torch.Tensor,
    ay: torch.Tensor,
    ay_h: torch.Tensor,
    ax: torch.Tensor,
    ax_h: torch.Tensor,
    by: torch.Tensor,
    by_h: torch.Tensor,
    bx: torch.Tensor,
    bx_h: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    dt: torch.Tensor,
    stencil: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Update magnetic fields Hx and Hz with CPML absorbing boundary conditions.

    For TM mode, the update equations are:
        Hx^{n+1} = Hx^n - Cq * dEy/dz
        Hz^{n+1} = Hz^n + Cq * dEy/dx

    With CPML, we use half-grid derivatives and auxiliary variables:
        dEy/dz -> dEy/dz / kappa_y_h + m_Ey_z
        dEy/dx -> dEy/dx / kappa_x_h + m_Ey_x

    Args:
        cq: Update coefficient (dt/mu)
        Hx, Hz: Magnetic field components to update
        Ey: Electric field component
        m_Ey_x, m_Ey_z: CPML auxiliary memory variables
        kappa_y, kappa_y_h: CPML kappa profiles in y direction (integer and half grid)
        kappa_x, kappa_x_h: CPML kappa profiles in x direction (integer and half grid)
        ay, ay_h, ax, ax_h: CPML a coefficients
        by, by_h, bx, bx_h: CPML b coefficients
        rdy, rdx: Reciprocal of grid spacing (1/dy, 1/dx)
        dt: Time step
        stencil: Finite difference stencil order (2, 4, 6, or 8)

    Returns:
        Updated Hx, Hz, m_Ey_x, m_Ey_z
    """

    # Compute spatial derivatives at half grid points (where H fields live)
    # dEy/dz at half grid points in z (for Hx update)
    dEy_dz = staggered.diffyh1(Ey, stencil, rdy)
    # dEy/dx at half grid points in x (for Hz update)
    dEy_dx = staggered.diffxh1(Ey, stencil, rdx)

    # Update CPML auxiliary variables using standard CPML recursion:
    # psi_new = b * psi_old + a * derivative
    # m_Ey_z stores the z-direction PML memory for Ey derivative (used in Hx update)
    m_Ey_z = by_h * m_Ey_z + ay_h * dEy_dz
    # m_Ey_x stores the x-direction PML memory for Ey derivative (used in Hz update)
    m_Ey_x = bx_h * m_Ey_x + ax_h * dEy_dx

    # Apply CPML correction to derivatives
    # In CPML: d/dz -> (1/kappa_h) * d/dz + m
    dEy_dz_pml = dEy_dz / kappa_y_h + m_Ey_z
    dEy_dx_pml = dEy_dx / kappa_x_h + m_Ey_x

    # Update Hx using the FDTD update equation
    # Hx^{n+1} = Hx^n - Cq * dEy/dz
    Hx = Hx - cq * dEy_dz_pml

    # Update Hz using the FDTD update equation
    # Hz^{n+1} = Hz^n + Cq * dEy/dx
    Hz = Hz + cq * dEy_dx_pml

    return Hx, Hz, m_Ey_x, m_Ey_z


# Initialize the optimized function pointers to the default implementations
_update_E_opt = update_E
_update_H_opt = update_H


def maxwell_c_cuda(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int,
    pml_width: Union[int, Sequence[int]],
    max_vel: Optional[float],
    Ey_0: Optional[torch.Tensor],
    Hx_0: Optional[torch.Tensor],
    Hz_0: Optional[torch.Tensor],
    m_Ey_x_0: Optional[torch.Tensor],
    m_Ey_z_0: Optional[torch.Tensor],
    m_Hx_z_0: Optional[torch.Tensor],
    m_Hz_x_0: Optional[torch.Tensor],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: Optional[bool],
    forward_callback: Optional[Callback],
    backward_callback: Optional[Callback],
    callback_frequency: int,
    gradient_mode: str = "snapshot",
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: bool = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    boundary_width: int = 0,
    alpha_rwii: float = 0.0,
):
    """Performs Maxwell propagation using C/CUDA backend.

    This function provides the interface to the compiled C/CUDA implementations
    for high-performance wave propagation.
    
    Padding strategy:
    - Models are padded by fd_pad + pml_width with replicate mode
    - Wavefields are padded by fd_pad only with zero padding
    - Output wavefields are cropped by fd_pad only (PML region is preserved)

    Args:
        Same as maxwell_python.

    Returns:
        Same as maxwell_python.
    """
    from . import backend_utils
    from . import staggered
    from .common import create_or_pad, zero_interior

    # Validate inputs
    if epsilon.ndim != 2:
        raise RuntimeError("epsilon must be 2D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    device = epsilon.device
    dtype = epsilon.dtype
    model_ny, model_nx = epsilon.shape  # Original model dimensions

    # Normalize grid_spacing to list
    if isinstance(grid_spacing, (int, float)):
        grid_spacing = [float(grid_spacing), float(grid_spacing)]
    else:
        grid_spacing = list(grid_spacing)
    dy, dx = grid_spacing

    # Normalize pml_width to list [top, bottom, left, right]
    if isinstance(pml_width, int):
        pml_width_list = [pml_width] * 4
    else:
        pml_width_list = list(pml_width)
        if len(pml_width_list) == 1:
            pml_width_list = pml_width_list * 4
        elif len(pml_width_list) == 2:
            pml_width_list = [pml_width_list[0], pml_width_list[0], pml_width_list[1], pml_width_list[1]]

    # Determine number of time steps
    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided")
        nt = source_amplitude.shape[-1]

    # Ensure nt is an integer for iteration
    nt_steps: int = int(nt)
    # Clamp gradient sampling interval to a sensible range for storage/backprop
    gradient_sampling_interval = int(model_gradient_sampling_interval)
    if gradient_sampling_interval < 1:
        gradient_sampling_interval = 1
    if nt_steps > 0:
        gradient_sampling_interval = min(gradient_sampling_interval, nt_steps)

    # Determine number of shots
    if source_amplitude is not None and source_amplitude.numel() > 0:
        n_shots = source_amplitude.shape[0]
    elif source_location is not None and source_location.numel() > 0:
        n_shots = source_location.shape[0]
    elif receiver_location is not None and receiver_location.numel() > 0:
        n_shots = receiver_location.shape[0]
    else:
        n_shots = 1

    # Compute maximum velocity for PML if not provided
    if max_vel is None:
        max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item())

    # Compute PML frequency
    pml_freq = 0.5 / dt

    # =========================================================================
    # Padding strategy:
    # - fd_pad: padding for finite difference stencil accuracy
    # - pml_width: padding for PML absorbing layers
    # - Total model padding = fd_pad + pml_width
    # - Wavefield padding = fd_pad only (wavefields include PML region)
    # =========================================================================
    
    # FD padding based on stencil: accuracy // 2
    fd_pad = stencil // 2
    # fd_pad_list: [y0, y1, x0, x1] - asymmetric for staggered grid
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
    
    # Total padding for models = fd_pad + pml_width
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]
    
    # Calculate padded dimensions
    padded_ny = model_ny + total_pad[0] + total_pad[1]
    padded_nx = model_nx + total_pad[2] + total_pad[3]
    
    # Pad model tensors with replicate mode (extend boundary values)
    padded_size = (padded_ny, padded_nx)
    epsilon_padded = create_or_pad(epsilon, total_pad, device, dtype, padded_size, mode='replicate')
    sigma_padded = create_or_pad(sigma, total_pad, device, dtype, padded_size, mode='replicate')
    mu_padded = create_or_pad(mu, total_pad, device, dtype, padded_size, mode='replicate')

    # Prepare update coefficients using padded models
    ca, cb, cq = prepare_parameters(epsilon_padded, sigma_padded, mu_padded, dt)

    # Initialize fields with padded dimensions
    size_with_batch = (n_shots, padded_ny, padded_nx)

    def init_wavefield(field_0: Optional[torch.Tensor]) -> torch.Tensor:
        """Initialize wavefield with fd_pad zero padding."""
        if field_0 is not None:
            if field_0.ndim == 2:
                field_0 = field_0[None, :, :].expand(n_shots, -1, -1)
            # Pad with asymmetric fd_pad_list for staggered grid
            return create_or_pad(field_0, fd_pad_list, device, dtype, size_with_batch).contiguous()
        return torch.zeros(size_with_batch, device=device, dtype=dtype)

    Ey = init_wavefield(Ey_0)
    Hx = init_wavefield(Hx_0)
    Hz = init_wavefield(Hz_0)
    m_Ey_x = init_wavefield(m_Ey_x_0)
    m_Ey_z = init_wavefield(m_Ey_z_0)
    m_Hx_z = init_wavefield(m_Hx_z_0)
    m_Hz_x = init_wavefield(m_Hz_x_0)
    
    # Zero out interior of PML auxiliary variables (optimization)
    # This works correctly with user-provided states (see forward pass comments)
    pml_aux_dims = [1, 0, 0, 1]  # [m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x]
    for wf, dim in zip([m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x], pml_aux_dims):
        zero_interior(wf, fd_pad_list, pml_width_list, dim)

    # Set up PML profiles for the padded domain
    pml_profiles, kappa_profiles = staggered.set_pml_profiles(
        pml_width=pml_width_list,
        accuracy=stencil,
        fd_pad=fd_pad_list,
        dt=dt,
        grid_spacing=grid_spacing,
        max_vel=max_vel,
        dtype=dtype,
        device=device,
        pml_freq=pml_freq,
        ny=padded_ny,
        nx=padded_nx,
    )
    ay, ay_h, ax, ax_h, by, by_h, bx, bx_h = pml_profiles
    ky, ky_h, kx, kx_h = kappa_profiles

    # Flatten PML profiles for C backend (remove batch dimensions)
    ay_flat = ay.squeeze().contiguous()
    ay_h_flat = ay_h.squeeze().contiguous()
    ax_flat = ax.squeeze().contiguous()
    ax_h_flat = ax_h.squeeze().contiguous()
    by_flat = by.squeeze().contiguous()
    by_h_flat = by_h.squeeze().contiguous()
    bx_flat = bx.squeeze().contiguous()
    bx_h_flat = bx_h.squeeze().contiguous()

    # Flatten kappa profiles for C backend
    ky_flat = ky.squeeze().contiguous()
    ky_h_flat = ky_h.squeeze().contiguous()
    kx_flat = kx.squeeze().contiguous()
    kx_h_flat = kx_h.squeeze().contiguous()

    # =========================================================================
    # Prepare source and receiver indices
    # Original positions are in the un-padded model coordinate system.
    # We need to offset by total_pad (fd_pad + pml_width) to get padded coords.
    # =========================================================================
    flat_model_shape = padded_ny * padded_nx

    if source_location is not None and source_location.numel() > 0:
        # Adjust source positions by total padding offset
        source_y = source_location[..., 0] + total_pad[0]
        source_x = source_location[..., 1] + total_pad[2]
        sources_i = (source_y * padded_nx + source_x).long().contiguous()
        n_sources = source_location.shape[1]
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        # Adjust receiver positions by total padding offset
        receiver_y = receiver_location[..., 0] + total_pad[0]
        receiver_x = receiver_location[..., 1] + total_pad[2]
        receivers_i = (receiver_y * padded_nx + receiver_x).long().contiguous()
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    # Prepare source amplitudes with proper scaling
    if source_amplitude is not None and source_amplitude.numel() > 0:
        source_coeff = -1.0 / (dx * dy)
        # Expand cb to batch dimension for gather
        cb_expanded = cb[None, :, :].expand(n_shots, -1, -1)
        cb_flat = cb_expanded.reshape(n_shots, flat_model_shape)
        cb_at_src = cb_flat.gather(1, sources_i)
        # Reshape source amplitude: [shot, source, time] -> [time, shot, source]
        f = source_amplitude.permute(2, 0, 1).contiguous()
        # Scale by cb and source coefficient
        f = f * cb_at_src[None, :, :] * source_coeff
        f = f.reshape(nt_steps * n_shots * n_sources)
    else:
        f = torch.empty(0, device=device, dtype=dtype)

    # Flatten fields for C backend
    Ey = Ey.contiguous()
    Hx = Hx.contiguous()
    Hz = Hz.contiguous()
    m_Ey_x = m_Ey_x.contiguous()
    m_Ey_z = m_Ey_z.contiguous()
    m_Hx_z = m_Hx_z.contiguous()
    m_Hz_x = m_Hz_x.contiguous()

    # Flatten coefficients (add batch dimension for consistency)
    ca = ca[None, :, :].contiguous()
    cb = cb[None, :, :].contiguous()
    cq = cq[None, :, :].contiguous()

    # PML boundaries (where PML starts in the padded domain)
    pml_y0 = fd_pad_list[0] + pml_width_list[0]
    pml_y1 = padded_ny - fd_pad_list[1] - pml_width_list[1]
    pml_x0 = fd_pad_list[2] + pml_width_list[2]
    pml_x1 = padded_nx - fd_pad_list[3] - pml_width_list[3]

    # Determine if any input requires gradients
    requires_grad = epsilon.requires_grad or sigma.requires_grad

    gradient_mode_str = gradient_mode.lower()
    if gradient_mode_str not in {"snapshot", "boundary", "rwii"}:
        raise ValueError(
            "gradient_mode must be 'snapshot', 'boundary', or 'rwii', "
            f"but got {gradient_mode!r}"
        )
    if gradient_mode_str == "boundary" and device.type != "cuda":
        raise NotImplementedError(
            "gradient_mode='boundary' is only supported on CUDA currently."
        )
    if gradient_mode_str == "rwii" and device.type != "cuda":
        raise NotImplementedError(
            "gradient_mode='rwii' is only supported on CUDA currently."
        )

    functorch_active = torch._C._are_functorch_transforms_active()
    if functorch_active and gradient_mode_str != "snapshot":
        raise NotImplementedError(
            "torch.func.jvp is only supported with gradient_mode='snapshot' "
            "for the C/CUDA backend."
        )

    boundary_indices: Optional[torch.Tensor] = None

    storage_bf16 = bool(storage_compression)
    if storage_bf16:
        if device.type != "cuda":
            raise NotImplementedError(
                "storage_compression (BF16 snapshot storage) is only supported on CUDA."
            )
        if dtype != torch.float32:
            raise NotImplementedError(
                "storage_compression (BF16 snapshot storage) is only supported for float32."
            )
    
    # Determine if we should save snapshots for backward pass
    if save_snapshots is None:
        do_save_snapshots = requires_grad
    else:
        do_save_snapshots = save_snapshots
        
    # If save_snapshots is False but requires_grad is True, warn user
    if requires_grad and save_snapshots is False:
        import warnings
        warnings.warn(
            "save_snapshots=False but model parameters require gradients. "
            "Backward pass will fail. Consider using randomized trace estimation "
            "or other gradient-free optimization methods.",
            UserWarning,
        )

    storage_mode_str = storage_mode.lower()
    if storage_mode_str not in {"device", "cpu", "disk", "none", "auto"}:
        raise ValueError(
            "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
            f"but got {storage_mode!r}"
        )
    if device.type == "cpu" and storage_mode_str == "cpu":
        storage_mode_str = "device"

    needs_storage = do_save_snapshots and requires_grad
    effective_storage_mode_str = storage_mode_str
    if not needs_storage:
        if effective_storage_mode_str == "auto":
            effective_storage_mode_str = "none"
    else:
        if effective_storage_mode_str == "none":
            raise ValueError(
                "storage_mode='none' is not compatible with "
                f"gradient_mode={gradient_mode!r} when gradients are required."
            )
        if effective_storage_mode_str == "auto":
            dtype_size = 2 if storage_bf16 else epsilon.element_size()
            if gradient_mode_str == "snapshot":
                # Estimate required bytes for storing Ey and curl_H.
                num_elements_per_shot = padded_ny * padded_nx
                shot_bytes_uncomp = num_elements_per_shot * dtype_size
                n_stored = (nt_steps + gradient_sampling_interval - 1) // gradient_sampling_interval
                total_bytes = n_stored * n_shots * shot_bytes_uncomp * 2  # Ey + curl_H
            else:
                # Boundary storage: store Ey/Hx/Hz boundary ring for every step (nt+1).
                if boundary_width <= 0:
                    boundary_width = fd_pad
                if boundary_width < fd_pad:
                    raise ValueError(
                        f"boundary_width must be >= {fd_pad} for stencil={stencil}."
                    )
                boundary_indices = _compute_boundary_indices_flat(
                    ny=padded_ny,
                    nx=padded_nx,
                    pml_y0=pml_y0,
                    pml_x0=pml_x0,
                    pml_y1=pml_y1,
                    pml_x1=pml_x1,
                    boundary_width=boundary_width,
                    device=device,
                )
                boundary_numel = int(boundary_indices.numel())
                shot_bytes_uncomp = boundary_numel * dtype_size
                total_bytes = (nt_steps + 1) * n_shots * shot_bytes_uncomp * 3  # Ey/Hx/Hz

            limit_device = (
                storage_bytes_limit_device
                if storage_bytes_limit_device is not None
                else float("inf")
            )
            limit_host = (
                storage_bytes_limit_host
                if storage_bytes_limit_host is not None
                else float("inf")
            )
            import warnings

            if device.type == "cuda" and total_bytes <= limit_device:
                effective_storage_mode_str = "device"
            elif total_bytes <= limit_host:
                effective_storage_mode_str = "cpu"
            else:
                effective_storage_mode_str = "disk"

            warnings.warn(
                f"storage_mode='auto' selected storage_mode='{effective_storage_mode_str}' "
                f"for estimated storage size {total_bytes / 1e9:.2f} GB.",
                RuntimeWarning,
            )

    # Callback fd_pad is the actual fd_pad used for wavefields
    callback_fd_pad = fd_pad_list
    
    # Callback models dict
    callback_models = {
        "epsilon": epsilon_padded,
        "sigma": sigma_padded,
        "mu": mu_padded,
        "ca": ca,
        "cb": cb,
        "cq": cq,
    }

    use_autograd_fn = (do_save_snapshots and requires_grad) or functorch_active
    if use_autograd_fn:
        # Use autograd Function for gradient computation
        if gradient_mode_str == "snapshot":
            result = MaxwellTMForwardFunc.apply(
                ca,
                cb,
                cq,
                f,
                ay_flat,
                by_flat,
                ay_h_flat,
                by_h_flat,
                ax_flat,
                bx_flat,
                ax_h_flat,
                bx_h_flat,
                ky_flat,
                ky_h_flat,
                kx_flat,
                kx_h_flat,
                sources_i,
                receivers_i,
                1.0 / dy,  # rdy
                1.0 / dx,  # rdx
                dt,
                nt_steps,
                n_shots,
                padded_ny,
                padded_nx,
                n_sources,
                n_receivers,
                gradient_sampling_interval,  # step_ratio
                stencil,  # accuracy
                False,  # ca_batched
                False,  # cb_batched
                False,  # cq_batched
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                tuple(fd_pad_list),  # fd_pad for callback
                tuple(pml_width_list),  # pml_width for callback
                callback_models,  # models dict for callback
                forward_callback,
                backward_callback,
                callback_frequency,
                effective_storage_mode_str,
                storage_path,
                storage_compression,
                Ey,
                Hx,
                Hz,
                m_Ey_x,
                m_Ey_z,
                m_Hx_z,
                m_Hz_x,
            )
        elif gradient_mode_str == "boundary":
            import warnings

            if forward_callback is not None or backward_callback is not None:
                raise NotImplementedError(
                    "Callbacks are not supported yet for gradient_mode='boundary'."
                )

            if boundary_width <= 0:
                boundary_width = fd_pad
            if boundary_width < fd_pad:
                raise ValueError(
                    f"boundary_width must be >= {fd_pad} for stencil={stencil}."
                )
            if gradient_sampling_interval != 1:
                warnings.warn(
                    "gradient_mode='boundary' requires model_gradient_sampling_interval=1; "
                    f"got {gradient_sampling_interval}, forcing to 1.",
                    RuntimeWarning,
                )
                gradient_sampling_interval = 1

            if boundary_indices is None:
                boundary_indices = _compute_boundary_indices_flat(
                    ny=padded_ny,
                    nx=padded_nx,
                    pml_y0=pml_y0,
                    pml_x0=pml_x0,
                    pml_y1=pml_y1,
                    pml_x1=pml_x1,
                    boundary_width=boundary_width,
                    device=device,
                )

            result = MaxwellTMForwardBoundaryFunc.apply(
                ca,
                cb,
                cq,
                f,
                boundary_indices,
                ay_flat,
                by_flat,
                ay_h_flat,
                by_h_flat,
                ax_flat,
                bx_flat,
                ax_h_flat,
                bx_h_flat,
                ky_flat,
                ky_h_flat,
                kx_flat,
                kx_h_flat,
                sources_i,
                receivers_i,
                1.0 / dy,  # rdy
                1.0 / dx,  # rdx
                dt,
                nt_steps,
                n_shots,
                padded_ny,
                padded_nx,
                n_sources,
                n_receivers,
                stencil,  # accuracy
                False,  # ca_batched
                False,  # cb_batched
                False,  # cq_batched
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                effective_storage_mode_str,
                storage_path,
                storage_compression,
                Ey,
                Hx,
                Hz,
                m_Ey_x,
                m_Ey_z,
                m_Hx_z,
                m_Hz_x,
            )
        else:
            import warnings

            if forward_callback is not None or backward_callback is not None:
                raise NotImplementedError(
                    "Callbacks are not supported yet for gradient_mode='rwii'."
                )
            if alpha_rwii == 0.0:
                raise ValueError("alpha_rwii must be non-zero for gradient_mode='rwii'.")
            if boundary_width <= 0:
                boundary_width = fd_pad
            if boundary_width < fd_pad:
                raise ValueError(
                    f"boundary_width must be >= {fd_pad} for stencil={stencil}."
                )
            if gradient_sampling_interval != 1:
                warnings.warn(
                    "gradient_mode='rwii' requires model_gradient_sampling_interval=1; "
                    f"got {gradient_sampling_interval}, forcing to 1.",
                    RuntimeWarning,
                )
                gradient_sampling_interval = 1

            if boundary_indices is None:
                boundary_indices = _compute_boundary_indices_flat(
                    ny=padded_ny,
                    nx=padded_nx,
                    pml_y0=pml_y0,
                    pml_x0=pml_x0,
                    pml_y1=pml_y1,
                    pml_x1=pml_x1,
                    boundary_width=boundary_width,
                    device=device,
                )

            # RWII assumes (approximately) time-reversible propagation.
            # For conductive media (sigma > 0), back-propagating by inverting the
            # update equations becomes anti-damped (division by ca<1), which can be
            # numerically unstable and is not guaranteed to match the true adjoint.
            # We allow it (with a warning) so users can experiment, but keep a hard
            # guard against near-zero ca which will blow up quickly.
            if sigma_padded.numel() > 0:
                sigma_max = float(sigma_padded.abs().max().item())
                if sigma_max != 0.0:
                    ca_min_abs = float(ca.abs().min().item())
                    if ca_min_abs < 1e-2:
                        raise ValueError(
                            "gradient_mode='rwii' is likely unstable for this sigma/dt/epsilon: "
                            f"min |ca|={ca_min_abs:.3e} (max |sigma|={sigma_max:.3e}). "
                            "Try smaller dt, smaller sigma, or use gradient_mode='boundary'/'snapshot'."
                        )
                    import warnings

                    warnings.warn(
                        "gradient_mode='rwii' with sigma!=0 is not guaranteed to be stable or correct "
                        f"(max |sigma|={sigma_max:.3e}, min |ca|={ca_min_abs:.3e}). "
                        "If you see NaNs/explosions, use gradient_mode='boundary'/'snapshot'.",
                        RuntimeWarning,
                    )

            result = MaxwellTMForwardRWIIFunc.apply(
                ca,
                cb,
                cq,
                f,
                boundary_indices,
                ay_flat,
                by_flat,
                ay_h_flat,
                by_h_flat,
                ax_flat,
                bx_flat,
                ax_h_flat,
                bx_h_flat,
                ky_flat,
                ky_h_flat,
                kx_flat,
                kx_h_flat,
                sources_i,
                receivers_i,
                1.0 / dy,  # rdy
                1.0 / dx,  # rdx
                dt,
                nt_steps,
                n_shots,
                padded_ny,
                padded_nx,
                n_sources,
                n_receivers,
                stencil,  # accuracy
                False,  # ca_batched
                False,  # cb_batched
                False,  # cq_batched
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                effective_storage_mode_str,
                storage_path,
                storage_compression,
                float(alpha_rwii),
                Ey,
                Hx,
                Hz,
                m_Ey_x,
                m_Ey_z,
                m_Hx_z,
                m_Hz_x,
            )
        
        # Unpack result (drop context handle if present)
        if len(result) == 9:
            (
                Ey_out,
                Hx_out,
                Hz_out,
                m_Ey_x_out,
                m_Ey_z_out,
                m_Hx_z_out,
                m_Hz_x_out,
                receiver_amplitudes,
                _ctx_handle,
            ) = result
        else:
            (
                Ey_out,
                Hx_out,
                Hz_out,
                m_Ey_x_out,
                m_Ey_z_out,
                m_Hx_z_out,
                m_Hz_x_out,
                receiver_amplitudes,
            ) = result  # type: ignore
        # Output cropping: only remove fd_pad, keep PML region
        s = (
            slice(None),  # batch dimension
            slice(fd_pad_list[0], padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None),
            slice(fd_pad_list[2], padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None),
        )
        
        return (
            Ey_out[s],
            Hx_out[s],
            Hz_out[s],
            m_Ey_x_out[s],
            m_Ey_z_out[s],
            m_Hx_z_out[s],
            m_Hz_x_out[s],
            receiver_amplitudes,
        )
    else:
        # Direct call without autograd for inference
        # Get the backend function
        try:
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm", "forward", stencil, dtype, device
            )
        except AttributeError as e:
            raise RuntimeError(
                f"C/CUDA backend function not available for accuracy={stencil}, "
                f"dtype={dtype}, device={device}. Error: {e}"
            )

        # Get device index for CUDA
        device_idx = device.index if device.type == "cuda" and device.index is not None else 0

        # Initialize receiver amplitudes
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(nt_steps, n_shots, n_receivers, device=device, dtype=dtype)
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        
        # If no callback is provided, run entire propagation in single call
        # Otherwise, chunk into callback_frequency steps
        if forward_callback is None:
            effective_callback_freq = nt_steps
        else:
            effective_callback_freq = callback_frequency
        
        # Main time-stepping loop with chunked calls for callback support
        for step in range(0, nt_steps, effective_callback_freq):
            # Call callback at the start of each chunk
            if forward_callback is not None:
                callback_wavefields = {
                    "Ey": Ey,
                    "Hx": Hx,
                    "Hz": Hz,
                    "m_Ey_x": m_Ey_x,
                    "m_Ey_z": m_Ey_z,
                    "m_Hx_z": m_Hx_z,
                    "m_Hz_x": m_Hz_x,
                }
                callback_state = CallbackState(
                    dt=dt,
                    step=step,
                    nt=nt_steps,
                    wavefields=callback_wavefields,
                    models=callback_models,
                    gradients=None,
                    fd_pad=callback_fd_pad,
                    pml_width=pml_width_list,
                    is_backward=False,
                    grid_spacing=[dy, dx],
                )
                forward_callback(callback_state)
            
            # Number of steps to propagate in this chunk
            step_nt = min(nt_steps - step, effective_callback_freq)
            
            # Call the C/CUDA function for this chunk
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(f),
                backend_utils.tensor_to_ptr(Ey),
                backend_utils.tensor_to_ptr(Hx),
                backend_utils.tensor_to_ptr(Hz),
                backend_utils.tensor_to_ptr(m_Ey_x),
                backend_utils.tensor_to_ptr(m_Ey_z),
                backend_utils.tensor_to_ptr(m_Hx_z),
                backend_utils.tensor_to_ptr(m_Hz_x),
                backend_utils.tensor_to_ptr(receiver_amplitudes),
                backend_utils.tensor_to_ptr(ay_flat),
                backend_utils.tensor_to_ptr(by_flat),
                backend_utils.tensor_to_ptr(ay_h_flat),
                backend_utils.tensor_to_ptr(by_h_flat),
                backend_utils.tensor_to_ptr(ax_flat),
                backend_utils.tensor_to_ptr(bx_flat),
                backend_utils.tensor_to_ptr(ax_h_flat),
                backend_utils.tensor_to_ptr(bx_h_flat),
                backend_utils.tensor_to_ptr(ky_flat),
                backend_utils.tensor_to_ptr(ky_h_flat),
                backend_utils.tensor_to_ptr(kx_flat),
                backend_utils.tensor_to_ptr(kx_h_flat),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                1.0 / dy,  # rdy
                1.0 / dx,  # rdx
                dt,
                step_nt,  # nt for this chunk
                n_shots,
                padded_ny,
                padded_nx,
                n_sources,
                n_receivers,
                gradient_sampling_interval,  # step_ratio
                False,  # ca_batched
                False,  # cb_batched
                False,  # cq_batched
                step,  # start_t - crucial for correct source injection timing
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                device_idx,
            )

        # Output cropping: only remove fd_pad, keep PML region
        s = (
            slice(None),  # batch dimension
            slice(fd_pad_list[0], padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None),
            slice(fd_pad_list[2], padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None),
        )

        return (
            Ey[s],
            Hx[s],
            Hz[s],
            m_Ey_x[s],
            m_Ey_z[s],
            m_Hx_z[s],
            m_Hz_x[s],
            receiver_amplitudes,
        )


class MaxwellTMForwardFunc(torch.autograd.Function):
    """Autograd function for the forward pass of Maxwell TM wave propagation.

    This class defines the forward and backward passes for the 2D TM mode
    Maxwell equations, allowing PyTorch to compute gradients through the wave
    propagation operation. It interfaces directly with the C/CUDA backend.
    
    The backward pass uses the Adjoint State Method (ASM) which requires
    storing forward wavefield values at each step_ratio interval for
    gradient computation.
    """

    @staticmethod
    def forward(
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
        ay: torch.Tensor,
        by: torch.Tensor,
        ay_h: torch.Tensor,
        by_h: torch.Tensor,
        ax: torch.Tensor,
        bx: torch.Tensor,
        ax_h: torch.Tensor,
        bx_h: torch.Tensor,
        ky: torch.Tensor,
        ky_h: torch.Tensor,
        kx: torch.Tensor,
        kx_h: torch.Tensor,
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        rdy: float,
        rdx: float,
        dt: float,
        nt: int,
        n_shots: int,
        ny: int,
        nx: int,
        n_sources: int,
        n_receivers: int,
        step_ratio: int,
        accuracy: int,
        ca_batched: bool,
        cb_batched: bool,
        cq_batched: bool,
        pml_y0: int,
        pml_x0: int,
        pml_y1: int,
        pml_x1: int,
        fd_pad: Tuple[int, int, int, int],
        pml_width: Tuple[int, int, int, int],
        models: dict,
        forward_callback: Optional[Callback],
        backward_callback: Optional[Callback],
        callback_frequency: int,
        storage_mode_str: str,
        storage_path: str,
        storage_compression: bool,
        Ey: torch.Tensor,
        Hx: torch.Tensor,
        Hz: torch.Tensor,
        m_Ey_x: torch.Tensor,
        m_Ey_z: torch.Tensor,
        m_Hx_z: torch.Tensor,
        m_Hz_x: torch.Tensor,
    ) -> Tuple[Any, ...]:
        """Performs the forward propagation of the Maxwell TM equations."""
        from . import backend_utils

        device = Ey.device
        dtype = Ey.dtype

        ca_requires_grad = ca.requires_grad
        cb_requires_grad = cb.requires_grad
        needs_grad = ca_requires_grad or cb_requires_grad

        jvp_initial = (
            Ey.detach().clone(),
            Hx.detach().clone(),
            Hz.detach().clone(),
            m_Ey_x.detach().clone(),
            m_Ey_z.detach().clone(),
            m_Hx_z.detach().clone(),
            m_Hz_x.detach().clone(),
        )

        # Initialize receiver amplitudes
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        # Get device index for CUDA
        device_idx = device.index if device.type == "cuda" and device.index is not None else 0

        backward_storage_tensors: list[torch.Tensor] = []
        backward_storage_objects: list[Any] = []
        backward_storage_filename_arrays: list[Any] = []
        storage_mode = STORAGE_NONE
        shot_bytes_uncomp = 0

        if needs_grad:
            import ctypes

            # Resolve storage mode and sizes
            if str(device) == "cpu" and storage_mode_str == "cpu":
                storage_mode_str = "device"
            storage_mode = storage_mode_to_int(storage_mode_str)

            num_elements_per_shot = ny * nx
            storage_bf16 = bool(storage_compression)
            if storage_bf16:
                if device.type != "cuda":
                    raise NotImplementedError(
                        "storage_compression (BF16 snapshot storage) is only supported on CUDA."
                    )
                if dtype != torch.float32:
                    raise NotImplementedError(
                        "storage_compression (BF16 snapshot storage) is only supported for float32."
                    )
                store_dtype = torch.bfloat16
            else:
                store_dtype = dtype

            shot_bytes_uncomp = num_elements_per_shot * store_dtype.itemsize

            num_steps_stored = (nt + step_ratio - 1) // step_ratio

            # Storage buffers and filename arrays (mirrors Deepwave)
            char_ptr_type = ctypes.c_char_p
            is_cuda = device.type == "cuda"

            def alloc_storage(requires_grad_cond: bool):
                store_1 = torch.empty(0)
                store_3 = torch.empty(0)
                filenames_arr = (char_ptr_type * 0)()

                if requires_grad_cond and storage_mode != STORAGE_NONE:
                    if storage_mode == STORAGE_DEVICE:
                        store_1 = torch.empty(
                            num_steps_stored,
                            n_shots,
                            ny,
                            nx,
                            device=device,
                            dtype=store_dtype,
                        )
                    elif storage_mode == STORAGE_CPU:
                        store_1 = torch.empty(
                            n_shots, ny, nx, device=device, dtype=store_dtype
                        )
                        store_3 = torch.empty(
                            num_steps_stored,
                            n_shots,
                            shot_bytes_uncomp // store_dtype.itemsize,
                            device="cpu",
                            pin_memory=True,
                            dtype=store_dtype,
                        )
                    elif storage_mode == STORAGE_DISK:
                        storage_obj = TemporaryStorage(
                            storage_path, 1 if is_cuda else n_shots
                        )
                        backward_storage_objects.append(storage_obj)
                        filenames_list = [
                            f.encode("utf-8") for f in storage_obj.get_filenames()
                        ]
                        filenames_arr = (char_ptr_type * len(filenames_list))()
                        for i_file, f_name in enumerate(filenames_list):
                            filenames_arr[i_file] = ctypes.cast(
                                ctypes.create_string_buffer(f_name), char_ptr_type
                            )

                        store_1 = torch.empty(
                            n_shots, ny, nx, device=device, dtype=store_dtype
                        )
                        if is_cuda:
                            store_3 = torch.empty(
                                n_shots,
                                shot_bytes_uncomp // store_dtype.itemsize,
                                device="cpu",
                                pin_memory=True,
                                dtype=store_dtype,
                            )

                backward_storage_tensors.extend([store_1, store_3])
                backward_storage_filename_arrays.append(filenames_arr)

                filenames_ptr = (
                    ctypes.cast(filenames_arr, ctypes.c_void_p)
                    if storage_mode == STORAGE_DISK
                    else 0
                )

                return store_1, store_3, filenames_ptr

            ey_store_1, ey_store_3, ey_filenames_ptr = alloc_storage(ca_requires_grad)
            curl_store_1, curl_store_3, curl_filenames_ptr = alloc_storage(cb_requires_grad)

            # Get the backend function with storage
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm", "forward_with_storage", accuracy, dtype, device
            )

            # Determine effective callback frequency
            if forward_callback is None:
                effective_callback_freq = nt // step_ratio
            else:
                effective_callback_freq = callback_frequency

            # Chunked forward propagation with callback support
            for step in range(0, nt // step_ratio, effective_callback_freq):
                step_nt = min(effective_callback_freq, nt // step_ratio - step) * step_ratio
                start_t = step * step_ratio
                
                # Call the C/CUDA function with storage for this chunk
                forward_func(
                    backend_utils.tensor_to_ptr(ca),
                    backend_utils.tensor_to_ptr(cb),
                    backend_utils.tensor_to_ptr(cq),
                    backend_utils.tensor_to_ptr(source_amplitudes_scaled),
                    backend_utils.tensor_to_ptr(Ey),
                    backend_utils.tensor_to_ptr(Hx),
                    backend_utils.tensor_to_ptr(Hz),
                    backend_utils.tensor_to_ptr(m_Ey_x),
                    backend_utils.tensor_to_ptr(m_Ey_z),
                    backend_utils.tensor_to_ptr(m_Hx_z),
                    backend_utils.tensor_to_ptr(m_Hz_x),
                    backend_utils.tensor_to_ptr(receiver_amplitudes),
                    backend_utils.tensor_to_ptr(ey_store_1),
                    backend_utils.tensor_to_ptr(ey_store_3),
                    ey_filenames_ptr,
                    backend_utils.tensor_to_ptr(curl_store_1),
                    backend_utils.tensor_to_ptr(curl_store_3),
                    curl_filenames_ptr,
                    backend_utils.tensor_to_ptr(ay),
                    backend_utils.tensor_to_ptr(by),
                    backend_utils.tensor_to_ptr(ay_h),
                    backend_utils.tensor_to_ptr(by_h),
                    backend_utils.tensor_to_ptr(ax),
                    backend_utils.tensor_to_ptr(bx),
                    backend_utils.tensor_to_ptr(ax_h),
                    backend_utils.tensor_to_ptr(bx_h),
                    backend_utils.tensor_to_ptr(ky),
                    backend_utils.tensor_to_ptr(ky_h),
                    backend_utils.tensor_to_ptr(kx),
                    backend_utils.tensor_to_ptr(kx_h),
                    backend_utils.tensor_to_ptr(sources_i),
                    backend_utils.tensor_to_ptr(receivers_i),
                    rdy,
                    rdx,
                    dt,
                    step_nt,  # number of steps in this chunk
                    n_shots,
                    ny,
                    nx,
                    n_sources,
                    n_receivers,
                    step_ratio,
                    storage_mode,
                    shot_bytes_uncomp,
                    ca_requires_grad,
                    cb_requires_grad,
                    ca_batched,
                    cb_batched,
                    cq_batched,
                    start_t,  # starting time step
                    pml_y0,
                    pml_x0,
                    pml_y1,
                    pml_x1,
                    device_idx,
                )
                
                # Call forward callback after each chunk
                if forward_callback is not None:
                    callback_wavefields = {
                        "Ey": Ey,
                        "Hx": Hx,
                        "Hz": Hz,
                        "m_Ey_x": m_Ey_x,
                        "m_Ey_z": m_Ey_z,
                        "m_Hx_z": m_Hx_z,
                        "m_Hz_x": m_Hz_x,
                    }
                    forward_callback(
                        CallbackState(
                            dt=dt,
                            step=step + step_nt // step_ratio,
                            nt=nt // step_ratio,
                            wavefields=callback_wavefields,
                            models=models,
                            gradients={},
                            fd_pad=list(fd_pad),
                            pml_width=list(pml_width),
                            is_backward=False,
                        )
                    )
        else:
            # Use regular forward without storage
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm", "forward", accuracy, dtype, device
            )

            # Call the C/CUDA function
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(source_amplitudes_scaled),
                backend_utils.tensor_to_ptr(Ey),
                backend_utils.tensor_to_ptr(Hx),
                backend_utils.tensor_to_ptr(Hz),
                backend_utils.tensor_to_ptr(m_Ey_x),
                backend_utils.tensor_to_ptr(m_Ey_z),
                backend_utils.tensor_to_ptr(m_Hx_z),
                backend_utils.tensor_to_ptr(m_Hz_x),
                backend_utils.tensor_to_ptr(receiver_amplitudes),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                rdy,
                rdx,
                dt,
                nt,
                n_shots,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                ca_batched,
                cb_batched,
                cq_batched,
                0,  # start_t
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                device_idx,
            )

        ctx_data = {
            "backward_storage_tensors": backward_storage_tensors,
            "backward_storage_objects": backward_storage_objects,
            "backward_storage_filename_arrays": backward_storage_filename_arrays,
            "storage_mode": storage_mode,
            "shot_bytes_uncomp": shot_bytes_uncomp,
            "jvp_initial": jvp_initial,
            "source_amplitudes_scaled": source_amplitudes_scaled,
            "ca_requires_grad": ca_requires_grad,
            "cb_requires_grad": cb_requires_grad,
        }
        ctx_handle = _register_ctx_handle(ctx_data)

        return (
            Ey,
            Hx,
            Hz,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            receiver_amplitudes,
            ctx_handle,
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], outputs: Tuple[Any, ...]) -> None:
        (
            ca,
            cb,
            cq,
            _source_amplitudes_scaled,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
            rdy,
            rdx,
            dt,
            nt,
            n_shots,
            ny,
            nx,
            n_sources,
            n_receivers,
            step_ratio,
            accuracy,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            fd_pad,
            pml_width,
            models,
            _forward_callback,
            backward_callback,
            callback_frequency,
            _storage_mode_str,
            _storage_path,
            _storage_compression,
            _Ey,
            _Hx,
            _Hz,
            _m_Ey_x,
            _m_Ey_z,
            _m_Hx_z,
            _m_Hz_x,
        ) = inputs

        if len(outputs) != 9:
            raise RuntimeError(
                "MaxwellTMForwardFunc expected a context handle output for setup_context."
            )
        ctx_handle = outputs[-1]
        if not isinstance(ctx_handle, torch.Tensor):
            raise RuntimeError("MaxwellTMForwardFunc context handle must be a Tensor.")

        ctx_handle_id = int(ctx_handle.item())
        ctx_data = _get_ctx_handle(ctx_handle_id)
        ctx._ctx_handle_id = ctx_handle_id
        backward_storage_tensors = ctx_data["backward_storage_tensors"]

        ctx.save_for_backward(
            ca,
            cb,
            cq,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
            *backward_storage_tensors,
        )
        ctx.save_for_forward(
            ca,
            cb,
            cq,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
        )
        ctx.backward_storage_objects = ctx_data["backward_storage_objects"]
        ctx.backward_storage_filename_arrays = ctx_data["backward_storage_filename_arrays"]
        ctx.rdy = rdy
        ctx.rdx = rdx
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.ny = ny
        ctx.nx = nx
        ctx.n_sources = n_sources
        ctx.n_receivers = n_receivers
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.ca_batched = ca_batched
        ctx.cb_batched = cb_batched
        ctx.cq_batched = cq_batched
        ctx.pml_y0 = pml_y0
        ctx.pml_x0 = pml_x0
        ctx.pml_y1 = pml_y1
        ctx.pml_x1 = pml_x1
        ctx.ca_requires_grad = ctx_data["ca_requires_grad"]
        ctx.cb_requires_grad = ctx_data["cb_requires_grad"]
        ctx.storage_mode = ctx_data["storage_mode"]
        ctx.shot_bytes_uncomp = ctx_data["shot_bytes_uncomp"]
        ctx.fd_pad = fd_pad
        ctx.pml_width = pml_width
        ctx.models = models
        ctx.backward_callback = backward_callback
        ctx.callback_frequency = callback_frequency
        ctx.source_amplitudes_scaled = ctx_data["source_amplitudes_scaled"]
        ctx.jvp_initial = ctx_data["jvp_initial"]

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass using ASM.

        Uses the Adjoint State Method (ASM) to compute gradients:
        - grad_ca = sum_t (E_y^n * lambda_Ey^{n+1})
        - grad_cb = sum_t (curl_H^n * lambda_Ey^{n+1})

        Args:
            ctx: A context object containing information saved during forward.
            grad_outputs: Gradients of the loss with respect to the outputs.

        Returns:
            Gradients with respect to the inputs of the forward pass.
        """
        from . import backend_utils

        grad_outputs = list(grad_outputs)
        if len(grad_outputs) == 9:
            grad_outputs.pop()  # drop context handle grad

        # Unpack grad_outputs
        (
            grad_Ey, grad_Hx, grad_Hz,
            grad_m_Ey_x, grad_m_Ey_z, grad_m_Hx_z, grad_m_Hz_x,
            grad_r,
        ) = grad_outputs

        # Retrieve saved tensors
        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        ay, by, ay_h, by_h = saved[3], saved[4], saved[5], saved[6]
        ax, bx, ax_h, bx_h = saved[7], saved[8], saved[9], saved[10]
        ky, ky_h, kx, kx_h = saved[11], saved[12], saved[13], saved[14]
        sources_i, receivers_i = saved[15], saved[16]
        ey_store_1, ey_store_3 = saved[17], saved[18]
        curl_store_1, curl_store_3 = saved[19], saved[20]

        device = ca.device
        dtype = ca.dtype

        rdy = ctx.rdy
        rdx = ctx.rdx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        ny = ctx.ny
        nx = ctx.nx
        n_sources = ctx.n_sources
        n_receivers = ctx.n_receivers
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        ca_batched = ctx.ca_batched
        cb_batched = ctx.cb_batched
        cq_batched = ctx.cq_batched
        pml_y0 = ctx.pml_y0
        pml_x0 = ctx.pml_x0
        pml_y1 = ctx.pml_y1
        pml_x1 = ctx.pml_x1
        ca_requires_grad = ctx.ca_requires_grad
        cb_requires_grad = ctx.cb_requires_grad
        pml_width = ctx.pml_width
        storage_mode = ctx.storage_mode
        shot_bytes_uncomp = ctx.shot_bytes_uncomp

        import ctypes

        if storage_mode == STORAGE_DISK:
            ey_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[0], ctypes.c_void_p
            )
            curl_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[1], ctypes.c_void_p
            )
        else:
            ey_filenames_ptr = 0
            curl_filenames_ptr = 0

        # Recalculate PML boundaries for gradient accumulation
        # 
        # For staggered grid schemes, the backward pass uses an extended PML region
        # compared to forward. This is because backward calculations
        # involve spatial derivatives of terms that are themselves spatial derivatives.
        #
        # In tide, the padded domain includes both fd_pad and pml_width:
        #   - pml_y0 = fd_pad + pml_width (start of interior, from forward)
        #   - pml_y1 = ny - (fd_pad-1) - pml_width (end of interior, from forward)
        # 
        # The gradient accumulation region is controlled by loop bounds in C/CUDA
        # with pml_bounds array and 3-region loop.

        # Ensure grad_r is contiguous
        if grad_r is None or grad_r.numel() == 0:
            grad_r = torch.zeros(nt, n_shots, n_receivers, device=device, dtype=dtype)
        else:
            grad_r = grad_r.contiguous()

        # Initialize adjoint fields (lambda fields)
        lambda_ey = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        lambda_hx = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        lambda_hz = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        # Initialize adjoint PML memory variables
        m_lambda_ey_x = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_ey_z = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_hx_z = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_hz_x = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        # Allocate gradient outputs
        if n_sources > 0:
            grad_f = torch.zeros(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad:
            if ca_batched:
                grad_ca = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            else:
                grad_ca = torch.zeros(ny, nx, device=device, dtype=dtype)
            # Per-shot workspace for gradient accumulation (needed for CUDA)
            grad_ca_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if cb_requires_grad:
            if cb_batched:
                grad_cb = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            else:
                grad_cb = torch.zeros(ny, nx, device=device, dtype=dtype)
            # Per-shot workspace for gradient accumulation (needed for CUDA)
            grad_cb_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad or cb_requires_grad:
            if ca_batched:
                grad_eps = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            else:
                grad_eps = torch.zeros(ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(ny, nx, device=device, dtype=dtype)
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)
            grad_sigma = torch.empty(0, device=device, dtype=dtype)

        # Get device index for CUDA
        device_idx = device.index if device.type == "cuda" and device.index is not None else 0

        # Get callback-related context
        backward_callback = ctx.backward_callback
        callback_frequency = ctx.callback_frequency
        fd_pad_ctx = ctx.fd_pad
        models = ctx.models

        # Get the backend function
        backward_func = backend_utils.get_backend_function(
            "maxwell_tm", "backward", accuracy, dtype, device
        )

        # Determine effective callback frequency
        if backward_callback is None:
            effective_callback_freq = nt // step_ratio
        else:
            effective_callback_freq = callback_frequency

        # Chunked backward propagation with callback support
        # Backward propagation goes from nt to 0
        for step in range(nt // step_ratio, 0, -effective_callback_freq):
            step_nt = min(step, effective_callback_freq) * step_ratio
            start_t = step * step_ratio
            
            # Call the C/CUDA backward function for this chunk
            backward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(grad_r),
                backend_utils.tensor_to_ptr(lambda_ey),
                backend_utils.tensor_to_ptr(lambda_hx),
                backend_utils.tensor_to_ptr(lambda_hz),
                backend_utils.tensor_to_ptr(m_lambda_ey_x),
                backend_utils.tensor_to_ptr(m_lambda_ey_z),
                backend_utils.tensor_to_ptr(m_lambda_hx_z),
                backend_utils.tensor_to_ptr(m_lambda_hz_x),
                backend_utils.tensor_to_ptr(ey_store_1),
                backend_utils.tensor_to_ptr(ey_store_3),
                ey_filenames_ptr,
                backend_utils.tensor_to_ptr(curl_store_1),
                backend_utils.tensor_to_ptr(curl_store_3),
                curl_filenames_ptr,
                backend_utils.tensor_to_ptr(grad_f),
                backend_utils.tensor_to_ptr(grad_ca),
                backend_utils.tensor_to_ptr(grad_cb),
                backend_utils.tensor_to_ptr(grad_eps),
                backend_utils.tensor_to_ptr(grad_sigma),
                backend_utils.tensor_to_ptr(grad_ca_shot),
                backend_utils.tensor_to_ptr(grad_cb_shot),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                rdy,
                rdx,
                dt,
                step_nt,  # number of steps to run in this chunk
                n_shots,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                storage_mode,
                shot_bytes_uncomp,
                ca_requires_grad,
                cb_requires_grad,
                ca_batched,
                cb_batched,
                cq_batched,
                start_t,  # starting time step for this chunk
                pml_y0,  # Use original PML boundaries for adjoint propagation
                pml_x0,
                pml_y1,
                pml_x1,
                device_idx,
            )

            # Call backward callback after each chunk
            if backward_callback is not None:
                # The time step index is step - 1 because the callback is
                # executed after the calculations for the current backward
                # step are complete
                callback_wavefields = {
                    "lambda_Ey": lambda_ey,
                    "lambda_Hx": lambda_hx,
                    "lambda_Hz": lambda_hz,
                    "m_lambda_Ey_x": m_lambda_ey_x,
                    "m_lambda_Ey_z": m_lambda_ey_z,
                    "m_lambda_Hx_z": m_lambda_hx_z,
                    "m_lambda_Hz_x": m_lambda_hz_x,
                }
                callback_gradients = {}
                if ca_requires_grad:
                    callback_gradients["ca"] = grad_ca
                if cb_requires_grad:
                    callback_gradients["cb"] = grad_cb
                if ca_requires_grad or cb_requires_grad:
                    callback_gradients["epsilon"] = grad_eps
                    callback_gradients["sigma"] = grad_sigma
                
                backward_callback(
                    CallbackState(
                        dt=dt,
                        step=step - 1,
                        nt=nt // step_ratio,
                        wavefields=callback_wavefields,
                        models=models,
                        gradients=callback_gradients,
                        fd_pad=list(fd_pad_ctx),
                        pml_width=list(pml_width),
                        is_backward=True,
                    )
                )

        # Return gradients for all inputs
        # Order: ca, cb, cq, source_amplitudes_scaled, 
        #        ay, by, ay_h, by_h, ax, bx, ax_h, bx_h,
        #        ky, ky_h, kx, kx_h,
        #        sources_i, receivers_i,
        #        rdy, rdx, dt, nt, n_shots, ny, nx, n_sources, n_receivers,
        #        step_ratio, accuracy, ca_batched, cb_batched, cq_batched,
        #        pml_y0, pml_x0, pml_y1, pml_x1,
        #        fd_pad, pml_width, models, backward_callback, callback_frequency,
        #        Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
        
        # Flatten grad_f to match input shape [nt * n_shots * n_sources]
        if n_sources > 0:
            grad_f_flat = grad_f.reshape(nt * n_shots * n_sources)
        else:
            grad_f_flat = None
        
        # Match gradient shapes to input shapes
        # Input ca, cb are [1, ny, nx] but grad_ca, grad_cb are [ny, nx] when not batched
        if ca_requires_grad and not ca_batched:
            grad_ca = grad_ca.unsqueeze(0)  # [ny, nx] -> [1, ny, nx]
        if cb_requires_grad and not cb_batched:
            grad_cb = grad_cb.unsqueeze(0)  # [ny, nx] -> [1, ny, nx]
            
        _release_ctx_handle(getattr(ctx, "_ctx_handle_id", None))
        return (
            grad_ca if ca_requires_grad else None,  # ca
            grad_cb if cb_requires_grad else None,  # cb
            None,  # cq
            grad_f_flat,  # source_amplitudes_scaled
            None, None, None, None,  # ay, by, ay_h, by_h
            None, None, None, None,  # ax, bx, ax_h, bx_h
            None, None, None, None,  # ky, ky_h, kx, kx_h
            None, None,  # sources_i, receivers_i
            None, None, None,  # rdy, rdx, dt
            None, None, None, None,  # nt, n_shots, ny, nx
            None, None,  # n_sources, n_receivers
            None,  # step_ratio
            None,  # accuracy
            None, None, None,  # ca_batched, cb_batched, cq_batched
            None, None, None, None,  # pml_y0, pml_x0, pml_y1, pml_x1
            None, None, None,  # fd_pad, pml_width, models
            None, None, None,  # forward_callback, backward_callback, callback_frequency
            None, None, None,  # storage_mode_str, storage_path, storage_compression
            None, None, None,  # Ey, Hx, Hz
            None, None, None, None,  # m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:


        expected = 52
        if len(tangents) != expected:
            raise RuntimeError(
                f"MaxwellTMForwardFunc.jvp expected {expected} tangents, got {len(tangents)}."
            )

        (
            d_ca,
            d_cb,
            d_cq,
            d_f,
            d_ay,
            d_by,
            d_ay_h,
            d_by_h,
            d_ax,
            d_bx,
            d_ax_h,
            d_bx_h,
            d_ky,
            d_ky_h,
            d_kx,
            d_kx_h,
            d_sources_i,
            d_receivers_i,
            d_rdy,
            d_rdx,
            d_dt,
            d_nt,
            d_n_shots,
            d_ny,
            d_nx,
            d_n_sources,
            d_n_receivers,
            d_step_ratio,
            d_accuracy,
            d_ca_batched,
            d_cb_batched,
            d_cq_batched,
            d_pml_y0,
            d_pml_x0,
            d_pml_y1,
            d_pml_x1,
            d_fd_pad,
            d_pml_width,
            d_models,
            d_forward_callback,
            d_backward_callback,
            d_callback_frequency,
            d_storage_mode_str,
            d_storage_path,
            d_storage_compression,
            d_Ey,
            d_Hx,
            d_Hz,
            d_m_Ey_x,
            d_m_Ey_z,
            d_m_Hx_z,
            d_m_Hz_x,
        ) = tangents

        unsupported = (
            d_ay,
            d_by,
            d_ay_h,
            d_by_h,
            d_ax,
            d_bx,
            d_ax_h,
            d_bx_h,
            d_ky,
            d_ky_h,
            d_kx,
            d_kx_h,
            d_sources_i,
            d_receivers_i,
            d_rdy,
            d_rdx,
            d_dt,
            d_nt,
            d_n_shots,
            d_ny,
            d_nx,
            d_n_sources,
            d_n_receivers,
            d_step_ratio,
            d_accuracy,
            d_ca_batched,
            d_cb_batched,
            d_cq_batched,
            d_pml_y0,
            d_pml_x0,
            d_pml_y1,
            d_pml_x1,
        )
        ignored = (
            d_fd_pad,
            d_pml_width,
            d_models,
            d_forward_callback,
            d_backward_callback,
            d_callback_frequency,
            d_storage_mode_str,
            d_storage_path,
            d_storage_compression,
        )
        def _has_unsupported_tangent(t: Optional[torch.Tensor]) -> bool:
            if t is None:
                return False
            if isinstance(t, (int, float, bool)):
                return t != 0
            if isinstance(t, torch.Tensor):
                return t.numel() != 0 and bool(torch.any(t != 0))
            return True

        if any(_has_unsupported_tangent(t) for t in unsupported):
            raise NotImplementedError(
                "JVP tangents are only supported for ca, cb, cq, "
                "source_amplitudes_scaled, and initial wavefields."
            )

        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        ay, by, ay_h, by_h = saved[3], saved[4], saved[5], saved[6]
        ax, bx, ax_h, bx_h = saved[7], saved[8], saved[9], saved[10]
        ky, ky_h, kx, kx_h = saved[11], saved[12], saved[13], saved[14]
        sources_i, receivers_i = saved[15], saved[16]

        device = ca.device
        dtype = ca.dtype
        if device.type != "cuda":
            raise NotImplementedError("MaxwellTM JVP is only supported on CUDA.")

        if hasattr(ctx, "jvp_initial"):
            (
                Ey_0,
                Hx_0,
                Hz_0,
                m_Ey_x_0,
                m_Ey_z_0,
                m_Hx_z_0,
                m_Hz_x_0,
            ) = ctx.jvp_initial
        else:
            size = (ctx.n_shots, ctx.ny, ctx.nx)
            Ey_0 = torch.zeros(size, device=device, dtype=dtype)
            Hx_0 = torch.zeros(size, device=device, dtype=dtype)
            Hz_0 = torch.zeros(size, device=device, dtype=dtype)
            m_Ey_x_0 = torch.zeros(size, device=device, dtype=dtype)
            m_Ey_z_0 = torch.zeros(size, device=device, dtype=dtype)
            m_Hx_z_0 = torch.zeros(size, device=device, dtype=dtype)
            m_Hz_x_0 = torch.zeros(size, device=device, dtype=dtype)

        Ey = Ey_0.clone()
        Hx = Hx_0.clone()
        Hz = Hz_0.clone()
        m_Ey_x = m_Ey_x_0.clone()
        m_Ey_z = m_Ey_z_0.clone()
        m_Hx_z = m_Hx_z_0.clone()
        m_Hz_x = m_Hz_x_0.clone()

        dey = torch.zeros_like(Ey) if d_Ey is None else d_Ey.contiguous()
        dhx = torch.zeros_like(Hx) if d_Hx is None else d_Hx.contiguous()
        dhz = torch.zeros_like(Hz) if d_Hz is None else d_Hz.contiguous()
        dm_Ey_x = torch.zeros_like(m_Ey_x) if d_m_Ey_x is None else d_m_Ey_x.contiguous()
        dm_Ey_z = torch.zeros_like(m_Ey_z) if d_m_Ey_z is None else d_m_Ey_z.contiguous()
        dm_Hx_z = torch.zeros_like(m_Hx_z) if d_m_Hx_z is None else d_m_Hx_z.contiguous()
        dm_Hz_x = torch.zeros_like(m_Hz_x) if d_m_Hz_x is None else d_m_Hz_x.contiguous()

        d_ca = torch.zeros_like(ca) if d_ca is None else d_ca.contiguous()
        d_cb = torch.zeros_like(cb) if d_cb is None else d_cb.contiguous()
        d_cq = torch.zeros_like(cq) if d_cq is None else d_cq.contiguous()

        source_amplitudes_scaled = ctx.source_amplitudes_scaled
        d_f = (
            torch.zeros_like(source_amplitudes_scaled)
            if d_f is None
            else d_f.contiguous()
        )

        if ctx.n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                ctx.nt, ctx.n_shots, ctx.n_receivers, device=device, dtype=dtype
            )
            d_receiver_amplitudes = torch.zeros_like(receiver_amplitudes)
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
            d_receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        device_idx = device.index if device.type == "cuda" and device.index is not None else 0
        forward_jvp = backend_utils.get_backend_function(
            "maxwell_tm", "forward_jvp", ctx.accuracy, dtype, device
        )

        forward_jvp(
            backend_utils.tensor_to_ptr(ca),
            backend_utils.tensor_to_ptr(cb),
            backend_utils.tensor_to_ptr(cq),
            backend_utils.tensor_to_ptr(d_ca),
            backend_utils.tensor_to_ptr(d_cb),
            backend_utils.tensor_to_ptr(d_cq),
            backend_utils.tensor_to_ptr(source_amplitudes_scaled),
            backend_utils.tensor_to_ptr(d_f),
            backend_utils.tensor_to_ptr(Ey),
            backend_utils.tensor_to_ptr(Hx),
            backend_utils.tensor_to_ptr(Hz),
            backend_utils.tensor_to_ptr(dey),
            backend_utils.tensor_to_ptr(dhx),
            backend_utils.tensor_to_ptr(dhz),
            backend_utils.tensor_to_ptr(m_Ey_x),
            backend_utils.tensor_to_ptr(m_Ey_z),
            backend_utils.tensor_to_ptr(m_Hx_z),
            backend_utils.tensor_to_ptr(m_Hz_x),
            backend_utils.tensor_to_ptr(dm_Ey_x),
            backend_utils.tensor_to_ptr(dm_Ey_z),
            backend_utils.tensor_to_ptr(dm_Hx_z),
            backend_utils.tensor_to_ptr(dm_Hz_x),
            backend_utils.tensor_to_ptr(receiver_amplitudes),
            backend_utils.tensor_to_ptr(d_receiver_amplitudes),
            backend_utils.tensor_to_ptr(ay),
            backend_utils.tensor_to_ptr(by),
            backend_utils.tensor_to_ptr(ay_h),
            backend_utils.tensor_to_ptr(by_h),
            backend_utils.tensor_to_ptr(ax),
            backend_utils.tensor_to_ptr(bx),
            backend_utils.tensor_to_ptr(ax_h),
            backend_utils.tensor_to_ptr(bx_h),
            backend_utils.tensor_to_ptr(ky),
            backend_utils.tensor_to_ptr(ky_h),
            backend_utils.tensor_to_ptr(kx),
            backend_utils.tensor_to_ptr(kx_h),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            ctx.rdy,
            ctx.rdx,
            ctx.dt,
            ctx.nt,
            ctx.n_shots,
            ctx.ny,
            ctx.nx,
            ctx.n_sources,
            ctx.n_receivers,
            ctx.step_ratio,
            ctx.ca_batched,
            ctx.cb_batched,
            ctx.cq_batched,
            0,  # start_t
            ctx.pml_y0,
            ctx.pml_x0,
            ctx.pml_y1,
            ctx.pml_x1,
            device_idx,
        )

        _release_ctx_handle(getattr(ctx, "_ctx_handle_id", None))
        return (
            dey,
            dhx,
            dhz,
            dm_Ey_x,
            dm_Ey_z,
            dm_Hx_z,
            dm_Hz_x,
            d_receiver_amplitudes,
            None,
        )


class MaxwellTMForwardBoundaryFunc(torch.autograd.Function):
    """Autograd function for Maxwell TM with boundary storage reconstruction.

    Stores a compact boundary ring of (Ey, Hx, Hz) for every time step and
    reconstructs forward wavefields during the backward pass by inverting the
    update equations in the non-PML region.
    """

    @staticmethod
    def forward(
        ctx: Any,
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
        boundary_indices: torch.Tensor,
        ay: torch.Tensor,
        by: torch.Tensor,
        ay_h: torch.Tensor,
        by_h: torch.Tensor,
        ax: torch.Tensor,
        bx: torch.Tensor,
        ax_h: torch.Tensor,
        bx_h: torch.Tensor,
        ky: torch.Tensor,
        ky_h: torch.Tensor,
        kx: torch.Tensor,
        kx_h: torch.Tensor,
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        rdy: float,
        rdx: float,
        dt: float,
        nt: int,
        n_shots: int,
        ny: int,
        nx: int,
        n_sources: int,
        n_receivers: int,
        accuracy: int,
        ca_batched: bool,
        cb_batched: bool,
        cq_batched: bool,
        pml_y0: int,
        pml_x0: int,
        pml_y1: int,
        pml_x1: int,
        storage_mode_str: str,
        storage_path: str,
        storage_compression: bool,
        Ey: torch.Tensor,
        Hx: torch.Tensor,
        Hz: torch.Tensor,
        m_Ey_x: torch.Tensor,
        m_Ey_z: torch.Tensor,
        m_Hx_z: torch.Tensor,
        m_Hz_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        from . import backend_utils

        import ctypes

        device = Ey.device
        dtype = Ey.dtype

        if device.type != "cuda":
            raise NotImplementedError(
                "MaxwellTMForwardBoundaryFunc is only supported on CUDA."
            )

        # Initialize receiver amplitudes
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        storage_mode = storage_mode_to_int(storage_mode_str)
        if storage_mode == STORAGE_NONE:
            raise ValueError(
                "storage_mode='none' is not compatible with gradient_mode='boundary' "
                "when gradients are required."
            )

        boundary_indices = boundary_indices.to(device=device, dtype=torch.int64).contiguous()
        boundary_numel = int(boundary_indices.numel())
        if boundary_numel <= 0:
            raise ValueError("boundary_indices must be non-empty for boundary storage.")

        storage_bf16 = bool(storage_compression)
        if storage_bf16:
            if dtype != torch.float32:
                raise NotImplementedError(
                    "storage_compression (BF16 boundary storage) is only supported for float32."
                )
            store_dtype = torch.bfloat16
        else:
            store_dtype = dtype

        shot_bytes_uncomp = boundary_numel * store_dtype.itemsize
        num_steps_stored = nt + 1

        char_ptr_type = ctypes.c_char_p

        ctx.boundary_storage_objects = []
        ctx.boundary_storage_filename_arrays = []

        def alloc_boundary_storage():
            store_1 = torch.empty(0)
            store_3 = torch.empty(0)
            filenames_arr = (char_ptr_type * 0)()

            if storage_mode == STORAGE_DEVICE:
                store_1 = torch.empty(
                    num_steps_stored,
                    n_shots,
                    boundary_numel,
                    device=device,
                    dtype=store_dtype,
                )
            elif storage_mode == STORAGE_CPU:
                # Double-buffer device staging to enable safe async copies.
                store_1 = torch.empty(
                    2, n_shots, boundary_numel, device=device, dtype=store_dtype
                )
                store_3 = torch.empty(
                    num_steps_stored,
                    n_shots,
                    boundary_numel,
                    device="cpu",
                    pin_memory=True,
                    dtype=store_dtype,
                )
            elif storage_mode == STORAGE_DISK:
                storage_obj = TemporaryStorage(storage_path, 1)
                ctx.boundary_storage_objects.append(storage_obj)
                filenames_list = [
                    f.encode("utf-8") for f in storage_obj.get_filenames()
                ]
                filenames_arr = (char_ptr_type * len(filenames_list))()
                for i_file, f_name in enumerate(filenames_list):
                    filenames_arr[i_file] = ctypes.cast(
                        ctypes.create_string_buffer(f_name), char_ptr_type
                    )

                store_1 = torch.empty(
                    n_shots, boundary_numel, device=device, dtype=store_dtype
                )
                store_3 = torch.empty(
                    n_shots,
                    boundary_numel,
                    device="cpu",
                    pin_memory=True,
                    dtype=store_dtype,
                )

            ctx.boundary_storage_filename_arrays.append(filenames_arr)

            filenames_ptr = (
                ctypes.cast(filenames_arr, ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0
            )
            return store_1, store_3, filenames_ptr

        boundary_ey_store_1, boundary_ey_store_3, boundary_ey_filenames_ptr = (
            alloc_boundary_storage()
        )
        boundary_hx_store_1, boundary_hx_store_3, boundary_hx_filenames_ptr = (
            alloc_boundary_storage()
        )
        boundary_hz_store_1, boundary_hz_store_3, boundary_hz_filenames_ptr = (
            alloc_boundary_storage()
        )

        device_idx = device.index if device.index is not None else 0

        forward_func = backend_utils.get_backend_function(
            "maxwell_tm", "forward_with_boundary_storage", accuracy, dtype, device
        )

        forward_func(
            backend_utils.tensor_to_ptr(ca),
            backend_utils.tensor_to_ptr(cb),
            backend_utils.tensor_to_ptr(cq),
            backend_utils.tensor_to_ptr(source_amplitudes_scaled),
            backend_utils.tensor_to_ptr(Ey),
            backend_utils.tensor_to_ptr(Hx),
            backend_utils.tensor_to_ptr(Hz),
            backend_utils.tensor_to_ptr(m_Ey_x),
            backend_utils.tensor_to_ptr(m_Ey_z),
            backend_utils.tensor_to_ptr(m_Hx_z),
            backend_utils.tensor_to_ptr(m_Hz_x),
            backend_utils.tensor_to_ptr(receiver_amplitudes),
            backend_utils.tensor_to_ptr(boundary_ey_store_1),
            backend_utils.tensor_to_ptr(boundary_ey_store_3),
            boundary_ey_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hx_store_1),
            backend_utils.tensor_to_ptr(boundary_hx_store_3),
            boundary_hx_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hz_store_1),
            backend_utils.tensor_to_ptr(boundary_hz_store_3),
            boundary_hz_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_indices),
            boundary_numel,
            backend_utils.tensor_to_ptr(ay),
            backend_utils.tensor_to_ptr(by),
            backend_utils.tensor_to_ptr(ay_h),
            backend_utils.tensor_to_ptr(by_h),
            backend_utils.tensor_to_ptr(ax),
            backend_utils.tensor_to_ptr(bx),
            backend_utils.tensor_to_ptr(ax_h),
            backend_utils.tensor_to_ptr(bx_h),
            backend_utils.tensor_to_ptr(ky),
            backend_utils.tensor_to_ptr(ky_h),
            backend_utils.tensor_to_ptr(kx),
            backend_utils.tensor_to_ptr(kx_h),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            rdy,
            rdx,
            dt,
            nt,
            n_shots,
            ny,
            nx,
            n_sources,
            n_receivers,
            storage_mode,
            shot_bytes_uncomp,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            device_idx,
        )

        ctx.save_for_backward(
            ca,
            cb,
            cq,
            source_amplitudes_scaled,
            boundary_indices,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
            boundary_ey_store_1,
            boundary_ey_store_3,
            boundary_hx_store_1,
            boundary_hx_store_3,
            boundary_hz_store_1,
            boundary_hz_store_3,
            Ey,
            Hx,
            Hz,
        )

        ctx.rdy = rdy
        ctx.rdx = rdx
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.ny = ny
        ctx.nx = nx
        ctx.n_sources = n_sources
        ctx.n_receivers = n_receivers
        ctx.accuracy = accuracy
        ctx.ca_batched = ca_batched
        ctx.cb_batched = cb_batched
        ctx.cq_batched = cq_batched
        ctx.pml_y0 = pml_y0
        ctx.pml_x0 = pml_x0
        ctx.pml_y1 = pml_y1
        ctx.pml_x1 = pml_x1
        ctx.ca_requires_grad = ca.requires_grad
        ctx.cb_requires_grad = cb.requires_grad
        ctx.storage_mode = storage_mode
        ctx.shot_bytes_uncomp = shot_bytes_uncomp
        ctx.boundary_numel = boundary_numel

        return (
            Ey,
            Hx,
            Hz,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            receiver_amplitudes,
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        from . import backend_utils

        (
            grad_Ey,
            grad_Hx,
            grad_Hz,
            grad_m_Ey_x,
            grad_m_Ey_z,
            grad_m_Hx_z,
            grad_m_Hz_x,
            grad_r,
        ) = grad_outputs

        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        source_amplitudes_scaled = saved[3]
        boundary_indices = saved[4]
        ay, by, ay_h, by_h = saved[5], saved[6], saved[7], saved[8]
        ax, bx, ax_h, bx_h = saved[9], saved[10], saved[11], saved[12]
        ky, ky_h, kx, kx_h = saved[13], saved[14], saved[15], saved[16]
        sources_i, receivers_i = saved[17], saved[18]
        boundary_ey_store_1, boundary_ey_store_3 = saved[19], saved[20]
        boundary_hx_store_1, boundary_hx_store_3 = saved[21], saved[22]
        boundary_hz_store_1, boundary_hz_store_3 = saved[23], saved[24]
        Ey_end, Hx_end, Hz_end = saved[25], saved[26], saved[27]

        device = ca.device
        dtype = ca.dtype

        rdy = ctx.rdy
        rdx = ctx.rdx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        ny = ctx.ny
        nx = ctx.nx
        n_sources = ctx.n_sources
        n_receivers = ctx.n_receivers
        accuracy = ctx.accuracy
        ca_batched = ctx.ca_batched
        cb_batched = ctx.cb_batched
        cq_batched = ctx.cq_batched
        pml_y0 = ctx.pml_y0
        pml_x0 = ctx.pml_x0
        pml_y1 = ctx.pml_y1
        pml_x1 = ctx.pml_x1
        ca_requires_grad = ctx.ca_requires_grad
        cb_requires_grad = ctx.cb_requires_grad
        storage_mode = ctx.storage_mode
        shot_bytes_uncomp = ctx.shot_bytes_uncomp
        boundary_numel = ctx.boundary_numel

        import ctypes

        if storage_mode == STORAGE_DISK:
            boundary_ey_filenames_ptr = ctypes.cast(
                ctx.boundary_storage_filename_arrays[0], ctypes.c_void_p
            )
            boundary_hx_filenames_ptr = ctypes.cast(
                ctx.boundary_storage_filename_arrays[1], ctypes.c_void_p
            )
            boundary_hz_filenames_ptr = ctypes.cast(
                ctx.boundary_storage_filename_arrays[2], ctypes.c_void_p
            )
        else:
            boundary_ey_filenames_ptr = 0
            boundary_hx_filenames_ptr = 0
            boundary_hz_filenames_ptr = 0

        if grad_r is None or grad_r.numel() == 0:
            grad_r = torch.zeros(nt, n_shots, n_receivers, device=device, dtype=dtype)
        else:
            grad_r = grad_r.contiguous()

        # Reconstruction buffers (initialized from final forward fields)
        ey_recon = Ey_end.clone()
        hx_recon = Hx_end.clone()
        hz_recon = Hz_end.clone()
        curl_recon = torch.empty_like(ey_recon)

        # Adjoint fields
        lambda_ey = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        lambda_hx = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        lambda_hz = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_ey_x = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_ey_z = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_hx_z = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_hz_x = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        if n_sources > 0:
            grad_f = torch.zeros(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad:
            grad_ca = torch.zeros(ny, nx, device=device, dtype=dtype)
            grad_ca_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if cb_requires_grad:
            grad_cb = torch.zeros(ny, nx, device=device, dtype=dtype)
            grad_cb_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad or cb_requires_grad:
            if ca_batched:
                grad_eps = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            else:
                grad_eps = torch.zeros(ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(ny, nx, device=device, dtype=dtype)
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)
            grad_sigma = torch.empty(0, device=device, dtype=dtype)

        device_idx = device.index if device.type == "cuda" and device.index is not None else 0

        backward_func = backend_utils.get_backend_function(
            "maxwell_tm", "backward_with_boundary", accuracy, dtype, device
        )

        backward_func(
            backend_utils.tensor_to_ptr(ca),
            backend_utils.tensor_to_ptr(cb),
            backend_utils.tensor_to_ptr(cq),
            backend_utils.tensor_to_ptr(source_amplitudes_scaled),
            backend_utils.tensor_to_ptr(grad_r),
            backend_utils.tensor_to_ptr(ey_recon),
            backend_utils.tensor_to_ptr(hx_recon),
            backend_utils.tensor_to_ptr(hz_recon),
            backend_utils.tensor_to_ptr(curl_recon),
            backend_utils.tensor_to_ptr(lambda_ey),
            backend_utils.tensor_to_ptr(lambda_hx),
            backend_utils.tensor_to_ptr(lambda_hz),
            backend_utils.tensor_to_ptr(m_lambda_ey_x),
            backend_utils.tensor_to_ptr(m_lambda_ey_z),
            backend_utils.tensor_to_ptr(m_lambda_hx_z),
            backend_utils.tensor_to_ptr(m_lambda_hz_x),
            backend_utils.tensor_to_ptr(boundary_ey_store_1),
            backend_utils.tensor_to_ptr(boundary_ey_store_3),
            boundary_ey_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hx_store_1),
            backend_utils.tensor_to_ptr(boundary_hx_store_3),
            boundary_hx_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hz_store_1),
            backend_utils.tensor_to_ptr(boundary_hz_store_3),
            boundary_hz_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_indices),
            boundary_numel,
            backend_utils.tensor_to_ptr(grad_f),
            backend_utils.tensor_to_ptr(grad_ca),
            backend_utils.tensor_to_ptr(grad_cb),
            backend_utils.tensor_to_ptr(grad_eps),
            backend_utils.tensor_to_ptr(grad_sigma),
            backend_utils.tensor_to_ptr(grad_ca_shot),
            backend_utils.tensor_to_ptr(grad_cb_shot),
            backend_utils.tensor_to_ptr(ay),
            backend_utils.tensor_to_ptr(by),
            backend_utils.tensor_to_ptr(ay_h),
            backend_utils.tensor_to_ptr(by_h),
            backend_utils.tensor_to_ptr(ax),
            backend_utils.tensor_to_ptr(bx),
            backend_utils.tensor_to_ptr(ax_h),
            backend_utils.tensor_to_ptr(bx_h),
            backend_utils.tensor_to_ptr(ky),
            backend_utils.tensor_to_ptr(ky_h),
            backend_utils.tensor_to_ptr(kx),
            backend_utils.tensor_to_ptr(kx_h),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            rdy,
            rdx,
            dt,
            nt,
            n_shots,
            ny,
            nx,
            n_sources,
            n_receivers,
            storage_mode,
            shot_bytes_uncomp,
            ca_requires_grad,
            cb_requires_grad,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            device_idx,
        )

        if n_sources > 0:
            grad_f_flat = grad_f.reshape(nt * n_shots * n_sources)
        else:
            grad_f_flat = None

        if ca_requires_grad and not ca_batched:
            grad_ca = grad_ca.unsqueeze(0)
        if cb_requires_grad and not cb_batched:
            grad_cb = grad_cb.unsqueeze(0)

        return (
            grad_ca if ca_requires_grad else None,  # ca
            grad_cb if cb_requires_grad else None,  # cb
            None,  # cq
            grad_f_flat,  # source_amplitudes_scaled
            None,  # boundary_indices
            None, None, None, None,  # ay, by, ay_h, by_h
            None, None, None, None,  # ax, bx, ax_h, bx_h
            None, None, None, None,  # ky, ky_h, kx, kx_h
            None, None,  # sources_i, receivers_i
            None, None, None,  # rdy, rdx, dt
            None, None, None, None, None, None,  # nt, n_shots, ny, nx, n_sources, n_receivers
            None,  # accuracy
            None, None, None,  # ca_batched, cb_batched, cq_batched
            None, None, None, None,  # pml_y0, pml_x0, pml_y1, pml_x1
            None, None, None,  # storage_mode_str, storage_path, storage_compression
            None, None, None,  # Ey, Hx, Hz
            None, None, None, None,  # m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
        )


class MaxwellTMForwardRWIIFunc(torch.autograd.Function):
    """Autograd function for a reduced-wavefield (RWII) gradient mode.

    Implements a two-pass, snapshot-free strategy inspired by Robertsson et al. (2021):
    - Forward: propagate u, store a compact boundary ring and accumulate self-correlations.
    - Backward: back-propagate a composite wavefield w = u + αλ using boundary forcing and
      receiver residual injection, accumulate self-correlations, and form gradients from
      (Γ_w - Γ_u) / (2α).

    Notes:
    - CUDA only.
    - Currently requires sigma==0 (lossless) for stability.
    - Callbacks are not supported (same limitation as boundary mode).
    """

    @staticmethod
    def forward(
        ctx: Any,
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
        boundary_indices: torch.Tensor,
        ay: torch.Tensor,
        by: torch.Tensor,
        ay_h: torch.Tensor,
        by_h: torch.Tensor,
        ax: torch.Tensor,
        bx: torch.Tensor,
        ax_h: torch.Tensor,
        bx_h: torch.Tensor,
        ky: torch.Tensor,
        ky_h: torch.Tensor,
        kx: torch.Tensor,
        kx_h: torch.Tensor,
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        rdy: float,
        rdx: float,
        dt: float,
        nt: int,
        n_shots: int,
        ny: int,
        nx: int,
        n_sources: int,
        n_receivers: int,
        accuracy: int,
        ca_batched: bool,
        cb_batched: bool,
        cq_batched: bool,
        pml_y0: int,
        pml_x0: int,
        pml_y1: int,
        pml_x1: int,
        storage_mode_str: str,
        storage_path: str,
        storage_compression: bool,
        alpha_rwii: float,
        Ey: torch.Tensor,
        Hx: torch.Tensor,
        Hz: torch.Tensor,
        m_Ey_x: torch.Tensor,
        m_Ey_z: torch.Tensor,
        m_Hx_z: torch.Tensor,
        m_Hz_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        from . import backend_utils

        import ctypes

        device = Ey.device
        dtype = Ey.dtype

        if device.type != "cuda":
            raise NotImplementedError(
                "MaxwellTMForwardRWIIFunc is only supported on CUDA."
            )
        if alpha_rwii == 0.0:
            raise ValueError("alpha_rwii must be non-zero for RWII.")

        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        storage_mode = storage_mode_to_int(storage_mode_str)
        if storage_mode == STORAGE_NONE:
            raise ValueError(
                "storage_mode='none' is not compatible with gradient_mode='rwii' "
                "when gradients are required."
            )

        boundary_indices = boundary_indices.to(device=device, dtype=torch.int64).contiguous()
        boundary_numel = int(boundary_indices.numel())
        if boundary_numel <= 0:
            raise ValueError("boundary_indices must be non-empty for boundary storage.")

        ca_requires_grad = bool(ca.requires_grad)
        cb_requires_grad = bool(cb.requires_grad)

        storage_bf16 = bool(storage_compression)
        if storage_bf16:
            if dtype != torch.float32:
                raise NotImplementedError(
                    "storage_compression (BF16 boundary storage) is only supported for float32."
                )
            store_dtype = torch.bfloat16
        else:
            store_dtype = dtype
        shot_bytes_uncomp = boundary_numel * store_dtype.itemsize

        num_steps_stored = nt + 1
        char_ptr_type = ctypes.c_char_p

        ctx.boundary_storage_objects = []
        ctx.boundary_storage_filename_arrays = []

        def alloc_boundary_storage():
            store_1 = torch.empty(0)
            store_3 = torch.empty(0)
            filenames_arr = (char_ptr_type * 0)()

            if storage_mode == STORAGE_DEVICE:
                store_1 = torch.empty(
                    num_steps_stored,
                    n_shots,
                    boundary_numel,
                    device=device,
                    dtype=store_dtype,
                )
            elif storage_mode == STORAGE_CPU:
                # Double-buffer device staging to enable safe async copies.
                store_1 = torch.empty(
                    2, n_shots, boundary_numel, device=device, dtype=store_dtype
                )
                store_3 = torch.empty(
                    num_steps_stored,
                    n_shots,
                    boundary_numel,
                    device="cpu",
                    pin_memory=True,
                    dtype=store_dtype,
                )
            elif storage_mode == STORAGE_DISK:
                storage_obj = TemporaryStorage(storage_path, 1)
                ctx.boundary_storage_objects.append(storage_obj)
                filenames_list = [
                    f.encode("utf-8") for f in storage_obj.get_filenames()
                ]
                filenames_arr = (char_ptr_type * len(filenames_list))()
                for i_file, f_name in enumerate(filenames_list):
                    filenames_arr[i_file] = ctypes.cast(
                        ctypes.create_string_buffer(f_name), char_ptr_type
                    )

                store_1 = torch.empty(
                    n_shots, boundary_numel, device=device, dtype=store_dtype
                )
                store_3 = torch.empty(
                    n_shots,
                    boundary_numel,
                    device="cpu",
                    pin_memory=True,
                    dtype=store_dtype,
                )

            ctx.boundary_storage_filename_arrays.append(filenames_arr)

            filenames_ptr = (
                ctypes.cast(filenames_arr, ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0
            )
            return store_1, store_3, filenames_ptr

        boundary_ey_store_1, boundary_ey_store_3, boundary_ey_filenames_ptr = (
            alloc_boundary_storage()
        )
        boundary_hx_store_1, boundary_hx_store_3, boundary_hx_filenames_ptr = (
            alloc_boundary_storage()
        )
        boundary_hz_store_1, boundary_hz_store_3, boundary_hz_filenames_ptr = (
            alloc_boundary_storage()
        )

        gamma_u_ey = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        gamma_u_curl = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        if n_sources > 0:
            u_src = torch.zeros(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            u_src = torch.empty(0, device=device, dtype=dtype)

        device_idx = device.index if device.index is not None else 0

        forward_func = backend_utils.get_backend_function(
            "maxwell_tm", "forward_with_boundary_storage_rwii", accuracy, dtype, device
        )

        forward_func(
            backend_utils.tensor_to_ptr(ca),
            backend_utils.tensor_to_ptr(cb),
            backend_utils.tensor_to_ptr(cq),
            backend_utils.tensor_to_ptr(source_amplitudes_scaled),
            backend_utils.tensor_to_ptr(Ey),
            backend_utils.tensor_to_ptr(Hx),
            backend_utils.tensor_to_ptr(Hz),
            backend_utils.tensor_to_ptr(m_Ey_x),
            backend_utils.tensor_to_ptr(m_Ey_z),
            backend_utils.tensor_to_ptr(m_Hx_z),
            backend_utils.tensor_to_ptr(m_Hz_x),
            backend_utils.tensor_to_ptr(receiver_amplitudes),
            backend_utils.tensor_to_ptr(boundary_ey_store_1),
            backend_utils.tensor_to_ptr(boundary_ey_store_3),
            boundary_ey_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hx_store_1),
            backend_utils.tensor_to_ptr(boundary_hx_store_3),
            boundary_hx_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hz_store_1),
            backend_utils.tensor_to_ptr(boundary_hz_store_3),
            boundary_hz_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_indices),
            boundary_numel,
            backend_utils.tensor_to_ptr(gamma_u_ey),
            backend_utils.tensor_to_ptr(gamma_u_curl),
            backend_utils.tensor_to_ptr(u_src),
            backend_utils.tensor_to_ptr(ay),
            backend_utils.tensor_to_ptr(by),
            backend_utils.tensor_to_ptr(ay_h),
            backend_utils.tensor_to_ptr(by_h),
            backend_utils.tensor_to_ptr(ax),
            backend_utils.tensor_to_ptr(bx),
            backend_utils.tensor_to_ptr(ax_h),
            backend_utils.tensor_to_ptr(bx_h),
            backend_utils.tensor_to_ptr(ky),
            backend_utils.tensor_to_ptr(ky_h),
            backend_utils.tensor_to_ptr(kx),
            backend_utils.tensor_to_ptr(kx_h),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            rdy,
            rdx,
            dt,
            nt,
            n_shots,
            ny,
            nx,
            n_sources,
            n_receivers,
            storage_mode,
            shot_bytes_uncomp,
            ca_requires_grad,
            cb_requires_grad,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            device_idx,
        )

        ctx.save_for_backward(
            ca,
            cb,
            cq,
            source_amplitudes_scaled,
            boundary_indices,
            sources_i,
            receivers_i,
            gamma_u_ey,
            gamma_u_curl,
            u_src,
            boundary_ey_store_1,
            boundary_ey_store_3,
            boundary_hx_store_1,
            boundary_hx_store_3,
            boundary_hz_store_1,
            boundary_hz_store_3,
            Ey,
            Hx,
            Hz,
        )
        ctx.rdy = rdy
        ctx.rdx = rdx
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.ny = ny
        ctx.nx = nx
        ctx.n_sources = n_sources
        ctx.n_receivers = n_receivers
        ctx.accuracy = accuracy
        ctx.ca_batched = ca_batched
        ctx.cb_batched = cb_batched
        ctx.cq_batched = cq_batched
        ctx.pml_y0 = pml_y0
        ctx.pml_x0 = pml_x0
        ctx.pml_y1 = pml_y1
        ctx.pml_x1 = pml_x1
        ctx.storage_mode = storage_mode
        ctx.shot_bytes_uncomp = shot_bytes_uncomp
        ctx.ca_requires_grad = ca_requires_grad
        ctx.cb_requires_grad = cb_requires_grad
        ctx.alpha_rwii = float(alpha_rwii)

        return (
            Ey,
            Hx,
            Hz,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            receiver_amplitudes,
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor):
        from . import backend_utils

        import ctypes

        (
            ca,
            cb,
            cq,
            source_amplitudes_scaled,
            boundary_indices,
            sources_i,
            receivers_i,
            gamma_u_ey,
            gamma_u_curl,
            u_src,
            boundary_ey_store_1,
            boundary_ey_store_3,
            boundary_hx_store_1,
            boundary_hx_store_3,
            boundary_hz_store_1,
            boundary_hz_store_3,
            Ey_final,
            Hx_final,
            Hz_final,
        ) = ctx.saved_tensors

        device = Ey_final.device
        dtype = Ey_final.dtype

        grad_r = grad_outputs[-1].contiguous() if grad_outputs[-1] is not None else None
        if grad_r is None or grad_r.numel() == 0:
            # No receiver gradient -> no parameter gradients.
            return (
                None,  # ca
                None,  # cb
                None,  # cq
                None,  # source_amplitudes_scaled
                None,  # boundary_indices
                None, None, None, None,  # ay, by, ay_h, by_h
                None, None, None, None,  # ax, bx, ax_h, bx_h
                None, None, None, None,  # ky, ky_h, kx, kx_h
                None, None,  # sources_i, receivers_i
                None, None, None,  # rdy, rdx, dt
                None, None, None, None, None, None,  # nt, n_shots, ny, nx, n_sources, n_receivers
                None,  # accuracy
                None, None, None,  # ca_batched, cb_batched, cq_batched
                None, None, None, None,  # pml_y0, pml_x0, pml_y1, pml_x1
                None, None, None,  # storage_mode_str, storage_path, storage_compression
                None,  # alpha_rwii
                None, None, None,  # Ey, Hx, Hz
                None, None, None, None,  # m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
            )

        ny = ctx.ny
        nx = ctx.nx
        nt = ctx.nt
        n_shots = ctx.n_shots
        n_sources = ctx.n_sources
        n_receivers = ctx.n_receivers
        accuracy = ctx.accuracy
        ca_batched = ctx.ca_batched
        cb_batched = ctx.cb_batched
        cq_batched = ctx.cq_batched
        ca_requires_grad = ctx.ca_requires_grad
        cb_requires_grad = ctx.cb_requires_grad
        storage_mode = ctx.storage_mode
        shot_bytes_uncomp = ctx.shot_bytes_uncomp
        pml_y0 = ctx.pml_y0
        pml_x0 = ctx.pml_x0
        pml_y1 = ctx.pml_y1
        pml_x1 = ctx.pml_x1
        alpha_rwii = float(ctx.alpha_rwii)

        if storage_mode == STORAGE_DISK:
            boundary_ey_filenames_ptr = ctypes.cast(
                ctx.boundary_storage_filename_arrays[0], ctypes.c_void_p
            ).value
            boundary_hx_filenames_ptr = ctypes.cast(
                ctx.boundary_storage_filename_arrays[1], ctypes.c_void_p
            ).value
            boundary_hz_filenames_ptr = ctypes.cast(
                ctx.boundary_storage_filename_arrays[2], ctypes.c_void_p
            ).value
        else:
            boundary_ey_filenames_ptr = 0
            boundary_hx_filenames_ptr = 0
            boundary_hz_filenames_ptr = 0

        ey_w = Ey_final.detach().clone()
        hx_w = Hx_final.detach().clone()
        hz_w = Hz_final.detach().clone()
        curl_w = None

        if n_sources > 0:
            grad_f = torch.zeros(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad:
            grad_ca = torch.zeros(ny, nx, device=device, dtype=dtype)
            grad_ca_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if cb_requires_grad:
            grad_cb = torch.zeros(ny, nx, device=device, dtype=dtype)
            grad_cb_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad or cb_requires_grad:
            if ca_batched:
                grad_eps = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            else:
                grad_eps = torch.zeros(ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(ny, nx, device=device, dtype=dtype)
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)
            grad_sigma = torch.empty(0, device=device, dtype=dtype)

        device_idx = device.index if device.index is not None else 0

        backward_func = backend_utils.get_backend_function(
            "maxwell_tm", "backward_rwii", accuracy, dtype, device
        )

        backward_func(
            backend_utils.tensor_to_ptr(ca),
            backend_utils.tensor_to_ptr(cb),
            backend_utils.tensor_to_ptr(cq),
            backend_utils.tensor_to_ptr(source_amplitudes_scaled),
            backend_utils.tensor_to_ptr(grad_r),
            backend_utils.tensor_to_ptr(ey_w),
            backend_utils.tensor_to_ptr(hx_w),
            backend_utils.tensor_to_ptr(hz_w),
            backend_utils.tensor_to_ptr(curl_w),
            backend_utils.tensor_to_ptr(boundary_ey_store_1),
            backend_utils.tensor_to_ptr(boundary_ey_store_3),
            boundary_ey_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hx_store_1),
            backend_utils.tensor_to_ptr(boundary_hx_store_3),
            boundary_hx_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hz_store_1),
            backend_utils.tensor_to_ptr(boundary_hz_store_3),
            boundary_hz_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_indices),
            int(boundary_indices.numel()),
            backend_utils.tensor_to_ptr(u_src),
            backend_utils.tensor_to_ptr(gamma_u_ey),
            backend_utils.tensor_to_ptr(gamma_u_curl),
            backend_utils.tensor_to_ptr(grad_f),
            backend_utils.tensor_to_ptr(grad_ca),
            backend_utils.tensor_to_ptr(grad_cb),
            backend_utils.tensor_to_ptr(grad_eps),
            backend_utils.tensor_to_ptr(grad_sigma),
            backend_utils.tensor_to_ptr(grad_ca_shot),
            backend_utils.tensor_to_ptr(grad_cb_shot),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            ctx.rdy,
            ctx.rdx,
            ctx.dt,
            nt,
            n_shots,
            ny,
            nx,
            n_sources,
            n_receivers,
            storage_mode,
            shot_bytes_uncomp,
            ca_requires_grad,
            cb_requires_grad,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            alpha_rwii,
            device_idx,
        )

        grad_f_flat = grad_f.reshape(nt * n_shots * n_sources) if n_sources > 0 else None

        if ca_requires_grad and not ca_batched:
            grad_ca = grad_ca.unsqueeze(0)
        if cb_requires_grad and not cb_batched:
            grad_cb = grad_cb.unsqueeze(0)

        # Inputs order (same as forward signature).
        return (
            grad_ca if ca_requires_grad else None,  # ca
            grad_cb if cb_requires_grad else None,  # cb
            None,  # cq
            grad_f_flat,  # source_amplitudes_scaled
            None,  # boundary_indices
            None, None, None, None,  # ay, by, ay_h, by_h
            None, None, None, None,  # ax, bx, ax_h, bx_h
            None, None, None, None,  # ky, ky_h, kx, kx_h
            None, None,  # sources_i, receivers_i
            None, None, None,  # rdy, rdx, dt
            None, None, None, None, None, None,  # nt, n_shots, ny, nx, n_sources, n_receivers
            None,  # accuracy
            None, None, None,  # ca_batched, cb_batched, cq_batched
            None, None, None, None,  # pml_y0, pml_x0, pml_y1, pml_x1
            None, None, None,  # storage_mode_str, storage_path, storage_compression
            None,  # alpha_rwii
            None, None, None,  # Ey, Hx, Hz
            None, None, None, None,  # m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
        )
