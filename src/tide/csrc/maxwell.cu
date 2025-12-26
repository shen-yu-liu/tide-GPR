/*
 * Maxwell wave equation propagator (CUDA implementation) 
 * 
 * This file contains the CUDA implementation of the 2D TM Maxwell equations
 * propagator with complete Adjoint State Method (ASM) support for gradient
 * computation.
 *
 * TM mode fields: Ey (electric), Hx, Hz (magnetic)
 *
 * Adjoint State Method for Maxwell TM equations:
 * ================================================
 * Forward equations (discrete):
 *   E_y^{n+1} = C_a * E_y^n + C_b * (∂H_z/∂x - ∂H_x/∂z)
 *   H_x^{n+1/2} = H_x^{n-1/2} - C_q * ∂E_y/∂z
 *   H_z^{n+1/2} = H_z^{n-1/2} + C_q * ∂E_y/∂x
 *
 * Adjoint equations (time-reversed):
 *   λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * (∂λ_Hz/∂x - ∂λ_Hx/∂z)
 *   λ_Hx^{n-1/2} = λ_Hx^{n+1/2} - C_b * ∂λ_Ey/∂z
 *   λ_Hz^{n-1/2} = λ_Hz^{n+1/2} + C_b * ∂λ_Ey/∂x
 *
 * Model gradients:
 *   ∂J/∂C_a = Σ_n E_y^n * λ_Ey^{n+1}
 *   ∂J/∂C_b = Σ_n curl_H^n * λ_Ey^{n+1}
 *
 * Gradient accumulation strategy:
 *   - Use per-shot gradient arrays (grad_ca_shot, grad_cb_shot)
 *   - Each shot writes to its own memory region (no race condition)
 *   - Use combine_grad kernel to sum across shots at the end
 */

#include <stdio.h>
#include <cstdint>
#include <cuda_bf16.h>
#include "common_gpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#ifndef TIDE_DEVICE
#define TIDE_DEVICE cuda
#endif

#define CAT_I(name, accuracy, dtype, device) \
  maxwell_tm_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) \
  CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, TIDE_DEVICE)

// 2D indexing macros
#define ND_INDEX(i, dy, dx) (i + (dy)*nx + (dx))
#define ND_INDEX_J(j, dy, dx) (j + (dy)*nx + (dx))

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

// Field access macros
#define EY(dy, dx) ey[ND_INDEX(i, dy, dx)]
#define HX(dy, dx) hx[ND_INDEX(i, dy, dx)]
#define HZ(dy, dx) hz[ND_INDEX(i, dy, dx)]

// Adjoint field access macros
#define LAMBDA_EY(dy, dx) lambda_ey[ND_INDEX(i, dy, dx)]
#define LAMBDA_HX(dy, dx) lambda_hx[ND_INDEX(i, dy, dx)]
#define LAMBDA_HZ(dy, dx) lambda_hz[ND_INDEX(i, dy, dx)]

// Material parameter access macros
#define CA(dy, dx) ca_shot[ND_INDEX_J(j, dy, dx)]
#define CB(dy, dx) cb_shot[ND_INDEX_J(j, dy, dx)]
#define CQ(dy, dx) cq_shot[ND_INDEX_J(j, dy, dx)]

// PML memory variable macros
#define M_HX_Z(dy, dx) m_hx_z[ND_INDEX(i, dy, dx)]
#define M_HZ_X(dy, dx) m_hz_x[ND_INDEX(i, dy, dx)]
#define M_EY_X(dy, dx) m_ey_x[ND_INDEX(i, dy, dx)]
#define M_EY_Z(dy, dx) m_ey_z[ND_INDEX(i, dy, dx)]

// Tangent field access macros (for JVP)
#define DEY(dy, dx) dey[ND_INDEX(i, dy, dx)]
#define DHX(dy, dx) dhx[ND_INDEX(i, dy, dx)]
#define DHZ(dy, dx) dhz[ND_INDEX(i, dy, dx)]

// Tangent PML memory variable macros (for JVP)
#define DM_HX_Z(dy, dx) dm_hx_z[ND_INDEX(i, dy, dx)]
#define DM_HZ_X(dy, dx) dm_hz_x[ND_INDEX(i, dy, dx)]
#define DM_EY_X(dy, dx) dm_ey_x[ND_INDEX(i, dy, dx)]
#define DM_EY_Z(dy, dx) dm_ey_z[ND_INDEX(i, dy, dx)]

// Adjoint PML memory variable macros
#define M_LAMBDA_EY_X(dy, dx) m_lambda_ey_x[ND_INDEX(i, dy, dx)]
#define M_LAMBDA_EY_Z(dy, dx) m_lambda_ey_z[ND_INDEX(i, dy, dx)]
#define M_LAMBDA_HX_Z(dy, dx) m_lambda_hx_z[ND_INDEX(i, dy, dx)]
#define M_LAMBDA_HZ_X(dy, dx) m_lambda_hz_x[ND_INDEX(i, dy, dx)]

#define MAX(a, b) (a > b ? a : b)

// Vacuum permittivity (F/m) to convert dL/d(epsilon_abs) -> dL/d(epsilon_r)
#define EP0 ((TIDE_DTYPE)8.8541878128e-12)

namespace {

// Device constants
__constant__ TIDE_DTYPE rdy;
__constant__ TIDE_DTYPE rdx;
__constant__ int64_t n_shots;
__constant__ int64_t ny;
__constant__ int64_t nx;
__constant__ int64_t shot_numel;
__constant__ int64_t n_sources_per_shot;
__constant__ int64_t n_receivers_per_shot;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;
__constant__ bool ca_batched;
__constant__ bool cb_batched;
__constant__ bool cq_batched;

// Add source to field
__global__ void add_sources_ey(TIDE_DTYPE *__restrict const ey,
                               TIDE_DTYPE const *__restrict const f,
                               int64_t const *__restrict const sources_i) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (0 <= src) {
      ey[shot_idx * shot_numel + src] += f[k];
    }
  }
}

// Add adjoint source at receiver locations (for backward pass)
__global__ void add_adjoint_sources_ey(TIDE_DTYPE *__restrict const ey,
                                       TIDE_DTYPE const *__restrict const f,
                                       int64_t const *__restrict const receivers_i) {
  int64_t receiver_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
    int64_t const rec = receivers_i[k];
    if (0 <= rec) {
      ey[shot_idx * shot_numel + rec] += f[k];
    }
  }
}

// Add scaled adjoint source at receiver locations (RWII composite wavefield).
__global__ void add_adjoint_sources_ey_scaled(
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE const *__restrict const f,
    int64_t const *__restrict const receivers_i,
    TIDE_DTYPE const alpha) {
  int64_t receiver_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
    int64_t const rec = receivers_i[k];
    if (0 <= rec) {
      ey[shot_idx * shot_numel + rec] += alpha * f[k];
    }
  }
}

// Record field at receiver locations
__global__ void record_receivers_ey(TIDE_DTYPE *__restrict const r,
                                   TIDE_DTYPE const *__restrict const ey,
                                   int64_t const *__restrict receivers_i) {
  int64_t receiver_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
    int64_t const rec = receivers_i[k];
    if (0 <= rec) {
      r[k] = ey[shot_idx * shot_numel + rec];
    }
  }
}

// Record field at source locations (for RWII source trace storage).
__global__ void record_sources_ey(TIDE_DTYPE *__restrict const u_src,
                                  TIDE_DTYPE const *__restrict const ey,
                                  int64_t const *__restrict const sources_i) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (0 <= src) {
      u_src[k] = ey[shot_idx * shot_numel + src];
    }
  }
}

// Record adjoint field at source locations (for backward pass - source gradient)
__global__ void record_adjoint_at_sources(TIDE_DTYPE *__restrict const grad_f,
                                          TIDE_DTYPE const *__restrict const lambda_ey,
                                          int64_t const *__restrict sources_i) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (0 <= src) {
      grad_f[k] = lambda_ey[shot_idx * shot_numel + src];
    }
  }
}

// RWII source gradient: grad_f = (w - u) / alpha at source locations.
__global__ void rwii_record_grad_f(TIDE_DTYPE *__restrict const grad_f,
                                   TIDE_DTYPE const *__restrict const w_ey,
                                   int64_t const *__restrict const sources_i,
                                   TIDE_DTYPE const *__restrict const u_src,
                                   TIDE_DTYPE const inv_alpha) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (0 <= src) {
      TIDE_DTYPE w_val = w_ey[shot_idx * shot_numel + src];
      grad_f[k] = (w_val - u_src[k]) * inv_alpha;
    }
  }
}

// Finalize RWII gradients: (Gamma_w - Gamma_u) / (2 * alpha).
__global__ void rwii_finalize_grads(TIDE_DTYPE *__restrict const gamma_w_ey,
                                    TIDE_DTYPE *__restrict const gamma_w_curl,
                                    TIDE_DTYPE const *__restrict const gamma_u_ey,
                                    TIDE_DTYPE const *__restrict const gamma_u_curl,
                                    TIDE_DTYPE const inv_2alpha,
                                    bool const ca_requires_grad,
                                    bool const cb_requires_grad) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    if (pml_y || pml_x) return;

    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;
    if (ca_requires_grad) {
      gamma_w_ey[i] = (gamma_w_ey[i] - gamma_u_ey[i]) * inv_2alpha;
    }
    if (cb_requires_grad) {
      gamma_w_curl[i] = (gamma_w_curl[i] - gamma_u_curl[i]) * inv_2alpha;
    }
  }
}


// Forward kernel: Update H fields (Hx and Hz)
__global__ __launch_bounds__(256) void forward_kernel_h(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE *__restrict const m_ey_z,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {
  
#if FD_PAD > 1
  // Shared-memory tiling for Ey stencil loads.
  // Assumes blockDim.z == 1 (one shot per block).
  extern __shared__ TIDE_DTYPE shmem[];
  TIDE_DTYPE *__restrict const tile_ey = shmem;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;

  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  int64_t const tile_numel = tile_w * tile_h;
  // Original scalar loading (optimization 2.1: vectorized loading disabled due to overhead)
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      tile_ey[ly * tile_pitch + lx] = __ldg(&ey[base + gy * nx + gx]);
    } else {
      tile_ey[ly * tile_pitch + lx] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define EY_L(dy, dx) tile_ey[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define EY_L(dy, dx) EY(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const pml_y0h = pml_y0;
    int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
    int64_t const pml_x0h = pml_x0;
    int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const cq_shot_i = cq_batched ? cq[i] : cq[j];

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE byh_val = __ldg(&byh[y]);
    TIDE_DTYPE ayh_val = __ldg(&ayh[y]);
    TIDE_DTYPE kyh_val = __ldg(&kyh[y]);
    TIDE_DTYPE bxh_val = __ldg(&bxh[x]);
    TIDE_DTYPE axh_val = __ldg(&axh[x]);
    TIDE_DTYPE kxh_val = __ldg(&kxh[x]);

    // Update Hx: Hx = Hx - cq * dEy/dz
    if (y < ny - FD_PAD) {
      bool pml_y = y < pml_y0h || y >= pml_y1h;

      TIDE_DTYPE dey_dz = DIFFYH1(EY_L);

      if (pml_y) {
        m_ey_z[i] = byh_val * m_ey_z[i] + ayh_val * dey_dz;
        dey_dz = dey_dz / kyh_val + m_ey_z[i];
      }

      hx[i] -= cq_shot_i * dey_dz;
    }

    // Update Hz: Hz = Hz + cq * dEy/dx
    if (x < nx - FD_PAD) {
      bool pml_x = x < pml_x0h || x >= pml_x1h;

      TIDE_DTYPE dey_dx = DIFFXH1(EY_L);

      if (pml_x) {
        m_ey_x[i] = bxh_val * m_ey_x[i] + axh_val * dey_dx;
        dey_dx = dey_dx / kxh_val + m_ey_x[i];
      }

      hz[i] += cq_shot_i * dey_dx;
    }
  }

#undef EY_L
}

// Forward kernel: Update E field (Ey) - standard version
__global__ __launch_bounds__(256) void forward_kernel_e(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {

#if FD_PAD > 1
  // Shared-memory tiling for Hx/Hz stencil loads.
  // Assumes blockDim.z == 1 (one shot per block).
  extern __shared__ TIDE_DTYPE shmem[];
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const tile_numel = tile_w * tile_h;
  TIDE_DTYPE *__restrict const tile_hx = shmem;
  TIDE_DTYPE *__restrict const tile_hz = shmem + tile_numel;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;
  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  // Original scalar loading (optimization 2.1: vectorized loading disabled due to overhead)
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      int64_t const g = base + gy * nx + gx;
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = __ldg(&hx[g]);
      tile_hz[offset] = __ldg(&hz[g]);
    } else {
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = (TIDE_DTYPE)0;
      tile_hz[offset] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define HX_L(dy, dx) tile_hx[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define HZ_L(dy, dx) tile_hz[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define HX_L(dy, dx) HX(dy, dx)
#define HZ_L(dy, dx) HZ(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cb_shot_i = cb_batched ? cb[i] : cb[j];

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    TIDE_DTYPE dhz_dx = DIFFX1(HZ_L);
    TIDE_DTYPE dhx_dz = DIFFY1(HX_L);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      m_hz_x[i] = bx_val * m_hz_x[i] + ax_val * dhz_dx;
      dhz_dx = dhz_dx / kx_val + m_hz_x[i];
    }

    if (pml_y) {
      m_hx_z[i] = by_val * m_hx_z[i] + ay_val * dhx_dz;
      dhx_dz = dhx_dz / ky_val + m_hx_z[i];
    }

    ey[i] = ca_shot_i * ey[i] + cb_shot_i * (dhz_dx - dhx_dz);
  }

#undef HX_L
#undef HZ_L
}

// Forward kernel: Update H fields with JVP (tangent propagation)
__global__ __launch_bounds__(256) void forward_kernel_h_jvp(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const dcq,
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const dey,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE *__restrict const dhx,
    TIDE_DTYPE *__restrict const dhz,
    TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE *__restrict const m_ey_z,
    TIDE_DTYPE *__restrict const dm_ey_x,
    TIDE_DTYPE *__restrict const dm_ey_z,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {

#if FD_PAD > 1
  // Shared-memory tiling for Ey/dey stencil loads.
  // Assumes blockDim.z == 1 (one shot per block).
  extern __shared__ TIDE_DTYPE shmem[];
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const tile_numel = tile_w * tile_h;
  TIDE_DTYPE *__restrict const tile_ey = shmem;
  TIDE_DTYPE *__restrict const tile_dey = shmem + tile_numel;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;

  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      int64_t const offset = ly * tile_pitch + lx;
      int64_t const g = base + gy * nx + gx;
      tile_ey[offset] = __ldg(&ey[g]);
      tile_dey[offset] = __ldg(&dey[g]);
    } else {
      int64_t const offset = ly * tile_pitch + lx;
      tile_ey[offset] = (TIDE_DTYPE)0;
      tile_dey[offset] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define EY_L(dy, dx) tile_ey[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define DEY_L(dy, dx) tile_dey[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define EY_L(dy, dx) EY(dy, dx)
#define DEY_L(dy, dx) DEY(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const pml_y0h = pml_y0;
    int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
    int64_t const pml_x0h = pml_x0;
    int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const cq_shot_i = cq_batched ? cq[i] : cq[j];
    TIDE_DTYPE const dcq_shot_i = cq_batched ? dcq[i] : dcq[j];

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE byh_val = __ldg(&byh[y]);
    TIDE_DTYPE ayh_val = __ldg(&ayh[y]);
    TIDE_DTYPE kyh_val = __ldg(&kyh[y]);
    TIDE_DTYPE bxh_val = __ldg(&bxh[x]);
    TIDE_DTYPE axh_val = __ldg(&axh[x]);
    TIDE_DTYPE kxh_val = __ldg(&kxh[x]);

    // Update Hx: Hx = Hx - cq * dEy/dz
    if (y < ny - FD_PAD) {
      bool pml_y = y < pml_y0h || y >= pml_y1h;

      TIDE_DTYPE dey_dz = DIFFYH1(EY_L);
      TIDE_DTYPE ddey_dz = DIFFYH1(DEY_L);

      if (pml_y) {
        m_ey_z[i] = byh_val * m_ey_z[i] + ayh_val * dey_dz;
        dm_ey_z[i] = byh_val * dm_ey_z[i] + ayh_val * ddey_dz;
        dey_dz = dey_dz / kyh_val + m_ey_z[i];
        ddey_dz = ddey_dz / kyh_val + dm_ey_z[i];
      }

      hx[i] -= cq_shot_i * dey_dz;
      dhx[i] -= dcq_shot_i * dey_dz + cq_shot_i * ddey_dz;
    }

    // Update Hz: Hz = Hz + cq * dEy/dx
    if (x < nx - FD_PAD) {
      bool pml_x = x < pml_x0h || x >= pml_x1h;

      TIDE_DTYPE dey_dx = DIFFXH1(EY_L);
      TIDE_DTYPE ddey_dx = DIFFXH1(DEY_L);

      if (pml_x) {
        m_ey_x[i] = bxh_val * m_ey_x[i] + axh_val * dey_dx;
        dm_ey_x[i] = bxh_val * dm_ey_x[i] + axh_val * ddey_dx;
        dey_dx = dey_dx / kxh_val + m_ey_x[i];
        ddey_dx = ddey_dx / kxh_val + dm_ey_x[i];
      }

      hz[i] += cq_shot_i * dey_dx;
      dhz[i] += dcq_shot_i * dey_dx + cq_shot_i * ddey_dx;
    }
  }

#undef EY_L
#undef DEY_L
}

// Forward kernel: Update E field with JVP (tangent propagation)
__global__ __launch_bounds__(256) void forward_kernel_e_jvp(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE const *__restrict const dhx,
    TIDE_DTYPE const *__restrict const dhz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const dey,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const dm_hx_z,
    TIDE_DTYPE *__restrict const dm_hz_x,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {

#if FD_PAD > 1
  // Shared-memory tiling for Hx/Hz and dHx/dHz stencil loads.
  // Assumes blockDim.z == 1 (one shot per block).
  extern __shared__ TIDE_DTYPE shmem[];
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const tile_numel = tile_w * tile_h;
  TIDE_DTYPE *__restrict const tile_hx = shmem;
  TIDE_DTYPE *__restrict const tile_hz = shmem + tile_numel;
  TIDE_DTYPE *__restrict const tile_dhx = shmem + 2 * tile_numel;
  TIDE_DTYPE *__restrict const tile_dhz = shmem + 3 * tile_numel;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;
  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      int64_t const offset = ly * tile_pitch + lx;
      int64_t const g = base + gy * nx + gx;
      tile_hx[offset] = __ldg(&hx[g]);
      tile_hz[offset] = __ldg(&hz[g]);
      tile_dhx[offset] = __ldg(&dhx[g]);
      tile_dhz[offset] = __ldg(&dhz[g]);
    } else {
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = (TIDE_DTYPE)0;
      tile_hz[offset] = (TIDE_DTYPE)0;
      tile_dhx[offset] = (TIDE_DTYPE)0;
      tile_dhz[offset] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define HX_L(dy, dx) tile_hx[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define HZ_L(dy, dx) tile_hz[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define DHX_L(dy, dx) tile_dhx[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define DHZ_L(dy, dx) tile_dhz[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define HX_L(dy, dx) HX(dy, dx)
#define HZ_L(dy, dx) HZ(dy, dx)
#define DHX_L(dy, dx) DHX(dy, dx)
#define DHZ_L(dy, dx) DHZ(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cb_shot_i = cb_batched ? cb[i] : cb[j];
    TIDE_DTYPE const dca_shot_i = ca_batched ? dca[i] : dca[j];
    TIDE_DTYPE const dcb_shot_i = cb_batched ? dcb[i] : dcb[j];

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    TIDE_DTYPE dhz_dx = DIFFX1(HZ_L);
    TIDE_DTYPE dhx_dz = DIFFY1(HX_L);
    TIDE_DTYPE ddhz_dx = DIFFX1(DHZ_L);
    TIDE_DTYPE ddhx_dz = DIFFY1(DHX_L);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      m_hz_x[i] = bx_val * m_hz_x[i] + ax_val * dhz_dx;
      dm_hz_x[i] = bx_val * dm_hz_x[i] + ax_val * ddhz_dx;
      dhz_dx = dhz_dx / kx_val + m_hz_x[i];
      ddhz_dx = ddhz_dx / kx_val + dm_hz_x[i];
    }

    if (pml_y) {
      m_hx_z[i] = by_val * m_hx_z[i] + ay_val * dhx_dz;
      dm_hx_z[i] = by_val * dm_hx_z[i] + ay_val * ddhx_dz;
      dhx_dz = dhx_dz / ky_val + m_hx_z[i];
      ddhx_dz = ddhx_dz / ky_val + dm_hx_z[i];
    }

    TIDE_DTYPE curl_h = dhz_dx - dhx_dz;
    TIDE_DTYPE d_curl_h = ddhz_dx - ddhx_dz;

    TIDE_DTYPE ey_val = ey[i];
    TIDE_DTYPE dey_val = dey[i];

    ey[i] = ca_shot_i * ey_val + cb_shot_i * curl_h;
    dey[i] = dca_shot_i * ey_val + ca_shot_i * dey_val + dcb_shot_i * curl_h + cb_shot_i * d_curl_h;
  }

#undef HX_L
#undef HZ_L
#undef DHX_L
#undef DHZ_L
}

// Forward kernel: Update E field (Ey) with optional RWII self-correlation accumulation
__global__ __launch_bounds__(256) void forward_kernel_e_rwii(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const gamma_u_ey,
    TIDE_DTYPE *__restrict const gamma_u_curl,
    bool const accum_ey,
    bool const accum_curl,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {

#if FD_PAD > 1
  // Shared-memory tiling for Hx/Hz stencil loads.
  // Assumes blockDim.z == 1 (one shot per block).
  extern __shared__ TIDE_DTYPE shmem[];
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const tile_numel = tile_w * tile_h;
  TIDE_DTYPE *__restrict const tile_hx = shmem;
  TIDE_DTYPE *__restrict const tile_hz = shmem + tile_numel;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;
  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  // Original scalar loading (optimization 2.1: vectorized loading disabled due to overhead)
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      int64_t const g = base + gy * nx + gx;
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = __ldg(&hx[g]);
      tile_hz[offset] = __ldg(&hz[g]);
    } else {
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = (TIDE_DTYPE)0;
      tile_hz[offset] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define HX_L(dy, dx) tile_hx[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define HZ_L(dy, dx) tile_hz[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define HX_L(dy, dx) HX(dy, dx)
#define HZ_L(dy, dx) HZ(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const pml_y0h = pml_y0;
    int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
    int64_t const pml_x0h = pml_x0;
    int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cb_shot_i = cb_batched ? cb[i] : cb[j];

    bool pml_y = y < pml_y0h || y >= pml_y1h;
    bool pml_x = x < pml_x0h || x >= pml_x1h;

    TIDE_DTYPE dhz_dx = DIFFX1(HZ_L);
    TIDE_DTYPE dhx_dz = DIFFY1(HX_L);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      m_hz_x[i] = bx_val * m_hz_x[i] + ax_val * dhz_dx;
      dhz_dx = dhz_dx / kx_val + m_hz_x[i];
    }

    if (pml_y) {
      m_hx_z[i] = by_val * m_hx_z[i] + ay_val * dhx_dz;
      dhx_dz = dhx_dz / ky_val + m_hx_z[i];
    }

    TIDE_DTYPE curl_h = dhz_dx - dhx_dz;

    if (!pml_y && !pml_x) {
      if (accum_ey && gamma_u_ey != nullptr) {
        TIDE_DTYPE v = ey[i];
        gamma_u_ey[i] += v * v;
      }
      if (accum_curl && gamma_u_curl != nullptr) {
        gamma_u_curl[i] += curl_h * curl_h;
      }
    }

    ey[i] = ca_shot_i * ey[i] + cb_shot_i * curl_h;
  }

#undef HX_L
#undef HZ_L
}

// Forward kernel: Update E field (Ey) with storage for gradient computation
__global__ void forward_kernel_e_with_storage(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const ey_store,      // Can be NULL
    TIDE_DTYPE *__restrict const curl_h_store,  // Can be NULL
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    bool const ca_requires_grad,
    bool const cb_requires_grad) {

#if FD_PAD > 1
  // Shared-memory tiling for Hx/Hz stencil loads.
  // Assumes blockDim.z == 1 (one shot per block).
  extern __shared__ TIDE_DTYPE shmem[];
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const tile_numel = tile_w * tile_h;
  TIDE_DTYPE *__restrict const tile_hx = shmem;
  TIDE_DTYPE *__restrict const tile_hz = shmem + tile_numel;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;
  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  // Original scalar loading (optimization 2.1: vectorized loading disabled due to overhead)
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      int64_t const g = base + gy * nx + gx;
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = __ldg(&hx[g]);
      tile_hz[offset] = __ldg(&hz[g]);
    } else {
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = (TIDE_DTYPE)0;
      tile_hz[offset] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define HX_L(dy, dx) tile_hx[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define HZ_L(dy, dx) tile_hz[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define HX_L(dy, dx) HX(dy, dx)
#define HZ_L(dy, dx) HZ(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cb_shot_i = cb_batched ? cb[i] : cb[j];

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    TIDE_DTYPE dhz_dx = DIFFX1(HZ_L);
    TIDE_DTYPE dhx_dz = DIFFY1(HX_L);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      m_hz_x[i] = bx_val * m_hz_x[i] + ax_val * dhz_dx;
      dhz_dx = dhz_dx / kx_val + m_hz_x[i];
    }

    if (pml_y) {
      m_hx_z[i] = by_val * m_hx_z[i] + ay_val * dhx_dz;
      dhx_dz = dhx_dz / ky_val + m_hx_z[i];
    }

    TIDE_DTYPE curl_h = dhz_dx - dhx_dz;

    // Store values for gradient computation (before E update)
    if (ca_requires_grad && ey_store != nullptr) {
      ey_store[i] = ey[i];
    }
    if (cb_requires_grad && curl_h_store != nullptr) {
      curl_h_store[i] = curl_h;
    }

    ey[i] = ca_shot_i * ey[i] + cb_shot_i * curl_h;
  }

#undef HX_L
#undef HZ_L
}

// Forward kernel: Update E field (Ey) with BF16 storage for gradient computation
// Stores Ey and curl_H in __nv_bfloat16 to reduce snapshot bandwidth/size.
__global__ void forward_kernel_e_with_storage_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    __nv_bfloat16 *__restrict const ey_store,      // Can be NULL
    __nv_bfloat16 *__restrict const curl_h_store,  // Can be NULL
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    bool const ca_requires_grad,
    bool const cb_requires_grad) {

#if FD_PAD > 1
  // Shared-memory tiling for Hx/Hz stencil loads.
  // Assumes blockDim.z == 1 (one shot per block).
  extern __shared__ TIDE_DTYPE shmem[];
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const tile_numel = tile_w * tile_h;
  TIDE_DTYPE *__restrict const tile_hx = shmem;
  TIDE_DTYPE *__restrict const tile_hz = shmem + tile_numel;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;
  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  // Original scalar loading (optimization 2.1: vectorized loading disabled due to overhead)
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      int64_t const g = base + gy * nx + gx;
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = __ldg(&hx[g]);
      tile_hz[offset] = __ldg(&hz[g]);
    } else {
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = (TIDE_DTYPE)0;
      tile_hz[offset] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define HX_L(dy, dx) tile_hx[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define HZ_L(dy, dx) tile_hz[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define HX_L(dy, dx) HX(dy, dx)
#define HZ_L(dy, dx) HZ(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cb_shot_i = cb_batched ? cb[i] : cb[j];

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    TIDE_DTYPE dhz_dx = DIFFX1(HZ_L);
    TIDE_DTYPE dhx_dz = DIFFY1(HX_L);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      m_hz_x[i] = bx_val * m_hz_x[i] + ax_val * dhz_dx;
      dhz_dx = dhz_dx / kx_val + m_hz_x[i];
    }

    if (pml_y) {
      m_hx_z[i] = by_val * m_hx_z[i] + ay_val * dhx_dz;
      dhx_dz = dhx_dz / ky_val + m_hx_z[i];
    }

    TIDE_DTYPE curl_h = dhz_dx - dhx_dz;

    if (ca_requires_grad && ey_store != nullptr) {
      ey_store[i] = __float2bfloat16((float)ey[i]);
    }
    if (cb_requires_grad && curl_h_store != nullptr) {
      curl_h_store[i] = __float2bfloat16((float)curl_h);
    }

    ey[i] = ca_shot_i * ey[i] + cb_shot_i * curl_h;
  }

#undef HX_L
#undef HZ_L
}

// Forward kernel: Update E field (Ey) with storage and device-side time.
// Backward kernel: Update adjoint λ_H fields
__global__ void backward_kernel_lambda_h(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const lambda_hx,
    TIDE_DTYPE *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const m_lambda_ey_x,
    TIDE_DTYPE *__restrict const m_lambda_ey_z,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {
  
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const pml_y0h = pml_y0;
    int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
    int64_t const pml_x0h = pml_x0;
    int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const cb_shot_i = cb_batched ? cb[i] : cb[j];

    // Update λ_Hx: λ_Hx = λ_Hx - cb * d(λ_Ey)/dz
    if (y < ny - FD_PAD) {
      bool pml_y = y < pml_y0h || y >= pml_y1h;

      TIDE_DTYPE d_lambda_ey_dz = DIFFYH1(LAMBDA_EY);

      if (pml_y) {
        m_lambda_ey_z[i] = __ldg(&byh[y]) * m_lambda_ey_z[i] + __ldg(&ayh[y]) * d_lambda_ey_dz;
        d_lambda_ey_dz = d_lambda_ey_dz / __ldg(&kyh[y]) + m_lambda_ey_z[i];
      }

      lambda_hx[i] -= cb_shot_i * d_lambda_ey_dz;
    }

    // Update λ_Hz: λ_Hz = λ_Hz + cb * d(λ_Ey)/dx
    if (x < nx - FD_PAD) {
      bool pml_x = x < pml_x0h || x >= pml_x1h;

      TIDE_DTYPE d_lambda_ey_dx = DIFFXH1(LAMBDA_EY);

      if (pml_x) {
        m_lambda_ey_x[i] = __ldg(&bxh[x]) * m_lambda_ey_x[i] + __ldg(&axh[x]) * d_lambda_ey_dx;
        d_lambda_ey_dx = d_lambda_ey_dx / __ldg(&kxh[x]) + m_lambda_ey_x[i];
      }

      lambda_hz[i] += cb_shot_i * d_lambda_ey_dx;
    }
  }
}

// Backward kernel: Update adjoint λ_Ey field with per-shot gradient accumulation
// Uses pml_y0/pml_y1/pml_x0/pml_x1 for both adjoint propagation and gradient masking
// NO atomicAdd - each shot writes to its own memory region
__global__ void backward_kernel_lambda_e_with_grad(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE const *__restrict const ey_store,
    TIDE_DTYPE const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,   // [n_shots, ny, nx] - per-shot gradient
    TIDE_DTYPE *__restrict const grad_cb_shot,   // [n_shots, ny, nx] - per-shot gradient
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    int64_t const step_ratio_val) {
  
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cq_shot_i = cq_batched ? cq[i] : cq[j];

    // Determine PML region (pml_y/pml_x = true means in PML region)
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    // Compute d(λ_Hz)/dx at integer grid points
    TIDE_DTYPE d_lambda_hz_dx = DIFFX1(LAMBDA_HZ);
    // Compute d(λ_Hx)/dz at integer grid points
    TIDE_DTYPE d_lambda_hx_dz = DIFFY1(LAMBDA_HX);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    // Apply adjoint CPML for d(λ_Hz)/dx (only in PML region)
    if (pml_x) {
      m_lambda_hz_x[i] = bx_val * m_lambda_hz_x[i] + ax_val * d_lambda_hz_dx;
      d_lambda_hz_dx = d_lambda_hz_dx / kx_val + m_lambda_hz_x[i];
    }

    // Apply adjoint CPML for d(λ_Hx)/dz (only in PML region)
    if (pml_y) {
      m_lambda_hx_z[i] = by_val * m_lambda_hx_z[i] + ay_val * d_lambda_hx_dz;
      d_lambda_hx_dz = d_lambda_hx_dz / ky_val + m_lambda_hx_z[i];
    }

    // curl_λH = d(λ_Hz)/dx - d(λ_Hx)/dz
    TIDE_DTYPE curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;

    // Store current λ_Ey before update (this is λ_Ey^{n+1})
    TIDE_DTYPE lambda_ey_curr = lambda_ey[i];

    // Update λ_Ey: λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * curl_λH
    lambda_ey[i] = ca_shot_i * lambda_ey_curr + cq_shot_i * curl_lambda_h;

    // Accumulate per-shot gradients only in interior region (!pml_y && !pml_x)
    if (!pml_y && !pml_x) {
      // grad_ca_shot[shot_idx, y, x] += λ_Ey^{n+1} * E_y^n
      // Convert from BF16 back to FP32 for computation
      if (ca_requires_grad && ey_store != nullptr) {
        TIDE_DTYPE ey_n = ey_store[i];
        grad_ca_shot[i] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio_val;
      }

      // grad_cb_shot[shot_idx, y, x] += λ_Ey^{n+1} * curl_H^n
      if (cb_requires_grad && curl_h_store != nullptr) {
        TIDE_DTYPE curl_h_n = curl_h_store[i];
        grad_cb_shot[i] += lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio_val;
      }
    }
  }
}

// Backward kernel: Update adjoint λ_Ey field with BF16 snapshot loads.
__global__ void backward_kernel_lambda_e_with_grad_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    __nv_bfloat16 const *__restrict const ey_store,
    __nv_bfloat16 const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    int64_t const step_ratio_val) {
  
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cq_shot_i = cq_batched ? cq[i] : cq[j];

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    TIDE_DTYPE d_lambda_hz_dx = DIFFX1(LAMBDA_HZ);
    TIDE_DTYPE d_lambda_hx_dz = DIFFY1(LAMBDA_HX);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      m_lambda_hz_x[i] = bx_val * m_lambda_hz_x[i] + ax_val * d_lambda_hz_dx;
      d_lambda_hz_dx = d_lambda_hz_dx / kx_val + m_lambda_hz_x[i];
    }

    if (pml_y) {
      m_lambda_hx_z[i] = by_val * m_lambda_hx_z[i] + ay_val * d_lambda_hx_dz;
      d_lambda_hx_dz = d_lambda_hx_dz / ky_val + m_lambda_hx_z[i];
    }

    TIDE_DTYPE curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;

    TIDE_DTYPE lambda_ey_curr = lambda_ey[i];
    lambda_ey[i] = ca_shot_i * lambda_ey_curr + cq_shot_i * curl_lambda_h;

    if (!pml_y && !pml_x) {
      if (ca_requires_grad && ey_store != nullptr) {
        TIDE_DTYPE ey_n = (TIDE_DTYPE)__bfloat162float(ey_store[i]);
        grad_ca_shot[i] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio_val;
      }
      if (cb_requires_grad && curl_h_store != nullptr) {
        TIDE_DTYPE curl_h_n = (TIDE_DTYPE)__bfloat162float(curl_h_store[i]);
        grad_cb_shot[i] += lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio_val;
      }
    }
  }
}

// Backward kernel: Update adjoint λ_Ey field (no gradient accumulation).
__global__ void backward_kernel_lambda_e(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    (void)ayh;
    (void)axh;
    (void)byh;
    (void)bxh;
    (void)kyh;
    (void)kxh;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cq_shot_i = cq_batched ? cq[i] : cq[j];

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    TIDE_DTYPE d_lambda_hz_dx = DIFFX1(LAMBDA_HZ);
    TIDE_DTYPE d_lambda_hx_dz = DIFFY1(LAMBDA_HX);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      m_lambda_hz_x[i] = bx_val * m_lambda_hz_x[i] + ax_val * d_lambda_hz_dx;
      d_lambda_hz_dx = d_lambda_hz_dx / kx_val + m_lambda_hz_x[i];
    }

    if (pml_y) {
      m_lambda_hx_z[i] = by_val * m_lambda_hx_z[i] + ay_val * d_lambda_hx_dz;
      d_lambda_hx_dz = d_lambda_hx_dz / ky_val + m_lambda_hx_z[i];
    }

    TIDE_DTYPE curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;

    TIDE_DTYPE lambda_ey_curr = lambda_ey[i];
    lambda_ey[i] = ca_shot_i * lambda_ey_curr + cq_shot_i * curl_lambda_h;
  }
}

// Combine per-shot gradients into final gradient (sum across shots)
__global__ void combine_grad(TIDE_DTYPE *__restrict const grad,
                             TIDE_DTYPE const *__restrict const grad_shot) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t j = y * nx + x;
    int64_t const stride = shot_numel;
    TIDE_DTYPE sum = 0;
    #pragma unroll 4
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      sum += grad_shot[shot_idx * stride + j];
    }
    grad[j] += sum;
  }
}

__global__ void convert_grad_ca_cb_to_eps_sigma(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const grad_ca,
    TIDE_DTYPE const *__restrict const grad_cb,
    TIDE_DTYPE const *__restrict const grad_ca_shot,
    TIDE_DTYPE const *__restrict const grad_cb_shot,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE const dt,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y;
  if (x >= nx || y >= ny) {
    return;
  }

  int64_t shot_idx = (int64_t)blockIdx.z;
  if (!ca_batched_h) {
    shot_idx = 0;
  }

  int64_t const j = y * nx + x;
  int64_t const idx_shot = shot_idx * shot_numel + j;
  int64_t const out_idx = ca_batched_h ? idx_shot : j;
  int64_t const ca_idx = ca_batched_h ? idx_shot : j;
  int64_t const cb_idx = cb_batched_h ? idx_shot : j;

  TIDE_DTYPE const ca_val = ca[ca_idx];
  TIDE_DTYPE const cb_val = cb[cb_idx];
  TIDE_DTYPE const cb_sq = cb_val * cb_val;
  TIDE_DTYPE const inv_dt = (TIDE_DTYPE)1 / dt;

  TIDE_DTYPE grad_ca_val = 0;
  if (ca_requires_grad) {
    grad_ca_val = ca_batched_h ? grad_ca_shot[idx_shot] : grad_ca[j];
  }

  TIDE_DTYPE grad_cb_val = 0;
  if (cb_requires_grad) {
    grad_cb_val = cb_batched_h ? grad_cb_shot[idx_shot] : grad_cb[j];
  }

  TIDE_DTYPE const dca_de = ((TIDE_DTYPE)1 - ca_val) * cb_val * inv_dt;
  TIDE_DTYPE const dcb_de = -cb_sq * inv_dt;
  TIDE_DTYPE const dca_ds = -((TIDE_DTYPE)0.5) * ((TIDE_DTYPE)1 + ca_val) * cb_val;
  TIDE_DTYPE const dcb_ds = -((TIDE_DTYPE)0.5) * cb_sq;

  if (grad_eps != nullptr) {
    TIDE_DTYPE const grad_e = grad_ca_val * dca_de + grad_cb_val * dcb_de;
    grad_eps[out_idx] = grad_e * EP0;
  }
  if (grad_sigma != nullptr) {
    grad_sigma[out_idx] = grad_ca_val * dca_ds + grad_cb_val * dcb_ds;
  }
}

// Gather a boundary ring (flat indices) into a compact [n_shots, boundary_numel] buffer.
__global__ void gather_boundary(
    TIDE_DTYPE const *__restrict const field,
    TIDE_DTYPE *__restrict const store,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t grid_idx = boundary_indices[bi];
    store[shot_idx * boundary_numel + bi] = field[shot_idx * shot_numel + grid_idx];
  }
}

__global__ void gather_boundary_bf16(
    TIDE_DTYPE const *__restrict const field,
    __nv_bfloat16 *__restrict const store,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t grid_idx = boundary_indices[bi];
    store[shot_idx * boundary_numel + bi] =
        __float2bfloat16((float)field[shot_idx * shot_numel + grid_idx]);
  }
}

// Gather Ey/Hx/Hz boundary ring in a single kernel to reduce launch overhead.
__global__ void gather_boundary_3(
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const bey,
    TIDE_DTYPE *__restrict const bhx,
    TIDE_DTYPE *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t grid_idx = boundary_indices[bi];
    int64_t const field_offset = shot_idx * shot_numel + grid_idx;
    int64_t const store_offset = shot_idx * boundary_numel + bi;
    bey[store_offset] = ey[field_offset];
    bhx[store_offset] = hx[field_offset];
    bhz[store_offset] = hz[field_offset];
  }
}

__global__ void gather_boundary_3_bf16(
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    __nv_bfloat16 *__restrict const bey,
    __nv_bfloat16 *__restrict const bhx,
    __nv_bfloat16 *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t grid_idx = boundary_indices[bi];
    int64_t const field_offset = shot_idx * shot_numel + grid_idx;
    int64_t const store_offset = shot_idx * boundary_numel + bi;
    bey[store_offset] = __float2bfloat16((float)ey[field_offset]);
    bhx[store_offset] = __float2bfloat16((float)hx[field_offset]);
    bhz[store_offset] = __float2bfloat16((float)hz[field_offset]);
  }
}

__device__ __forceinline__ int64_t boundary_dense_width_from_numel(
    int64_t const boundary_numel) {
  int64_t const interior_ny = pml_y1 - pml_y0;
  int64_t const denom = 2 * (nx + interior_ny);
  if (denom <= 0) return 0;
  return boundary_numel / denom;
}

__device__ __forceinline__ int64_t boundary_dense_grid_idx(
    int64_t bi, int64_t const boundary_numel) {
  int64_t const w = boundary_dense_width_from_numel(boundary_numel);
  if (w <= 0) return 0;

  int64_t const top_n = w * nx;
  if (bi < top_n) {
    int64_t const row = bi / nx;
    int64_t const col = bi - row * nx;
    int64_t const y = (pml_y0 - w) + row;
    return y * nx + col;
  }
  bi -= top_n;

  int64_t const bottom_n = top_n;
  if (bi < bottom_n) {
    int64_t const row = bi / nx;
    int64_t const col = bi - row * nx;
    int64_t const y = pml_y1 + row;
    return y * nx + col;
  }
  bi -= bottom_n;

  int64_t const row_width = 2 * w;
  int64_t const row = bi / row_width;
  int64_t const col = bi - row * row_width;
  int64_t const y = pml_y0 + row;
  int64_t const x =
      (col < w) ? ((pml_x0 - w) + col) : (pml_x1 + (col - w));
  return y * nx + x;
}

// Dense boundary gather: avoids indirect addressing through boundary_indices.
// Assumes boundary buffer layout matches the Python-side dense ordering.
__global__ void gather_boundary_3_dense(
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const bey,
    TIDE_DTYPE *__restrict const bhx,
    TIDE_DTYPE *__restrict const bhz,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t const grid_idx = boundary_dense_grid_idx(bi, boundary_numel);
    int64_t const field_offset = shot_idx * shot_numel + grid_idx;
    int64_t const store_offset = shot_idx * boundary_numel + bi;
    bey[store_offset] = ey[field_offset];
    bhx[store_offset] = hx[field_offset];
    bhz[store_offset] = hz[field_offset];
  }
}

__global__ void gather_boundary_3_dense_bf16(
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    __nv_bfloat16 *__restrict const bey,
    __nv_bfloat16 *__restrict const bhx,
    __nv_bfloat16 *__restrict const bhz,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t const grid_idx = boundary_dense_grid_idx(bi, boundary_numel);
    int64_t const field_offset = shot_idx * shot_numel + grid_idx;
    int64_t const store_offset = shot_idx * boundary_numel + bi;
    bey[store_offset] = __float2bfloat16((float)ey[field_offset]);
    bhx[store_offset] = __float2bfloat16((float)hx[field_offset]);
    bhz[store_offset] = __float2bfloat16((float)hz[field_offset]);
  }
}

// Scatter a compact [n_shots, boundary_numel] buffer into a boundary ring.
__global__ void scatter_boundary(
    TIDE_DTYPE *__restrict const field,
    TIDE_DTYPE const *__restrict const store,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t grid_idx = boundary_indices[bi];
    field[shot_idx * shot_numel + grid_idx] = store[shot_idx * boundary_numel + bi];
  }
}

__global__ void scatter_boundary_bf16(
    TIDE_DTYPE *__restrict const field,
    __nv_bfloat16 const *__restrict const store,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t grid_idx = boundary_indices[bi];
    field[shot_idx * shot_numel + grid_idx] =
        (TIDE_DTYPE)__bfloat162float(store[shot_idx * boundary_numel + bi]);
  }
}

__global__ void scatter_boundary_dense(
    TIDE_DTYPE *__restrict const field,
    TIDE_DTYPE const *__restrict const store,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t const grid_idx = boundary_dense_grid_idx(bi, boundary_numel);
    field[shot_idx * shot_numel + grid_idx] =
        store[shot_idx * boundary_numel + bi];
  }
}

__global__ void scatter_boundary_dense_bf16(
    TIDE_DTYPE *__restrict const field,
    __nv_bfloat16 const *__restrict const store,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t const grid_idx = boundary_dense_grid_idx(bi, boundary_numel);
    field[shot_idx * shot_numel + grid_idx] =
        (TIDE_DTYPE)__bfloat162float(store[shot_idx * boundary_numel + bi]);
  }
}

// Scatter Hx/Hz boundary ring in a single kernel to reduce launch overhead.
__global__ void scatter_boundary_2(
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE const *__restrict const bhx,
    TIDE_DTYPE const *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t grid_idx = boundary_indices[bi];
    int64_t const field_offset = shot_idx * shot_numel + grid_idx;
    int64_t const store_offset = shot_idx * boundary_numel + bi;
    hx[field_offset] = bhx[store_offset];
    hz[field_offset] = bhz[store_offset];
  }
}

__global__ void scatter_boundary_2_bf16(
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    __nv_bfloat16 const *__restrict const bhx,
    __nv_bfloat16 const *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t grid_idx = boundary_indices[bi];
    int64_t const field_offset = shot_idx * shot_numel + grid_idx;
    int64_t const store_offset = shot_idx * boundary_numel + bi;
    hx[field_offset] = (TIDE_DTYPE)__bfloat162float(bhx[store_offset]);
    hz[field_offset] = (TIDE_DTYPE)__bfloat162float(bhz[store_offset]);
  }
}

__global__ void scatter_boundary_2_dense(
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE const *__restrict const bhx,
    TIDE_DTYPE const *__restrict const bhz,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t const grid_idx = boundary_dense_grid_idx(bi, boundary_numel);
    int64_t const field_offset = shot_idx * shot_numel + grid_idx;
    int64_t const store_offset = shot_idx * boundary_numel + bi;
    hx[field_offset] = bhx[store_offset];
    hz[field_offset] = bhz[store_offset];
  }
}

__global__ void scatter_boundary_2_dense_bf16(
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    __nv_bfloat16 const *__restrict const bhx,
    __nv_bfloat16 const *__restrict const bhz,
    int64_t const boundary_numel) {
  int64_t bi =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx = (int64_t)blockIdx.y;
  if (shot_idx < n_shots && bi < boundary_numel) {
    int64_t const grid_idx = boundary_dense_grid_idx(bi, boundary_numel);
    int64_t const field_offset = shot_idx * shot_numel + grid_idx;
    int64_t const store_offset = shot_idx * boundary_numel + bi;
    hx[field_offset] = (TIDE_DTYPE)__bfloat162float(bhx[store_offset]);
    hz[field_offset] = (TIDE_DTYPE)__bfloat162float(bhz[store_offset]);
  }
}

__global__ void subtract_sources_ey(TIDE_DTYPE *__restrict const ey,
                                    TIDE_DTYPE const *__restrict const f,
                                    int64_t const *__restrict const sources_i) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (0 <= src) {
      ey[shot_idx * shot_numel + src] -= f[k];
    }
  }
}

// Invert the E update in the physical domain (non-PML), producing Ey^t and curl(H) for gradients.
__global__ void inverse_kernel_e_and_curl(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,        // in: Ey^{t+1} (after removing source), out: Ey^t
    TIDE_DTYPE *__restrict const curl_h_out  // out: curl(H) at time t
) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    if (pml_y || pml_x) return;

    TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];

    TIDE_DTYPE dhz_dx = DIFFX1(HZ);
    TIDE_DTYPE dhx_dz = DIFFY1(HX);
    TIDE_DTYPE curl_h = dhz_dx - dhx_dz;

    curl_h_out[i] = curl_h;
    ey[i] = (ey[i] - cb_val * curl_h) / ca_val;
  }
}

// Invert E update (non-PML) and optionally accumulate curl(H)^2 for RWII.
__global__ __launch_bounds__(256) void inverse_kernel_e_and_curl_rwii(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const curl_h_out,
    TIDE_DTYPE *__restrict const gamma_w_curl,
    bool const accum_curl) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    if (pml_y || pml_x) return;

    TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];

    TIDE_DTYPE dhz_dx = DIFFX1(HZ);
    TIDE_DTYPE dhx_dz = DIFFY1(HX);
    TIDE_DTYPE curl_h = dhz_dx - dhx_dz;

    if (curl_h_out != nullptr) {
      curl_h_out[i] = curl_h;
    }
    if (accum_curl && gamma_w_curl != nullptr) {
      gamma_w_curl[i] += curl_h * curl_h;
    }
    ey[i] = (ey[i] - cb_val * curl_h) / ca_val;
  }
}

// Invert the H update in the physical domain (non-PML), producing H^t from H^{t+1} and Ey^t.
__global__ void inverse_kernel_h(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    if (pml_y || pml_x) return;

    TIDE_DTYPE const cq_val = cq_batched ? cq[i] : cq[j];

    if (y < ny - FD_PAD) {
      TIDE_DTYPE dey_dz = DIFFYH1(EY);
      hx[i] += cq_val * dey_dz;
    }
    if (x < nx - FD_PAD) {
      TIDE_DTYPE dey_dx = DIFFXH1(EY);
      hz[i] -= cq_val * dey_dx;
    }
  }
}

// Invert H update (non-PML) and optionally accumulate Ey^2 for RWII.
__global__ __launch_bounds__(256) void inverse_kernel_h_rwii(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE *__restrict const gamma_w_ey,
    bool const accum_ey) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    if (pml_y || pml_x) return;

    if (accum_ey && gamma_w_ey != nullptr) {
      TIDE_DTYPE v = ey[i];
      gamma_w_ey[i] += v * v;
    }

    TIDE_DTYPE const cq_val = cq_batched ? cq[i] : cq[j];

    if (y < ny - FD_PAD) {
      TIDE_DTYPE dey_dz = DIFFYH1(EY);
      hx[i] += cq_val * dey_dz;
    }
    if (x < nx - FD_PAD) {
      TIDE_DTYPE dey_dx = DIFFXH1(EY);
      hz[i] -= cq_val * dey_dx;
    }
  }
}

}  // namespace

// Forward propagation function
extern "C" void FUNC(forward)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const f,
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE *const r,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const device) {
  
  cudaSetDevice(device);
  (void)dt_h;
  (void)step_ratio_h;

  int64_t const shot_numel_h = ny_h * nx_h;

  // Copy constants to device with caching to avoid redundant copies
  static TIDE_DTYPE cached_rdy = 0, cached_rdx = 0;
  static int64_t cached_n_shots = -1, cached_ny = -1, cached_nx = -1;
  static int64_t cached_shot_numel = -1, cached_n_sources_per_shot = -1, cached_n_receivers_per_shot = -1;
  static int64_t cached_pml_y0 = -1, cached_pml_y1 = -1;
  static int64_t cached_pml_x0 = -1, cached_pml_x1 = -1;
  static bool cached_ca_batched = false, cached_cb_batched = false, cached_cq_batched = false;
  static bool first_call = true;
  
  if (first_call || cached_rdy != rdy_h || cached_rdx != rdx_h ||
      cached_n_shots != n_shots_h || cached_ny != ny_h || cached_nx != nx_h ||
      cached_shot_numel != shot_numel_h || cached_n_sources_per_shot != n_sources_per_shot_h ||
      cached_n_receivers_per_shot != n_receivers_per_shot_h ||
      cached_pml_y0 != pml_y0_h || cached_pml_y1 != pml_y1_h ||
      cached_pml_x0 != pml_x0_h || cached_pml_x1 != pml_x1_h ||
      cached_ca_batched != ca_batched_h || cached_cb_batched != cb_batched_h ||
      cached_cq_batched != cq_batched_h) {
    
    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));
    
    cached_rdy = rdy_h; cached_rdx = rdx_h;
    cached_n_shots = n_shots_h; cached_ny = ny_h; cached_nx = nx_h;
    cached_shot_numel = shot_numel_h; cached_n_sources_per_shot = n_sources_per_shot_h;
    cached_n_receivers_per_shot = n_receivers_per_shot_h;
    cached_pml_y0 = pml_y0_h; cached_pml_y1 = pml_y1_h;
    cached_pml_x0 = pml_x0_h; cached_pml_x1 = pml_x1_h;
    cached_ca_batched = ca_batched_h; cached_cb_batched = cb_batched_h;
    cached_cq_batched = cq_batched_h;
    first_call = false;
  }

  dim3 dimBlock(32, 8, 1);
  int64_t gridx = (nx_h - 2 * FD_PAD + 2 + dimBlock.x - 1) / dimBlock.x;
  int64_t gridy = (ny_h - 2 * FD_PAD + 2 + dimBlock.y - 1) / dimBlock.y;
  int64_t gridz = n_shots_h;
  dim3 dimGrid(gridx, gridy, gridz);
#if FD_PAD > 1
  size_t const shmem_h_bytes =
      (size_t)(dimBlock.x + 2 * FD_PAD) * (size_t)(dimBlock.y + 2 * FD_PAD) *
      sizeof(TIDE_DTYPE);
  size_t const shmem_e_bytes = 2 * shmem_h_bytes;
#else
  size_t const shmem_h_bytes = 0;
  size_t const shmem_e_bytes = 0;
#endif

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources(
      (n_sources_per_shot_h + dimBlock_sources.x - 1) / dimBlock_sources.x,
      n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers(
      (n_receivers_per_shot_h + dimBlock_receivers.x - 1) / dimBlock_receivers.x,
      n_shots_h, 1);

  auto run_step = [&](int64_t t) {
    forward_kernel_h<<<dimGrid, dimBlock, shmem_h_bytes>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);
    forward_kernel_e<<<dimGrid, dimBlock, shmem_e_bytes>>>(
        ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  };

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    run_step(t);
  }

  gpuErrchk(cudaPeekAtLastError());
}

// Forward propagation with JVP (tangent propagation)
extern "C" void FUNC(forward_jvp)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const dca,
    TIDE_DTYPE const *const dcb,
    TIDE_DTYPE const *const dcq,
    TIDE_DTYPE const *const f,
    TIDE_DTYPE const *const df,
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const dey,
    TIDE_DTYPE *const dhx,
    TIDE_DTYPE *const dhz,
    TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE *const dm_ey_x,
    TIDE_DTYPE *const dm_ey_z,
    TIDE_DTYPE *const dm_hx_z,
    TIDE_DTYPE *const dm_hz_x,
    TIDE_DTYPE *const r,
    TIDE_DTYPE *const dr,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const device) {
  
  cudaSetDevice(device);
  (void)dt_h;
  (void)step_ratio_h;

  int64_t const shot_numel_h = ny_h * nx_h;

  // Copy constants to device with caching to avoid redundant copies
  static TIDE_DTYPE cached_rdy = 0, cached_rdx = 0;
  static int64_t cached_n_shots = -1, cached_ny = -1, cached_nx = -1;
  static int64_t cached_shot_numel = -1, cached_n_sources_per_shot = -1, cached_n_receivers_per_shot = -1;
  static int64_t cached_pml_y0 = -1, cached_pml_y1 = -1;
  static int64_t cached_pml_x0 = -1, cached_pml_x1 = -1;
  static bool cached_ca_batched = false, cached_cb_batched = false, cached_cq_batched = false;
  static bool first_call = true;
  
  if (first_call || cached_rdy != rdy_h || cached_rdx != rdx_h ||
      cached_n_shots != n_shots_h || cached_ny != ny_h || cached_nx != nx_h ||
      cached_shot_numel != shot_numel_h || cached_n_sources_per_shot != n_sources_per_shot_h ||
      cached_n_receivers_per_shot != n_receivers_per_shot_h ||
      cached_pml_y0 != pml_y0_h || cached_pml_y1 != pml_y1_h ||
      cached_pml_x0 != pml_x0_h || cached_pml_x1 != pml_x1_h ||
      cached_ca_batched != ca_batched_h || cached_cb_batched != cb_batched_h ||
      cached_cq_batched != cq_batched_h) {
    
    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));
    
    cached_rdy = rdy_h; cached_rdx = rdx_h;
    cached_n_shots = n_shots_h; cached_ny = ny_h; cached_nx = nx_h;
    cached_shot_numel = shot_numel_h; cached_n_sources_per_shot = n_sources_per_shot_h;
    cached_n_receivers_per_shot = n_receivers_per_shot_h;
    cached_pml_y0 = pml_y0_h; cached_pml_y1 = pml_y1_h;
    cached_pml_x0 = pml_x0_h; cached_pml_x1 = pml_x1_h;
    cached_ca_batched = ca_batched_h; cached_cb_batched = cb_batched_h;
    cached_cq_batched = cq_batched_h;
    first_call = false;
  }

  dim3 dimBlock(32, 8, 1);
  int64_t gridx = (nx_h - 2 * FD_PAD + 2 + dimBlock.x - 1) / dimBlock.x;
  int64_t gridy = (ny_h - 2 * FD_PAD + 2 + dimBlock.y - 1) / dimBlock.y;
  int64_t gridz = n_shots_h;
  dim3 dimGrid(gridx, gridy, gridz);
#if FD_PAD > 1
  size_t const shmem_base_bytes =
      (size_t)(dimBlock.x + 2 * FD_PAD) * (size_t)(dimBlock.y + 2 * FD_PAD) *
      sizeof(TIDE_DTYPE);
  size_t const shmem_h_bytes = 2 * shmem_base_bytes;
  size_t const shmem_e_bytes = 4 * shmem_base_bytes;
#else
  size_t const shmem_h_bytes = 0;
  size_t const shmem_e_bytes = 0;
#endif

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources(
      (n_sources_per_shot_h + dimBlock_sources.x - 1) / dimBlock_sources.x,
      n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers(
      (n_receivers_per_shot_h + dimBlock_receivers.x - 1) / dimBlock_receivers.x,
      n_shots_h, 1);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h_jvp<<<dimGrid, dimBlock, shmem_h_bytes>>>(
        cq, dcq, ey, dey, hx, hz, dhx, dhz,
        m_ey_x, m_ey_z, dm_ey_x, dm_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);
    forward_kernel_e_jvp<<<dimGrid, dimBlock, shmem_e_bytes>>>(
        ca, cb, dca, dcb, hx, hz, dhx, dhz, ey, dey,
        m_hx_z, m_hz_x, dm_hx_z, dm_hz_x,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
      if (df != nullptr) {
        add_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
            dey, df + t * n_shots_h * n_sources_per_shot_h, sources_i);
      }
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
      if (dr != nullptr) {
        record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
            dr + t * n_shots_h * n_receivers_per_shot_h, dey, receivers_i);
      }
    }
  }

  gpuErrchk(cudaPeekAtLastError());
}

// Forward with storage for backward pass
extern "C" void FUNC(forward_with_storage)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const f,
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE *const r,
    void *const ey_store_1,
    void *const ey_store_3,
    char const *const *const ey_filenames,
    void *const curl_store_1,
    void *const curl_store_3,
    char const *const *const curl_filenames,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode_h,
    int64_t const shot_bytes_uncomp_h,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const device) {
  
  cudaSetDevice(device);

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h = (shot_bytes_uncomp_h == shot_numel_h * 2);
  int64_t const store_stride_elems =
      (storage_mode_h == STORAGE_DEVICE) ? (n_shots_h * shot_numel_h) : 0;

  // Copy constants to device with caching to avoid redundant copies
  static TIDE_DTYPE cached_rdy2 = 0, cached_rdx2 = 0;
  static int64_t cached_n_shots2 = -1, cached_ny2 = -1, cached_nx2 = -1;
  static int64_t cached_shot_numel2 = -1, cached_n_sources_per_shot2 = -1, cached_n_receivers_per_shot2 = -1;
  static int64_t cached_pml_y02 = -1, cached_pml_y12 = -1;
  static int64_t cached_pml_x02 = -1, cached_pml_x12 = -1;
  static bool cached_ca_batched2 = false, cached_cb_batched2 = false, cached_cq_batched2 = false;
  static bool first_call2 = true;
  
  if (first_call2 || cached_rdy2 != rdy_h || cached_rdx2 != rdx_h ||
      cached_n_shots2 != n_shots_h || cached_ny2 != ny_h || cached_nx2 != nx_h ||
      cached_shot_numel2 != shot_numel_h || cached_n_sources_per_shot2 != n_sources_per_shot_h ||
      cached_n_receivers_per_shot2 != n_receivers_per_shot_h ||
      cached_pml_y02 != pml_y0_h || cached_pml_y12 != pml_y1_h ||
      cached_pml_x02 != pml_x0_h || cached_pml_x12 != pml_x1_h ||
      cached_ca_batched2 != ca_batched_h || cached_cb_batched2 != cb_batched_h ||
      cached_cq_batched2 != cq_batched_h) {
    
    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));
    
    cached_rdy2 = rdy_h; cached_rdx2 = rdx_h;
    cached_n_shots2 = n_shots_h; cached_ny2 = ny_h; cached_nx2 = nx_h;
    cached_shot_numel2 = shot_numel_h; cached_n_sources_per_shot2 = n_sources_per_shot_h;
    cached_n_receivers_per_shot2 = n_receivers_per_shot_h;
    cached_pml_y02 = pml_y0_h; cached_pml_y12 = pml_y1_h;
    cached_pml_x02 = pml_x0_h; cached_pml_x12 = pml_x1_h;
    cached_ca_batched2 = ca_batched_h; cached_cb_batched2 = cb_batched_h;
    cached_cq_batched2 = cq_batched_h;
    first_call2 = false;
  }

  dim3 dimBlock(32, 8, 1);
  int64_t gridx = (nx_h - 2 * FD_PAD + 2 + dimBlock.x - 1) / dimBlock.x;
  int64_t gridy = (ny_h - 2 * FD_PAD + 2 + dimBlock.y - 1) / dimBlock.y;
  int64_t gridz = n_shots_h;
  dim3 dimGrid(gridx, gridy, gridz);
#if FD_PAD > 1
  size_t const shmem_h_bytes =
      (size_t)(dimBlock.x + 2 * FD_PAD) * (size_t)(dimBlock.y + 2 * FD_PAD) *
      sizeof(TIDE_DTYPE);
  size_t const shmem_e_bytes = 2 * shmem_h_bytes;
#else
  size_t const shmem_h_bytes = 0;
  size_t const shmem_e_bytes = 0;
#endif

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources(
      (n_sources_per_shot_h + dimBlock_sources.x - 1) / dimBlock_sources.x,
      n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers(
      (n_receivers_per_shot_h + dimBlock_receivers.x - 1) / dimBlock_receivers.x,
      n_shots_h, 1);

  FILE *fp_ey = nullptr;
  FILE *fp_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad) fp_ey = fopen(ey_filenames[0], "wb");
    if (cb_requires_grad) fp_curl = fopen(curl_filenames[0], "wb");
  }

  auto run_step = [&](int64_t t) {
    forward_kernel_h<<<dimGrid, dimBlock, shmem_h_bytes>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    bool const store_step = ((t % step_ratio_h) == 0);
    bool const store_ey = store_step && ca_requires_grad;
    bool const store_curl = store_step && cb_requires_grad;
    bool const want_store = store_ey || store_curl;
    if (want_store) {
      int64_t const step_idx = t / step_ratio_h;

      void *__restrict const ey_store_1_t =
          (uint8_t *)ey_store_1 +
          (storage_mode_h == STORAGE_DEVICE ? (size_t)step_idx * bytes_per_step_store : 0);
      void *__restrict const ey_store_3_t =
          (uint8_t *)ey_store_3 +
          (storage_mode_h == STORAGE_CPU
               ? (size_t)step_idx * bytes_per_step_store
               : 0);

      void *__restrict const curl_store_1_t =
          (uint8_t *)curl_store_1 +
          (storage_mode_h == STORAGE_DEVICE ? (size_t)step_idx * bytes_per_step_store : 0);
      void *__restrict const curl_store_3_t =
          (uint8_t *)curl_store_3 +
          (storage_mode_h == STORAGE_CPU
               ? (size_t)step_idx * bytes_per_step_store
               : 0);

      if (storage_bf16_h) {
        forward_kernel_e_with_storage_bf16<<<dimGrid, dimBlock, shmem_e_bytes>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
            store_ey ? (__nv_bfloat16 *)ey_store_1_t : nullptr,
            store_curl ? (__nv_bfloat16 *)curl_store_1_t : nullptr, ay, ayh, ax,
            axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey, store_curl);
      } else {
        forward_kernel_e_with_storage<<<dimGrid, dimBlock, shmem_e_bytes>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
            store_ey ? (TIDE_DTYPE *)ey_store_1_t : nullptr,
            store_curl ? (TIDE_DTYPE *)curl_store_1_t : nullptr, ay, ayh, ax,
            axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey, store_curl);
      }

      if (store_ey) {
        storage_save_snapshot_gpu(
            ey_store_1_t, ey_store_3_t, fp_ey, storage_mode_h, step_idx,
            (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
      }
      if (store_curl) {
        storage_save_snapshot_gpu(
            curl_store_1_t, curl_store_3_t, fp_curl, storage_mode_h, step_idx,
            (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
      }
    } else {
      forward_kernel_e<<<dimGrid, dimBlock, shmem_e_bytes>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx,
          bxh, ky, kyh, kx, kxh);
    }

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  };

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    run_step(t);
  }

  if (fp_ey != nullptr) fclose(fp_ey);
  if (fp_curl != nullptr) fclose(fp_curl);

  gpuErrchk(cudaPeekAtLastError());
}

// Forward with boundary storage (for boundary/RWII gradient modes)
extern "C" void FUNC(forward_with_boundary_storage)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const f,
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE *const r,
    void *const boundary_ey_store_1,
    void *const boundary_ey_store_3,
    char const *const *const boundary_ey_filenames,
    void *const boundary_hx_store_1,
    void *const boundary_hx_store_3,
    char const *const *const boundary_hx_filenames,
    void *const boundary_hz_store_1,
    void *const boundary_hz_store_3,
    char const *const *const boundary_hz_filenames,
    int64_t const *const boundary_indices,
    int64_t const boundary_numel_h,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const storage_mode_h,
    int64_t const shot_bytes_uncomp_h,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const device) {
  
  cudaSetDevice(device);

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h = (shot_bytes_uncomp_h == boundary_numel_h * 2);

  // Copy constants to device with caching to avoid redundant copies
  static TIDE_DTYPE cached_rdy4 = 0, cached_rdx4 = 0;
  static int64_t cached_n_shots4 = -1, cached_ny4 = -1, cached_nx4 = -1;
  static int64_t cached_shot_numel4 = -1, cached_n_sources_per_shot4 = -1, cached_n_receivers_per_shot4 = -1;
  static int64_t cached_pml_y04 = -1, cached_pml_y14 = -1;
  static int64_t cached_pml_x04 = -1, cached_pml_x14 = -1;
  static bool cached_ca_batched4 = false, cached_cb_batched4 = false, cached_cq_batched4 = false;
  static bool first_call4 = true;
  
  if (first_call4 || cached_rdy4 != rdy_h || cached_rdx4 != rdx_h ||
      cached_n_shots4 != n_shots_h || cached_ny4 != ny_h || cached_nx4 != nx_h ||
      cached_shot_numel4 != shot_numel_h || cached_n_sources_per_shot4 != n_sources_per_shot_h ||
      cached_n_receivers_per_shot4 != n_receivers_per_shot_h ||
      cached_pml_y04 != pml_y0_h || cached_pml_y14 != pml_y1_h ||
      cached_pml_x04 != pml_x0_h || cached_pml_x14 != pml_x1_h ||
      cached_ca_batched4 != ca_batched_h || cached_cb_batched4 != cb_batched_h ||
      cached_cq_batched4 != cq_batched_h) {
    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));

    cached_rdy4 = rdy_h; cached_rdx4 = rdx_h;
    cached_n_shots4 = n_shots_h; cached_ny4 = ny_h; cached_nx4 = nx_h;
    cached_shot_numel4 = shot_numel_h; cached_n_sources_per_shot4 = n_sources_per_shot_h;
    cached_n_receivers_per_shot4 = n_receivers_per_shot_h;
    cached_pml_y04 = pml_y0_h; cached_pml_y14 = pml_y1_h;
    cached_pml_x04 = pml_x0_h; cached_pml_x14 = pml_x1_h;
    cached_ca_batched4 = ca_batched_h; cached_cb_batched4 = cb_batched_h;
    cached_cq_batched4 = cq_batched_h;
    first_call4 = false;
  }

  dim3 dimBlock(32, 8, 1);
  int64_t gridx = (nx_h - 2 * FD_PAD + 2 + dimBlock.x - 1) / dimBlock.x;
  int64_t gridy = (ny_h - 2 * FD_PAD + 2 + dimBlock.y - 1) / dimBlock.y;
  int64_t gridz = n_shots_h;
  dim3 dimGrid(gridx, gridy, gridz);
#if FD_PAD > 1
  size_t const shmem_h_bytes =
      (size_t)(dimBlock.x + 2 * FD_PAD) * (size_t)(dimBlock.y + 2 * FD_PAD) *
      sizeof(TIDE_DTYPE);
  size_t const shmem_e_bytes = 2 * shmem_h_bytes;
#else
  size_t const shmem_h_bytes = 0;
  size_t const shmem_e_bytes = 0;
#endif

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources(
      (n_sources_per_shot_h + dimBlock_sources.x - 1) / dimBlock_sources.x,
      n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers(
      (n_receivers_per_shot_h + dimBlock_receivers.x - 1) / dimBlock_receivers.x,
      n_shots_h, 1);

  dim3 dimBlock_boundary(256, 1, 1);
  dim3 dimGrid_boundary(
      (boundary_numel_h + dimBlock_boundary.x - 1) / dimBlock_boundary.x,
      n_shots_h, 1);

  int64_t const boundary_interior_ny_h = pml_y1_h - pml_y0_h;
  int64_t const boundary_denom_h = 2 * (nx_h + boundary_interior_ny_h);
  bool const boundary_dense_ok =
      (boundary_denom_h > 0) && (boundary_numel_h > 0) &&
      (boundary_numel_h % boundary_denom_h == 0) &&
      ((boundary_numel_h / boundary_denom_h) > 0) &&
      ((boundary_numel_h / boundary_denom_h) <= pml_y0_h) &&
      ((boundary_numel_h / boundary_denom_h) <= pml_x0_h) &&
      (pml_y1_h + (boundary_numel_h / boundary_denom_h) <= ny_h) &&
      (pml_x1_h + (boundary_numel_h / boundary_denom_h) <= nx_h);

  FILE *fp_bey = nullptr;
  FILE *fp_bhx = nullptr;
  FILE *fp_bhz = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    fp_bey = fopen(boundary_ey_filenames[0], "wb");
    fp_bhx = fopen(boundary_hx_filenames[0], "wb");
    fp_bhz = fopen(boundary_hz_filenames[0], "wb");
  }

  auto boundary_store1_offset = [&](int64_t step_idx) -> size_t {
    if (storage_mode_h == STORAGE_DEVICE) {
      return (size_t)step_idx * bytes_per_step_store;
    }
    if (storage_mode_h == STORAGE_CPU) {
      // CPU mode uses device staging; Python allocates a 2-buffer ping-pong tensor.
      return (size_t)(step_idx & 1) * bytes_per_step_store;
    }
    return 0;
  };

  auto store_boundary_step = [&](int64_t step_idx) {
    void *bey_store_1_t =
        (uint8_t *)boundary_ey_store_1 +
        boundary_store1_offset(step_idx);
    void *bhx_store_1_t =
        (uint8_t *)boundary_hx_store_1 +
        boundary_store1_offset(step_idx);
    void *bhz_store_1_t =
        (uint8_t *)boundary_hz_store_1 +
        boundary_store1_offset(step_idx);

    void *bey_store_3_t =
        (uint8_t *)boundary_ey_store_3 +
        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);
    void *bhx_store_3_t =
        (uint8_t *)boundary_hx_store_3 +
        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);
    void *bhz_store_3_t =
        (uint8_t *)boundary_hz_store_3 +
        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);

	    if (storage_bf16_h) {
        if (boundary_dense_ok) {
	        gather_boundary_3_dense_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
	            ey, hx, hz,
	            (__nv_bfloat16 *)bey_store_1_t,
	            (__nv_bfloat16 *)bhx_store_1_t,
	            (__nv_bfloat16 *)bhz_store_1_t,
	            boundary_numel_h);
        } else {
	        gather_boundary_3_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
	            ey, hx, hz,
	            (__nv_bfloat16 *)bey_store_1_t,
	            (__nv_bfloat16 *)bhx_store_1_t,
	            (__nv_bfloat16 *)bhz_store_1_t,
	            boundary_indices, boundary_numel_h);
        }
	    } else {
        if (boundary_dense_ok) {
	        gather_boundary_3_dense<<<dimGrid_boundary, dimBlock_boundary>>>(
	            ey, hx, hz,
	            (TIDE_DTYPE *)bey_store_1_t,
	            (TIDE_DTYPE *)bhx_store_1_t,
	            (TIDE_DTYPE *)bhz_store_1_t,
	            boundary_numel_h);
        } else {
	        gather_boundary_3<<<dimGrid_boundary, dimBlock_boundary>>>(
	            ey, hx, hz,
	            (TIDE_DTYPE *)bey_store_1_t,
	            (TIDE_DTYPE *)bhx_store_1_t,
	            (TIDE_DTYPE *)bhz_store_1_t,
	            boundary_indices, boundary_numel_h);
        }
	    }

    if (storage_mode_h == STORAGE_CPU || storage_mode_h == STORAGE_DISK) {
      storage_save_snapshot_gpu(
          bey_store_1_t, bey_store_3_t, fp_bey, storage_mode_h, step_idx,
          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
      storage_save_snapshot_gpu(
          bhx_store_1_t, bhx_store_3_t, fp_bhx, storage_mode_h, step_idx,
          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
      storage_save_snapshot_gpu(
          bhz_store_1_t, bhz_store_3_t, fp_bhz, storage_mode_h, step_idx,
          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
    }
  };

  // Store initial boundary (t=0)
  store_boundary_step(0);

  for (int64_t t = 0; t < nt; ++t) {
    forward_kernel_h<<<dimGrid, dimBlock, shmem_h_bytes>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);
    forward_kernel_e<<<dimGrid, dimBlock, shmem_e_bytes>>>(
        ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }

    // Store boundary after source injection: this is the state at time (t+1).
    store_boundary_step(t + 1);
  }

  if (fp_bey != nullptr) fclose(fp_bey);
  if (fp_bhx != nullptr) fclose(fp_bhx);
  if (fp_bhz != nullptr) fclose(fp_bhz);

  gpuErrchk(cudaPeekAtLastError());
}

// Forward with boundary storage + RWII accumulators (CUDA only)
extern "C" void FUNC(forward_with_boundary_storage_rwii)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const f,
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE *const r,
    void *const boundary_ey_store_1,
    void *const boundary_ey_store_3,
    char const *const *const boundary_ey_filenames,
    void *const boundary_hx_store_1,
    void *const boundary_hx_store_3,
    char const *const *const boundary_hx_filenames,
    void *const boundary_hz_store_1,
    void *const boundary_hz_store_3,
    char const *const *const boundary_hz_filenames,
    int64_t const *const boundary_indices,
    int64_t const boundary_numel_h,
    TIDE_DTYPE *const gamma_u_ey,
    TIDE_DTYPE *const gamma_u_curl,
    TIDE_DTYPE *const u_src,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const storage_mode_h,
    int64_t const shot_bytes_uncomp_h,
    bool const accum_ey_h,
    bool const accum_curl_h,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const device) {

  cudaSetDevice(device);

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h = (shot_bytes_uncomp_h == boundary_numel_h * 2);

  // Copy constants to device with caching to avoid redundant copies
  static TIDE_DTYPE cached_rdy6 = 0, cached_rdx6 = 0;
  static int64_t cached_n_shots6 = -1, cached_ny6 = -1, cached_nx6 = -1;
  static int64_t cached_shot_numel6 = -1, cached_n_sources_per_shot6 = -1, cached_n_receivers_per_shot6 = -1;
  static int64_t cached_pml_y06 = -1, cached_pml_y16 = -1;
  static int64_t cached_pml_x06 = -1, cached_pml_x16 = -1;
  static bool cached_ca_batched6 = false, cached_cb_batched6 = false, cached_cq_batched6 = false;
  static bool first_call6 = true;

  if (first_call6 || cached_rdy6 != rdy_h || cached_rdx6 != rdx_h ||
      cached_n_shots6 != n_shots_h || cached_ny6 != ny_h || cached_nx6 != nx_h ||
      cached_shot_numel6 != shot_numel_h || cached_n_sources_per_shot6 != n_sources_per_shot_h ||
      cached_n_receivers_per_shot6 != n_receivers_per_shot_h ||
      cached_pml_y06 != pml_y0_h || cached_pml_y16 != pml_y1_h ||
      cached_pml_x06 != pml_x0_h || cached_pml_x16 != pml_x1_h ||
      cached_ca_batched6 != ca_batched_h || cached_cb_batched6 != cb_batched_h ||
      cached_cq_batched6 != cq_batched_h) {
    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));

    cached_rdy6 = rdy_h;
    cached_rdx6 = rdx_h;
    cached_n_shots6 = n_shots_h;
    cached_ny6 = ny_h;
    cached_nx6 = nx_h;
    cached_shot_numel6 = shot_numel_h;
    cached_n_sources_per_shot6 = n_sources_per_shot_h;
    cached_n_receivers_per_shot6 = n_receivers_per_shot_h;
    cached_pml_y06 = pml_y0_h;
    cached_pml_y16 = pml_y1_h;
    cached_pml_x06 = pml_x0_h;
    cached_pml_x16 = pml_x1_h;
    cached_ca_batched6 = ca_batched_h;
    cached_cb_batched6 = cb_batched_h;
    cached_cq_batched6 = cq_batched_h;
    first_call6 = false;
  }

  dim3 dimBlock(32, 8, 1);
  dim3 dimGrid((nx_h - 2 * FD_PAD + dimBlock.x - 1) / dimBlock.x,
               (ny_h - 2 * FD_PAD + dimBlock.y - 1) / dimBlock.y, n_shots_h);

#if FD_PAD > 1
  size_t const shmem_h_bytes =
      (size_t)(dimBlock.x + 2 * FD_PAD) * (size_t)(dimBlock.y + 2 * FD_PAD) *
      sizeof(TIDE_DTYPE);
  size_t const shmem_e_bytes =
      2 * (size_t)(dimBlock.x + 2 * FD_PAD) * (size_t)(dimBlock.y + 2 * FD_PAD) *
      sizeof(TIDE_DTYPE);
#else
  size_t const shmem_h_bytes = 0;
  size_t const shmem_e_bytes = 0;
#endif

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources(
      (n_sources_per_shot_h + dimBlock_sources.x - 1) / dimBlock_sources.x,
      n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers(
      (n_receivers_per_shot_h + dimBlock_receivers.x - 1) / dimBlock_receivers.x,
      n_shots_h, 1);

  dim3 dimBlock_boundary(256, 1, 1);
  dim3 dimGrid_boundary(
      (boundary_numel_h + dimBlock_boundary.x - 1) / dimBlock_boundary.x,
      n_shots_h, 1);

  int64_t const boundary_interior_ny_h = pml_y1_h - pml_y0_h;
  int64_t const boundary_denom_h = 2 * (nx_h + boundary_interior_ny_h);
  bool const boundary_dense_ok =
      (boundary_denom_h > 0) && (boundary_numel_h > 0) &&
      (boundary_numel_h % boundary_denom_h == 0) &&
      ((boundary_numel_h / boundary_denom_h) > 0) &&
      ((boundary_numel_h / boundary_denom_h) <= pml_y0_h) &&
      ((boundary_numel_h / boundary_denom_h) <= pml_x0_h) &&
      (pml_y1_h + (boundary_numel_h / boundary_denom_h) <= ny_h) &&
      (pml_x1_h + (boundary_numel_h / boundary_denom_h) <= nx_h);

  FILE *fp_bey = nullptr;
  FILE *fp_bhx = nullptr;
  FILE *fp_bhz = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    fp_bey = fopen(boundary_ey_filenames[0], "wb");
    fp_bhx = fopen(boundary_hx_filenames[0], "wb");
    fp_bhz = fopen(boundary_hz_filenames[0], "wb");
  }

  auto boundary_store1_offset = [&](int64_t step_idx) -> size_t {
    if (storage_mode_h == STORAGE_DEVICE) {
      return (size_t)step_idx * bytes_per_step_store;
    }
    if (storage_mode_h == STORAGE_CPU) {
      // CPU mode uses device staging; Python allocates a 2-buffer ping-pong tensor.
      return (size_t)(step_idx & 1) * bytes_per_step_store;
    }
    return 0;
  };

  auto store_boundary_step = [&](int64_t step_idx) {
    void *bey_store_1_t =
        (uint8_t *)boundary_ey_store_1 +
        boundary_store1_offset(step_idx);
    void *bhx_store_1_t =
        (uint8_t *)boundary_hx_store_1 +
        boundary_store1_offset(step_idx);
    void *bhz_store_1_t =
        (uint8_t *)boundary_hz_store_1 +
        boundary_store1_offset(step_idx);

    void *bey_store_3_t =
        (uint8_t *)boundary_ey_store_3 +
        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);
    void *bhx_store_3_t =
        (uint8_t *)boundary_hx_store_3 +
        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);
    void *bhz_store_3_t =
        (uint8_t *)boundary_hz_store_3 +
        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);

    if (storage_bf16_h) {
      if (boundary_dense_ok) {
        gather_boundary_3_dense_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
            ey, hx, hz,
            (__nv_bfloat16 *)bey_store_1_t,
            (__nv_bfloat16 *)bhx_store_1_t,
            (__nv_bfloat16 *)bhz_store_1_t,
            boundary_numel_h);
      } else {
        gather_boundary_3_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
            ey, hx, hz,
            (__nv_bfloat16 *)bey_store_1_t,
            (__nv_bfloat16 *)bhx_store_1_t,
            (__nv_bfloat16 *)bhz_store_1_t,
            boundary_indices, boundary_numel_h);
      }
    } else {
      if (boundary_dense_ok) {
        gather_boundary_3_dense<<<dimGrid_boundary, dimBlock_boundary>>>(
            ey, hx, hz,
            (TIDE_DTYPE *)bey_store_1_t,
            (TIDE_DTYPE *)bhx_store_1_t,
            (TIDE_DTYPE *)bhz_store_1_t,
            boundary_numel_h);
      } else {
        gather_boundary_3<<<dimGrid_boundary, dimBlock_boundary>>>(
            ey, hx, hz,
            (TIDE_DTYPE *)bey_store_1_t,
            (TIDE_DTYPE *)bhx_store_1_t,
            (TIDE_DTYPE *)bhz_store_1_t,
            boundary_indices, boundary_numel_h);
      }
    }

    if (storage_mode_h == STORAGE_CPU || storage_mode_h == STORAGE_DISK) {
      storage_save_snapshot_gpu(
          bey_store_1_t, bey_store_3_t, fp_bey, storage_mode_h, step_idx,
          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
      storage_save_snapshot_gpu(
          bhx_store_1_t, bhx_store_3_t, fp_bhx, storage_mode_h, step_idx,
          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
      storage_save_snapshot_gpu(
          bhz_store_1_t, bhz_store_3_t, fp_bhz, storage_mode_h, step_idx,
          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
    }
  };

  // Store initial boundary (t=0)
  store_boundary_step(0);

  for (int64_t t = 0; t < nt; ++t) {
    if (n_sources_per_shot_h > 0 && u_src != nullptr) {
      record_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          u_src + t * n_shots_h * n_sources_per_shot_h, ey, sources_i);
    }

    forward_kernel_h<<<dimGrid, dimBlock, shmem_h_bytes>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    forward_kernel_e_rwii<<<dimGrid, dimBlock, shmem_e_bytes>>>(
        ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
        gamma_u_ey, gamma_u_curl, accum_ey_h, accum_curl_h,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }

    store_boundary_step(t + 1);
  }

  if (fp_bey != nullptr) fclose(fp_bey);
  if (fp_bhx != nullptr) fclose(fp_bhx);
  if (fp_bhz != nullptr) fclose(fp_bhz);

  gpuErrchk(cudaPeekAtLastError());
}

// Backward with boundary storage (reconstruct forward fields from boundary ring)
extern "C" void FUNC(backward_with_boundary)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const f,
    TIDE_DTYPE const *const grad_r,
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const curl_h,
    TIDE_DTYPE *const lambda_ey,
    TIDE_DTYPE *const lambda_hx,
    TIDE_DTYPE *const lambda_hz,
    TIDE_DTYPE *const m_lambda_ey_x,
    TIDE_DTYPE *const m_lambda_ey_z,
    TIDE_DTYPE *const m_lambda_hx_z,
    TIDE_DTYPE *const m_lambda_hz_x,
    void *const boundary_ey_store_1,
    void *const boundary_ey_store_3,
    char const *const *const boundary_ey_filenames,
    void *const boundary_hx_store_1,
    void *const boundary_hx_store_3,
    char const *const *const boundary_hx_filenames,
    void *const boundary_hz_store_1,
    void *const boundary_hz_store_3,
    char const *const *const boundary_hz_filenames,
    int64_t const *const boundary_indices,
    int64_t const boundary_numel_h,
    TIDE_DTYPE *const grad_f,
    TIDE_DTYPE *const grad_ca,
    TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE *const grad_eps,
    TIDE_DTYPE *const grad_sigma,
    TIDE_DTYPE *const grad_ca_shot,
    TIDE_DTYPE *const grad_cb_shot,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const storage_mode_h,
    int64_t const shot_bytes_uncomp_h,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const device) {

  cudaSetDevice(device);
  (void)dt_h;

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h = (shot_bytes_uncomp_h == boundary_numel_h * 2);

  // Copy constants to device with caching to avoid redundant copies
  static TIDE_DTYPE cached_rdy5 = 0, cached_rdx5 = 0;
  static int64_t cached_n_shots5 = -1, cached_ny5 = -1, cached_nx5 = -1;
  static int64_t cached_shot_numel5 = -1, cached_n_sources_per_shot5 = -1, cached_n_receivers_per_shot5 = -1;
  static int64_t cached_pml_y05 = -1, cached_pml_y15 = -1;
  static int64_t cached_pml_x05 = -1, cached_pml_x15 = -1;
  static bool cached_ca_batched5 = false, cached_cb_batched5 = false, cached_cq_batched5 = false;
  static bool first_call5 = true;

  if (first_call5 || cached_rdy5 != rdy_h ||
      cached_rdx5 != rdx_h || cached_n_shots5 != n_shots_h ||
      cached_ny5 != ny_h || cached_nx5 != nx_h ||
      cached_shot_numel5 != shot_numel_h ||
      cached_n_sources_per_shot5 != n_sources_per_shot_h ||
      cached_n_receivers_per_shot5 != n_receivers_per_shot_h ||
      cached_pml_y05 != pml_y0_h || cached_pml_y15 != pml_y1_h ||
      cached_pml_x05 != pml_x0_h || cached_pml_x15 != pml_x1_h ||
      cached_ca_batched5 != ca_batched_h ||
      cached_cb_batched5 != cb_batched_h ||
      cached_cq_batched5 != cq_batched_h) {

    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));

    cached_rdy5 = rdy_h;
    cached_rdx5 = rdx_h;
    cached_n_shots5 = n_shots_h;
    cached_ny5 = ny_h;
    cached_nx5 = nx_h;
    cached_shot_numel5 = shot_numel_h;
    cached_n_sources_per_shot5 = n_sources_per_shot_h;
    cached_n_receivers_per_shot5 = n_receivers_per_shot_h;
    cached_pml_y05 = pml_y0_h;
    cached_pml_y15 = pml_y1_h;
    cached_pml_x05 = pml_x0_h;
    cached_pml_x15 = pml_x1_h;
    cached_ca_batched5 = ca_batched_h;
    cached_cb_batched5 = cb_batched_h;
    cached_cq_batched5 = cq_batched_h;
    first_call5 = false;
  }

  dim3 dimBlock(32, 8, 1);
  int64_t gridx = (nx_h - 2 * FD_PAD + 2 + dimBlock.x - 1) / dimBlock.x;
  int64_t gridy = (ny_h - 2 * FD_PAD + 2 + dimBlock.y - 1) / dimBlock.y;
  int64_t gridz = n_shots_h;
  dim3 dimGrid(gridx, gridy, gridz);

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources(
      (n_sources_per_shot_h + dimBlock_sources.x - 1) / dimBlock_sources.x,
      n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers(
      (n_receivers_per_shot_h + dimBlock_receivers.x - 1) / dimBlock_receivers.x,
      n_shots_h, 1);

  dim3 dimBlock_boundary(256, 1, 1);
  dim3 dimGrid_boundary(
      (boundary_numel_h + dimBlock_boundary.x - 1) / dimBlock_boundary.x,
      n_shots_h, 1);

  int64_t const boundary_interior_ny_h = pml_y1_h - pml_y0_h;
  int64_t const boundary_denom_h = 2 * (nx_h + boundary_interior_ny_h);
  bool const boundary_dense_ok =
      (boundary_denom_h > 0) && (boundary_numel_h > 0) &&
      (boundary_numel_h % boundary_denom_h == 0) &&
      ((boundary_numel_h / boundary_denom_h) > 0) &&
      ((boundary_numel_h / boundary_denom_h) <= pml_y0_h) &&
      ((boundary_numel_h / boundary_denom_h) <= pml_x0_h) &&
      (pml_y1_h + (boundary_numel_h / boundary_denom_h) <= ny_h) &&
      (pml_x1_h + (boundary_numel_h / boundary_denom_h) <= nx_h);

  FILE *fp_bey = nullptr;
  FILE *fp_bhx = nullptr;
  FILE *fp_bhz = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    fp_bey = fopen(boundary_ey_filenames[0], "rb");
    fp_bhx = fopen(boundary_hx_filenames[0], "rb");
    fp_bhz = fopen(boundary_hz_filenames[0], "rb");
  }

  auto boundary_store1_offset = [&](int64_t step_idx) -> size_t {
    if (storage_mode_h == STORAGE_DEVICE) {
      return (size_t)step_idx * bytes_per_step_store;
    }
    if (storage_mode_h == STORAGE_CPU) {
      // CPU mode uses device staging; Python allocates a 2-buffer ping-pong tensor.
      return (size_t)(step_idx & 1) * bytes_per_step_store;
    }
    return 0;
  };

	  auto load_and_scatter = [&](TIDE_DTYPE *field, void *store_1, void *store_3,
	                              FILE *fp, int64_t step_idx) {
	    void *store_1_t =
	        (uint8_t *)store_1 +
	        boundary_store1_offset(step_idx);
	    void *store_3_t =
        (uint8_t *)store_3 +
	        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);

    if (storage_mode_h != STORAGE_DEVICE) {
      storage_load_snapshot_gpu(
          store_1_t, store_3_t, fp, storage_mode_h, step_idx,
          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
    }

	    if (storage_bf16_h) {
        if (boundary_dense_ok) {
	        scatter_boundary_dense_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
	            field, (__nv_bfloat16 const *)store_1_t, boundary_numel_h);
        } else {
	        scatter_boundary_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
	            field, (__nv_bfloat16 const *)store_1_t, boundary_indices,
	            boundary_numel_h);
        }
	    } else {
        if (boundary_dense_ok) {
	        scatter_boundary_dense<<<dimGrid_boundary, dimBlock_boundary>>>(
	            field, (TIDE_DTYPE const *)store_1_t, boundary_numel_h);
        } else {
	        scatter_boundary<<<dimGrid_boundary, dimBlock_boundary>>>(
	            field, (TIDE_DTYPE const *)store_1_t, boundary_indices, boundary_numel_h);
        }
	    }
	  };

		  auto load_and_scatter_h = [&](TIDE_DTYPE *hx_field, TIDE_DTYPE *hz_field,
		                                void *hx_store_1, void *hx_store_3, FILE *hx_fp,
		                                void *hz_store_1, void *hz_store_3, FILE *hz_fp,
		                                int64_t step_idx) {
		    void *hx_store_1_t =
		        (uint8_t *)hx_store_1 +
		        boundary_store1_offset(step_idx);
		    void *hx_store_3_t =
		        (uint8_t *)hx_store_3 +
		        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);

		    void *hz_store_1_t =
		        (uint8_t *)hz_store_1 +
		        boundary_store1_offset(step_idx);
		    void *hz_store_3_t =
		        (uint8_t *)hz_store_3 +
		        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);

	    if (storage_mode_h != STORAGE_DEVICE) {
	      storage_load_snapshot_gpu(
	          hx_store_1_t, hx_store_3_t, hx_fp, storage_mode_h, step_idx,
	          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
	      storage_load_snapshot_gpu(
	          hz_store_1_t, hz_store_3_t, hz_fp, storage_mode_h, step_idx,
	          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
	    }

	    if (storage_bf16_h) {
        if (boundary_dense_ok) {
	        scatter_boundary_2_dense_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
	            hx_field, hz_field,
	            (__nv_bfloat16 const *)hx_store_1_t,
	            (__nv_bfloat16 const *)hz_store_1_t,
	            boundary_numel_h);
        } else {
	        scatter_boundary_2_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
	            hx_field, hz_field,
	            (__nv_bfloat16 const *)hx_store_1_t,
	            (__nv_bfloat16 const *)hz_store_1_t,
	            boundary_indices, boundary_numel_h);
        }
	    } else {
        if (boundary_dense_ok) {
	        scatter_boundary_2_dense<<<dimGrid_boundary, dimBlock_boundary>>>(
	            hx_field, hz_field,
	            (TIDE_DTYPE const *)hx_store_1_t,
	            (TIDE_DTYPE const *)hz_store_1_t,
	            boundary_numel_h);
        } else {
	        scatter_boundary_2<<<dimGrid_boundary, dimBlock_boundary>>>(
	            hx_field, hz_field,
	            (TIDE_DTYPE const *)hx_store_1_t,
	            (TIDE_DTYPE const *)hz_store_1_t,
	            boundary_indices, boundary_numel_h);
        }
	    }
	  };

  // Time reversed loop: reconstruct forward fields from boundary ring and
  // propagate adjoint fields to accumulate gradients.
  for (int64_t t = nt - 1; t >= 0; --t) {
    if (n_sources_per_shot_h > 0) {
      subtract_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    // Invert E update (interior only) and compute curl(H) at time t.
    inverse_kernel_e_and_curl<<<dimGrid, dimBlock>>>(
        ca, cb, hx, hz, ey, curl_h);

    // Overwrite boundary Ey values at time t for correct inverse H update.
    load_and_scatter(ey, boundary_ey_store_1, boundary_ey_store_3, fp_bey, t);

    // Inject adjoint source (receiver residual) at receiver locations.
    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
    }

    // Record adjoint field at source locations for source gradient.
    if (n_sources_per_shot_h > 0) {
      record_adjoint_at_sources<<<dimGrid_sources, dimBlock_sources>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h, lambda_ey, sources_i);
    }

    // Backward λ_H fields update.
    backward_kernel_lambda_h<<<dimGrid, dimBlock>>>(
        cb, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x, m_lambda_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);

    // Backward λ_Ey update with per-shot gradient accumulation using reconstructed
    // Ey^t and curl(H)^t (both in full precision).
    backward_kernel_lambda_e_with_grad<<<dimGrid, dimBlock>>>(
        ca, cq, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z, m_lambda_hz_x,
        ca_requires_grad ? (TIDE_DTYPE const *)ey : nullptr,
        cb_requires_grad ? (TIDE_DTYPE const *)curl_h : nullptr,
        grad_ca_shot, grad_cb_shot, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh,
        kx, kxh, ca_requires_grad, cb_requires_grad, 1);

	    // Invert H update (interior only).
	    inverse_kernel_h<<<dimGrid, dimBlock>>>(cq, ey, hx, hz);

	    // Overwrite boundary H values at time t (this is H^{t-1/2}).
	    load_and_scatter_h(
	        hx, hz,
	        boundary_hx_store_1, boundary_hx_store_3, fp_bhx,
	        boundary_hz_store_1, boundary_hz_store_3, fp_bhz,
	        t);
	  }

  if (fp_bey != nullptr) fclose(fp_bey);
  if (fp_bhx != nullptr) fclose(fp_bhx);
  if (fp_bhz != nullptr) fclose(fp_bhz);

  // Combine per-shot gradients (only if not batched - batched case keeps per-shot grads)
  dim3 dimBlock_combine(32, 32, 1);
  dim3 dimGrid_combine(
      (nx_h - 2 * FD_PAD + dimBlock_combine.x - 1) / dimBlock_combine.x,
      (ny_h - 2 * FD_PAD + dimBlock_combine.y - 1) / dimBlock_combine.y, 1);

  if (ca_requires_grad && !ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_ca, grad_ca_shot);
  }
  if (cb_requires_grad && !cb_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_cb, grad_cb_shot);
  }

  if ((grad_eps != nullptr || grad_sigma != nullptr) && (ca_requires_grad || cb_requires_grad)) {
    dim3 dimBlock_conv(32, 8, 1);
    dim3 dimGrid_conv(
        (nx_h + dimBlock_conv.x - 1) / dimBlock_conv.x,
        (ny_h + dimBlock_conv.y - 1) / dimBlock_conv.y,
        ca_batched_h ? n_shots_h : 1);
    convert_grad_ca_cb_to_eps_sigma<<<dimGrid_conv, dimBlock_conv>>>(
        ca, cb, grad_ca, grad_cb, grad_ca_shot, grad_cb_shot,
        grad_eps, grad_sigma, dt_h,
        ca_requires_grad, cb_requires_grad,
        ca_batched_h, cb_batched_h);
  }

  gpuErrchk(cudaPeekAtLastError());
}

// RWII backward: propagate composite wavefield w and form gradients from self-correlations.
extern "C" void FUNC(backward_rwii)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const f,
    TIDE_DTYPE const *const grad_r,
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const curl_h,
    void *const boundary_ey_store_1,
    void *const boundary_ey_store_3,
    char const *const *const boundary_ey_filenames,
    void *const boundary_hx_store_1,
    void *const boundary_hx_store_3,
    char const *const *const boundary_hx_filenames,
    void *const boundary_hz_store_1,
    void *const boundary_hz_store_3,
    char const *const *const boundary_hz_filenames,
    int64_t const *const boundary_indices,
    int64_t const boundary_numel_h,
    TIDE_DTYPE const *const u_src,
    TIDE_DTYPE const *const gamma_u_ey,
    TIDE_DTYPE const *const gamma_u_curl,
    TIDE_DTYPE *const grad_f,
    TIDE_DTYPE *const grad_ca,
    TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE *const grad_eps,
    TIDE_DTYPE *const grad_sigma,
    TIDE_DTYPE *const grad_ca_shot,
    TIDE_DTYPE *const grad_cb_shot,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const storage_mode_h,
    int64_t const shot_bytes_uncomp_h,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    TIDE_DTYPE const alpha_rwii_h,
    int64_t const device) {

  cudaSetDevice(device);
  (void)dt_h;

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h = (shot_bytes_uncomp_h == boundary_numel_h * 2);

  // Copy constants to device with caching to avoid redundant copies
  static TIDE_DTYPE cached_rdy7 = 0, cached_rdx7 = 0;
  static int64_t cached_n_shots7 = -1, cached_ny7 = -1, cached_nx7 = -1;
  static int64_t cached_shot_numel7 = -1, cached_n_sources_per_shot7 = -1, cached_n_receivers_per_shot7 = -1;
  static int64_t cached_pml_y07 = -1, cached_pml_y17 = -1;
  static int64_t cached_pml_x07 = -1, cached_pml_x17 = -1;
  static bool cached_ca_batched7 = false, cached_cb_batched7 = false, cached_cq_batched7 = false;
  static bool first_call7 = true;

  if (first_call7 || cached_rdy7 != rdy_h ||
      cached_rdx7 != rdx_h || cached_n_shots7 != n_shots_h ||
      cached_ny7 != ny_h || cached_nx7 != nx_h ||
      cached_shot_numel7 != shot_numel_h ||
      cached_n_sources_per_shot7 != n_sources_per_shot_h ||
      cached_n_receivers_per_shot7 != n_receivers_per_shot_h ||
      cached_pml_y07 != pml_y0_h || cached_pml_y17 != pml_y1_h ||
      cached_pml_x07 != pml_x0_h || cached_pml_x17 != pml_x1_h ||
      cached_ca_batched7 != ca_batched_h ||
      cached_cb_batched7 != cb_batched_h ||
      cached_cq_batched7 != cq_batched_h) {

    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));

    cached_rdy7 = rdy_h;
    cached_rdx7 = rdx_h;
    cached_n_shots7 = n_shots_h;
    cached_ny7 = ny_h;
    cached_nx7 = nx_h;
    cached_shot_numel7 = shot_numel_h;
    cached_n_sources_per_shot7 = n_sources_per_shot_h;
    cached_n_receivers_per_shot7 = n_receivers_per_shot_h;
    cached_pml_y07 = pml_y0_h;
    cached_pml_y17 = pml_y1_h;
    cached_pml_x07 = pml_x0_h;
    cached_pml_x17 = pml_x1_h;
    cached_ca_batched7 = ca_batched_h;
    cached_cb_batched7 = cb_batched_h;
    cached_cq_batched7 = cq_batched_h;
    first_call7 = false;
  }

  dim3 dimBlock(32, 8, 1);
  dim3 dimGrid((nx_h - 2 * FD_PAD + dimBlock.x - 1) / dimBlock.x,
               (ny_h - 2 * FD_PAD + dimBlock.y - 1) / dimBlock.y, n_shots_h);

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources(
      (n_sources_per_shot_h + dimBlock_sources.x - 1) / dimBlock_sources.x,
      n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers(
      (n_receivers_per_shot_h + dimBlock_receivers.x - 1) / dimBlock_receivers.x,
      n_shots_h, 1);

  dim3 dimBlock_boundary(256, 1, 1);
  dim3 dimGrid_boundary(
      (boundary_numel_h + dimBlock_boundary.x - 1) / dimBlock_boundary.x,
      n_shots_h, 1);

  int64_t const boundary_interior_ny_h = pml_y1_h - pml_y0_h;
  int64_t const boundary_denom_h = 2 * (nx_h + boundary_interior_ny_h);
  bool const boundary_dense_ok =
      (boundary_denom_h > 0) && (boundary_numel_h > 0) &&
      (boundary_numel_h % boundary_denom_h == 0) &&
      ((boundary_numel_h / boundary_denom_h) > 0) &&
      ((boundary_numel_h / boundary_denom_h) <= pml_y0_h) &&
      ((boundary_numel_h / boundary_denom_h) <= pml_x0_h) &&
      (pml_y1_h + (boundary_numel_h / boundary_denom_h) <= ny_h) &&
      (pml_x1_h + (boundary_numel_h / boundary_denom_h) <= nx_h);

  FILE *fp_bey = nullptr;
  FILE *fp_bhx = nullptr;
  FILE *fp_bhz = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    fp_bey = fopen(boundary_ey_filenames[0], "rb");
    fp_bhx = fopen(boundary_hx_filenames[0], "rb");
    fp_bhz = fopen(boundary_hz_filenames[0], "rb");
  }

  auto boundary_store1_offset = [&](int64_t step_idx) -> size_t {
    if (storage_mode_h == STORAGE_DEVICE) {
      return (size_t)step_idx * bytes_per_step_store;
    }
    if (storage_mode_h == STORAGE_CPU) {
      // CPU mode uses device staging; Python allocates a 2-buffer ping-pong tensor.
      return (size_t)(step_idx & 1) * bytes_per_step_store;
    }
    return 0;
  };

  auto load_and_scatter = [&](TIDE_DTYPE *__restrict const field,
                              void *store_1, void *store_3, FILE *fp,
                              int64_t step_idx) {
    void *store_1_t =
        (uint8_t *)store_1 +
        boundary_store1_offset(step_idx);
    void *store_3_t =
        (uint8_t *)store_3 +
        (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);

    if (storage_mode_h != STORAGE_DEVICE) {
      storage_load_snapshot_gpu(
          store_1_t, store_3_t, fp, storage_mode_h, step_idx,
          (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
    }

    if (storage_bf16_h) {
      if (boundary_dense_ok) {
        scatter_boundary_dense_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
            field, (__nv_bfloat16 const *)store_1_t, boundary_numel_h);
      } else {
        scatter_boundary_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
            field, (__nv_bfloat16 const *)store_1_t, boundary_indices, boundary_numel_h);
      }
    } else {
      if (boundary_dense_ok) {
        scatter_boundary_dense<<<dimGrid_boundary, dimBlock_boundary>>>(
            field, (TIDE_DTYPE const *)store_1_t, boundary_numel_h);
      } else {
        scatter_boundary<<<dimGrid_boundary, dimBlock_boundary>>>(
            field, (TIDE_DTYPE const *)store_1_t, boundary_indices, boundary_numel_h);
      }
    }
  };

  auto load_and_scatter_h =
      [&](TIDE_DTYPE *__restrict const hx_field,
          TIDE_DTYPE *__restrict const hz_field,
          void *hx_store_1, void *hx_store_3, FILE *hx_fp,
          void *hz_store_1, void *hz_store_3, FILE *hz_fp,
          int64_t step_idx) {

        void *hx_store_1_t =
            (uint8_t *)hx_store_1 +
            boundary_store1_offset(step_idx);
        void *hx_store_3_t =
            (uint8_t *)hx_store_3 +
            (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);

        void *hz_store_1_t =
            (uint8_t *)hz_store_1 +
            boundary_store1_offset(step_idx);
        void *hz_store_3_t =
            (uint8_t *)hz_store_3 +
            (storage_mode_h == STORAGE_CPU ? (size_t)step_idx * bytes_per_step_store : 0);

        if (storage_mode_h != STORAGE_DEVICE) {
          storage_load_snapshot_gpu(
              hx_store_1_t, hx_store_3_t, hx_fp, storage_mode_h, step_idx,
              (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
          storage_load_snapshot_gpu(
              hz_store_1_t, hz_store_3_t, hz_fp, storage_mode_h, step_idx,
              (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
        }

        if (storage_bf16_h) {
          if (boundary_dense_ok) {
            scatter_boundary_2_dense_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
                hx_field, hz_field,
                (__nv_bfloat16 const *)hx_store_1_t,
                (__nv_bfloat16 const *)hz_store_1_t,
                boundary_numel_h);
          } else {
            scatter_boundary_2_bf16<<<dimGrid_boundary, dimBlock_boundary>>>(
                hx_field, hz_field,
                (__nv_bfloat16 const *)hx_store_1_t,
                (__nv_bfloat16 const *)hz_store_1_t,
                boundary_indices, boundary_numel_h);
          }
        } else {
          if (boundary_dense_ok) {
            scatter_boundary_2_dense<<<dimGrid_boundary, dimBlock_boundary>>>(
                hx_field, hz_field,
                (TIDE_DTYPE const *)hx_store_1_t,
                (TIDE_DTYPE const *)hz_store_1_t,
                boundary_numel_h);
          } else {
            scatter_boundary_2<<<dimGrid_boundary, dimBlock_boundary>>>(
                hx_field, hz_field,
                (TIDE_DTYPE const *)hx_store_1_t,
                (TIDE_DTYPE const *)hz_store_1_t,
                boundary_indices, boundary_numel_h);
          }
        }
      };

  TIDE_DTYPE const inv_alpha_h = (TIDE_DTYPE)1 / alpha_rwii_h;
  TIDE_DTYPE const inv_2alpha_h = (TIDE_DTYPE)0.5 * inv_alpha_h;

  // Time reversed loop: reconstruct u part via boundary forcing and inject residual.
  for (int64_t t = nt - 1; t >= 0; --t) {
    if (n_sources_per_shot_h > 0) {
      subtract_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    inverse_kernel_e_and_curl_rwii<<<dimGrid, dimBlock>>>(
        ca, cb, hx, hz, ey, curl_h, grad_cb_shot, cb_requires_grad);

    load_and_scatter(ey, boundary_ey_store_1, boundary_ey_store_3, fp_bey, t);

    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey_scaled<<<dimGrid_receivers, dimBlock_receivers>>>(
          ey, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i, alpha_rwii_h);
    }

    if (n_sources_per_shot_h > 0 && grad_f != nullptr && u_src != nullptr) {
      rwii_record_grad_f<<<dimGrid_sources, dimBlock_sources>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h,
          ey,
          sources_i,
          u_src + t * n_shots_h * n_sources_per_shot_h,
          inv_alpha_h);
    }

    inverse_kernel_h_rwii<<<dimGrid, dimBlock>>>(cq, ey, hx, hz, grad_ca_shot, ca_requires_grad);

    load_and_scatter_h(
        hx, hz,
        boundary_hx_store_1, boundary_hx_store_3, fp_bhx,
        boundary_hz_store_1, boundary_hz_store_3, fp_bhz,
        t);
  }

  if (fp_bey != nullptr) fclose(fp_bey);
  if (fp_bhx != nullptr) fclose(fp_bhx);
  if (fp_bhz != nullptr) fclose(fp_bhz);

  if ((ca_requires_grad || cb_requires_grad) && gamma_u_ey != nullptr && gamma_u_curl != nullptr) {
    rwii_finalize_grads<<<dimGrid, dimBlock>>>(
        grad_ca_shot, grad_cb_shot,
        gamma_u_ey, gamma_u_curl,
        inv_2alpha_h, ca_requires_grad, cb_requires_grad);
  }

  dim3 dimBlock_combine(32, 32, 1);
  dim3 dimGrid_combine(
      (nx_h - 2 * FD_PAD + dimBlock_combine.x - 1) / dimBlock_combine.x,
      (ny_h - 2 * FD_PAD + dimBlock_combine.y - 1) / dimBlock_combine.y, 1);

  if (ca_requires_grad && !ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_ca, grad_ca_shot);
  }
  if (cb_requires_grad && !cb_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_cb, grad_cb_shot);
  }

  if ((grad_eps != nullptr || grad_sigma != nullptr) && (ca_requires_grad || cb_requires_grad)) {
    dim3 dimBlock_conv(32, 8, 1);
    dim3 dimGrid_conv(
        (nx_h + dimBlock_conv.x - 1) / dimBlock_conv.x,
        (ny_h + dimBlock_conv.y - 1) / dimBlock_conv.y,
        ca_batched_h ? n_shots_h : 1);
    convert_grad_ca_cb_to_eps_sigma<<<dimGrid_conv, dimBlock_conv>>>(
        ca, cb, grad_ca, grad_cb, grad_ca_shot, grad_cb_shot,
        grad_eps, grad_sigma, dt_h,
        ca_requires_grad, cb_requires_grad,
        ca_batched_h, cb_batched_h);
  }

  gpuErrchk(cudaPeekAtLastError());
}

// Backward propagation function 
extern "C" void FUNC(backward)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const grad_r,
    TIDE_DTYPE *const lambda_ey,
    TIDE_DTYPE *const lambda_hx,
    TIDE_DTYPE *const lambda_hz,
    TIDE_DTYPE *const m_lambda_ey_x,
    TIDE_DTYPE *const m_lambda_ey_z,
    TIDE_DTYPE *const m_lambda_hx_z,
    TIDE_DTYPE *const m_lambda_hz_x,
    void *const ey_store_1,
    void *const ey_store_3,
    char const *const *const ey_filenames,
    void *const curl_store_1,
    void *const curl_store_3,
    char const *const *const curl_filenames,
    TIDE_DTYPE *const grad_f,
    TIDE_DTYPE *const grad_ca,
    TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE *const grad_eps,
    TIDE_DTYPE *const grad_sigma,
    TIDE_DTYPE *const grad_ca_shot,    // [n_shots, ny, nx] - per-shot gradient workspace
    TIDE_DTYPE *const grad_cb_shot,    // [n_shots, ny, nx] - per-shot gradient workspace
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode_h,
    int64_t const shot_bytes_uncomp_h,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const device) {
  
  cudaSetDevice(device);
  (void)dt_h;

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h = (shot_bytes_uncomp_h == shot_numel_h * 2);

  // Copy constants to device with caching to avoid redundant copies
  static TIDE_DTYPE cached_rdy3 = 0, cached_rdx3 = 0;
  static int64_t cached_n_shots3 = -1, cached_ny3 = -1, cached_nx3 = -1;
  static int64_t cached_shot_numel3 = -1, cached_n_sources_per_shot3 = -1, cached_n_receivers_per_shot3 = -1;
  static int64_t cached_pml_y03 = -1, cached_pml_y13 = -1;
  static int64_t cached_pml_x03 = -1, cached_pml_x13 = -1;
  static bool cached_ca_batched3 = false, cached_cb_batched3 = false, cached_cq_batched3 = false;
  static bool first_call3 = true;
  
  if (first_call3 || cached_rdy3 != rdy_h || cached_rdx3 != rdx_h ||
      cached_n_shots3 != n_shots_h || cached_ny3 != ny_h || cached_nx3 != nx_h ||
      cached_shot_numel3 != shot_numel_h || cached_n_sources_per_shot3 != n_sources_per_shot_h ||
      cached_n_receivers_per_shot3 != n_receivers_per_shot_h ||
      cached_pml_y03 != pml_y0_h || cached_pml_y13 != pml_y1_h ||
      cached_pml_x03 != pml_x0_h || cached_pml_x13 != pml_x1_h ||
      cached_ca_batched3 != ca_batched_h || cached_cb_batched3 != cb_batched_h ||
      cached_cq_batched3 != cq_batched_h) {
    
    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));
    
    cached_rdy3 = rdy_h; cached_rdx3 = rdx_h;
    cached_n_shots3 = n_shots_h; cached_ny3 = ny_h; cached_nx3 = nx_h;
    cached_shot_numel3 = shot_numel_h; cached_n_sources_per_shot3 = n_sources_per_shot_h;
    cached_n_receivers_per_shot3 = n_receivers_per_shot_h;
    cached_pml_y03 = pml_y0_h; cached_pml_y13 = pml_y1_h;
    cached_pml_x03 = pml_x0_h; cached_pml_x13 = pml_x1_h;
    cached_ca_batched3 = ca_batched_h; cached_cb_batched3 = cb_batched_h;
    cached_cq_batched3 = cq_batched_h;
    first_call3 = false;
  }

  dim3 dimBlock(32, 8, 1);
  int64_t gridx = (nx_h - 2 * FD_PAD + 2 + dimBlock.x - 1) / dimBlock.x;
  int64_t gridy = (ny_h - 2 * FD_PAD + 2 + dimBlock.y - 1) / dimBlock.y;
  int64_t gridz = n_shots_h;
  dim3 dimGrid(gridx, gridy, gridz);

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources(
      (n_sources_per_shot_h + dimBlock_sources.x - 1) / dimBlock_sources.x,
      n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers(
      (n_receivers_per_shot_h + dimBlock_receivers.x - 1) / dimBlock_receivers.x,
      n_shots_h, 1);

  FILE *fp_ey = nullptr;
  FILE *fp_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad) fp_ey = fopen(ey_filenames[0], "rb");
    if (cb_requires_grad) fp_curl = fopen(curl_filenames[0], "rb");
  }

  // Time reversed loop
  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    // Inject adjoint source (receiver residual) at receiver locations
    // Use add_adjoint_sources_ey which checks n_receivers_per_shot
    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
    }

    // Record adjoint field at source locations for source gradient
    // Use record_adjoint_at_sources which checks n_sources_per_shot
    if (n_sources_per_shot_h > 0) {
      record_adjoint_at_sources<<<dimGrid_sources, dimBlock_sources>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h,
          lambda_ey, sources_i);
    }

    int64_t const store_idx = t / step_ratio_h;
    bool const do_grad = (t % step_ratio_h) == 0;
    bool const grad_ey = do_grad && ca_requires_grad;
    bool const grad_curl = do_grad && cb_requires_grad;

    void const *__restrict const ey_store_1_t =
        (uint8_t const *)ey_store_1 +
        (storage_mode_h == STORAGE_DEVICE ? (size_t)store_idx * bytes_per_step_store : 0);
    void const *__restrict const ey_store_3_t =
        (uint8_t const *)ey_store_3 +
        (storage_mode_h == STORAGE_CPU
             ? (size_t)store_idx * bytes_per_step_store
             : 0);

    void const *__restrict const curl_store_1_t =
        (uint8_t const *)curl_store_1 +
        (storage_mode_h == STORAGE_DEVICE ? (size_t)store_idx * bytes_per_step_store : 0);
    void const *__restrict const curl_store_3_t =
        (uint8_t const *)curl_store_3 +
        (storage_mode_h == STORAGE_CPU
             ? (size_t)store_idx * bytes_per_step_store
             : 0);

    if (grad_ey && storage_mode_h != STORAGE_DEVICE) {
      storage_load_snapshot_gpu(
          (void *)ey_store_1_t, (void *)ey_store_3_t, fp_ey, storage_mode_h,
          store_idx, (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
    }
    if (grad_curl && storage_mode_h != STORAGE_DEVICE) {
      storage_load_snapshot_gpu(
          (void *)curl_store_1_t, (void *)curl_store_3_t, fp_curl, storage_mode_h,
          store_idx, (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
    }

    // Backward λ_H fields update
    backward_kernel_lambda_h<<<dimGrid, dimBlock>>>(
        cb, lambda_ey, lambda_hx, lambda_hz,
        m_lambda_ey_x, m_lambda_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    // Backward λ_Ey update (specialized kernel when no gradient is needed).
    if (grad_ey || grad_curl) {
      if (storage_bf16_h) {
        backward_kernel_lambda_e_with_grad_bf16<<<dimGrid, dimBlock>>>(
            ca, cq, lambda_hx, lambda_hz, lambda_ey,
            m_lambda_hx_z, m_lambda_hz_x,
            grad_ey ? (__nv_bfloat16 const *)ey_store_1_t : nullptr,
            grad_curl ? (__nv_bfloat16 const *)curl_store_1_t : nullptr,
            grad_ca_shot, grad_cb_shot,
            ay, ayh, ax, axh, by, byh, bx, bxh,
            ky, kyh, kx, kxh,
            grad_ey, grad_curl,
            step_ratio_h);
      } else {
        backward_kernel_lambda_e_with_grad<<<dimGrid, dimBlock>>>(
            ca, cq, lambda_hx, lambda_hz, lambda_ey,
            m_lambda_hx_z, m_lambda_hz_x,
            grad_ey ? (TIDE_DTYPE const *)ey_store_1_t : nullptr,
            grad_curl ? (TIDE_DTYPE const *)curl_store_1_t : nullptr,
            grad_ca_shot, grad_cb_shot,
            ay, ayh, ax, axh, by, byh, bx, bxh,
            ky, kyh, kx, kxh,
            grad_ey, grad_curl,
            step_ratio_h);
      }
    } else {
      backward_kernel_lambda_e<<<dimGrid, dimBlock>>>(
          ca, cq, lambda_hx, lambda_hz, lambda_ey,
          m_lambda_hx_z, m_lambda_hz_x,
          ay, ayh, ax, axh, by, byh, bx, bxh,
          ky, kyh, kx, kxh);
    }
  }

  if (fp_ey != nullptr) fclose(fp_ey);
  if (fp_curl != nullptr) fclose(fp_curl);

  // Combine per-shot gradients (only if not batched - batched case keeps per-shot grads)
  dim3 dimBlock_combine(32, 32, 1);
  dim3 dimGrid_combine(
      (nx_h - 2 * FD_PAD + dimBlock_combine.x - 1) / dimBlock_combine.x,
      (ny_h - 2 * FD_PAD + dimBlock_combine.y - 1) / dimBlock_combine.y, 1);

  if (ca_requires_grad && !ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_ca, grad_ca_shot);
  }
  if (cb_requires_grad && !cb_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_cb, grad_cb_shot);
  }

  if ((grad_eps != nullptr || grad_sigma != nullptr) && (ca_requires_grad || cb_requires_grad)) {
    dim3 dimBlock_conv(32, 8, 1);
    dim3 dimGrid_conv(
        (nx_h + dimBlock_conv.x - 1) / dimBlock_conv.x,
        (ny_h + dimBlock_conv.y - 1) / dimBlock_conv.y,
        ca_batched_h ? n_shots_h : 1);
    convert_grad_ca_cb_to_eps_sigma<<<dimGrid_conv, dimBlock_conv>>>(
        ca, cb, grad_ca, grad_cb, grad_ca_shot, grad_cb_shot,
        grad_eps, grad_sigma, dt_h,
        ca_requires_grad, cb_requires_grad,
        ca_batched_h, cb_batched_h);
  }

  gpuErrchk(cudaPeekAtLastError());
}
