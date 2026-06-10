"""
Reconstruction methods for TV-regularised super-resolution.

Methods implemented
-------------------
1. LeastSquaresReconstruction  — gradient descent on the data-fidelity term
2. PrimalDualTVReconstruction  — Chambolle–Pock PDHG (Algorithm 1)

Reference for PDHG:
    Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm
    for convex problems with applications to imaging.
    Journal of Mathematical Imaging and Vision, 40(1), 120-145.
"""

import time
import numpy as np
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.forward_model import ForwardModel
from src.gradient_operators import GradientOperators


# ============================================================================
# Baseline: Least Squares
# ============================================================================

class LeastSquaresReconstruction:
    """
    Gradient descent on the data-fidelity term.

    Minimises:  min_x  ½ ‖MHx - y‖²

    No regularisation.  Used as the baseline for comparison.

    Parameters
    ----------
    forward_model : ForwardModel
    step_size : float
        Gradient descent step size.  Typical range: 0.005 – 0.05.
    """

    def __init__(self, forward_model: ForwardModel, step_size: float = 0.01):
        self.forward = forward_model
        self.step_size = step_size

    def reconstruct(self,
                    y: np.ndarray,
                    original_shape: tuple,
                    num_iters: int = 300,
                    verbose: bool = False) -> np.ndarray:
        """
        Run gradient descent reconstruction.

        Parameters
        ----------
        y             : low-resolution measurement
        original_shape: (H, W) of the target high-resolution image
        num_iters     : number of gradient steps
        verbose       : print loss every 100 iterations

        Returns
        -------
        x : reconstructed image, values in [0, 1]
        """
        x = resize(y, original_shape, order=3, anti_aliasing=False)
        x = np.clip(x, 0.0, 1.0)

        for i in range(num_iters):
            # gradient of ½‖MHx - y‖²  =  A^T(Ax - y)
            residual = self.forward.forward(x) - y
            grad = self.forward.adjoint(residual, original_shape)
            x = np.clip(x - self.step_size * grad, 0.0, 1.0)

            if verbose and i % 100 == 0:
                loss = 0.5 * np.mean(residual ** 2)
                print(f"  LS iter {i:4d}  |  data loss = {loss:.6f}")

        return x


# ============================================================================
# TV regularisation: Primal-Dual Hybrid Gradient (PDHG)
# ============================================================================

class PrimalDualTVReconstruction:
    """
    Primal-Dual Hybrid Gradient algorithm for TV-regularised super-resolution.

    Solves:  min_x  ½ ‖MHx - y‖²  +  λ · TV(x)

    Algorithm (Chambolle & Pock 2011, Algorithm 1)
    -----------------------------------------------
    Initialise x⁰, x̄⁰ = x⁰,  p⁰ = q⁰ = 0

    for k = 0, 1, 2, …:
        # dual ascent + projection onto ‖·‖₂ ≤ λ
        (p, q) ← project( (p, q) + σ · ∇x̄,  ball(λ) )

        # primal descent
        x⁺ ← clip( x − τ · ( A^T(Ax − y) − div(p, q) ),  [0, 1] )

        # over-relaxation / extrapolation
        x̄  ← x⁺ + θ · (x⁺ − x)
        x  ← x⁺

    Convergence condition:  τ · σ ≤ 1 / ‖∇‖²
    For 2-D forward differences:  ‖∇‖² ≤ 8  →  τσ ≤ 1/8 = 0.125

    Parameters
    ----------
    forward_model : ForwardModel
    lambda_tv     : float   TV regularisation weight (default 0.001)
    tau           : float   primal step size          (default 0.1)
    sigma         : float   dual   step size          (default 0.1)
    theta         : float   extrapolation parameter   (default 1.0)
    """

    def __init__(self,
                 forward_model: ForwardModel,
                 lambda_tv: float = 0.001,
                 tau: float = 0.1,
                 sigma: float = 0.1,
                 theta: float = 1.0):
        self.forward = forward_model
        self.lambda_tv = lambda_tv
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        self.grad_ops = GradientOperators()
        self._check_step_sizes()

    def _check_step_sizes(self):
        product = self.tau * self.sigma
        if product >= 0.125:
            print(f"[Warning] τσ = {product:.4f} ≥ 0.125  "
                  f"— convergence is not guaranteed.  "
                  f"Reduce τ or σ.")

    def set_ground_truth(self, gt: np.ndarray):
        """Provide ground truth to track PSNR / SSIM during optimisation."""
        self._gt = gt

    def reconstruct(self,
                    y: np.ndarray,
                    original_shape: tuple,
                    num_iters: int = 500,
                    verbose: bool = False) -> tuple:
        """
        Run PDHG reconstruction.

        Parameters
        ----------
        y             : low-resolution measurement
        original_shape: (H, W) of the target high-resolution image
        num_iters     : maximum number of iterations
        verbose       : print progress table

        Returns
        -------
        x       : reconstructed image, values in [0, 1]
        history : dict with keys loss, data_term, tv_norm, psnr, ssim
        """
        # ---- Initialisation ------------------------------------------------
        x     = np.clip(resize(y, original_shape, order=3, anti_aliasing=False),
                        0.0, 1.0)
        x_bar = x.copy()
        p     = np.zeros_like(x)   # horizontal dual variable
        q     = np.zeros_like(x)   # vertical   dual variable

        history = dict(loss=[], data_term=[], tv_norm=[], psnr=[], ssim=[])
        has_gt  = hasattr(self, '_gt')

        if verbose:
            print("\n" + "=" * 72)
            print("  PRIMAL-DUAL TV RECONSTRUCTION")
            print("=" * 72)
            print(f"  λ={self.lambda_tv}  τ={self.tau}  σ={self.sigma}  θ={self.theta}")
            print(f"  {'iter':<6} {'loss':>12} {'data':>12} {'TV':>12} {'PSNR':>8}")
            print("-" * 72)

        t0 = time.time()

        # ---- Main loop -----------------------------------------------------
        for k in range(num_iters):

            # -- Dual update (gradient ascent + projection) --
            gx, gy   = self.grad_ops.gradient(x_bar)
            p_new    = p + self.sigma * gx
            q_new    = q + self.sigma * gy
            norm_pq  = self.grad_ops.gradient_norm(p_new, q_new)
            proj     = np.maximum(1.0, norm_pq / self.lambda_tv)
            p_new   /= proj
            q_new   /= proj

            # -- Primal update (gradient descent + box constraint) --
            div_pq   = self.grad_ops.divergence(p_new, q_new)
            Ax       = self.forward.forward(x)
            data_grad = self.forward.adjoint(Ax - y, original_shape)
            x_new    = np.clip(x - self.tau * (data_grad - div_pq), 0.0, 1.0)

            # -- Extrapolation --
            x_bar = x_new + self.theta * (x_new - x)

            # -- Update state --
            p, q, x = p_new, q_new, x_new

            # -- Track metrics --
            data_term  = float(0.5 * np.mean((self.forward.forward(x) - y) ** 2))
            tv         = self.grad_ops.tv_norm(x)
            n_pixels   = x.shape[0] * x.shape[1]
            loss       = data_term + self.lambda_tv * tv / n_pixels

            history['loss'].append(loss)
            history['data_term'].append(data_term)
            history['tv_norm'].append(tv)

            if has_gt:
                psnr = peak_signal_noise_ratio(self._gt, x, data_range=1.0)
                ssim = structural_similarity(self._gt, x, data_range=1.0)
            else:
                psnr = ssim = 0.0

            history['psnr'].append(psnr)
            history['ssim'].append(ssim)

            if verbose and (k % 50 == 0 or k == num_iters - 1):
                print(f"  {k:<6} {loss:>12.4e} {data_term:>12.4e} "
                      f"{self.lambda_tv*tv/n_pixels:>12.4e} {psnr:>8.2f}")

            # -- Early stopping --
            if k > 100:
                recent = history['loss'][-100:]
                if (max(recent) - min(recent)) < 1e-6 * min(recent):
                    if verbose:
                        print(f"\n  Converged at iteration {k}.")
                    break

        if verbose:
            print("=" * 72)
            print(f"  Elapsed: {time.time()-t0:.1f}s  |  "
                  f"Final PSNR: {history['psnr'][-1]:.2f} dB  |  "
                  f"SSIM: {history['ssim'][-1]:.4f}")

        return x, history
