"""
Hyperparameter tuning for both reconstruction methods.

Least Squares
-------------
  - step_size     : gradient descent step (too large → diverge, too small → slow)
  - num_iters     : number of gradient steps

TV · Primal-Dual
----------------
  - lambda_tv     : THE key parameter — controls smoothness vs edge sharpness
  - num_iters     : convergence iterations (less critical; 300-500 usually enough)

All figures are saved as  {img_stem}_hp_*.png  so different images never clash.

Usage (from main.py with --tune flag):
    python main.py --tune
    python main.py --tune --image data/test042.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from src.forward_model import ForwardModel
from src.reconstruction import LeastSquaresReconstruction, PrimalDualTVReconstruction


# ============================================================================
# Shared helpers
# ============================================================================

def _metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    if reconstructed.shape != original.shape:
        reconstructed = resize(reconstructed, original.shape, anti_aliasing=False)
    gt  = original.flatten()
    rec = reconstructed.flatten()
    return {
        'MSE':  float(mean_squared_error(gt, rec)),
        'PSNR': float(peak_signal_noise_ratio(original, reconstructed, data_range=1.0)),
        'SSIM': float(structural_similarity(original, reconstructed, data_range=1.0)),
    }


def _overlay(ax, image: np.ndarray, title: str,
             metrics: dict = None, show_metrics: bool = True,
             title_color: str = 'black'):
    """Greyscale image with optional yellow metric overlay (top-right)."""
    ax.imshow(image, cmap='gray')
    ax.set_title(title, fontsize=8, fontweight='bold', pad=4, color=title_color)
    ax.axis('off')
    if show_metrics and metrics:
        text = (f"MSE:  {metrics['MSE']:.6f}\n"
                f"PSNR: {metrics['PSNR']:.2f} dB\n"
                f"SSIM: {metrics['SSIM']:.4f}")
        ax.text(0.98, 0.98, text,
                transform=ax.transAxes,
                fontsize=6.5, color='yellow', fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          alpha=0.75, edgecolor='yellow', linewidth=0.8))


def _effect_chart(param_vals, psnr_vals, ssim_vals, xlabel,
                  best_idx, save_path, is_log=False):
    """Two-panel PSNR + SSIM vs hyperparameter, with optimal marked."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, ylabel, color in zip(
            axes,
            [psnr_vals, ssim_vals],
            ['PSNR (dB)', 'SSIM'],
            ['#185FA5', '#0F6E56']):
        plot_fn = ax.semilogx if is_log else ax.plot
        plot_fn(param_vals, vals, 'o-', linewidth=2, markersize=6, color=color)
        ax.plot(param_vals[best_idx], vals[best_idx],
                'r*', markersize=14, zorder=5,
                label=f'Best: {param_vals[best_idx]}')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} vs {xlabel}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


def _print_section(title: str):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ============================================================================
# Least Squares tuning
# ============================================================================

def tune_least_squares(img: np.ndarray,
                       forward: ForwardModel,
                       y: np.ndarray,
                       img_stem: str = 'image',
                       save_dir: str = 'outputs'):
    """
    Sweep step_size (fixed iters=300) then num_iters (fixed step=0.01).

    Each sweep produces:
      - 3-row image grid : GT | reconstruction | error map
      - PSNR + SSIM vs parameter chart
    """
    _print_section("LEAST SQUARES — step_size sweep  (fixed iters = 300)")

    step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    n = len(step_sizes)

    # ---- Step size grid ----------------------------------------------------
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)

    step_psnr, step_ssim, step_results = [], [], []

    for idx, step in enumerate(step_sizes):
        print(f"  step_size={step} …", end=' ', flush=True)
        ls = LeastSquaresReconstruction(forward, step_size=step)
        x  = ls.reconstruct(y, img.shape, num_iters=300)
        m  = _metrics(img, x)
        step_results.append(m)
        step_psnr.append(m['PSNR'])
        step_ssim.append(m['SSIM'])
        print(f"PSNR={m['PSNR']:.2f} dB")

        # Row 0 — GT (first col only; others blank)
        if idx == 0:
            _overlay(axes[0, 0], img, 'GROUND TRUTH\n(Reference)',
                     show_metrics=False, title_color='green')
        else:
            axes[0, idx].axis('off')

        # Row 1 — reconstruction
        _overlay(axes[1, idx], x, f'LS  step={step}', m)

        # Row 2 — error map
        err = np.abs(img - x)
        axes[2, idx].imshow(err, cmap='hot', vmin=0, vmax=0.15)
        axes[2, idx].set_title(f'Error map\nMAE: {np.mean(err):.4f}', fontsize=8)
        axes[2, idx].axis('off')

    plt.suptitle('Least Squares — step_size tuning  (iters = 300)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(save_dir, f'{img_stem}_hp_ls_step_grid.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.show(); plt.close()

    best_step_idx = int(np.argmax(step_psnr))
    _effect_chart(step_sizes, step_psnr, step_ssim,
                  xlabel='Step size',
                  best_idx=best_step_idx,
                  save_path=os.path.join(save_dir, f'{img_stem}_hp_ls_step_chart.png'))

    # ---- Iterations grid ---------------------------------------------------
    _print_section("LEAST SQUARES — num_iters sweep  (fixed step_size = 0.01)")

    iterations = [50, 100, 200, 300, 500, 800]
    n = len(iterations)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)

    iter_psnr, iter_ssim, iter_results = [], [], []

    for idx, n_iter in enumerate(iterations):
        print(f"  iters={n_iter} …", end=' ', flush=True)
        ls = LeastSquaresReconstruction(forward, step_size=0.01)
        x  = ls.reconstruct(y, img.shape, num_iters=n_iter)
        m  = _metrics(img, x)
        iter_results.append(m)
        iter_psnr.append(m['PSNR'])
        iter_ssim.append(m['SSIM'])
        print(f"PSNR={m['PSNR']:.2f} dB")

        if idx == 0:
            _overlay(axes[0, 0], img, 'GROUND TRUTH\n(Reference)',
                     show_metrics=False, title_color='green')
        else:
            axes[0, idx].axis('off')

        _overlay(axes[1, idx], x, f'LS  iters={n_iter}', m)

        err = np.abs(img - x)
        axes[2, idx].imshow(err, cmap='hot', vmin=0, vmax=0.15)
        axes[2, idx].set_title(f'Error map\nMAE: {np.mean(err):.4f}', fontsize=8)
        axes[2, idx].axis('off')

    plt.suptitle('Least Squares — num_iters tuning  (step_size = 0.01)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(save_dir, f'{img_stem}_hp_ls_iters_grid.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.show(); plt.close()

    best_iter_idx = int(np.argmax(iter_psnr))
    _effect_chart(iterations, iter_psnr, iter_ssim,
                  xlabel='Num iterations',
                  best_idx=best_iter_idx,
                  save_path=os.path.join(save_dir, f'{img_stem}_hp_ls_iters_chart.png'))

    best_step = step_sizes[best_step_idx]
    best_iter = iterations[best_iter_idx]
    print(f"\n  ✓ Best LS step_size  = {best_step}  "
          f"(PSNR {step_psnr[best_step_idx]:.2f} dB)")
    print(f"  ✓ Best LS num_iters  = {best_iter}  "
          f"(PSNR {iter_psnr[best_iter_idx]:.2f} dB)")

    return dict(best_step=best_step, best_iters=best_iter,
                step_results=step_results, iter_results=iter_results)


# ============================================================================
# TV · Primal-Dual tuning
# ============================================================================

def tune_tv(img: np.ndarray,
            forward: ForwardModel,
            y: np.ndarray,
            img_stem: str = 'image',
            save_dir: str = 'outputs'):
    """
    Sweep lambda_tv (fixed iters=500) then num_iters (fixed lambda=0.001).

    lambda_tv is the most important parameter:
      too small → under-regularised, noise remains
      too large  → over-smoothed, edges lost
    """
    _print_section("TV · PRIMAL-DUAL — lambda_tv sweep  (fixed iters = 500)")

    lambda_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    n = len(lambda_values)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)

    lam_psnr, lam_ssim, lam_results = [], [], []

    for idx, lam in enumerate(lambda_values):
        print(f"  λ={lam} …", end=' ', flush=True)
        tv    = PrimalDualTVReconstruction(forward, lambda_tv=lam)
        x, _  = tv.reconstruct(y, img.shape, num_iters=500)
        m     = _metrics(img, x)
        lam_results.append(m)
        lam_psnr.append(m['PSNR'])
        lam_ssim.append(m['SSIM'])
        print(f"PSNR={m['PSNR']:.2f} dB")

        if idx == 0:
            _overlay(axes[0, 0], img, 'GROUND TRUTH\n(Reference)',
                     show_metrics=False, title_color='green')
        else:
            axes[0, idx].axis('off')

        _overlay(axes[1, idx], x, f'TV  λ={lam}', m)

        err = np.abs(img - x)
        axes[2, idx].imshow(err, cmap='hot', vmin=0, vmax=0.15)
        axes[2, idx].set_title(f'Error map\nMAE: {np.mean(err):.4f}', fontsize=8)
        axes[2, idx].axis('off')

    plt.suptitle('TV · Primal-Dual — lambda_tv tuning  (iters = 500)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(save_dir, f'{img_stem}_hp_tv_lambda_grid.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.show(); plt.close()

    best_lam_idx = int(np.argmax(lam_psnr))
    _effect_chart(lambda_values, lam_psnr, lam_ssim,
                  xlabel='lambda_tv',
                  best_idx=best_lam_idx,
                  is_log=True,
                  save_path=os.path.join(save_dir, f'{img_stem}_hp_tv_lambda_chart.png'))

    # ---- Iterations grid ---------------------------------------------------
    _print_section("TV · PRIMAL-DUAL — num_iters sweep  (fixed λ = 0.001)")

    iterations = [50, 100, 200, 300, 500, 800, 1000, 1500]
    n = len(iterations)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)

    iter_psnr, iter_ssim, iter_results = [], [], []

    for idx, n_iter in enumerate(iterations):
        print(f"  iters={n_iter} …", end=' ', flush=True)
        tv    = PrimalDualTVReconstruction(forward, lambda_tv=0.001)
        x, _  = tv.reconstruct(y, img.shape, num_iters=n_iter)
        m     = _metrics(img, x)
        iter_results.append(m)
        iter_psnr.append(m['PSNR'])
        iter_ssim.append(m['SSIM'])
        print(f"PSNR={m['PSNR']:.2f} dB")

        if idx == 0:
            _overlay(axes[0, 0], img, 'GROUND TRUTH\n(Reference)',
                     show_metrics=False, title_color='green')
        else:
            axes[0, idx].axis('off')

        _overlay(axes[1, idx], x, f'TV  iters={n_iter}', m)

        err = np.abs(img - x)
        axes[2, idx].imshow(err, cmap='hot', vmin=0, vmax=0.15)
        axes[2, idx].set_title(f'Error map\nMAE: {np.mean(err):.4f}', fontsize=8)
        axes[2, idx].axis('off')

    plt.suptitle('TV · Primal-Dual — num_iters tuning  (λ = 0.001)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = os.path.join(save_dir, f'{img_stem}_hp_tv_iters_grid.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.show(); plt.close()

    best_iter_idx = int(np.argmax(iter_psnr))
    _effect_chart(iterations, iter_psnr, iter_ssim,
                  xlabel='Num iterations',
                  best_idx=best_iter_idx,
                  save_path=os.path.join(save_dir, f'{img_stem}_hp_tv_iters_chart.png'))

    best_lam  = lambda_values[best_lam_idx]
    best_iter = iterations[best_iter_idx]
    print(f"\n  ✓ Best TV lambda_tv  = {best_lam}  "
          f"(PSNR {lam_psnr[best_lam_idx]:.2f} dB)")
    print(f"  ✓ Best TV num_iters  = {best_iter}  "
          f"(PSNR {iter_psnr[best_iter_idx]:.2f} dB)")

    return dict(best_lambda=best_lam, best_iters=best_iter,
                lam_results=lam_results, iter_results=iter_results)


# ============================================================================
# Summary: default vs optimal side-by-side
# ============================================================================

def tuning_summary(img: np.ndarray,
                   forward: ForwardModel,
                   y: np.ndarray,
                   ls_best: dict,
                   tv_best: dict,
                   img_stem: str = 'image',
                   save_dir: str = 'outputs'):
    """
    2×4 grid:
      Row 0 — GT | Measurement | LS default | LS optimal
      Row 1 — GT | Measurement | TV default | TV optimal
    """
    _print_section("SUMMARY — default vs optimal hyperparameters")

    y_up = resize(y, img.shape, order=3, anti_aliasing=False)

    print("  Reconstructing (4 variants) …")

    ls_def = LeastSquaresReconstruction(forward, step_size=0.01)
    x_ls_def = ls_def.reconstruct(y, img.shape, num_iters=300)

    ls_opt = LeastSquaresReconstruction(forward, step_size=ls_best['best_step'])
    x_ls_opt = ls_opt.reconstruct(y, img.shape, num_iters=ls_best['best_iters'])

    tv_def    = PrimalDualTVReconstruction(forward, lambda_tv=0.01)
    x_tv_def, _ = tv_def.reconstruct(y, img.shape, num_iters=500)

    tv_opt    = PrimalDualTVReconstruction(forward, lambda_tv=tv_best['best_lambda'])
    x_tv_opt, _ = tv_opt.reconstruct(y, img.shape, num_iters=tv_best['best_iters'])

    m_meas    = _metrics(img, y_up)
    m_ls_def  = _metrics(img, x_ls_def)
    m_ls_opt  = _metrics(img, x_ls_opt)
    m_tv_def  = _metrics(img, x_tv_def)
    m_tv_opt  = _metrics(img, x_tv_opt)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)

    # Row 0 — LS
    _overlay(axes[0, 0], img,    'GROUND TRUTH',          show_metrics=False, title_color='green')
    _overlay(axes[0, 1], y_up,   'Measurement\n(input)',  m_meas)
    _overlay(axes[0, 2], x_ls_def, f'LS default\nstep=0.01  iters=300', m_ls_def)
    _overlay(axes[0, 3], x_ls_opt,
             f'LS optimal\nstep={ls_best["best_step"]}  iters={ls_best["best_iters"]}',
             m_ls_opt)

    # Row 1 — TV
    _overlay(axes[1, 0], img,    'GROUND TRUTH',          show_metrics=False, title_color='green')
    _overlay(axes[1, 1], y_up,   'Measurement\n(input)',  m_meas)
    _overlay(axes[1, 2], x_tv_def, f'TV default\nλ=0.01  iters=500',    m_tv_def)
    _overlay(axes[1, 3], x_tv_opt,
             f'TV optimal\nλ={tv_best["best_lambda"]}  iters={tv_best["best_iters"]}',
             m_tv_opt)

    plt.suptitle('Hyperparameter tuning summary: default vs optimal',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(save_dir, f'{img_stem}_hp_summary.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.show(); plt.close()

    # Print improvement table
    print(f"\n  {'Method':<22} {'Config':<22} {'PSNR (dB)':>10} {'SSIM':>8}")
    print(f"  {'─'*65}")
    for label, cfg, m in [
        ('Least Squares', 'Default  (step=0.01, i=300)', m_ls_def),
        ('Least Squares', f'Optimal  (step={ls_best["best_step"]}, i={ls_best["best_iters"]})', m_ls_opt),
        ('TV · PD',       'Default  (λ=0.01, i=500)',   m_tv_def),
        ('TV · PD',       f'Optimal  (λ={tv_best["best_lambda"]}, i={tv_best["best_iters"]})', m_tv_opt),
    ]:
        print(f"  {label:<22} {cfg:<22} {m['PSNR']:>10.2f} {m['SSIM']:>8.4f}")

    ls_gain = m_ls_opt['PSNR'] - m_ls_def['PSNR']
    tv_gain = m_tv_opt['PSNR'] - m_tv_def['PSNR']
    print(f"\n  LS gain from tuning:  {ls_gain:+.2f} dB PSNR")
    print(f"  TV gain from tuning:  {tv_gain:+.2f} dB PSNR")


# ============================================================================
# Top-level entry point (called from main.py --tune)
# ============================================================================

def run_tuning(img: np.ndarray,
               blur_sigma: float = 1.0,
               factor: int = 2,
               noise_sigma: float = 0.01,
               img_stem: str = 'image',
               save_dir: str = 'outputs'):
    """
    Full tuning pipeline — called by main.py when --tune is passed.

    Generates one shared noisy measurement so both methods see the same y,
    then sweeps each hyperparameter independently.
    """
    print("\n" + "=" * 70)
    print("  HYPERPARAMETER TUNING")
    print(f"  Image: {img_stem}  |  blur σ={blur_sigma}  "
          f"downsample ×{factor}  noise σ={noise_sigma}")
    print("=" * 70)

    fwd = ForwardModel(blur_sigma, factor, noise_sigma)
    y   = fwd.forward_noisy(img)

    os.makedirs(save_dir, exist_ok=True)

    ls_best = tune_least_squares(img, fwd, y, img_stem=img_stem, save_dir=save_dir)
    tv_best = tune_tv(img, fwd, y, img_stem=img_stem, save_dir=save_dir)
    tuning_summary(img, fwd, y, ls_best, tv_best,
                   img_stem=img_stem, save_dir=save_dir)

    print("\n" + "=" * 70)
    print("  TUNING COMPLETE")
    print(f"  Recommended  LS : step_size={ls_best['best_step']}, "
          f"num_iters={ls_best['best_iters']}")
    print(f"  Recommended  TV : lambda_tv={tv_best['best_lambda']}, "
          f"num_iters={tv_best['best_iters']}")
    print("=" * 70)

    return ls_best, tv_best