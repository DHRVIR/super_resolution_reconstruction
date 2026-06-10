"""
Parameter sensitivity analysis.

Each of the three main sweeps (noise, blur, downsampling) now produces:
  1. A 4-row image grid per parameter value
       Row 0 — Ground truth (reference, no overlay)
       Row 1 — Measurement  (bicubic-upsampled, yellow metric overlay)
       Row 2 — Least Squares reconstruction  (yellow metric overlay)
       Row 3 — TV · Primal-Dual reconstruction  (yellow metric overlay)
  2. A PSNR/SSIM vs parameter line-chart

The lambda sweep produces a 3-row grid (GT / TV reconstruction / error map)
plus the PSNR/SSIM vs λ chart.

All files are saved as  {image_stem}_{analysis_type}.png  so different input
images never overwrite each other.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.metrics import (mean_squared_error,
                              peak_signal_noise_ratio,
                              structural_similarity)

from src.forward_model import ForwardModel
from src.reconstruction import LeastSquaresReconstruction, PrimalDualTVReconstruction


# ============================================================================
# Internal helpers
# ============================================================================

def _metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """MSE, PSNR, SSIM for a single (original, reconstructed) pair."""
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
             metrics: dict = None, show_metrics: bool = True):
    """
    Display a greyscale image with an optional yellow metric overlay in the
    top-right corner, matching the style from the reference implementation.
    """
    ax.imshow(image, cmap='gray')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.axis('off')

    if show_metrics and metrics is not None:
        text = (f"MSE:  {metrics['MSE']:.6f}\n"
                f"PSNR: {metrics['PSNR']:.2f} dB\n"
                f"SSIM: {metrics['SSIM']:.4f}")
        ax.text(0.98, 0.98, text,
                transform=ax.transAxes,
                fontsize=7, color='yellow', fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          alpha=0.75, edgecolor='yellow', linewidth=0.8))


def _run_pair(img, forward, lam=0.001, ls_iters=300, tv_iters=500):
    """
    Generate noisy measurement, reconstruct with both methods, return
    (y, x_ls, x_tv, m_meas, m_ls, m_tv).
    """
    y    = forward.forward_noisy(img)
    y_up = resize(y, img.shape, order=3, anti_aliasing=False)

    ls   = LeastSquaresReconstruction(forward)
    x_ls = ls.reconstruct(y, img.shape, num_iters=ls_iters)

    tv       = PrimalDualTVReconstruction(forward, lambda_tv=lam)
    x_tv, _  = tv.reconstruct(y, img.shape, num_iters=tv_iters)

    return (y_up, x_ls, x_tv,
            _metrics(img, y_up),
            _metrics(img, x_ls),
            _metrics(img, x_tv))


def _line_chart(param_vals, ls_psnr, tv_psnr, xlabel, save_path):
    """PSNR vs parameter, both methods on one axis."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(param_vals, ls_psnr, 'o--', linewidth=2, markersize=6,
            color='#BA7517', label='Least Squares')
    ax.plot(param_vals, tv_psnr, 's-',  linewidth=2, markersize=6,
            color='#185FA5', label='TV · Primal-Dual')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title(f'PSNR vs {xlabel}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


def _print_table(rows, col1_header, col_width=12):
    sep = "-" * (12 + 4 * col_width + 10)
    print("\n" + sep)
    print(f"  {col1_header:<12} {'LS PSNR':>{col_width}} "
          f"{'LS SSIM':>{col_width}} {'TV PSNR':>{col_width}} "
          f"{'TV SSIM':>{col_width}}")
    print(sep)
    for r in rows:
        print(f"  {str(r[0]):<12} {r[1]:>{col_width}.2f} "
              f"{r[2]:>{col_width}.4f} {r[3]:>{col_width}.2f} "
              f"{r[4]:>{col_width}.4f}")
    print(sep)


# ============================================================================
# Noise sensitivity
# ============================================================================

def analyze_noise_sensitivity(img: np.ndarray,
                               blur_sigma: float = 1.0,
                               factor: int = 2,
                               noise_levels=None,
                               img_stem: str = 'image',
                               save_dir: str = 'outputs') -> list:
    """
    4-row image grid for each noise level  +  PSNR line chart.

    Parameters
    ----------
    img_stem : base name (no extension) of the source image,
               used as filename prefix so runs on different images don't clash.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]

    print("\n== Noise sensitivity analysis ==")
    n = len(noise_levels)

    # ---- 4-row image grid --------------------------------------------------
    fig, axes = plt.subplots(4, n, figsize=(3 * n, 12), squeeze=False)

    rows, ls_psnr, tv_psnr = [], [], []

    for idx, sigma_n in enumerate(noise_levels):
        print(f"  σ_n={sigma_n:.3f} …", end=' ', flush=True)

        fwd = ForwardModel(blur_sigma, factor, sigma_n)
        y_up, x_ls, x_tv, m_meas, m_ls, m_tv = _run_pair(img, fwd)

        rows.append((sigma_n, m_ls['PSNR'], m_ls['SSIM'],
                     m_tv['PSNR'], m_tv['SSIM']))
        ls_psnr.append(m_ls['PSNR'])
        tv_psnr.append(m_tv['PSNR'])

        print(f"LS={m_ls['PSNR']:.2f} dB  TV={m_tv['PSNR']:.2f} dB")

        col_title = f'σ_n = {sigma_n}'

        # Row 0 — ground truth (no overlay)
        _overlay(axes[0, idx], img,
                 'Ground truth' if idx == 0 else 'Reference',
                 show_metrics=False)
        if idx == 0:
            axes[0, idx].set_title('GROUND TRUTH\n(Reference)',
                                   fontsize=9, fontweight='bold', color='green', pad=4)

        # Row 1 — measurement
        _overlay(axes[1, idx], y_up,
                 f'Measurement\n{col_title}', m_meas)

        # Row 2 — least squares
        _overlay(axes[2, idx], x_ls,
                 f'Least Squares\n{col_title}', m_ls)

        # Row 3 — TV
        _overlay(axes[3, idx], x_tv,
                 f'TV (Primal-Dual)\n{col_title}', m_tv)

    # Row labels on left edge
    for row_idx, label in enumerate(
            ['Ground truth', 'Measurement', 'Least Squares', 'TV · Primal-Dual']):
        axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=90,
                                    labelpad=6, color='#333333')

    plt.suptitle(
        'Noise sensitivity: reconstruction quality vs noise level\n'
        'Metrics (MSE / PSNR / SSIM) overlaid in yellow on each image',
        fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    grid_path = os.path.join(save_dir, f'{img_stem}_noise_sensitivity_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {grid_path}")
    plt.show()
    plt.close()

    # ---- PSNR line chart ---------------------------------------------------
    _print_table(rows, 'Noise σ_n')
    chart_path = os.path.join(save_dir, f'{img_stem}_noise_sensitivity_chart.png')
    _line_chart(noise_levels, ls_psnr, tv_psnr,
                xlabel='Noise σ_n', save_path=chart_path)
    return rows


# ============================================================================
# Blur sensitivity
# ============================================================================

def analyze_blur_sensitivity(img: np.ndarray,
                              noise_sigma: float = 0.01,
                              factor: int = 2,
                              blur_levels=None,
                              img_stem: str = 'image',
                              save_dir: str = 'outputs') -> list:
    """4-row image grid for each blur level  +  PSNR line chart."""
    if blur_levels is None:
        blur_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print("\n== Blur sensitivity analysis ==")
    n = len(blur_levels)

    fig, axes = plt.subplots(4, n, figsize=(3 * n, 12), squeeze=False)

    rows, ls_psnr, tv_psnr = [], [], []

    for idx, sigma_b in enumerate(blur_levels):
        print(f"  σ_b={sigma_b:.1f} …", end=' ', flush=True)

        fwd = ForwardModel(sigma_b, factor, noise_sigma)
        y_up, x_ls, x_tv, m_meas, m_ls, m_tv = _run_pair(img, fwd)

        rows.append((sigma_b, m_ls['PSNR'], m_ls['SSIM'],
                     m_tv['PSNR'], m_tv['SSIM']))
        ls_psnr.append(m_ls['PSNR'])
        tv_psnr.append(m_tv['PSNR'])

        print(f"LS={m_ls['PSNR']:.2f} dB  TV={m_tv['PSNR']:.2f} dB")

        col_title = f'σ_b = {sigma_b}'

        _overlay(axes[0, idx], img,
                 'Ground truth' if idx == 0 else 'Reference',
                 show_metrics=False)
        if idx == 0:
            axes[0, idx].set_title('GROUND TRUTH\n(Reference)',
                                   fontsize=9, fontweight='bold', color='green', pad=4)

        _overlay(axes[1, idx], y_up,
                 f'Measurement\n{col_title}', m_meas)
        _overlay(axes[2, idx], x_ls,
                 f'Least Squares\n{col_title}', m_ls)
        _overlay(axes[3, idx], x_tv,
                 f'TV (Primal-Dual)\n{col_title}', m_tv)

    for row_idx, label in enumerate(
            ['Ground truth', 'Measurement', 'Least Squares', 'TV · Primal-Dual']):
        axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=90,
                                    labelpad=6, color='#333333')

    plt.suptitle(
        'Blur sensitivity: reconstruction quality vs blur strength\n'
        'Metrics (MSE / PSNR / SSIM) overlaid in yellow on each image',
        fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    grid_path = os.path.join(save_dir, f'{img_stem}_blur_sensitivity_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {grid_path}")
    plt.show()
    plt.close()

    _print_table(rows, 'Blur σ_b')
    chart_path = os.path.join(save_dir, f'{img_stem}_blur_sensitivity_chart.png')
    _line_chart(blur_levels, ls_psnr, tv_psnr,
                xlabel='Blur σ_b', save_path=chart_path)
    return rows


# ============================================================================
# Downsampling sensitivity
# ============================================================================

def analyze_downsampling_sensitivity(img: np.ndarray,
                                     blur_sigma: float = 1.0,
                                     noise_sigma: float = 0.01,
                                     factors=None,
                                     img_stem: str = 'image',
                                     save_dir: str = 'outputs') -> list:
    """4-row image grid for each downsampling factor  +  PSNR line chart."""
    if factors is None:
        factors = [1, 2, 3, 4]

    print("\n== Downsampling sensitivity analysis ==")
    n = len(factors)

    fig, axes = plt.subplots(4, n, figsize=(3 * n, 12), squeeze=False)

    rows, ls_psnr, tv_psnr = [], [], []

    for idx, s in enumerate(factors):
        print(f"  factor={s} …", end=' ', flush=True)

        fwd = ForwardModel(blur_sigma, s, noise_sigma)
        y_up, x_ls, x_tv, m_meas, m_ls, m_tv = _run_pair(img, fwd)

        rows.append((s, m_ls['PSNR'], m_ls['SSIM'],
                     m_tv['PSNR'], m_tv['SSIM']))
        ls_psnr.append(m_ls['PSNR'])
        tv_psnr.append(m_tv['PSNR'])

        print(f"LS={m_ls['PSNR']:.2f} dB  TV={m_tv['PSNR']:.2f} dB")

        col_title = f'{s}× down'

        _overlay(axes[0, idx], img,
                 'Ground truth' if idx == 0 else 'Reference',
                 show_metrics=False)
        if idx == 0:
            axes[0, idx].set_title('GROUND TRUTH\n(Reference)',
                                   fontsize=9, fontweight='bold', color='green', pad=4)

        _overlay(axes[1, idx], y_up,
                 f'Measurement\n{col_title}', m_meas)
        _overlay(axes[2, idx], x_ls,
                 f'Least Squares\n{col_title}', m_ls)
        _overlay(axes[3, idx], x_tv,
                 f'TV (Primal-Dual)\n{col_title}', m_tv)

    for row_idx, label in enumerate(
            ['Ground truth', 'Measurement', 'Least Squares', 'TV · Primal-Dual']):
        axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=90,
                                    labelpad=6, color='#333333')

    plt.suptitle(
        'Downsampling sensitivity: reconstruction quality vs downsampling factor\n'
        'Metrics (MSE / PSNR / SSIM) overlaid in yellow on each image',
        fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    grid_path = os.path.join(save_dir, f'{img_stem}_downsampling_sensitivity_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {grid_path}")
    plt.show()
    plt.close()

    _print_table(rows, 'Factor')
    chart_path = os.path.join(save_dir, f'{img_stem}_downsampling_sensitivity_chart.png')
    _line_chart(factors, ls_psnr, tv_psnr,
                xlabel='Downsampling factor', save_path=chart_path)
    return rows


# ============================================================================
# Lambda sensitivity  (TV-only — 3-row grid: GT / reconstruction / error map)
# ============================================================================

def analyze_lambda_sensitivity(img: np.ndarray,
                                blur_sigma: float = 1.0,
                                factor: int = 2,
                                noise_sigma: float = 0.01,
                                lambda_values=None,
                                img_stem: str = 'image',
                                save_dir: str = 'outputs') -> list:
    """
    3-row image grid (GT / TV reconstruction / error map) for each λ
    plus a PSNR/SSIM vs λ chart.
    """
    if lambda_values is None:
        lambda_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    print("\n== TV λ parameter sweep ==")
    fwd = ForwardModel(blur_sigma, factor, noise_sigma)
    y   = fwd.forward_noisy(img)
    n   = len(lambda_values)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)

    rows, psnr_vals, ssim_vals = [], [], []

    for idx, lam in enumerate(lambda_values):
        print(f"  λ={lam:.4f} …", end=' ', flush=True)

        tv       = PrimalDualTVReconstruction(fwd, lambda_tv=lam)
        x_tv, _  = tv.reconstruct(y, img.shape, num_iters=500)
        m        = _metrics(img, x_tv)

        rows.append((lam, m['PSNR'], m['SSIM']))
        psnr_vals.append(m['PSNR'])
        ssim_vals.append(m['SSIM'])
        print(f"PSNR={m['PSNR']:.2f} dB  SSIM={m['SSIM']:.4f}")

        # Row 0 — ground truth
        _overlay(axes[0, idx], img,
                 'Ground truth' if idx == 0 else 'Reference',
                 show_metrics=False)
        if idx == 0:
            axes[0, idx].set_title('GROUND TRUTH\n(Reference)',
                                   fontsize=9, fontweight='bold', color='green', pad=4)

        # Row 1 — TV reconstruction
        _overlay(axes[1, idx], x_tv,
                 f'TV reconstruction\nλ = {lam}', m)

        # Row 2 — absolute error map
        error = np.abs(img - x_tv)
        axes[2, idx].imshow(error, cmap='hot', vmin=0, vmax=0.15)
        axes[2, idx].set_title(f'Error map\nMSE: {m["MSE"]:.5f}', fontsize=8)
        axes[2, idx].axis('off')

    for row_idx, label in enumerate(
            ['Ground truth', 'TV reconstruction', 'Error map']):
        axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=90,
                                    labelpad=6, color='#333333')

    plt.suptitle(
        'TV regularisation parameter sweep: finding optimal λ\n'
        'Metrics (MSE / PSNR / SSIM) overlaid in yellow on each reconstruction',
        fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    grid_path = os.path.join(save_dir, f'{img_stem}_lambda_sensitivity_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {grid_path}")
    plt.show()
    plt.close()

    # ---- PSNR/SSIM vs λ chart ----------------------------------------------
    best_idx = int(np.argmax(psnr_vals))
    print(f"\n  Optimal λ = {lambda_values[best_idx]}  "
          f"→  PSNR = {psnr_vals[best_idx]:.2f} dB")

    fig, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, ylabel, color in zip(
            axes2,
            [psnr_vals, ssim_vals],
            ['PSNR (dB)', 'SSIM'],
            ['#185FA5', '#0F6E56']):
        ax.semilogx(lambda_values, vals, 'o-', linewidth=2,
                    markersize=6, color=color)
        ax.axvline(lambda_values[best_idx], color='#D85A30',
                   linestyle='--', linewidth=1,
                   label=f'Optimal λ = {lambda_values[best_idx]}')
        ax.set_xlabel('λ (log scale)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} vs λ', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle('TV regularisation parameter optimisation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(save_dir, f'{img_stem}_lambda_sensitivity_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {chart_path}")
    plt.show()
    plt.close()

    return rows