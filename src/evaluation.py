"""
Quantitative and qualitative evaluation of reconstruction methods.

Provides
--------
- compute_metrics()   — MSE, PSNR, SSIM, MAE
- print_results_table()
- plot_comparison_grid()
- plot_convergence()
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import (mean_squared_error,
                              peak_signal_noise_ratio,
                              structural_similarity)
from skimage.transform import resize


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(original: np.ndarray,
                    reconstructed: np.ndarray) -> dict:
    """
    Compute standard image quality metrics.

    Parameters
    ----------
    original      : ground-truth image,  float in [0, 1]
    reconstructed : estimated image,      float in [0, 1]

    Returns
    -------
    dict with keys: MSE, MAE, PSNR, SSIM
    """
    if reconstructed.shape != original.shape:
        reconstructed = resize(reconstructed, original.shape,
                               anti_aliasing=False)

    gt  = original.flatten()
    rec = reconstructed.flatten()

    return {
        'MSE':  float(mean_squared_error(gt, rec)),
        'MAE':  float(np.mean(np.abs(gt - rec))),
        'PSNR': float(peak_signal_noise_ratio(original, reconstructed,
                                               data_range=1.0)),
        'SSIM': float(structural_similarity(original, reconstructed,
                                             data_range=1.0)),
    }


def print_results_table(results: dict):
    """
    Pretty-print a comparison table.

    Parameters
    ----------
    results : dict  {method_name: metrics_dict}
    """
    header = f"\n{'Method':<22} {'MSE':>10} {'PSNR (dB)':>11} {'SSIM':>8} {'MAE':>10}"
    print("\n" + "=" * 65)
    print("  QUANTITATIVE RESULTS")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for name, m in results.items():
        print(f"  {name:<20} {m['MSE']:>10.6f} {m['PSNR']:>11.2f} "
              f"{m['SSIM']:>8.4f} {m['MAE']:>10.6f}")
    print("=" * 65)


# ============================================================================
# Visualisation helpers
# ============================================================================

def _add_metrics_overlay(ax, metrics: dict):
    """Overlay MSE / PSNR / SSIM in the top-right corner of an axes."""
    text = (f"MSE:  {metrics['MSE']:.5f}\n"
            f"PSNR: {metrics['PSNR']:.2f} dB\n"
            f"SSIM: {metrics['SSIM']:.4f}")
    ax.text(0.98, 0.98, text,
            transform=ax.transAxes,
            fontsize=7, color='yellow', fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                      alpha=0.7, edgecolor='yellow', linewidth=0.8))


def plot_comparison_grid(ground_truth: np.ndarray,
                         measurement: np.ndarray,
                         reconstructions: dict,
                         save_path: str = None):
    """
    Three-row comparison grid:
      Row 1 — images
      Row 2 — absolute error maps
      Row 3 — zoomed detail

    Parameters
    ----------
    ground_truth    : reference HR image
    measurement     : LR measurement (will be upsampled for display)
    reconstructions : {name: image}  ordered dict
    save_path       : if given, save figure to this path
    """
    methods   = list(reconstructions.items())
    n_methods = len(methods)
    n_cols    = n_methods + 2   # GT + measurement + reconstructions

    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 9))

    y_up = resize(measurement, ground_truth.shape, anti_aliasing=False)
    meas_metrics = compute_metrics(ground_truth, y_up)

    # ---- Row 1: images ----
    axes[0, 0].imshow(ground_truth, cmap='gray')
    axes[0, 0].set_title('Ground truth', fontsize=9, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(y_up, cmap='gray')
    axes[0, 1].set_title('Measurement\n(bicubic up)', fontsize=9)
    _add_metrics_overlay(axes[0, 1], meas_metrics)
    axes[0, 1].axis('off')

    for col, (name, img) in enumerate(methods, start=2):
        m = compute_metrics(ground_truth, img)
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(name, fontsize=9, fontweight='bold')
        _add_metrics_overlay(axes[0, col], m)
        axes[0, col].axis('off')

    # ---- Row 2: error maps ----
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.abs(ground_truth - y_up), cmap='hot', vmin=0, vmax=0.2)
    axes[1, 1].set_title('Error map', fontsize=9)
    axes[1, 1].axis('off')

    for col, (_, img) in enumerate(methods, start=2):
        axes[1, col].imshow(np.abs(ground_truth - img),
                            cmap='hot', vmin=0, vmax=0.2)
        axes[1, col].set_title('Error map', fontsize=9)
        axes[1, col].axis('off')

    # ---- Row 3: zoomed patch ----
    h, w  = ground_truth.shape
    zoom  = (slice(h // 3, h // 2), slice(w // 4, w // 3))

    axes[2, 0].imshow(ground_truth[zoom], cmap='gray')
    axes[2, 0].set_title('GT (zoom)', fontsize=9)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(y_up[zoom], cmap='gray')
    axes[2, 1].set_title('Meas. (zoom)', fontsize=9)
    axes[2, 1].axis('off')

    for col, (name, img) in enumerate(methods, start=2):
        axes[2, col].imshow(img[zoom], cmap='gray')
        axes[2, col].set_title(f'{name} (zoom)', fontsize=9)
        axes[2, col].axis('off')

    plt.suptitle('Super-resolution reconstruction comparison',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_convergence(history: dict, save_path: str = None):
    """
    Plot objective value and PSNR over PDHG iterations.

    Parameters
    ----------
    history   : dict returned by PrimalDualTVReconstruction.reconstruct()
    save_path : optional file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.semilogy(history['loss'], color='#D85A30', linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective value (log scale)')
    ax1.set_title('Convergence — objective')
    ax1.grid(True, alpha=0.3)

    if any(v > 0 for v in history['psnr']):
        ax2.plot(history['psnr'], color='#185FA5', linewidth=1.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('PSNR vs iteration')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'PSNR not tracked\n(no ground truth provided)',
                 ha='center', va='center', transform=ax2.transAxes,
                 color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
