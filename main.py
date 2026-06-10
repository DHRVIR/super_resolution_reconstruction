"""
main.py — Image Super-Resolution via TV Regularisation

Usage
-----
    python main.py                             # reconstruct, full sensitivity
    python main.py --no-sensitivity            # reconstruct only (~1 min)
    python main.py --tune                      # hyperparameter tuning only
    python main.py --tune --no-sensitivity     # tune + skip sensitivity
    python main.py --image data/test042.png    # use a specific image
    python main.py --help

Outputs (saved to outputs/, prefixed with image name)
------------------------------------------------------
    {stem}_comparison.png
    {stem}_convergence.png
    {stem}_noise_sensitivity_grid.png / _chart.png
    {stem}_blur_sensitivity_grid.png  / _chart.png
    {stem}_downsampling_sensitivity_grid.png / _chart.png
    {stem}_lambda_sensitivity_grid.png / _chart.png
    {stem}_hp_ls_step_grid.png / _chart.png     (--tune only)
    {stem}_hp_ls_iters_grid.png / _chart.png    (--tune only)
    {stem}_hp_tv_lambda_grid.png / _chart.png   (--tune only)
    {stem}_hp_tv_iters_grid.png / _chart.png    (--tune only)
    {stem}_hp_summary.png                       (--tune only)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # remove this line to show windows interactively

from skimage import io, img_as_float
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(__file__))

from src.forward_model import ForwardModel
from src.reconstruction import LeastSquaresReconstruction, PrimalDualTVReconstruction
from src.evaluation import (compute_metrics, print_results_table,
                             plot_comparison_grid, plot_convergence)
from src.sensitivity_analysis import (analyze_noise_sensitivity,
                                       analyze_blur_sensitivity,
                                       analyze_downsampling_sensitivity,
                                       analyze_lambda_sensitivity)
from src.hyperparameter_tuning import run_tuning


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='TV-regularised super-resolution (Chambolle & Pock 2011)',
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--image',          default=None,
                   help='Path to a greyscale PNG.\n'
                        'Defaults to the first image found in data/.')
    p.add_argument('--blur-sigma',     type=float, default=1.0,
                   help='Gaussian blur σ (default: 1.0)')
    p.add_argument('--downsample',     type=int,   default=2,
                   help='Downsampling factor (default: 2)')
    p.add_argument('--noise-sigma',    type=float, default=0.01,
                   help='Measurement noise σ (default: 0.01)')
    p.add_argument('--lambda-tv',      type=float, default=0.001,
                   help='TV regularisation weight (default: 0.001)')
    p.add_argument('--ls-iters',       type=int,   default=800,
                   help='Least Squares gradient steps (default: 800)')
    p.add_argument('--ls-step',        type=float, default=0.05,
                   help='Least Squares gradient descent step size (default: 0.05)')
    p.add_argument('--tv-iters',       type=int,   default=300,
                   help='PDHG iterations (default: 300)')
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--tune',           action='store_true',
                   help='Run hyperparameter tuning (step_size, lambda_tv, num_iters).\n'
                        'Skips the normal reconstruction pipeline.')
    p.add_argument('--no-sensitivity', action='store_true',
                   help='Skip sensitivity sweeps (noise / blur / downsample / lambda).')
    return p.parse_args()


# ============================================================================
# Helper
# ============================================================================

def find_image(path_hint=None) -> str:
    if path_hint and os.path.isfile(path_hint):
        return path_hint
    for d in ['data', 'BSD68_images', '.']:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith('.png'):
                return os.path.join(d, fname)
    print("[Error] No image found. Place a PNG in data/ or pass --image <path>.")
    sys.exit(1)


# ============================================================================
# Reconstruction pipeline
# ============================================================================

def run_reconstruction(args, img, img_path):
    img_stem = os.path.splitext(os.path.basename(img_path))[0]

    fwd = ForwardModel(blur_sigma=args.blur_sigma,
                       downsample_factor=args.downsample,
                       noise_sigma=args.noise_sigma)
    y   = fwd.forward_noisy(img)
    y_up = resize(y, img.shape, order=3, anti_aliasing=False)

    print(f"Measurement: {y.shape}  "
          f"(blur σ={args.blur_sigma}, ×{args.downsample}, noise σ={args.noise_sigma})")

    # ---- Reconstruct -------------------------------------------------------
    print("\n[1/2] Least Squares reconstruction …")
    ls_solver = LeastSquaresReconstruction(fwd, step_size=args.ls_step)
    x_ls = ls_solver.reconstruct(y, img.shape, num_iters=args.ls_iters, verbose=True)

    print("\n[2/2] TV · Primal-Dual reconstruction …")
    tv_solver = PrimalDualTVReconstruction(fwd, lambda_tv=args.lambda_tv)
    tv_solver.set_ground_truth(img)
    x_tv, tv_history = tv_solver.reconstruct(y, img.shape,
                                              num_iters=args.tv_iters, verbose=True)

    # ---- Evaluate ----------------------------------------------------------
    results = {
        'Bicubic (input)':  compute_metrics(img, y_up),
        'Least Squares':    compute_metrics(img, x_ls),
        'TV · Primal-Dual': compute_metrics(img, x_tv),
    }
    print_results_table(results)

    ls_m, tv_m = results['Least Squares'], results['TV · Primal-Dual']
    print(f"\n  TV vs LS:  PSNR {tv_m['PSNR']-ls_m['PSNR']:+.2f} dB  |  "
          f"SSIM {tv_m['SSIM']-ls_m['SSIM']:+.4f}  |  "
          f"MSE {(ls_m['MSE']-tv_m['MSE'])/ls_m['MSE']*100:+.1f}% reduction")

    # ---- Save figures ------------------------------------------------------
    plot_comparison_grid(
        ground_truth=img, measurement=y,
        reconstructions={'Least Squares': x_ls, 'TV · Primal-Dual': x_tv},
        save_path=f'outputs/{img_stem}_comparison.png')

    plot_convergence(tv_history, save_path=f'outputs/{img_stem}_convergence.png')

    # ---- Sensitivity sweeps (optional) -------------------------------------
    if not args.no_sensitivity:
        print(f"\n── Sensitivity analysis  [{img_stem}] ────────────────────────")
        kw = dict(img_stem=img_stem, save_dir='outputs')
        analyze_noise_sensitivity(img,
                                  blur_sigma=args.blur_sigma,
                                  factor=args.downsample, **kw)
        analyze_blur_sensitivity(img,
                                  noise_sigma=args.noise_sigma,
                                  factor=args.downsample, **kw)
        analyze_downsampling_sensitivity(img,
                                         blur_sigma=args.blur_sigma,
                                         noise_sigma=args.noise_sigma, **kw)
        analyze_lambda_sensitivity(img,
                                   blur_sigma=args.blur_sigma,
                                   factor=args.downsample,
                                   noise_sigma=args.noise_sigma, **kw)


# ============================================================================
# Entry point
# ============================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs('outputs', exist_ok=True)

    img_path = find_image(args.image)
    img      = img_as_float(io.imread(img_path, as_gray=True))
    img_stem = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\nImage: {img_path}  {img.shape}")

    if args.tune:
        # ---- Hyperparameter tuning mode ------------------------------------
        run_tuning(img,
                   blur_sigma=args.blur_sigma,
                   factor=args.downsample,
                   noise_sigma=args.noise_sigma,
                   img_stem=img_stem,
                   save_dir='outputs')
    else:
        # ---- Normal reconstruction + sensitivity mode ----------------------
        run_reconstruction(args, img, img_path)

    print("\nDone. All outputs saved to outputs/")


if __name__ == '__main__':
    main()