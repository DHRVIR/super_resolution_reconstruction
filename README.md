# Image Super-Resolution via TV-Regularised Reconstruction

Reconstructs a high-resolution image from a blurred, downsampled, noisy
measurement using two methods:

1. **Least Squares (LS)** — projected gradient descent on the data-fidelity
   term (baseline)
2. **TV-PDHG** — Total Variation regularisation solved via the
   Primal-Dual Hybrid Gradient algorithm of Chambolle & Pock (2011)

> **Reference:** Chambolle, A. & Pock, T. (2011). *A first-order primal-dual
> algorithm for convex problems with applications to imaging.*
> Journal of Mathematical Imaging and Vision, 40(1), 120–145.

---

## Problem formulation

```
y = MHx + n
```

| Symbol | Meaning |
|--------|---------|
| `x` | Unknown high-resolution image |
| `H` | Gaussian blur operator (σ configurable) |
| `M` | Downsampling operator (factor configurable) |
| `n` | Additive Gaussian noise |
| `y` | Observed low-resolution measurement |

**TV-PDHG objective:**

```
min_x  ½‖MHx - y‖²  +  λ · TV(x)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/DHRVIR/super_resolution_reconstruction.git
cd super-resolution-reconstruction
pip install -r requirements.txt
```

### 2. Add data

Download BSD68 from
[Kaggle](https://www.kaggle.com/code/mpwolke/berkeley-segmentation-dataset-68/input?select=BSD68)
and place `.png` files in `data/`:

```
data/
  test001.png
  test002.png
  ...
```

---

## Usage

```bash
# Full pipeline — reconstruction + all sensitivity sweeps (~15 min)
python main.py

# Reconstruction only, no sensitivity sweeps (~1 min)
python main.py --no-sensitivity

# Hyperparameter tuning grids (~30 min)
python main.py --tune

# Specific image
python main.py --image data/test042.png

# Custom degradation parameters
python main.py --blur-sigma 1.5 --downsample 4 --noise-sigma 0.02 --lambda-tv 0.005

# All options
python main.py --help
```

---

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | first PNG in `data/` | Input image path |
| `--blur-sigma` | 1.0 | Gaussian blur σ |
| `--downsample` | 2 | Downsampling factor |
| `--noise-sigma` | 0.01 | Measurement noise σ |
| `--lambda-tv` | 0.001 | TV regularisation weight |
| `--ls-step` | 0.05 | LS gradient descent step size |
| `--ls-iters` | 800 | LS iterations |
| `--tv-iters` | 300 | PDHG iterations |
| `--tune` | — | Run hyperparameter tuning mode |
| `--no-sensitivity` | — | Skip sensitivity sweeps |

---

## Results

Evaluated on BSD68 `test001.png`
(blur σ=1.0, 2× downsample, noise σ=0.01, tuned parameters):

| Method | MSE ↓ | PSNR ↑ | SSIM ↑ |
|--------|-------|--------|--------|
| Bicubic (input) | 0.004877 | 23.12 dB | 0.633 |
| Least Squares | 0.003910 | 24.08 dB | 0.705 |
| **TV-PDHG** | **0.003613** | **24.42 dB** | **0.722** |

TV-PDHG outperforms LS at low-to-moderate noise. At high noise (σ > 0.02),
λ should be increased proportionally — see the report for details.

---

## Outputs

All figures are saved to `outputs/`, prefixed with the image name.

| File | Contents |
|------|----------|
| `{stem}_comparison.png` | Side-by-side visual + error maps + zoom |
| `{stem}_convergence.png` | PDHG objective and PSNR vs iteration |
| `{stem}_noise_sensitivity_grid.png` | Image grid across noise levels |
| `{stem}_noise_sensitivity_chart.png` | PSNR vs noise σ |
| `{stem}_blur_sensitivity_grid.png` | Image grid across blur levels |
| `{stem}_blur_sensitivity_chart.png` | PSNR vs blur σ |
| `{stem}_downsampling_sensitivity_grid.png` | Image grid across downsample factors |
| `{stem}_downsampling_sensitivity_chart.png` | PSNR vs downsample factor |
| `{stem}_lambda_sensitivity_grid.png` | Image grid across λ values |
| `{stem}_lambda_sensitivity_chart.png` | PSNR/SSIM vs λ |
| `{stem}_hp_ls_step_grid.png` | LS step-size tuning grid |
| `{stem}_hp_ls_iters_grid.png` | LS iteration tuning grid |
| `{stem}_hp_tv_lambda_grid.png` | TV-PDHG λ tuning grid |
| `{stem}_hp_tv_iters_grid.png` | TV-PDHG iteration tuning grid |
| `{stem}_hp_summary.png` | Before/after tuning comparison |

---

## Project structure

```
super-resolution-reconstruction/
├── main.py                      ← entry point
├── requirements.txt
├── data/                        ← place BSD68 images here
├── outputs/                     ← all figures saved here (gitignored)
├── report/
│   └── report.pdf
└── src/
    ├── forward_model.py         ← ForwardModel (H, M, adjoint)
    ├── gradient_operators.py    ← GradientOperators (∇, div, TV norm)
    ├── reconstruction.py        ← LeastSquaresReconstruction, PrimalDualTVReconstruction
    ├── evaluation.py            ← metrics, comparison grid, convergence plot
    ├── sensitivity_analysis.py  ← parameter sweep functions
    └── hyperparameter_tuning.py ← step-size, λ, iteration sweeps
```

---

## Algorithm summary

### PDHG (Chambolle & Pock 2011, Algorithm 1)

```
Initialise  x⁰,  x̄⁰ = x⁰,  p⁰ = q⁰ = 0

for k = 0, 1, 2, …:
    (p, q) ← project( (p,q) + σ·∇x̄,   ‖·‖₂ ≤ λ )    # dual ascent
    x⁺     ← clip( x − τ·(AᵀAx − Aᵀy − div(p,q)),  [0,1] )  # primal descent
    x̄      ← x⁺ + θ·(x⁺ − x)                           # extrapolation
    x      ← x⁺
```

Convergence guaranteed when **τ · σ ≤ 1/8**
(with τ=σ=0.1, product = 0.01 ✓).

---

## License

MIT