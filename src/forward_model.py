"""
Forward degradation model: y = MHx + n

H  = Gaussian blur operator
M  = downsampling operator
n  = additive Gaussian noise
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


class ForwardModel:
    """
    Implements the forward degradation pipeline and its adjoint.

    Pipeline:  x  --[H]--> blurred  --[M]--> downsampled  --[+n]--> y

    Parameters
    ----------
    blur_sigma : float
        Standard deviation of the Gaussian blur kernel.
    downsample_factor : int
        Integer downsampling factor (e.g. 2 for 2x super-resolution).
    noise_sigma : float
        Standard deviation of additive Gaussian measurement noise.
    """

    def __init__(self, blur_sigma: float = 1.0,
                 downsample_factor: int = 2,
                 noise_sigma: float = 0.01):
        self.blur_sigma = blur_sigma
        self.factor = downsample_factor
        self.noise_sigma = noise_sigma

    # ------------------------------------------------------------------
    # Forward operators
    # ------------------------------------------------------------------

    def blur_operator(self, x: np.ndarray) -> np.ndarray:
        """H: Gaussian blur with Neumann boundary conditions."""
        return gaussian_filter(x, sigma=self.blur_sigma)

    def sampling_operator(self, x: np.ndarray) -> np.ndarray:
        """M: anti-aliased downsampling."""
        h, w = x.shape
        return resize(x,
                      (h // self.factor, w // self.factor),
                      anti_aliasing=True)

    def upsample_operator(self, z: np.ndarray,
                          target_shape: tuple) -> np.ndarray:
        """M^T: bicubic upsampling (adjoint of downsampling)."""
        return resize(z, target_shape, order=3, anti_aliasing=False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward model without noise: y = MHx."""
        return self.sampling_operator(self.blur_operator(x))

    def forward_noisy(self, x: np.ndarray) -> np.ndarray:
        """Full forward model with noise: y = MHx + n."""
        y = self.forward(x)
        noise = self.noise_sigma * np.random.randn(*y.shape)
        return np.clip(y + noise, 0.0, 1.0)

    def adjoint(self, r: np.ndarray, original_shape: tuple) -> np.ndarray:
        """
        Adjoint operator A^T = H^T M^T.

        Used to compute the data-fidelity gradient:
            grad_f(x) = A^T (Ax - y)
        """
        return self.blur_operator(self.upsample_operator(r, original_shape))
