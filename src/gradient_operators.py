"""
Discrete gradient and divergence operators for Total Variation regularisation.

Reference:
    Chambolle, A. (2004). An algorithm for total variation minimization
    and applications. Journal of Mathematical Imaging and Vision, 20(1-2), 89-97.
"""

import numpy as np


class GradientOperators:
    """
    Forward-difference gradient and its adjoint (negative divergence).

    Boundary conditions are Neumann (zero-flux), so the gradient vanishes
    at the last row/column and the divergence mirrors this symmetrically.
    This ensures the discrete adjoint identity:

        <∇x, (p, q)> = -<x, div(p, q)>
    """

    @staticmethod
    def gradient(x: np.ndarray):
        """
        Compute forward differences (discrete gradient).

            (∇x)_{i,j} = ( x_{i+1,j} - x_{i,j},  x_{i,j+1} - x_{i,j} )

        Returns
        -------
        grad_x : ndarray   horizontal (row) component
        grad_y : ndarray   vertical  (col) component
        """
        grad_x = np.zeros_like(x)
        grad_y = np.zeros_like(x)
        grad_x[:-1, :] = x[1:, :] - x[:-1, :]   # Neumann: last row stays 0
        grad_y[:, :-1] = x[:, 1:] - x[:, :-1]   # Neumann: last col stays 0
        return grad_x, grad_y

    @staticmethod
    def divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute negative divergence (adjoint of gradient).

            (div p)_{i,j} = (p_{i,j} - p_{i-1,j}) + (q_{i,j} - q_{i,j-1})

        Parameters
        ----------
        p : ndarray   horizontal dual variable
        q : ndarray   vertical  dual variable
        """
        div = np.zeros_like(p)

        # Backward differences — row direction
        div[0, :]    =  p[0, :]
        div[1:-1, :] =  p[1:-1, :] - p[:-2, :]
        div[-1, :]   = -p[-2, :]

        # Backward differences — column direction
        div[:, 0]    +=  q[:, 0]
        div[:, 1:-1] +=  q[:, 1:-1] - q[:, :-2]
        div[:, -1]   += -q[:, -2]

        return div

    @staticmethod
    def gradient_norm(grad_x: np.ndarray,
                      grad_y: np.ndarray,
                      eps: float = 1e-8) -> np.ndarray:
        """Pointwise Euclidean norm of the gradient: sqrt(gx^2 + gy^2)."""
        return np.sqrt(grad_x ** 2 + grad_y ** 2 + eps)

    @staticmethod
    def tv_norm(x: np.ndarray) -> float:
        """Isotropic total variation: sum_ij ||(∇x)_{ij}||_2."""
        gx, gy = GradientOperators.gradient(x)
        return float(np.sum(GradientOperators.gradient_norm(gx, gy)))
