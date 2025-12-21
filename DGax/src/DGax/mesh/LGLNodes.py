import numpy as np
from scipy.special import legendre, eval_legendre
import warnings

def legendre_gauss_lobatto_nodes(N):
    """
    Compute Legendre-Gauss-Lobatto (LGL) nodes.
    
    LGL nodes are the roots of (1-x^2)P_{N-1}'(x) = 0 plus the endpoints ±1.
    
    Parameters:
    -----------
    N : int
        Number of nodes (order = N-1)
        
    Returns:
    --------
    x : ndarray
        LGL nodes in [-1, 1], sorted in ascending order
    """
    if N < 2:
        raise ValueError("N must be at least 2")
    
    # For small N, use explicit formulas
    if N == 2:
        return np.array([-1., 1.])
    elif N == 3:
        return np.array([-1., 0., 1.])
    elif N == 4:
        return np.array([-1., -np.sqrt(1/5), np.sqrt(1/5), 1.])
    elif N == 5:
        return np.array([-1., -np.sqrt(3/7), 0., np.sqrt(3/7), 1.])
    
    # For larger N, use Newton's method on initial guess from Chebyshev points
    # Initial guess: Chebyshev-Gauss-Lobatto nodes
    k = np.arange(N)
    x = -np.cos(np.pi * k / (N - 1))
    
    # Legendre polynomial of degree N-1 and its derivative
    P = legendre(N-1)
    dP = P.deriv()
    
    # Newton iteration to find roots of (1-x^2)P_{N-1}'(x)
    for _ in range(20):  # Maximum 20 iterations
        # Compute function value and derivative
        f = (1 - x**2) * dP(x)
        
        # Derivative of f(x) = (1-x^2)P_{N-1}'(x)
        # f'(x) = -2x P_{N-1}'(x) + (1-x^2)P_{N-1}''(x)
        df = -2 * x * dP(x) + (1 - x**2) * dP.deriv()(x)
        
        # Newton update (skip endpoints which are already fixed)
        mask = np.abs(x) < 1 - 1e-10  # Interior points
        delta = np.zeros_like(x)
        delta[mask] = f[mask] / df[mask]
        
        # Update
        x -= delta
        
        # Check convergence
        if np.max(np.abs(delta[mask])) < 1e-15:
            break
    
    # Ensure endpoints are exactly -1 and 1
    x[0] = -1.0
    x[-1] = 1.0
    
    return np.sort(x)

def lagrange_derivative_matrix(x):
    """
    Compute derivative matrix for Lagrange polynomials at given nodes.
    
    D[i,j] = L_j'(x_i), where L_j is the j-th Lagrange basis polynomial.
    
    Parameters:
    -----------
    x : ndarray
        Nodes in [-1, 1]
        
    Returns:
    --------
    D : ndarray
        Derivative matrix of shape (N, N)
    """
    N = len(x)
    D = np.zeros((N, N))
    
    # Compute barycentric weights
    w = np.ones(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                w[i] *= (x[i] - x[j])
    w = 1.0 / w
    
    # Compute derivative matrix
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = w[j] / (w[i] * (x[i] - x[j]))
    
    # Diagonal entries: D[i,i] = -sum_{j≠i} D[i,j]
    D = D - np.diag(np.sum(D, axis=1))
    
    return D

def legendre_gauss_lobatto_weights(x):
    """
    Compute LGL quadrature weights.
    
    Parameters:
    -----------
    x : ndarray
        LGL nodes
        
    Returns:
    --------
    w : ndarray
        Quadrature weights
    """
    N = len(x)
    if N == 1:
        return np.array([2.0])
    
    # Weights formula: w_j = 2/(N(N-1)[P_{N-1}(x_j)]^2)
    P = legendre(N-1)
    Px = P(x)
    w = 2.0 / (N * (N-1) * Px**2)
    
    # Adjust for endpoints
    w[0] = 2.0 / (N * (N-1))
    w[-1] = 2.0 / (N * (N-1))
    
    return w

def GaussLobatto1D(N):
    """
    Compute all LGL properties: nodes, derivative matrix, and weights.
    
    Parameters:
    -----------
    N : int
        Number of nodes
        
    Returns:
    --------
    nodes : ndarray
        LGL nodes
    D : ndarray
        Derivative matrix
    weights : ndarray
        Quadrature weights
    """
    nodes = legendre_gauss_lobatto_nodes(N)
    D = lagrange_derivative_matrix_fast(nodes)
    weights = legendre_gauss_lobatto_weights(nodes)
    
    return nodes, D, weights

# Alternative: More efficient implementation using numpy broadcasting
def lagrange_derivative_matrix_fast(x):
    """
    Faster implementation using numpy broadcasting.
    """
    N = len(x)
    X = np.tile(x[:, None], (1, N))
    X_diff = X - X.T
    
    # Remove diagonal entries (set to 1 to avoid division by zero)
    np.fill_diagonal(X_diff, 1)
    
    # Compute weights
    w = 1.0 / np.prod(X_diff, axis=1)
    
    # Compute derivative matrix
    D = w[None, :] / (w[:, None] * X_diff)
    np.fill_diagonal(D, 0)
    
    # Diagonal entries
    D = D - np.diag(np.sum(D, axis=1))
    
    return D

# Test the implementation
if __name__ == "__main__":
    # Test with various N
    for N in [3, 5, 8, 10, 6]:
        print(f"\n{'='*50}")
        print(f"Testing with N = {N}")
        print('='*50)
        
        nodes, D, weights = GaussLobatto1D(N)
        
        print(f"\nNodes: {nodes}")
        print(f"\nWeights: {weights}")
        print(f"\nWeights sum: {np.sum(weights):.10f} (should be 2.0)")
        
        # Test derivative matrix on polynomials
        print("\nTesting derivative matrix:")
        for k in range(N):
            f = nodes**k
            df_exact = k * nodes**(max(0, k-1))
            df_approx = D @ f
            error = np.max(np.abs(df_exact - df_approx))
            print(f"  x^{k}: max error = {error:.2e}")
        
        # Test quadrature
        print("\nTesting quadrature:")
        for k in range(2*N-1):  # LGL is exact for polynomials up to degree 2N-3
            f = nodes**k
            integral = weights @ f
            exact = 2/(k+1) if k % 2 == 0 else 0
            error = np.abs(integral - exact)
            print(f"  ∫x^{k} dx: {integral:.10f}, exact: {exact:.10f}, error: {error:.2e}")