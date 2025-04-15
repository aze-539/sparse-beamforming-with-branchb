import cvxpy as cp
import numpy as np


def solve_ASP(H=None,
               z_sol=None,
               z_mask=None,
               sigma_sq=1.0,
               power_comsumption=10):
    """Solve the ASP optimization problem
    Args:
        H (ndarray): Channel matrix
        z_sol (ndarray): Current antenna selection solution
        z_mask (ndarray): Mask indicating available antennas
        sigma_sq (float): Noise variance
        power_comsumption (float): Maximum power consumption constraint

    Returns:
        tuple: (z_sol, W, Q_sol, optimal_value, status)
    """
    Ns, Nt = H.shape
    num_off_ants = np.sum(z_mask * (1 - z_sol))
    assert num_off_ants < Nt, 'Number of allowed antennas must be at least 1'

    # Define optimization variables and problem
    Q = cp.Variable((Nt, Nt), hermitian=True, name='Q')
    rate = cp.real(cp.log_det(np.eye(Ns) + H @ Q @ H.conj().T / sigma_sq ** 2))
    obj = cp.Maximize(rate)

    # Build constraints
    constraints = []
    for n in range(Nt - 1, -1, -1):
        if z_mask[n] and not z_sol[n]:
            constraints.append(Q[n, :] == 0)
    constraints.append(Q >> 0)  # Positive semidefinite constraint
    constraints.append(cp.real(cp.trace(Q)) <= power_comsumption)

    prob = cp.Problem(obj, constraints)

    # Try different solvers with fallback mechanism
    solvers = [cp.MOSEK, cp.ECOS, cp.SCS, cp.OSQP]
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=False)
            break
        except cp.SolverError:
            print(f"Solver {solver} failed, trying next solver.")

    Q_sol = Q.value
    threshold = 1e-8

    # Validate solution
    assert Q_sol.shape[0] == Nt, 'Q matrix has incorrect dimensions'
    W, Q_sol_1 = semi_definite_decomposition(Q_sol, Ns)
    W[np.abs(W) < threshold] = 0  # Threshold small values to zero

    return z_sol, W, Q_sol, prob.value, True


def semi_definite_decomposition(Q, k):
    """
    Perform semi-definite decomposition of matrix Q into A and A.H such that Q = A @ A.H
    Args:
        Q (ndarray): Input positive semidefinite matrix
        k (int): Number of largest eigenvalues to keep

    Returns:
        tuple: (A, Q_reconstructed) where:
            A: Decomposed matrix
            Q_reconstructed: Reconstructed matrix for verification
    """
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Q)

    # Keep top k non-negative eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[idx][:k]
    eigenvectors = eigenvectors[:, idx][:, :k]

    # Compute decomposition matrix
    A = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    # Reconstruct Q for verification
    Q_reconstructed = A @ A.conj().T

    return A, Q_reconstructed


if __name__ == '__main__':
    # Test configuration
    Ns, Nt = 4, 8
    np.random.seed(30)
    H = (np.random.randn(Ns, Nt) + 1j * np.random.randn(Ns, Nt)) / np.sqrt(2)
    z_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # All antennas available

    # Test different antenna selection patterns
    z_sol1 = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    z_sol2 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    z_sol3 = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    z_sol4 = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Run optimization for each selection pattern
    results = []
    for z_sol in [z_sol1, z_sol2, z_sol3, z_sol4]:
        results.append(solve_ASP(H=H, z_mask=z_mask, z_sol=z_sol))

    # Print results
    for result in results:
        z_sol, F, Q, obj, optimal = result
        print("Objective value:", obj)
        print("Antenna selection solution:", z_sol)
        print("Q matrix:", Q)
        print()