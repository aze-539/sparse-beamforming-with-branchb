import numpy as np
import cvxpy as cp
import time

# Define fixed test data
x = np.array([[-0.41675785],
              [-0.05626683],
              [-2.1361961],
              [1.64027081]])
y = np.array([[-3.02091013 - 2.28153521j],
              [-0.01189441 + 0.50224336j],
              [1.56515899 + 2.44947271j],
              [-1.79186692 + 1.89704246j]])
H = np.array([[-1.26815046+3.54866634e-01j, -0.59520527-5.96949235e-01j,
         0.35559086+6.90240276e-06j, -0.88055165+3.83501182e-01j,
        -0.74808519-2.21683772e-01j, -0.64276545+5.45187628e-01j,
         0.38993689-1.32093957e+00j,  1.62083583+1.22413242e+00j],
       [ 0.02937279+1.03780507e+00j, -0.79049266-2.37359722e-01j,
         0.38117179+4.32283211e-01j, -0.42154857+3.39203308e-02j,
        -0.0135273 -5.86287185e-01j,  0.83085133+6.20204902e-02j,
        -0.52882462+7.07365502e-01j,  0.00638182-2.69473103e-01j],
       [-0.62091605-2.65638397e-01j, -0.11061566-5.26587814e-02j,
         0.18142271+3.06528195e-01j, -0.69917237+9.03950623e-01j,
        -0.23958331-4.48786041e-01j, -0.16700733+3.59490431e-01j,
        -0.45089018+1.52817094e-01j, -0.8397687 -1.31423742e+00j],
       [-1.00495234-2.96501528e-01j, -0.10853749-9.35706614e-02j,
        -0.190252  -2.79803848e-02j,  1.57781459+2.30519238e-01j,
        -1.72164066-1.44272626e+00j,  0.07970968+3.27075941e-02j,
         0.26194384-4.79188996e-01j,  0.96140632-1.01783710e+00j]])  # Truncated here to save space; same as original


def ls_solve(x, y, W, H, max_ant=None):
    """
    Solves a least squares beamforming problem with antenna selection.
    Args:
        x (np.ndarray): Input signal vector.
        y (np.ndarray): Desired received signal vector.
        W (np.ndarray): Initial beamforming matrix.
        H (np.ndarray): Channel matrix.
        max_ant (int): Maximum number of active antennas.
    """
    Nt, Ns = W.shape
    a = cp.Variable(Nt)

    W_eff = cp.diag(a) @ W
    objective = cp.Minimize(cp.norm(y - H @ W_eff @ x, 'fro') ** 2)
    constraints = [
        cp.norm(W_eff, 'fro') ** 2 <= 20,
        a >= 0,
        a <= 1,
        cp.sum(a) == max_ant
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(verbose=False)
    except cp.SolverError:
        return None

    return a.value


def solve_ASP(H, a, max_ant=None, sigma_sq=1.0, power_comsumption=20):
    """
    Solves the ASP optimization problem.
    Args:
        H (np.ndarray): Channel matrix.
        a (np.ndarray): Antenna weights.
        max_ant (int): Maximum number of antennas (not used).
        sigma_sq (float): Noise variance.
        power_comsumption (float): Power budget.
    """
    Ns, Nt = H.shape
    Q = cp.Variable((Nt, Nt), hermitian=True, name='Q')

    rate = cp.real(cp.log_det(np.eye(Ns) + H @ cp.diag(a) @ Q @ cp.diag(a) @ H.conj().T / sigma_sq ** 2))
    objective = cp.Maximize(rate)

    constraints = [
        Q >> 0,
        cp.real(cp.trace(Q)) <= power_comsumption
    ]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except cp.SolverError:
        print("Solver failed.")
        return None, None, None, False

    Q_sol = Q.value
    assert Q_sol.shape[0] == Nt, 'Q has incorrect shape'

    W, Q_check = semi_definite_decomposition(Q_sol, Ns)
    W[np.abs(W) < 1e-8] = 0
    return W, Q_sol, prob.value, True


def semi_definite_decomposition(Q, k):
    """
    Decomposes a positive semi-definite matrix Q into A such that Q ≈ A @ A.H
    Args:
        Q (np.ndarray): Semi-definite matrix.
        k (int): Number of largest eigenvalues to retain.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:k]
    eigenvectors = eigenvectors[:, idx][:, :k]

    A = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    Q_reconstructed = A @ A.conj().T
    return A, Q_reconstructed


# === Simulation Loop ===
results = []

for i in range(10):
    np.random.seed(10 * i)
    W = (np.random.randn(8, 4) + 1j * np.random.randn(8, 4)) / np.sqrt(2)

    obj_result = []
    for j in range(1, 9):  # Try different number of active antennas
        for _ in range(20):  # Retry optimization in case of failure
            a = ls_solve(x, y, W, H, max_ant=j)
            if a is None:
                break
            W, Q_sol, obj, _ = solve_ASP(H, a)
        if obj is not None:
            obj_result.append(obj)
    print(obj_result)
    results.append(obj_result)

# Report average performance
average_result = np.mean(results, axis=0)
print("Average result over 10 runs:", average_result)



