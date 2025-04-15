import cvxpy as cp
import numpy as np


def solve_subASP(H=None,
               z_sol=None,
               z_mask=None,
               sigma_sq=1.0,
               power_comsumption=10):
    Ns, Nt = H.shape
    num_off_ants = np.sum(z_mask * (1 - z_sol))
    assert num_off_ants < Nt, 'number of allowed antennas < 1'
    I = np.eye(Nt)
    m = int(Nt - num_off_ants)
    indices = np.where(~((z_mask == 1) & (z_sol == 0)))[0]
    indices_swap = list(range(m))
    for ai, bi in zip(indices, indices_swap):
        # 交换行
        I[[ai, bi], :] = I[[bi, ai], :]
    H_short = (H@I.T)[:, :m]
    A = cp.Variable((m, m), hermitian=True, name='A')
    rate = cp.real(cp.log_det(np.eye(Ns) + H_short@A@H_short.conj().T / sigma_sq**2))
    obj = cp.Maximize(rate)
    constraints = []
    constraints.append(A >> 0)
    constraints.append(cp.real(cp.trace(A)) <= power_comsumption)

    prob = cp.Problem(obj, constraints)

    # 尝试不同的求解器并输出详细信息
    solvers = [cp.MOSEK, cp.ECOS, cp.SCS, cp.OSQP]
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=False)
            break
        except cp.SolverError:
            print(f"Solver {solver} failed, trying next solver.")

    # A_sol = A.value
    Q = np.zeros((Nt, Nt)) + 1j * np.zeros((Nt, Nt))
    Q[:m, :m] = A.value
    Q_sol = I.T @ Q @ I
    # 检查解是否找到
    # np.linalg.det(H@Q_sol@H.conj().T)
    assert Q_sol.shape[0] == Nt, 'Q not of correct shape'
    W, Q_sol_1 = semi_definite_decomposition(Q_sol, Ns)
    threshold = 1e-8
    W[np.abs(W) < threshold] = 0
    return z_sol, W, Q_sol, prob.value, True


def semi_definite_decomposition(Q, k):
    """
    将半正定矩阵 Q 分解成 A 和 A.H，使得 Q = A @ A.H
    参数:
    Q : numpy.ndarray
        半正定矩阵
    返回:
    A : numpy.ndarray
        分解后的矩阵 A
    Q_reconstructed : numpy.ndarray
        通过 A @ A.H 重建的矩阵，用于验证
    """
    # 进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(Q)

    # 保留最大的 k 个非负特征值及其对应的特征向量
    idx = np.argsort(eigenvalues)[::-1]  # 从大到小排序特征值
    eigenvalues = eigenvalues[idx][:k]
    eigenvectors = eigenvectors[:, idx][:, :k]
    # 计算 A 矩阵
    A = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    # 验证 Q 是否等于 A @ A.H
    Q_reconstructed = A @ A.conj().T

    return A, Q_reconstructed


if __name__ == '__main__':
    Ns, Nt = 4, 8
    np.random.seed(30)
    H = (np.random.randn(Ns, Nt) + 1j * np.random.randn(Ns, Nt)) / np.sqrt(2)
    z_mask = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    z_sol1 = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    z_sol2 = np.array([0, 0, 0, 0, 1, 0, 0, 0])
    z_sol3 = np.array([0, 1, 0, 0, 1, 0, 0, 0])
    z_sol4 = np.array([0, 0, 0, 0, 1, 0, 1, 0])

    results = []
    for z_sol in [z_sol1, z_sol2, z_sol3, z_sol4]:
        results.append(solve_subASP(H=H, z_mask=z_mask, z_sol=z_sol))

    for result in results:
        z_sol, F, Q, obj, optimal = result
        print("Objective value:", obj)
        print("a_sol:", z_sol)
        # print("F", F)
        print("Q:", Q)
        # print("Optimal:", optimal)
        print()
