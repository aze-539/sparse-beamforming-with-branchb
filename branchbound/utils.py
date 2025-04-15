import numpy as np
from branchbound.solve_ASP import solve_ASP as ASP_solve
from branchbound.solve_subproblem import solve_subASP as subASP_solve


def post_process(instance,
                 solution,
                 max_ant: int,
                 sigma_sq: float = 1.0,
                 power_consumption: float = 10) -> float:
    """
    when the solution consists of larger number of antennas than max_ant,
    select the antennas from the solution that is assigned the highest power when all antennas are turned on
    """
    Ns, Nt = instance.shape
    # _, W_sol, Q_sol, obj, solved = ASP_solve(H=instance,
    #                                           z_mask=np.ones(Nt),
    #                                           z_sol=solution,
    #                                           sigma_sq=sigma_sq,
    #                                           power_comsumption=power_consumption)
    _, W_sol, Q_sol, obj, solved = subASP_solve(H=instance,
                                                 z_mask=np.ones(Nt),
                                                 z_sol=solution,
                                                 sigma_sq=sigma_sq,
                                                 power_comsumption=power_consumption)
    if not solved:
        return {'objective': None, 'solution': solution}
    w_energy = np.diag(Q_sol)
    ind = np.argpartition(w_energy, -max_ant)[-max_ant:]
    z_hat = np.zeros(solution.shape)
    z_hat[ind] = 1

    # _, W_sol, Q_sol, obj, solved = solve_DFRC(H=instance,
    #                                    z_mask=np.ones(Nt),
    #                                    z_sol=z_hat,
    #                                    sigma_sq=sigma_sq,
    #                                    power_comsumption=power_consumption)
    _, W_sol, Q_sol, obj, solved = solve_subDFRC(H=instance,
                                                 z_mask=np.ones(Nt),
                                                 z_sol=solution,
                                                 sigma_sq=sigma_sq,
                                                 power_comsumption=power_consumption)

    if solved:
        return {'objective': obj, 'solution': z_hat}
    else:
        return {'objective': None, 'solution': z_hat}
