'''
Implementation of the Greedy approach for joint beamforming and antenna selection.
This code works for both beamforming and robust beamforming problem.
General Procedure:
    R = list of all antennas
    Loop until (N-L) antennas are removed:
        1. Compute the objective for all possible combinations of |R|-1 antennas in R.
        2. Remove the antenna that results in maximum objective (power).
'''

import numpy as np
import pickle
import os
import time
from collections import namedtuple
from dataclasses import dataclass
from branchbound.ob import Observation
from branchbound.bb_unified import BBenv as Environment, DefaultBranchingPolicy, solve_bb
from branchbound.helper import SolverException
from branchbound.solve_ASP import solve_ASP as ASP_solve
from branchbound.solve_subproblem import solve_subASP as subASP_solve
from branchbound.utils import post_process

from collections import namedtuple


def greedy(H=None,
           max_ant=None,
           sigma_sq=1.0,
           power_comsumption=10,
           timeout=np.inf,
           max_problems=np.inf,
           random_post_process=False):
    # TODO: Unify the input type (take complex tensor of [N,M])
    start_time = time.time()
    Candidate = namedtuple('Candidate', 'selection, objective')

    Ns, Nt = H.shape
    selected_antennas = np.ones((Nt))
    num_problems = 0
    for i in range(Nt - max_ant):
        print('selecting {} antennas'.format(Nt - i))
        # Construct the antenna combintions and evaluate the cost
        candidates = []
        for j in range(Nt):
            if selected_antennas[j] == 0:
                continue

            # specify the mask
            z = selected_antennas.copy()
            z[j] = 0
            # z, W_sol, Q_sol, obj, solved = ASP_solve(H=self.H,
            #                                           z_mask=z_mask,
            #                                           z_sol=z_sol,
            #                                           sigma_sq=self.sigma_sq,
            #                                           power_comsumption=self.power_comsumption)
            z, W_sol, Q_sol, obj, solved = subASP_solve(H=H,
                                                        z_mask=np.ones(Nt),
                                                        z_sol=z,
                                                        sigma_sq=sigma_sq,
                                                        power_comsumption=power_comsumption)
            num_problems += 1
            if not solved:
                obj = np.inf
            candidates.append(Candidate(z, obj))
            if num_problems > max_problems or time.time() - start_time > timeout:
                break
        best_candidate = min(candidates, key=lambda x: x.objective)
        selected_antennas = best_candidate.selection
        best_objective = best_candidate.objective
        time_taken = time.time() - start_time
        if num_problems > max_problems or time.time() - start_time > timeout:
            break
    if int(np.sum(selected_antennas)) > max_ant:
        if not random_post_process:
            post_result = post_process(H,
                                       selected_antennas,
                                       max_ant=max_ant,
                                       sigma_sq=sigma_sq,
                                       power_consumption=power_comsumption)
            selected_antennas = post_result['solution']
        else:
            indices = np.where(selected_antennas == 1)[0]
            np.random.shuffle(indices)
            selected_antennas[indices[:int(np.sum(selected_antennas)) - max_ant]] = 0

    assert int(np.sum(selected_antennas)) == max_ant, "Number of selected antennas {} != {}".format(
        int(np.sum(selected_antennas)), max_ant)
    # _, _, _, best_objective, solved = DFRC_solve(H=self.H,
    #                                              z_mask=z_mask,
    #                                              z_sol=z_sol,
    #                                              sigma_sq=self.sigma_sq,
    #                                              power_comsumption=self.power_comsumption)
    _, W, _, best_objective, solved = subASP_solve(H=H,
                                                   z_mask=np.ones(Nt),
                                                   z_sol=selected_antennas,
                                                   sigma_sq=sigma_sq,
                                                   power_comsumption=power_comsumption)

    # return {'optimal W': W, 'objective': best_objective, 'solution': selected_antennas, 'num_problems': num_problems,
    #         'time': time_taken}
    return (W, best_objective)


if __name__ == '__main__':
    # np.random.seed(seed=2)
    result = []
    for i in range(10):
        np.random.seed(seed=10*i)
        train_size = [
            [8, 4, 1],
            [8, 4, 2],
            [8, 4, 3],
            [8, 4, 4],
            [8, 4, 5],
            [8, 4, 6],
            [8, 4, 7],
            [8, 4, 8]
        ]
        sigma_sq = 1.0
        power_comsumption = 20
        result_list = []
        W_list = []
        for item in train_size:
            Nt, Ns, max_ant = item
            u_avg = 0
            t_avg = 0
            tstep_avg = 0
            # x = np.random.normal(0, 1, (1, Ns)).T
            for i in range(1):
                instance = (np.random.randn(Ns, Nt) + 1j * np.random.randn(Ns, Nt)) / np.sqrt(2)

                greedy_output_random_W, greedy_output_random_obj = greedy(H=instance,
                                              max_ant=max_ant,
                                              sigma_sq=sigma_sq,
                                              power_comsumption=power_comsumption,
                                              timeout=2,
                                              random_post_process=True
                                              )

                # greedy_output_largest_W, greedy_output_largest_obj = greedy(H=instance,
                #                                                             max_ant=max_ant,
                #                                                             sigma_sq=sigma_sq,
                #                                                             power_comsumption=power_comsumption,
                #                                                             timeout=2,
                #                                                             random_post_process=False
                #                                                             )
                result_list.append(greedy_output_random_obj)
                # if max_ant == 4 or max_ant == 8:
                #     W_list.append(greedy_output_random_W)
                # print('greedy random', greedy_output_random)
                # print('greedy largest', greedy_output_largest_W, greedy_output_largest_obj)
        result.append(result_list)
        print(result_list)
        # print(W_list)
        print('---------------------------')
    average_result = np.mean(result, axis=0)
    print("Average of result_list across all 10 runs:", average_result)
