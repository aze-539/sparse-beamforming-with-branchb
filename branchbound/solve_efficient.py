"""
Wrapper for solve_relaxation module. This module implements saving results for all the problems ever solved so that redundant computations can be avoided
"""
from branchbound.solve_ASP import solve_ASP as ASP_solve
from branchbound.solve_subproblem import solve_subASP as subASP_solve
import numpy as np


class EfficientRelaxation:
    def __init__(self,
                 H=None,
                 sigma_sq=1.0,
                 power_consumption=10):
        self.H = H.copy()
        self.power_comsumption =power_consumption
        self.sigma_sq = sigma_sq
        self.data = {}
        self.data['node'] = []
        self.data['solution'] = []
        self.num_problems = 0
        self.num_unique_problems = 0

    def _save_solutions(self, z_mask=None,
                        z_sol=None,
                        z_result=None,
                        W_sol=None,
                        Q_sol=None,
                        obj=None,
                        optimal=None):
        """
        Stores the solutions in RAM as a dictionary

        Does not save duplicate solutions. For example if the node is already present in the data, it does not store.
        """
        assert z_mask is not None and z_sol is not None, "Save solutions: one of the input is None"
        assert len(self.data['node']) == len(self.data['solution'])
        self.data['node'].append((z_mask.copy(), z_sol.copy()))
        self.data['solution'].append((z_result.copy(), W_sol.copy(), Q_sol.copy(), obj, optimal))

    def print_nodes(self):
        for item in self.data['node']:
            print(item[0] * (1 - item[1]))

    @staticmethod
    def _compare_nodes(z_mask_query, z_mask, z_sol_query, z_sol):
        """
        returns True if the two nodes are equivalent
        """
        # ensure that they are integral
        if np.sum(
                z_mask * (1 - z_mask_query)) == 0:  # check if z_mask is a subset of z_mask_query (A \subseteq A_query)
            if np.sum(np.abs(z_sol_query * z_mask - z_sol * z_mask)) == 0:  # on set A values of y should be the same
                remaining_antennas = z_mask_query - z_mask
                if np.sum(z_sol_query * remaining_antennas) == np.sum(
                        remaining_antennas):  # values of y_query should be 1 in the set A_query\A
                    return True
        return False

    def solve_efficient(self, z_mask=None,
                        z_sol=None):
        '''
        Wrapper for solving the relaxed problems for DFRC-beamforming
        First checks whether an equivalent node problem has already been solved.
        If so, it returns the stored solution, otherwise, it computes the new solution.
        '''
        assert z_mask is not None and z_sol is not None, "Solve efficient: one of the input is None"

        self.num_problems += 1
        for i in range(len(self.data['node'])):
            if self._compare_nodes(z_mask.copy(), self.data['node'][i][0], z_sol.copy(), self.data['node'][i][1]):
                return z_sol.copy(), self.data['solution'][i][1], self.data['solution'][i][2], \
                       self.data['solution'][i][3], self.data['solution'][i][4]

        self.num_unique_problems += 1
        # z, W, Q, obj, optimality = ASP_solve(H=self.H,
        #                                       z_mask=z_mask,
        #                                       z_sol=z_sol,
        #                                       sigma_sq=self.sigma_sq,
        #                                       power_comsumption=self.power_comsumption)
        z, W, Q, obj, optimality = subASP_solve(H=self.H,
                                                 z_mask=z_mask,
                                                 z_sol=z_sol,
                                                 sigma_sq=self.sigma_sq,
                                                 power_comsumption=self.power_comsumption)
        self._save_solutions(z_mask=z_mask.copy(),
                             z_sol=z_sol.copy(),
                             z_result=z.copy(),
                             W_sol=W.copy(),
                             Q_sol=Q.copy(),
                             obj=obj,
                             optimal=optimality)
        return z, W, Q, obj, optimality


    def get_total_problems(self):
        return self.num_unique_problems
