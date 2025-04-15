import numpy as np
import os
import re
import time
import matplotlib.pyplot as plt

from branchbound.solve_ASP import *
from branchbound.solve_subproblem import *
from branchbound.solve_efficient import EfficientRelaxation
from branchbound.ob import Observation
DEBUG = False


class Node(object):
    def __init__(self, z_mask=None, z_sol=None, z_feas=None, W_sol=None, Q_sol=None,
                 U=False, L=False, depth=0, parent_node=None, node_index=0):
        '''
        Args:
            z_mask: vector of boolean, 1 means that the corresponding variable(antenna) is decided (= A U B in the paper)
            z_sol: value of z at the solution of the cvx relaxation (1 if in A and 0 otherwise)
            z_feas: value of z after making z_sol feasible (i.e. boolean with constraint satisfaction)
            U: current global upper bound
            L: current global lower bound
            depth: depth of the node from the root of the BB tree
            node_index: unique index assigned to the node in the BB tree
            parent_node: reference to the parent Node objet
            node_index: unique index to identify the node (and count them)
        TODO: This could have been a named tuple.
        '''
        self.z_mask = z_mask.copy()
        self.z_sol = z_sol.copy()
        self.z_feas = z_feas.copy()
        self.W_sol = W_sol.copy()
        self.Q_sol = Q_sol.copy()
        self.U = U
        self.L = L
        self.depth = depth
        self.parent_node = parent_node
        self.node_index = node_index

    def copy(self):
        Nt, Ns = self.W_sol.shape
        new_node = Node(z_mask=self.z_mask,
                        z_sol=self.z_sol,
                        z_feas=self.z_feas,
                        W_sol=self.W_sol,
                        Q_sol=self.Q_sol,
                        U=self.U,
                        L=self.L,
                        depth=self.depth,
                        parent_node=None,
                        node_index=self.node_index)
        return new_node


class DefaultBranchingPolicy(object):
    '''
    Default Branching Policy: This policy returns the antenna index from the unselected antennas with the maximum power assigned.
    This is currently using Observation object in order to extract the current solution and the decided antenna set.
    (change this to Node, so the code is readable and and insensitive to change in Obervation class)
    TODO: Convert it into a function as it no longer requires storing data for future computation.
    '''

    def __init__(self):
        pass

    def select_variable(self, observation, candidates):
        # Fetch W_sol, z_mask (= A U B in the paper)
        Nt, Ns = observation.antenna_features.shape[0], observation.variable_features.shape[0]
        W_sol = observation.edge_features[:, 6] + 1j * observation.edge_features[:, 7]
        W_sol = W_sol.reshape((Nt, Ns))

        z_mask = observation.antenna_features[:, 2]
        z_sol = observation.antenna_features[:, 0]

        power_w = np.linalg.norm(W_sol, axis=1)

        power_w = (1 - z_mask) * power_w
        return np.argmax(power_w)


class BBenv(object):
    def __init__(self, observation_function=Observation, node_select_policy_path='default', epsilon=0.001):
        '''
        Initializes a B&B environment.
        For solving several B&B problem instances, one needs to call the reset function with the problem instance parameters

        Args:
            observation_function: What kind of features to use.
            node_select_policy_path: one of {'default', 'oracle'}
                                     if the value is 'oracle', optimal solution should be provided in the reset function
            epsilon: The maximum gap between the global upper bound and global lower bound for the termination of the B&B algorithm.
        '''
        self._is_reset = None
        self.epsilon = epsilon  # stopping criterion
        self.H = None

        self.nodes = []  # list of problems (nodes)
        self.num_nodes = 0
        self.num_active_nodes = 0
        self.all_nodes = []  # list of all nodes to serve as training data for node selection policy
        self.optimal_nodes = []
        self.node_index_count = 0

        self.L_list = []  # list of lower bounds on the problem
        self.U_list = []  # list of upper bounds on the problem

        self.global_L = np.nan  # global lower bound
        self.global_U = np.nan  # global upper bound

        self.action_set_indices = None
        self.active_node = None  # current active node

        self.global_U_ind = None
        self.failed_reward = -2000

        self.node_select_model = None

        self.init_U = 999999
        self.node_select_policy = self.default_node_select

        self.z_incumbent = None
        self.W_incumbent = None
        self.Q_incumbent = None
        self.current_opt_node = None
        self.min_bound_gap = None

        if node_select_policy_path == 'default':
            self.node_select_policy = self.default_node_select
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = self.oracle_node_select

        self.observation_function = observation_function
        self.include_heuristic_solutions = False
        self.heuristic_solutions = []

        self.bm_solver = None

    def reset(self,
              instance,
              max_ant,
              oracle_opt=None,
              sigma_sq=1.0,
              power_comsumption=10):
        '''
        Solve new problem instance with given max_ant, sigma_sq, and power_comsumption
        '''
        # clear all variables
        self.H = None
        self.nodes = []  # list of problems (nodes)
        self.all_nodes = []
        self.optimal_nodes = []
        self.node_index_count = 0

        self.L_list = []  # list of lower bounds on the problem
        self.U_list = []  # list of upper bounds on the problem
        self.global_L = np.nan  # global lower bound
        self.global_U = np.nan  # global upper bound
        self.action_set_indices = None
        self.active_node = None
        self.global_U_ind = None
        self.num_nodes = 1
        self.H = instance
        self.sigma_sq = sigma_sq

        # EfficientRelaxation saves the solutions so that the same lower or upper bound problem is not solved twice
        self.bm_solver = EfficientRelaxation(H=self.H,
                                             sigma_sq=self.sigma_sq,
                                             power_consumption=power_comsumption)

        self.min_bound_gap = np.ones(self.H.shape[-1]) * 0.01

        self.max_ant = max_ant

        # number of transmitters and users
        self.Ns, self.Nt = self.H.shape
        self._is_reset = True
        self.action_set_indices = np.arange(1, self.Nt)

        z_mask = np.zeros(self.Nt)
        # values of z (selection var) at the z_mask locations
        # for the root node it does not matter
        z_sol = np.zeros(self.Nt)

        [z, W, Q, upper_bound, optimal] = self.bm_solver.solve_efficient(z_mask=z_mask, z_sol=z_sol)

        self.global_U = upper_bound
        self.z_incumbent = self.get_feasible_z(W_sol=W,
                                               Q_sol=Q,
                                               z_sol=z,
                                               z_mask=z_mask,
                                               max_ant=self.max_ant,
                                               U=upper_bound)

        [_, W_feas, Q_feas, self.global_L, optimal] = self.bm_solver.solve_efficient(z_mask=np.ones(self.Nt),
                                                                                     z_sol=self.z_incumbent)

        if not self.global_L == np.inf:
            self.W_incumbent = W_feas.copy()
            self.Q_incumbent = Q_feas.copy()
        else:
            self.W_incumbent = np.zeros((self.Nt, self.Ns))
            self.Q_incumbent = np.zeros((self.Nt, self.Nt))

        self.active_node = Node(z_mask=z_mask, z_sol=z, z_feas=self.z_incumbent, W_sol=W, Q_sol=Q, U=self.global_U,
                                L=self.global_L, depth=1, node_index=self.node_index_count)
        self.current_opt_node = self.active_node

        self.active_node_index = 0
        self.nodes.append(self.active_node)
        self.L_list.append(self.global_L)
        self.U_list.append(self.global_U)
        self.all_nodes.append(self.active_node)

        if oracle_opt is not None:
            self.oracle_opt = oracle_opt
        else:
            self.oracle_opt = np.zeros(self.Nt)

    def push_children(self, var_id, node_id, parallel=False):
        '''
        Creates two children and appends it to the node list. Also executes fathom condition.
        Args:
            var_id: selected variable (in our case, antenna) to branch on
            node_id: selected node to branch on
            parallel: whether to run the node computations in parallel
        '''
        self.delete_node(node_id)
        if var_id == None:
            return
        if sum(self.active_node.z_mask * self.active_node.z_sol) == self.max_ant:
            print('\n #####################')
            print('current node is already determined')
            print()
            return

        max_possible_ant = sum(self.active_node.z_mask * self.active_node.z_sol) + sum(1 - self.active_node.z_mask)
        if max_possible_ant < self.max_ant:
            # this condition should never occur (node would be infeasible, sum(z) != Ns)
            print('\n*******************')
            print('exception: max antenna possible < Ns')
            print(self.active_node.z_mask)
            print(self.active_node.z_sol)
            return

        elif max_possible_ant == self.max_ant:
            print('\n*******************')
            print('exception: max antenna possible = Ns')
            self.active_node.z_sol = self.active_node.z_mask * self.active_node.z_sol + (
                    1 - self.active_node.z_mask) * np.ones(self.Nt)
            self.active_node.z_mask = np.ones(self.Nt)
            return

        else:
            z_mask_left = self.active_node.z_mask.copy()
            z_mask_left[var_id] = 1

            z_mask_right = self.active_node.z_mask.copy()
            z_mask_right[var_id] = 1

            z_sol_left = self.active_node.z_sol.copy()
            z_sol_left[var_id] = 0

            z_sol_right = self.active_node.z_sol.copy()
            z_sol_right[var_id] = 1

            if sum(z_sol_right * z_mask_right) == self.max_ant:
                z_sol_right = z_sol_right * z_mask_right
                z_mask_right = np.ones(self.Nt)

        children_sets = []
        children_sets.append([z_mask_left.copy(), z_sol_left.copy()])
        children_sets.append([z_mask_right.copy(), z_sol_right.copy()])

        if DEBUG:
            print('expanding node id {}, children {}, lb {}, z_inc {}'.format(self.active_node.node_index, (
                self.active_node.z_mask, self.active_node.z_sol), self.active_node.U, self.z_incumbent))

        children_stats = []
        t1 = time.time()
        for subset in children_sets:
            if DEBUG:
                print('\n creating children {}'.format(subset))
            children_stats.append(self.create_children(subset))
        if DEBUG:
            print('time taken by loop {}'.format(time.time() - t1))

        for stat in children_stats:
            U, L, _, _, _, new_node = stat
            if new_node is not None:
                self.L_list.append(L)
                self.U_list.append(U)
                self.nodes.append(new_node)
                self.all_nodes.append(new_node)

        if len(self.nodes) == 0:
            if DEBUG:
                print('all nodes exhausted')
            return

        # Update the global upper and lower bound
        # update the incumbent solutions

        max_U_child = max([children_stats[i][0] for i in range(len(children_stats))])
        self.global_U = max(max(self.U_list), max_U_child)
        max_L_index = np.argmax([children_stats[i][1] for i in range(len(children_stats))])
        if self.global_L < children_stats[max_L_index][1]:
            # print('node depth at global U update {}'.format(self.active_node.depth + 1))
            self.global_L = children_stats[max_L_index][1]
            self.z_incumbent = children_stats[max_L_index][2].copy()
            self.W_incumbent = children_stats[max_L_index][3].copy()
            self.Q_incumbent = children_stats[max_L_index][4].copy()

    def create_children(self, constraint_set):
        '''
        Create the Node with the constraint set
        Compute the local lower and upper bounds
        return the computed bounds to the calling function to update
        '''
        z_mask, z_sol = constraint_set

        # check if the maximum number of antennas are already selected or all antennas are already assigned (z is fully assigned)
        if np.sum(z_mask * np.round(z_sol)) == self.max_ant or np.sum(z_mask * (1 - np.round(z_sol))) == len(
                z_mask) - self.max_ant:
            if np.sum(z_mask * np.round(z_sol)) == self.max_ant:
                z_sol = np.round(z_sol) * z_mask
            elif np.sum(z_mask * (1 - np.round(z_sol))) == len(z_mask) - self.max_ant:
                z_sol = np.round(z_sol) * z_mask + (1 - np.round(z_sol)) * (1 - z_mask)

            [_, W, Q, U, optimal] = self.bm_solver.solve_efficient(z_mask=np.ones(self.Nt), z_sol=z_sol)

            # check this constraint
            if not optimal:
                print('antennas: {} not optimal, may be infeasible'.format(None))
                return np.inf, np.inf, np.zeros(self.Nt), np.zeros(self.Nt), np.zeros(self.Nt), None

            assert U <= self.active_node.U + self.epsilon, 'selected antennas: upper bound of child node more than that of parent'

            z_feas = z_sol.copy()

            L = self.get_objective(Q=Q, z_feas=z_feas)

            # create and append node
            self.node_index_count += 1
            new_node = Node(z_mask=z_mask,
                            z_sol=z_sol,
                            z_feas=z_feas,
                            W_sol=W,
                            Q_sol=Q,
                            U=U,
                            L=L,
                            depth=self.active_node.depth + 1,
                            node_index=self.node_index_count
                            )
            return U, L, z_feas, W, Q, new_node

        elif np.sum(z_mask * np.round(z_sol)) > self.max_ant:
            return np.inf, np.inf, np.zeros(self.Nt), np.zeros(self.Nt), np.zeros(self.Nt), None

        else:
            [z, W, Q, U, optimal] = self.bm_solver.solve_efficient(z_sol=z_sol, z_mask=z_mask)

            if not optimal:
                if DEBUG:
                    print('relaxed: not optimal', z, U, optimal)
                else:
                    print('relaxed: not optimal, may be infeasible')
                return np.inf, np.inf, np.zeros(self.Nt), np.zeros(self.Nt), np.zeros(self.Nt), None

            if U > self.active_node.U + self.epsilon:
                print('relaxed: upper bound of child node more than that of parent')

            if not U == np.inf:
                z_feas = self.get_feasible_z(W_sol=W,
                                             Q_sol=Q,
                                             z_sol=z,
                                             z_mask=z_mask,
                                             max_ant=self.max_ant,
                                             U=self.active_node.U)

                [_, W_feas, Q_feas, L_feas, optimal] = self.bm_solver.solve_efficient(z_mask=np.ones(self.Nt),
                                                                                      z_sol=z_feas)

                if optimal:
                    L = self.get_objective(Q=Q_feas, z_feas=z_feas)
                    # L = L_feas
                else:
                    L = np.inf

                # create and append node
                self.node_index_count += 1
                new_node = Node(z_mask=z_mask,
                                z_sol=z,
                                z_feas=z_feas,
                                W_sol=W,
                                Q_sol=Q,
                                U=U,
                                L=L,
                                depth=self.active_node.depth + 1,
                                node_index=self.node_index_count
                                )

                return U, L, z_feas, W_feas, Q_feas, new_node

            else:
                return np.inf, np.inf, np.zeros(self.Nt), np.zeros(self.Nt), np.zeros(self.Nt), None

    def get_objective(self, Q, z_feas):
        Q_short = Q.copy()
        Q_short = Q_short * np.expand_dims(z_feas, 1)
        rate = np.real(np.log(np.linalg.det(np.eye(self.Ns) + self.H @ Q_short @ self.H.conj().T / self.sigma_sq ** 2)))
        return np.real(rate)

    def set_node_select_policy(self, node_select_policy_path='default'):
        '''
        what policy to use for node selection
        Args:
            node_select_policy_path: one of ('default', 'oracle', gnn_node_policy_parameters)
                                        'default' -> use the lowest lower bound first policy
                                        'oracle' -> select the optimal node (optimal solution should be provided in the reset function)
        '''
        if node_select_policy_path == 'default':
            self.node_select_policy = 'default'
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = 'oracle'

    def select_variable_default(self):
        '''
        Currently this method is not being used for default variable selection.
        '''
        z_sol_rel = (1 - self.active_node.z_mask) * (np.abs(self.active_node.z_sol - 0.5))
        return np.argmax(z_sol_rel)

    def select_node(self):
        '''
        Default node selection method
        TODO: the fathom method has been moved from here. So the loop is not needed
        '''
        node_id = 0
        node_id = self.rank_nodes()
        self.active_node = self.nodes[node_id]
        return node_id, self.observation_function().extract(self), self.is_optimal(self.active_node)

    def prune(self, observation):
        if self.node_select_policy == 'oracle':
            return not self.is_optimal(self.active_node)
        elif self.node_select_policy == 'default':
            return False

    def rank_nodes(self):
        return np.argmax(self.U_list)

    def fathom_nodes(self):
        del_ind = np.argwhere(np.array(self.L_list) > self.global_U + self.epsilon)
        if len(del_ind) > 0:
            del_ind = sorted(list(del_ind.squeeze(axis=1)))
            for i in reversed(del_ind):
                # print('fathomed nodes')
                self.delete_node(i)

    def fathom(self, node_id):
        if self.nodes[node_id].L > self.global_U:
            self.delete_node(node_id)
            return True
        return False

    def delete_node(self, node_id):
        del self.nodes[node_id]
        del self.L_list[node_id]
        del self.U_list[node_id]

    def is_optimal(self, node, oracle_opt=None):
        if oracle_opt is None:
            oracle = self.oracle_opt
        else:
            oracle = oracle_opt
        if np.linalg.norm(node.z_mask * (node.z_sol - oracle)) < 0.0001:
            return True
        else:
            return False

    def is_terminal(self):
        if (self.global_U - self.global_L) < self.epsilon:
            return True
        else:
            return False

    def default_node_select(self):
        '''
        Use the node with the lowest lower bound
        '''
        return np.argmax(self.U_list)

    def get_feasible_z(self, W_sol=None, Q_sol=None, z_mask=None, z_sol=None, max_ant=None, U=None):
        '''
        Selects the antennas that have been assigned the max obj in the solution of W
        '''
        obj = []
        for i in range(self.Nt):
            Q_short = Q_sol.copy()
            Q_short[i, :] = 0
            Q_short[:, i] = 0
            rate = np.real(
                np.log(np.linalg.det(np.eye(self.Ns) + self.H @ Q_short @ self.H.conj().T / self.sigma_sq ** 2)))
            obj.append(U - rate)

        obj = (1 - z_mask) * obj
        used_ant = int(np.sum(z_mask * z_sol))
        assert used_ant <= max_ant, 'used antennas already larger than max allowed antennas'
        if used_ant == max_ant:
            return z_mask * z_sol
        z_feas = z_mask * z_sol

        # test the effect of power_w
        # power_w = np.random.permutation(power_w)

        z_feas[np.flip(np.argsort(obj))[:max_ant - used_ant]] = 1
        return z_feas


def solve_bb(instance,
             max_ant=4,
             max_iter=10000,
             policy_type='default',
             power_consumption=10,
             sigma_sq=1.0):
    t1 = time.time()
    if policy_type == 'default':
        env = BBenv(observation_function=Observation, epsilon=0.01)
    elif policy_type == 'oracle':
        env = BBenv(observation_function=Observation, epsilon=0.01)
        pass

    branching_policy = DefaultBranchingPolicy()

    t1 = time.time()

    env.reset(instance,
              max_ant=max_ant,
              sigma_sq=sigma_sq,
              power_comsumption=power_consumption)
    timestep = 0
    done = False
    lb_list = []
    ub_list = []
    print('\ntimestep', timestep, env.global_U, env.global_L)
    while timestep < max_iter and len(env.nodes) > 0 and not done:

        env.fathom_nodes()
        if len(env.nodes) == 0:
            break
        node_id, node_feats, label = env.select_node()

        if len(env.nodes) == 0:
            break

        branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
        done = env.push_children(branching_var, node_id, parallel=False)
        timestep = timestep + 1
        lb_list.append(env.global_L)
        ub_list.append(env.global_U)
        print('\ntimestep: {}, global U: {}, global L: {}'.format(timestep, env.global_U, env.global_L))
        if env.is_terminal():
            break
    return (env.z_incumbent.copy(), env.global_U, env.W_incumbent,
            time.time() - t1, env.bm_solver.get_total_problems(), ub_list, lb_list)


def plot_two_lists(ub_list, lb_list, filename='plot.png', folder='experiment results'):
    # Ensure the save directory exists; create it if it doesn't
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Compute the x-axis values
    x1 = range(len(ub_list))

    # Plot the lines
    plt.plot(x1, ub_list, marker='o', color='red', label='upper bound')  # Red line for List 1
    plt.plot(x1, lb_list, marker='x', color='blue', label='lower bound')  # Blue line for List 2

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')

    # Show the legend
    plt.legend()

    # Construct the full file path
    file_path = os.path.join(folder, filename)

    # Save the figure to file
    plt.savefig(file_path)  # Save the figure to the specified folder and filename
    plt.grid(True)
    # Display the figure
    plt.show()


if __name__ == '__main__':
    result = []
    # random_seed_list = [2, 24, 43, 56, 65, 67, 76, 88, 90, 91, 92, 101, 110, 124, 138, 146, 148, 154, 155, 180, 193]
    # for i in range(10):
    np.random.seed(seed=70)
    train_size = [
        # [6, 3, 3],
        # [6, 3, 4],
        [8, 4, 4],
        # [8, 4, 5],
        # [10, 5, 5],
        # [10, 5, 6],
        # [12, 6, 6],
        # [12, 6, 7]
    ]
    power_consumption = 20
    sigma_sq = 1.0
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
            _, global_U, W, t, num_problems, ub_list, lb_list = solve_bb(instance,
                                                                         max_ant=max_ant,
                                                                         max_iter=10000,
                                                                         power_consumption=power_consumption,
                                                                         sigma_sq=sigma_sq)
            u_avg += global_U
            t_avg += t
            plot_two_lists(ub_list=ub_list, lb_list=lb_list, filename='bbconvengence.eps')
        # y = instance@W@x
        # result_list.append(global_U)
        # if max_ant == 4 or max_ant == 8:
        #     W_list.append(W)
        print('\nAverage global U: {} avg time: {}, avg num problems: {}'.format(u_avg, t_avg, num_problems))
    # result.append(result_list)
    # print(result_list)
    # average_result = np.mean(result, axis=0)
    # print("Average of result_list across all 10 runs:", average_result)

