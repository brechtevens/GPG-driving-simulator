import numpy as np
import feature
import casadi.casadi as cs


class Trajectory(object):
    """
    A class used to represent general trajectory objects for dynamical systems

    Attributes
    ----------
    dyn : Dynamics object
        the dynamics of the trajectory
    N : int
        the length of the trajectory
    x0: list
        the current state of the trajectory
    u: list
        the control sequence along the trajectory
    """
    def __init__(self, N, dyn):
        self.dyn = dyn
        self.N = N
        self.x0 = [0] * dyn.nx
        self.u = [0] * (dyn.nu * N)

    def get_future_trajectory(self):
        """ Returns the future states of the trajectory for the current control sequence """
        trajectory = []
        z = self.x0
        for k in range(self.N):
            z = self.dyn(z, self.u[k * self.dyn.nu:(k + 1) * self.dyn.nu])
            trajectory.append(z)
        return trajectory

    def get_future_trajectory_given_u(self, u):
        """ Returns the future states of the trajectory for the given control sequence

        Parameters
        ----------
        u : CasADi SX or MX
            the given control sequence
        """
        trajectory = []
        z = self.x0
        for k in range(u.shape[0]//self.dyn.nu):
            z = self.dyn(z, u[k * self.dyn.nu:(k + 1) * self.dyn.nu])
            trajectory.append(z)
        return trajectory

    def quadratic_following_reward(self, d_des, target_vehicle):
        """ Returns a cost feature rewarding driving at a certain distance from a preceding vehicle

        Parameters
        ----------
        d_des : float
            the desired headway distance
        target_vehicle : Car object
            the preceding target vehicle
        """
        lr_front = target_vehicle.dyn.lr
        lf_back = self.dyn.lf
        headway = feature.headway(False, lr_front, lf_back)

        @feature.feature
        def f(x, u, x_other):
            return -(d_des - headway(x, u, x_other[target_vehicle.id])) ** 2
        return f

    def reward(self, stage_reward, x_robot, u_robot, x_humans, terminal_reward=None):
        """ Returns the total reward along the entire trajectory for an ego vehicle

        Parameters
        ----------
        stage_reward : Feature
            the stage reward of the ego vehicle
        x_robot : list
            the state variables of the ego vehicle along the trajectory
        u_robot : list
            the control variables of the ego vehicle along the trajectory
        x_humans : dict
            the state variables of other vehicles required to evaluate the reward along the trajectory
        terminal_reward : Feature, optional
            the terminal reward
        """
        reward = 0
        for k in range(self.N):
            x_k_humans = {}
            for i, value in x_humans.items():
                x_k_humans[i] = value[k]
            reward += stage_reward(x_robot[k], u_robot[k * self.dyn.nu:(k + 1) * self.dyn.nu], x_k_humans)
            if terminal_reward is not None and k == self.N-1:
                reward += terminal_reward(x_robot[k], u_robot[k * self.dyn.nu:(k + 1) * self.dyn.nu], x_k_humans)
        return reward

    def constraints(self, stage_constraints, x, u):
        """ Returns the constraints along the entire trajectory for an ego vehicle

        Parameters
        ----------
        stage_constraints : Constraints
            the stage constraints of the ego vehicle
        x : dict
            the state variables of all vehicles required to evaluate the constraints along the trajectory
        u: dict
            the control variables of all vehicles required to evaluate the constraints along the trajectory
        """
        if stage_constraints.length == 0:
            return np.array([])
        constraint_list = []
        for k in range(self.N):
            x_k = {}
            u_k = {}
            for i, value in x.items():
                x_k[i] = value[k]
            for i, value in u.items():
                u_k[i] = value[k]
            constraint_list.extend(stage_constraints(x_k, u_k))
        return constraint_list

    def constraints_dual_formulation(self, dual_formulation, x, symbolics):
        """ Returns the dual constraints along the entire trajectory for an ego vehicle

        Parameters
        ----------
        dual_formulation : Constraints
            the stage dual constraints of the ego vehicle
        x : dict
            the state variables of all vehicles required to evaluate the constraints along the trajectory
        symbolics: cs.SX or cs.MX attribute
            allows to generate additional SX or MX symbolics
        """
        lam = symbolics.sym('lambda', 4, 1)
        mu = symbolics.sym('mu', 4, 1)
        x_0 = {}
        for i, value in x.items():
            x_0[i] = value[0]
        if len(dual_formulation(x_0, lam, mu)) == 0:
            return np.array([]), np.array([]), symbolics()
        inequality_constraint_list = []
        equality_constraint_list = []
        optimization_parameters = symbolics()
        for k in range(self.N):
            lam = symbolics.sym('lambda_' + str(k), 4, 1)
            mu = symbolics.sym('mu_' + str(k), 4, 1)
            x_k = {}
            for i, value in x.items():
                x_k[i] = value[k]
            g, h = dual_formulation(x_k, lam, mu)
            equality_constraint_list.extend(g)
            inequality_constraint_list.extend(h)
            optimization_parameters = cs.vertcat(optimization_parameters, lam, mu)
        return equality_constraint_list, inequality_constraint_list, optimization_parameters
