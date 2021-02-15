import numpy as np
import gaussseidelsolver
import learning
import lagrangiansolver
from penalty import PenaltyParameters
from trajectory import Trajectory
import casadi.casadi as cs
from collections import deque
import time
import settings


class Car(object):
    """
    A class used to represent general car objects

    Attributes
    ----------
    id : int
        the id of the vehicle
    reset_x0 : list
        the initial state of the vehicle
    N : int
        the control horizon of the vehicle
    dyn : Dynamics object
        the dynamics of the vehicle
    traj : Trajectory object
        the trajectory object of the vehicle
    color : str
        the color of the vehicle
    default_u : list
        the default control sequence applied by the vehicle

    Properties
    ----------
    x : list
        the current state of the vehicle
    center_x : list
        the current position of the center of the vehicle
    u : list
        the current control action of the vehicle
    corners : list
        list of the current position of the four corners of the vehicle
    lr : float
        the length between the mass center and the rear end
    lf : float
        the length between the mass center and the front end
    len : float
        the length of the vehicle
    width : float
        the width of the vehicle
    u_2D : list
        the current control action of the vehicle, given by [acceleration, steering angle]
    """
    def __init__(self, car_dyn, x0, N, id, color):
        """
        Parameters
        ----------
        car_dyn : Dynamics object
            the dynamics of the vehicle
        x0 : list
            the initial state of the vehicle
        N : int
            the control horizon of the vehicle
        id : int
            the id of the vehicle
        color : str
            the color of the vehicle
        """
        self.id = id
        self.reset_x0 = x0
        self.N = N
        self.dyn = car_dyn
        self.traj = Trajectory(N, car_dyn)
        self.traj.x0 = x0
        self.color = color
        self.default_u = [0] * self.dyn.nu * self.N

    def reset(self):
        """ Resets the initial state and the control actions of the vehicle """
        self.traj.x0 = self.reset_x0
        self.traj.u = self.default_u

    def move(self):
        """ Moves the vehicle by one time step """
        self.traj.x0 = self.dyn(self.traj.x0, self.traj.u)

    @property
    def x(self):
        return self.traj.x0

    @property
    def center_x(self):
        return self.center_x(self.x)

    def center_x(self, current_x):
        """ Returns the center of the vehicle, given the current state of the vehicle.

        Parameters
        ----------
        current_x : list
            the current state of the vehicle
         """
        center_x = cs.vertcat(current_x[0] + (1 / 2) * (self.dyn.lf - self.dyn.lr) * cs.cos(current_x[2]),
                              current_x[1] + (1 / 2) * (self.dyn.lf - self.dyn.lr) * cs.sin(current_x[2]),
                              current_x[2],
                              current_x[3])
        return center_x

    @property
    def corners(self):
        return self.corners(self.x)

    def corners(self, current_x):
        """ Returns the four corners of the vehicle, given the current state of the vehicle.

        Parameters
        ----------
        current_x : list
            the current state of the vehicle
         """
        if isinstance(current_x[0], cs.SX) or isinstance(current_x[0], cs.MX):
            four_corners = [cs.vertcat(current_x[0] + self.lf * np.cos(current_x[2]) - self.width / 2. * np.sin(current_x[2]), current_x[1] + self.lf * np.sin(current_x[2]) + self.width / 2. * np.cos(current_x[2])),
                       cs.vertcat(current_x[0] + self.lf * np.cos(current_x[2]) + self.width / 2. * np.sin(current_x[2]), current_x[1] + self.lf * np.sin(current_x[2]) - self.width / 2. * np.cos(current_x[2])),
                       cs.vertcat(current_x[0] - self.lr * np.cos(current_x[2]) + self.width / 2. * np.sin(current_x[2]), current_x[1] - self.lr * np.sin(current_x[2]) - self.width / 2. * np.cos(current_x[2])),
                       cs.vertcat(current_x[0] - self.lr * np.cos(current_x[2]) - self.width / 2. * np.sin(current_x[2]), current_x[1] - self.lr * np.sin(current_x[2]) + self.width / 2. * np.cos(current_x[2]))]
        else:
            four_corners = [[current_x[0] + self.lf * np.cos(current_x[2]) - self.width / 2. * np.sin(current_x[2]), current_x[1] + self.lf * np.sin(current_x[2]) + self.width / 2. * np.cos(current_x[2])],
                       [current_x[0] + self.lf * np.cos(current_x[2]) + self.width / 2. * np.sin(current_x[2]), current_x[1] + self.lf * np.sin(current_x[2]) - self.width / 2. * np.cos(current_x[2])],
                       [current_x[0] - self.lr * np.cos(current_x[2]) + self.width / 2. * np.sin(current_x[2]), current_x[1] - self.lr * np.sin(current_x[2]) - self.width / 2. * np.cos(current_x[2])],
                       [current_x[0] - self.lr * np.cos(current_x[2]) - self.width / 2. * np.sin(current_x[2]), current_x[1] - self.lr * np.sin(current_x[2]) + self.width / 2. * np.cos(current_x[2])]]
        return four_corners

    @property
    def lr(self):
        return self.dyn.lr

    @property
    def lf(self):
        return self.dyn.lf

    @property
    def len(self):
        return self.dyn.lf + self.dyn.lr

    @property
    def width(self):
        return self.dyn.width

    @property
    def u(self):
        return self.traj.u[0:self.dyn.nu]

    @u.setter
    def u(self, value):
        self.traj.u[0:self.dyn.nu] = value

    @property
    def u_2D(self):
        if self.dyn.nu == 1:
            return cs.vertcat(self.traj.u[0:self.dyn.nu], 0)
        else:
            return self.traj.u[0:self.dyn.nu]

    def control(self, steer, gas):
        pass


class UserControlledCar(Car):
    """
        A class used to represent a user controlled vehicle

        Attributes
        ----------
        _fixed_control : list
            the fixed control action of the vehicle

        Properties
        ----------
        fixed_control : list
            the fixed control action of the vehicle
        """
    def __init__(self, *args, **vargs):
        """
        Parameters
        ----------
        *args : arguments for Car object
        **vargs : optional arguments for Car object
        """
        Car.__init__(self, *args, **vargs)
        self._fixed_control = None

    @property
    def fixed_control(self):
        return self._fixed_control

    def fix_control(self, ctrl):
        """ Sets the value of the fixed control of the vehicle

        Parameters
        ----------
        ctrl : list
            the fixed control action of the vehicle
        """
        self._fixed_control = ctrl

    def control(self, steer, gas):
        """ Sets the value of the control action of the vehicle

        Parameters
        ----------
        steer : float
            the current steering angle of the vehicle
        gas: float
            the current acceleration of the vehicle
        """
        if self._fixed_control is not None:
            self.u = self._fixed_control
        if self.dyn.nu == 1:
            self.u = [gas]
        elif self.dyn.nu == 2:
            self.u = [steer, gas]


class GPGOptimizerCar(Car):
    """
    A class used to represent a vehicle solving a GPG formulation

    Properties
    ----------
    reward : Feature object
        the reward of the vehicle
    terminal_reward
        the terminal reward of the vehicle
    mode : str
        the shooting mode of the solver, i.e. 'single' or 'multiple'
    solver : str
        the solver for for inner problems in the decomposition method, i.e. 'ipopt' or 'OpEn' (or 'qpoases')
    symbolics_type : str
        the symbolics type for the problem, i.e. 'SX' or 'MX'
    """
    def __init__(self, *args, gpg_solver_settings=None, online_learning_settings=None):
        """
        Parameters
        ----------
        gpg_solver_settings : GPGSolverSettings object, optional
            the settings for the GPG solver
        online_learning_settings : OnlineLearningSettings object, optional
            the online learning settings
        """
        Car.__init__(self, *args)

        # Solver settings
        if gpg_solver_settings is None:
            gpg_solver_settings = settings.GPGSolverSettings()
        if online_learning_settings is None:
            online_learning_settings = settings.OnlineLearningSettings()
        self.online_learning_settings = online_learning_settings
        self.gpg_solver_settings = gpg_solver_settings

        # CasADi variable type
        self._sym = getattr(cs, 'SX')

        # Initialize optimizer and observer
        self.optimizer = None
        self.observer = None

        # Optimization problem variables
        self._reward = None
        self._terminal_reward = None
        self.stage_g = []
        self.player_stage_g = {self.id: []}
        self.stage_h = []
        self.player_stage_h = {self.id: []}
        self.player_terminal_h = {self.id: []}
        self._stage_dual = []
        self.soft_stage_g = []
        self.soft_stage_h = []
        self._soft_stage_dual = []
        self.terminal_h = []
        self.soft_terminal_h = []
        self.g = []
        self.h = []
        self.player_g = {self.id: []}
        self.player_h = {self.id: []}
        self.v = self._sym()
        self.lbv = []
        self.ubv = []
        self.z = self._sym()
        self.lbz = []
        self.ubz = []
        self.h_original = []
        self.g_original = []
        self.x0_dict = {}
        self.u_dict = {}
        self.x_dict = {}
        self.reward_dict = {}
        self.reward_dict_original = {}
        self.g_dyn_dict = None
        self.x0_numeric_dict = {}
        self.p_numeric_dict = {}
        self.p_dict = {}
        self.optimum_v = self._sym()
        self.nb_bounded = {}
        self.penalty_parameters = PenaltyParameters(self._sym, self.N, gpg_solver_settings)
        self.dyn_dict = {self.id: self.dyn}

        # Humans data
        self.humans = {}
        self.human_trajectories = {}
        self._human_rewards = {}
        self._human_reward_params = {}
        self._human_reward_params_initial_belief = {}
        self._human_reward_params_current_belief = {}
        self._human_terminal_rewards = {}

        # Obstacles data
        self.obstacles = {}
        self.obstacle_trajectories = {}

        # Variables for storing obtained optimal values
        self.optimum_penalty_parameters = cs.DM()
        self.optimum_u_dict = {}
        self.optimum_lam = {}
        self.optimum_lam_g = {}

        # Solution times
        self.gpg_solution_time = 0
        self.observer_solution_time = 0

        # Deque for storing observations
        self._observations = deque([], online_learning_settings.nb_observations)

    def reset(self):
        Car.reset(self)

        # Reset initial parameter beliefs
        for id, value in self._human_reward_params_initial_belief.items():
            self._human_reward_params_current_belief[id] = value

        # Reset observations deque
        self._observations = deque([], self.online_learning_settings.nb_observations)

        # Reset variables for storing obtained optimal values
        self.optimum_penalty_parameters = cs.DM()
        self.optimum_u_dict = {}
        self.optimum_lam = {}
        self.optimum_lam_g = {}

        # Reset controller and observer
        if self.optimizer is not None:
            self.optimizer.reset()
        if self.observer is not None:
            self.observer.reset()

    def move(self):
        """ Moves the vehicle by one time step """
        Car.move(self)

    def get_human(self, i):
        """ Returns the human with the corresponding id

        Parameters
        ----------
        i : int
            the identifier of the human
        """
        return self.humans[i]

    def add_human(self, human, human_reward, human_terminal_reward=None, params=None, param_values=None):
        """ Adds a 'human', i.e. another GPGOptimizerCar, to the GPG formulation

        Parameters
        ----------
        human : GPGOptimizerCar object
            the added vehicle
        human_reward : Feature
            the stage reward of the added vehicle
        human_terminal_reward : Feature, optional
            the terminal reward of the added vehicle
        params : cs.SX or cs.MX, optional
            the cost function and constraint parameters
        param_values : list, optional
            the initial estimate for the parameters
        """
        self.humans[human.id] = human
        self.dyn_dict[human.id] = human.dyn
        self.human_trajectories[human.id] = Trajectory(self.N, human.dyn)
        self.player_stage_h[human.id] = []
        self.player_terminal_h[human.id] = []
        self.player_g[human.id] = []
        self.player_h[human.id] = []
        self._human_rewards[human.id] = human_reward
        self._human_terminal_rewards[human.id] = human_terminal_reward
        if params is not None and param_values is not None:
            self._human_reward_params[human.id] = params
            self._human_reward_params_initial_belief[human.id] = param_values
            self._human_reward_params_current_belief[human.id] = param_values
        elif params is not None or param_values is not None:
            raise Exception('Arguments not properly initialised for add_human()')

    def get_current_belief(self):
        """ Returns the dictionary of all the parameter estimates of the human parameters """
        return self._human_reward_params_current_belief

    def add_player_specific_g(self, i, g):
        """ Adds a player-specific stage equality constraint to the GPG formulation

        Parameters
        ----------
        i : int
            the identifier of the player
        g : Constraints object
            the equality constraints
        """
        self.player_stage_g[i].append(g)
        return

    def add_g(self, g):
        """ Adds a shared stage equality constraint to the GPG formulation

        Parameters
        ----------
        g : Constraints object
            the shared equality constraints
        """
        if self.gpg_solver_settings.constraint_mode == 'hard':
            self.stage_g.append(g)
        elif self.gpg_solver_settings.constraint_mode == 'soft':
            self.soft_stage_g.append(g)
        else:
            raise Exception('Given constraint mode is unknown: ' + str(self.gpg_solver_settings.constraint_mode))
        return

    def add_player_specific_h(self, i, h):
        """ Adds a player-specific stage inequality constraint to the GPG formulation

        Parameters
        ----------
        i : int
            the identifier of the player
        h : Constraints object
            the inequality constraints
        """
        self.player_stage_h[i].append(h)
        return

    def add_h(self, h):
        """ Adds a shared stage inequality constraint to the GPG formulation

        Parameters
        ----------
        h : Constraints object
            the shared inequality constraints
        """
        if self.gpg_solver_settings.constraint_mode == 'hard':
            self.stage_h.append(h)
        elif self.gpg_solver_settings.constraint_mode == 'soft':
            self.soft_stage_h.append(h)
        else:
            raise Exception('Given constraint mode is unknown: ' + str(self.gpg_solver_settings.constraint_mode))
        return

    def add_player_specific_terminal_h(self, i, h):
        """ Adds a player-specific terminal inequality constraint to the GPG formulation

        Parameters
        ----------
        i : int
            the identifier of the player
        h : Constraints object
            the inequality constraints
        """
        self.player_terminal_h[i].append(h)
        return

    def add_terminal_h(self, h):
        """ Adds a shared terminal inequality constraint to the GPG formulation

        Parameters
        ----------
        h : Constraints object
            the shared inequality constraints
        """
        if self.gpg_solver_settings.constraint_mode == 'hard':
            self.terminal_h.append(h)
        elif self.gpg_solver_settings.constraint_mode == 'soft':
            self.soft_terminal_h.append(h)
        else:
            raise Exception('Given constraint mode is unknown: ' + str(self.gpg_solver_settings.constraint_mode))
        return

    def add_dual(self, dual):
        """ Adds shared stage dual constraints to the GPG formulation

        Parameters
        ----------
        dual : Constraints object
            the shared dual constraints
        """
        if self.gpg_solver_settings.constraint_mode == 'hard':
            self._stage_dual.append(dual)
        elif self.gpg_solver_settings.constraint_mode == 'soft':
            self._soft_stage_dual.append(dual)
        else:
            raise Exception('Given constraint mode is unknown: ' + str(self.gpg_solver_settings.constraint_mode))
        return

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val
        self.optimizer = None

    def get_current_reward(self, x, u, x_dict):
        try:
            return cs.substitute(self._reward(x, u, x_dict), self.p_dict[self.id], self.p_numeric_dict[self.id])
        except KeyError:
            return self._reward(x, u, x_dict)

    @property
    def terminal_reward(self):
        return self._terminal_reward

    @terminal_reward.setter
    def terminal_reward(self, val):
        self._terminal_reward = val
        self.optimizer = None

    def set_ego_params(self, params, param_values):
        if params is not None:
            self.p_dict[self.id] = params
            self.p_numeric_dict[self.id] = param_values

    def add_obstacle(self, obstacle):
        """ Adds an 'obstacle', i.e. another Car object with e.g. fixed motion to the GPG formulation

        Parameters
        ----------
        obstacle : Car object
            the added vehicle
        """
        self.obstacles[obstacle.id] = obstacle
        self.obstacle_trajectories[obstacle.id] = Trajectory(self.N, obstacle.dyn)
        self.obstacle_trajectories[obstacle.id].x0 = obstacle.x
        self.obstacle_trajectories[obstacle.id].u = obstacle.fixed_control * self.N
        return

    @property
    def mode(self):
        return self.gpg_solver_settings.shooting_mode

    @property
    def solver(self):
        return self.gpg_solver_settings.solver

    @property
    def symbolics_type(self):
        return self._sym.type_name()

    @symbolics_type.setter
    def symbolics_type(self, val):
        assert(val == 'SX' or val == 'MX')
        self._sym = getattr(cs, val)

    def setup_x0_and_u(self):
        """ Initializes the CasADi symbolics for the initial states and control variables of the players in the GPG """
        self.x0_dict = {self.id: self._sym.sym('x0_r_id' + str(self.id), self.dyn.nx)}
        self.u_dict = {self.id: self._sym.sym('u_r_id' + str(self.id), self.dyn.nu * self.N)}
        for i in self.humans:
            self.x0_dict[i] = self._sym.sym('x0_h_id' + str(i), self.humans[i].dyn.nx)
            self.u_dict[i] = self._sym.sym('u_h_id' + str(i), self.humans[i].dyn.nu * self.N)
        for i in self.obstacles:
            self.x0_dict[i] = self._sym.sym('x0_h_id' + str(i), self.obstacles[i].dyn.nx)
            self.u_dict[i] = self._sym.sym('u_h_id' + str(i), self.obstacles[i].dyn.nu * self.N)
        return

    def x_single_shooting(self, car_dyn, x0, u):
        """ Returns a list of the state variables along a trajectory given initial state and control variables

        Parameters
        ----------
        car_dyn : Dynamics object
            the dynamics of the regarded vehicle
        x0 : cs.SX or cs.MX
            the initial state
        u : cs.SX or cs.MX
            the control variables
        """
        x = []
        z = x0
        for k in range(self.N):
            z = car_dyn(z, u[k * car_dyn.nu:(k + 1) * car_dyn.nu])
            x.append(z)
        return x

    def x_multiple_shooting(self, car_dyn, i):
        """ Returns a list of the state variables along a trajectory

        Parameters
        ----------
        car_dyn : Dynamics object
            the dynamics of the regarded vehicle
        i : int
            the identifier of the vehicle
        """
        x = []
        for k in range(self.N):
            z = []
            for state in range(car_dyn.nx):
                z.append(self._sym.sym('x_id' + str(i) + '_t' + str(k + 1) + '_' + str(state), 1))
            x.append(z)
        return x

    def setup_x(self):
        """ Initializes list of state variables for the players in the GPG """
        if self.mode == "single":
            self.x_dict = {self.id: self.x_single_shooting(self.dyn, self.x0_dict[self.id], self.u_dict[self.id])}
            for i in self.humans:
                self.x_dict[i] = self.x_single_shooting(self.humans[i].dyn, self.x0_dict[i], self.u_dict[i])
            for i in self.obstacles:
                self.x_dict[i] = self.x_single_shooting(self.obstacles[i].dyn, self.x0_dict[i], self.u_dict[i])
        else:
            self.x_dict = {self.id: self.x_multiple_shooting(self.dyn, self.id)}
            for i in self.humans:
                self.x_dict[i] = self.x_multiple_shooting(self.humans[i].dyn, i)
            for i in self.obstacles:
                self.x_dict[i] = self.x_single_shooting(self.obstacles[i].dyn, self.x0_dict[i], self.u_dict[i])
            self.setup_dynamics_constraints_multiple_shooting()
        return

    def dynamics_constraints_multiple_shooting(self, car_dyn, x0, u, x):
        """ Returns a list of CasADi expressions corresponding the the dynamics of the vehicle, i.e. equality constraints

        Parameters
        ----------
        car_dyn : Dynamics object
            the dynamics of the regarded vehicle
        x0 : cs.SX or cs.MX
            the initial state
        u : cs.SX or cs.MX
            the control variables
        x : cs.SX or cs.MX
            the state variables along the trajectory
        """
        dynamics_constraints = [x[0][i] - car_dyn(x0, u[0:car_dyn.nu])[i] for i in range(car_dyn.nx)]
        for k in range(1, self.N):
            dynamics_constraints += [x[k][i] - car_dyn(x[k - 1], u[k * car_dyn.nu:(k + 1) * car_dyn.nu])[i]
                                     for i in range(car_dyn.nx)]
        return dynamics_constraints

    def setup_dynamics_constraints_multiple_shooting(self):
        """ Initializes the dynamics constraints for the players in the GPG (only for multiple shooting) """
        self.g_dyn_dict = {self.id: self.dynamics_constraints_multiple_shooting(self.dyn, self.x0_dict[self.id], self.u_dict[self.id], self.x_dict[self.id])}
        for i in self.humans:
            self.g_dyn_dict[i] = self.dynamics_constraints_multiple_shooting(self.humans[i].dyn, self.x0_dict[i], self.u_dict[i], self.x_dict[i])
        return

    def setup_rewards(self):
        """ Initializes the reward function for the players in the GPG """
        self.reward_dict = {self.id: self.traj.reward(self.reward, self.x_dict[self.id], self.u_dict[self.id], self.x_dict, terminal_reward=self.terminal_reward)}
        for i in self.humans:
            self.reward_dict[i] = self.human_trajectories[i].reward(self._human_rewards[i], self.x_dict[i], self.u_dict[i], self.x_dict, terminal_reward=self._human_terminal_rewards[i])
        return

    def setup_constraints(self):
        """ Initializes the constraints for the players in the GPG

        When soft constraints are used, penalty parameters are introduced and the reward function adapted accordingly
        """
        x_N = {}
        for id, value in self.x_dict.items():
            x_N[id] = value[self.N-1]

        for con in self.stage_g:
            self.g.extend(self.traj.constraints(con, self.x_dict, self.u_dict))

        for con in self.stage_h:
            self.h.extend(self.traj.constraints(con, self.x_dict, self.u_dict))

        for con in self.terminal_h:
            self.h.extend(con(x_N))

        for i, constraint_list in self.player_stage_g.items():
            for con in constraint_list:
                self.player_g[i].extend(self.traj.constraints(con, self.x_dict, self.u_dict))

        for i, constraint_list in self.player_stage_h.items():
            for con in constraint_list:
                self.player_h[i].extend(self.traj.constraints(con, self.x_dict, self.u_dict))

        for i, constraint_list in self.player_terminal_h.items():
            for con in constraint_list:
                self.player_h[i].extend(con(x_N))

        for con in self._stage_dual:
            eq, ineq, additional_parameters = self.traj.constraints_dual_formulation(con, self.x_dict, self._sym)
            self.g.extend(eq)
            self.h.extend(ineq)
            self.v = cs.vertcat(self.v, additional_parameters)
            self.lbv += [-float("inf")] * additional_parameters.shape[0]
            self.ubv += [float("inf")] * additional_parameters.shape[0]

        self.reward_dict_original = self.reward_dict.copy()
        self.h_original = self.h.copy()
        self.g_original = self.g.copy()

        for con in self.soft_stage_g:
            current_constraint = self.traj.constraints(con, self.x_dict, self.u_dict)
            reward_contribution, z = self.penalty_parameters.add_penalty_constraints(current_constraint, 'stage_g')
            self.g_original.extend(current_constraint)
            for i in self.reward_dict:
                self.reward_dict[i] += reward_contribution

        for con in self.soft_stage_h:
            current_constraint = self.traj.constraints(con, self.x_dict, self.u_dict)
            reward_contribution, z = self.penalty_parameters.add_penalty_constraints(current_constraint, 'stage_h')
            self.h_original.extend(current_constraint)
            for i in self.reward_dict:
                self.reward_dict[i] += reward_contribution
            if z is not None:
                self.h.extend([current_constraint[i] + z[i] for i in range(current_constraint.length)])
                self.z = cs.vertcat(self.z, z)
                self.lbz += [0] * current_constraint.length
                self.ubz += [float("inf")] * current_constraint.length

        for con in self.soft_terminal_h:
            current_constraint = con(x_N)
            reward_contribution, z = self.penalty_parameters.add_penalty_constraints(current_constraint, 'terminal_h')
            self.h_original.extend(current_constraint)
            for i in self.reward_dict:
                self.reward_dict[i] += reward_contribution
            if z is not None:
                self.h.extend([current_constraint[i] + z[i] for i in range(current_constraint.length)])
                self.z = cs.vertcat(self.z, z)
                self.lbz += [0] * current_constraint.length
                self.ubz += [float("inf")] * current_constraint.length

        for con in self._soft_stage_dual:  # TODO
            eq, ineq, additional_parameters = self.traj.constraints_dual_formulation(con, self.x_dict, self._sym)
            self.g_original.extend(eq)
            self.h_original.extend(ineq)
            self.v = cs.vertcat(self.v, additional_parameters)
            self.lbv += [-float("inf")] * additional_parameters.shape[0]
            self.ubv += [float("inf")] * additional_parameters.shape[0]
        return

    def casadi_bounds(self, car_dyn, for_original_gpg):
        """ Returns a list of bounds for the variables of a player, given the respective dynamics object

        Parameters
        ----------
        car_dyn : Dynamics object
            the dynamics of the vehicle for which bounds are generated
        for_original_gpg : boolean
            indicates whether the bounds are for the original of the penalized game
        """
        if self.mode == 'single':
            lb_player = car_dyn.bounds[0] * self.N + self.lbv
            ub_player = car_dyn.bounds[1] * self.N + self.ubv
        else:
            lb_player = car_dyn.bounds[0] * self.N + [-float("inf")] * car_dyn.nx * self.N + self.lbv
            ub_player = car_dyn.bounds[1] * self.N + [float("inf")] * car_dyn.nx * self.N + self.ubv
        if not for_original_gpg:
            lb_player += self.lbz
            ub_player += self.ubz
        return lb_player, ub_player

    def get_bounds(self, for_original_gpg):
        """ Initializes the bounds for the players in the GPG

        Parameters
        ----------
        for_original_gpg : boolean
            indicates whether the bounds are for the original of the penalized game
        """
        lb, ub = {}, {}
        lb[self.id], ub[self.id] = self.casadi_bounds(self.dyn, for_original_gpg)
        for i in self.humans:
            lb[i], ub[i] = self.casadi_bounds(self.dyn_dict[i], for_original_gpg)
        return lb, ub

    def setup_parameters(self):
        """ Initializes the parameters required for obstacles in the GPG """
        for i in self.obstacles:
            self.p_dict[i] = cs.vertcat(self.x0_dict[i], self.u_dict[i])
            del self.x0_dict[i]
            del self.u_dict[i]
            del self.x_dict[i]
        return

    def initialize_solvers(self):
        """ Initializes the GPG solver and the online learning methodology of the ego vehicle """
        self.setup_x0_and_u()
        self.setup_x()
        self.setup_rewards()
        self.setup_constraints()
        self.setup_parameters()
        lb_pen, ub_pen = self.get_bounds(False)
        lb, ub = self.get_bounds(True)
        x_dict_solver = None if self.mode == 'single' else self.x_dict
        if self.gpg_solver_settings.use_gauss_seidel:
            self.optimizer = gaussseidelsolver.GPG_solver(self.id, self.reward_dict, self.u_dict, self.x0_dict,
                                                          lb_pen, ub_pen, self.g, self.h, self.player_g, self.player_h, cs.vertcat(self.v, self.z),
                                                          self._human_reward_params, self.p_dict, self._sym,
                                                          self.penalty_parameters, self.dyn_dict, self.gpg_solver_settings,
                                                          x_dict=x_dict_solver, g_dyn_dict=self.g_dyn_dict)
        else:
            self.optimizer = lagrangiansolver.GPG_solver(self.id, self.reward_dict, self.u_dict, self.x0_dict,
                                                         lb_pen, ub_pen, self.g, self.h, self.player_g, self.player_h, cs.vertcat(self.v, self.z),
                                                         self._human_reward_params, self.p_dict, self._sym,
                                                         self.dyn_dict, self.gpg_solver_settings,
                                                         x_dict=x_dict_solver, g_dyn_dict=self.g_dyn_dict)
        if self.online_learning_settings.based_on_original_gpg:
            self.observer = learning.online_learning_solver(self.id, self.reward_dict_original, self.u_dict,
                                                                     self.x0_dict, lb, ub, self.g_original, self.h_original,
                                                                     self.player_g, self.player_h, self._sym(), self.v,
                                                                     self._human_reward_params, self.p_dict, self._sym,
                                                                     1, self.online_learning_settings,
                                                                     x_dict=x_dict_solver, g_dyn_dict=self.g_dyn_dict) #TODO human nu hardcoded
        else:
            self.observer = learning.online_learning_solver(self.id, self.reward_dict, self.u_dict, self.x0_dict,
                                                                     lb_pen, ub_pen, self.g, self.h, self.player_g, self.player_h,
                                                                     self.penalty_parameters, cs.vertcat(self.v, self.z),
                                                                     self._human_reward_params, self.p_dict, self._sym, 1,
                                                                     self.online_learning_settings,
                                                                     x_dict=x_dict_solver, g_dyn_dict=self.g_dyn_dict)

    def control(self, steer, gas):
        """ Sets the value of the control action of the ego vehicle by solving a GPG formulation using the current
        belief in the parameters of the human drivers """
        # Initialize optimizers
        if self.optimizer is None:
            self.initialize_solvers()

        # Initialize current states and parameters
        self.x0_numeric_dict[self.id] = self.x
        for i, human in self.humans.items():
            self.x0_numeric_dict[i] = human.x
        for i, obstacle in self.obstacles.items():
            self.p_numeric_dict[i] = obstacle.x + obstacle.u * self.N

        # Solve the GPG
        start = time.perf_counter()
        self.optimum_u_dict, self.optimum_v, self.optimum_lam, self.optimum_lam_g, self.optimum_penalty_parameters =\
            self.optimizer.minimize(self.x0_numeric_dict, self._human_reward_params_current_belief, self.p_numeric_dict)
        end = time.perf_counter()
        self.gpg_solution_time = (end-start)
        self.traj.u = self.optimum_u_dict[self.id][0:self.N*self.dyn.nu]
        print('id: ' + str(self.id) + ', u_opt: ' + str(self.optimum_u_dict) + ', v_opt: ' + str(self.optimum_v))
        print('Solution time: ' + str(self.gpg_solution_time) + ', CPU time: ' + str(self.optimizer.cpu_time))

    def observe(self):
        """ Solves the online learning problem, updating the belief in the parameters of the 'human' players"""
        if self.online_learning_settings.nb_observations > 0:
            # Observe the human actions
            observed_actions = {}
            for id, human in self.humans.items():
                observed_actions[id] = human.u

            # Add data for warm-starting
            if self.online_learning_settings.based_on_original_gpg:
                optimum_lam_x_and_v = {}
                for key, list in self.optimum_lam.items():
                    if self.z.shape[0] == 0:
                        optimum_lam_x_and_v[key] = list
                    else:
                        optimum_lam_x_and_v[key] = list[:-self.z.shape[0]]
                self._observations.appendleft(
                    [self.x0_numeric_dict.copy(), self.optimum_u_dict.copy(), self.optimum_v[-self.z.shape[0]:],
                     observed_actions, optimum_lam_x_and_v, self.optimum_lam_g.copy(),
                     self.p_numeric_dict.copy(), cs.DM()])
            else:
                self._observations.appendleft(
                    [self.x0_numeric_dict.copy(), self.optimum_u_dict.copy(), self.optimum_v, observed_actions,
                     self.optimum_lam.copy(), self.optimum_lam_g.copy(), self.p_numeric_dict.copy(),
                     self.optimum_penalty_parameters])

            # Solve the online learning methodology
            start = time.perf_counter()
            self._human_reward_params_current_belief = self.observer.observe(self._human_reward_params_current_belief,
                                                                             self._observations)
            end = time.perf_counter()
            self.observer_solution_time = end - start
            print('Observation time: ' + str(self.observer_solution_time) + ', CPU time: ' + str(self.observer.cpu_time))