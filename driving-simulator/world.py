import car
import constraints
import feature
import dynamics
import numpy as np
import road
import casadi as cs
import collision
import settings


class Object(object):
    def __init__(self, name, x):
        self.name = name
        self.x = np.asarray(x)


class World(object):
    """
    A class used to represent traffic scenarios

    Attributes
    ----------
    cars : list
        the cars in the traffic scenario
    lanes : list
        the lanes of the traffic scenario
    roads : list
        the roads of the traffic scenario
    highway : Highway object
        the highway of the traffic scenario, if any
    _colors : list
        the possible colors for the cars in the scenario
    Ts : float
        the sampling time of the experiment, required to save the timestamps of the current experiment correctly
    """
    def __init__(self):
        self.cars = []
        self.lanes = []
        self.roads = []
        self.highway = None
        self._colors = ['red', 'yellow', 'blue', 'white', 'orange', 'purple', 'gray']
        self.Ts = 0

    def set_nb_lanes(self, nb_lanes, width=3.0, length_list=None):
        """ Sets the number of lanes for a traffic scenario on a highway section

        Parameters
        ----------
        nb_lanes : int
            the number of lanes on the highway section (> 0)
        width : float
            the width of the lanes
        length_list : list
            the lengths of the different lanes
         """
        assert (nb_lanes > 0)
        self.highway = road.Highway([-1., 0.], [0., 0.], width, nb_lanes, length_list)
        self.lanes = self.highway.get_lanes()
        self.roads.append(self.highway)

    def add_vehicle(self, type, dynamics, x0, horizon=None, color=None, **vargs):
        """ Adds a vehicle to the traffic scenario and returns its identifier

        Parameters
        ----------
        type : str
            the vehicle type, eg 'UserControlledCar'
        dynamics : Dynamics object
            the dynamics of the vehicle
        x0 : list
            the initial state of the vehicle
        horizon : int, optional
            the control horizon of the vehicle
        color : str, optional
            the color of the vehicle
         """
        id = len(self.cars)
        if horizon is None:
            horizon = 1
        if color is None:
            color = self._colors[id]
        vehicle_initializer = getattr(car, type)
        self.cars.append(vehicle_initializer(dynamics, x0, horizon, id, color, **vargs))
        return id

    def set_reward(self, id, reward, terminal_reward=None, params=None, param_values=None):
        """ Sets the stage reward and terminal reward of a given vehicle in the traffic scenario

        Parameters
        ----------
        id : int
            the identifier of the vehicle
        reward : Feature
            the stage reward of the vehicle
        terminal_reward : Feature
            the terminal reward of the vehicle
         """
        if isinstance(self.cars[id], car.GPGOptimizerCar):
            self.cars[id].reward = reward
            self.cars[id].terminal_reward = terminal_reward
            self.cars[id].set_ego_params(params, param_values)
        else:
            print('Could not add reward!')
        return

    def thesis_reward(self, C_v_des, v_des, C_road=0., additional_reward=None):
        """ Returns a commonly used stage reward, a parameterized version and the cost function parameters

        consists of control feature, velocity feature and optionally a feature for driving at the center of the road, i.e.
            reward = C_v_des * (v - v_des)^2 + C_road * highway.quadratic() + feature.control()

        Parameters
        ----------
        C_v_des : float
            parameter value for keeping the desired velocity
        v_des : float
            the desired velocity
        C_road : float, optional
            parameter value for driving at the center of the road
         """
        if C_road == 0:
            params = cs.SX.sym('theta_human', 2, 1)
            reward = feature.control() + C_v_des * feature.speed(v_des)
            reward_parametrized = feature.control() + params[0] * feature.speed(params[1])
        else:
            params = cs.SX.sym('theta_human', 3, 1)
            reward = feature.control() + C_v_des * feature.speed(v_des) - C_road * self.highway.quadratic()
            reward_parametrized = feature.control() + params[0] * feature.speed(params[1]) - params[2] * self.highway.quadratic()
        if additional_reward is not None:
            reward += additional_reward
            reward_parametrized += additional_reward
        return reward, reward_parametrized, params

    def add_human(self, id_player, id_human, human_reward, human_terminal_reward=None, params=None, param_values=None):
        """ Adds a 'human' vehicle to a GPGOptimizerCar

        Parameters
        ----------
        id_player : int
            the id of the GPGOptimizerCar
        id_human : int
            the id of the human
        human_reward : Feature
            the reward of the human
        human_terminal_reward : Feature
            the terminal reward of the human
        params : cs.SX or cs.MX
            the parameters of the human subproblem
        param_values : list or float
            the initial estimate for the human parameters
         """
        if isinstance(self.cars[id_player], car.GPGOptimizerCar):
            self.cars[id_player].add_human(self.cars[id_human], human_reward, human_terminal_reward, params, param_values)
        return

    def add_obstacle(self, id_player, id_obstacle):
        """ Adds a 'obstacle' to a GPGOptimizerCar

        Parameters
        ----------
        id_player : int
            the id of the GPGOptimizerCar
        id_obstacle : int
            the id of the obstacle
        """
        if isinstance(self.cars[id_player], car.GPGOptimizerCar):
            self.cars[id_player].add_obstacle(self.cars[id_obstacle])
        return

    def set_collision_avoidance_mode(self, mode, *args):
        """ Sets the collision avoidance mode for all vehicles

        Parameters
        ----------
        mode : str
            the string for the collision avoidance formulation, i.e. rectangle, product or dual
        """
        if mode == 'rectangle':
            self.add_common_constraints(collision.rectangle_formulation_inequality_constraints, 'add_h', *args)
        elif mode == 'product':
            self.add_common_constraints(collision.product_formulation_equality_constraints, 'add_h', *args)
        elif mode == 'dual':
            self.add_common_constraints(collision.dual_formulation_constraints, 'add_dual', *args)
        else:
            raise Exception('The given collision avoidance mode is unknown')

    def add_common_constraints(self, constraint_formulation, method, *args):
        """ Adds the common constraints for all vehicles

        Parameters
        ----------
        constraint formulation : Constraints object
            the common constraints
        method : str
            determines whether equality or inequality constraints are added, equals 'add_h' or 'add_g'
        """
        for id in range(len(self.cars)):
            self.set_common_constraints(id, constraint_formulation, method, *args)
        return

    def set_common_constraints(self, id, constraint_formulation, method, epsilon=0, *args):
        """ Sets the common constraints of a single vehicle

        Parameters
        ----------
        id : int
            the identifier of the regarded vehicle
        constraint formulation : Constraints object
            the common constraints
        method : str
            determines whether equality or inequality constraints are added, equals 'add_h' or 'add_g'
        epsilon : float, optional
            the epsilon parameter for the collision avoidance constraints, i.e. the virtual enlargement
        """
        method_to_call = getattr(car.GPGOptimizerCar, method)
        vehicle = self.cars[id]
        if isinstance(vehicle, car.GPGOptimizerCar):
            for i, other_vehicle in vehicle.humans.items():
                method_to_call(vehicle, constraint_formulation(vehicle, other_vehicle, epsilon), *args)
                print('added constraint between ' + str(vehicle.id) + ' and ' + str(other_vehicle.id) + ' for ' + str(vehicle.id))
                if isinstance(other_vehicle, car.GPGOptimizerCar):
                    for j, other_other_vehicle in other_vehicle.humans.items():
                        if other_other_vehicle.id != id:
                            method_to_call(vehicle, constraint_formulation(other_vehicle, other_other_vehicle, epsilon), *args)
                            print('added constraint between ' + str(other_vehicle.id) + ' and ' + str(other_other_vehicle.id) + ' for ' + str(vehicle.id))
                    for j, other_other_vehicle in other_vehicle.obstacles.items():
                        method_to_call(vehicle, constraint_formulation(other_vehicle, other_other_vehicle, epsilon), *args)
                        print('added constraint between ' + str(other_vehicle.id) + ' and ' + str(other_other_vehicle.id) + ' for ' + str(vehicle.id))
            for i, other_vehicle in vehicle.obstacles.items():
                method_to_call(vehicle, constraint_formulation(vehicle, other_vehicle, epsilon), *args)
                print('added constraint between ' + str(vehicle.id) + ' and ' + str(other_vehicle.id) + ' for ' + str(vehicle.id))
        return

    def add_boundary_constraint(self, id):
        """ Adds a boundary constraint for a single given vehicle

        Parameters
        ----------
        id : int
            the identifier of the regarded vehicle
        """
        for vehicle in self.cars:
            if isinstance(vehicle, car.GPGOptimizerCar):
                vehicle.add_player_specific_h(id, self.highway.boundary_constraint(self.cars[id]))
        return


def world_illustrative_example(Ts=1, N=1):
    """ Defines the set-up of the illustrative example

    Parameters
    ----------
    Ts : float
        the sampling time
    N : int, optional
        the control horizon
    """
    # Initialize the world with a highway with a single lane
    world = World()
    world.set_nb_lanes(1)

    # Initialize the vehicles
    d_min = 0
    world.Ts = Ts
    dyn1 = dynamics.CarDynamicsLongitudinal(Ts)
    dyn2 = dynamics.CarDynamicsLongitudinal(Ts)
    solver_settings = settings.GPGSolverSettings()
    solver_settings.solver = 'ipopt'
    solver_settings.constraint_mode = 'hard'

    id1 = world.add_vehicle('GPGOptimizerCar', dyn1, [0., 0., 0., 5], N, gpg_solver_settings=solver_settings)
    id2 = world.add_vehicle('GPGOptimizerCar', dyn2, [14., 0., 0., 5], N, gpg_solver_settings=solver_settings)

    # Select the rewards
    r1 = world.thesis_reward(0.1, 7.)[0]
    r2 = world.thesis_reward(0.1, 5.)[0]

    # Set the rewards
    world.set_reward(id1, r1)
    world.set_reward(id2, r2)

    # Add the 'humans' to the corresponding vehicles
    world.add_human(id1, id2, r2)
    world.add_human(id2, id1, r1)

    # Add common headway constraint
    world.add_common_constraints(collision.headway_formulation_constraint, 'add_h', d_min)

    return world


def world_one_dimensional_gpg_terminal(Ts=0.25, N=40):
    """ Defines the set-up of a one-dimensional GPG

    Parameters
    ----------
    Ts : float
        the sampling time
    N : int, optional
        the control horizon
    """
    # Initialize the world with a highway with a single lane
    world = World()
    world.set_nb_lanes(1)

    # Initialize the vehicles
    d_min = 10
    world.Ts = Ts
    dyn1 = dynamics.CarDynamicsLongitudinal(Ts)
    dyn2 = dynamics.CarDynamicsLongitudinal(Ts)
    solver_settings = settings.GPGSolverSettings()
    solver_settings.solver = 'ipopt'
    solver_settings.constraint_mode = 'hard'

    id1 = world.add_vehicle('GPGOptimizerCar', dyn1, [0., 0., 0., 5], N, gpg_solver_settings=solver_settings)
    id2 = world.add_vehicle('GPGOptimizerCar', dyn2, [29., 0., 0., 5], N, gpg_solver_settings=solver_settings)

    # Select the rewards
    r1 = world.thesis_reward(0.1, 7.)[0]
    r2 = world.thesis_reward(0.1, 5.)[0]

    # Set the rewards
    world.set_reward(id1, r1)
    world.set_reward(id2, r2)

    # Add the 'humans' to the corresponding vehicles
    world.add_human(id1, id2, r2)
    world.add_human(id2, id1, r1)

    # Add common headway constraint
    world.add_common_constraints(collision.headway_formulation_constraint, 'add_h', d_min)

    # Add robust control invariant terminal constraint set
    A, b = [[0, -1, 0, 1],], [0]

    # Add terminal constraints
    world.cars[id1].add_terminal_h(collision.terminal_constraint(world.cars[id1], world.cars[id2], A, b))
    world.cars[id2].add_terminal_h(collision.terminal_constraint(world.cars[id1], world.cars[id2], A, b))
    return world


def world_one_dimensional_gpg(solver_settings, learning_settings, human_belief='inexact'):
    """ Defines the set-up of a one-dimensional GPG with parameter estimation """
    Ts = 0.25
    N = 12

    # Initialize the world with a highway with a single lane
    world = World()
    world.set_nb_lanes(1)

    # Initialize the cost function parameters
    d_min = 0
    a_min = 3
    a_max = 3
    a_min_p = cs.SX.sym('a_min', 1, 1)
    a_max_p = cs.SX.sym('a_max', 1, 1)

    # Initialize the vehicles
    world.Ts = Ts
    dyn1 = dynamics.CarDynamicsLongitudinal(Ts)
    dyn2 = dynamics.CarDynamicsLongitudinal(Ts, bounds=[[-a_min_p], [a_max_p]])

    id1 = world.add_vehicle('GPGOptimizerCar', dyn1, [0., 0., 0., 5], N,
                            gpg_solver_settings=solver_settings, online_learning_settings=learning_settings)
    id2 = world.add_vehicle('GPGOptimizerCar', dyn2, [19., 0., 0., 5], N,
                            gpg_solver_settings=solver_settings)

    # Select the rewards
    r1, r_p1, p1 = world.thesis_reward(0.1, 7)
    r2, r_p2, p2 = world.thesis_reward(0.1, 5)

    # Set the rewards
    world.set_reward(id1, r1)
    world.set_reward(id2, r2, params=cs.vertcat(a_min_p, a_max_p), param_values=[a_min, a_max])

    # Add the 'humans' to the corresponding vehicles
    if human_belief == 'inexact':
        world.add_human(id1, id2, r_p2, params=cs.vertcat(p2, a_min_p, a_max_p), param_values=[0.2, 7., 0.1, 0.1])
    elif human_belief == 'exact':
        world.add_human(id1, id2, r_p2, params=cs.vertcat(p2, a_min_p, a_max_p), param_values=[0.1, 5., 3.0, 3.0])
    else:
        raise Exception("Could not add human: unknown variant")
    world.add_human(id2, id1, r1)

    # Add the common constraints and the bounds for the human
    world.add_common_constraints(collision.headway_formulation_constraint, 'add_h', d_min)
    return world


def world_overtaking_scenario(Ts=0.25, N=12, collision_mode='rectangle'):
    """ Defines the set-up of an overtaking scenario

    Parameters
    ----------
    collision_mode : str, optional
        the constraint formulation, i.e. either 'rectangle', 'product' or 'dual'
    """
    # Initialize the world with a highway with 2 lanes
    world = World()
    world.set_nb_lanes(2)

    # Initialize the vehicles
    world.Ts = Ts
    dyn = dynamics.CarDynamics(Ts)

    solver_settings = settings.GPGSolverSettings()

    id1 = world.add_vehicle('GPGOptimizerCar', dyn, [0., 0., 0., 6.5], N, gpg_solver_settings=solver_settings)
    id2 = world.add_vehicle('GPGOptimizerCar', dyn, [15, 0., 0., 5], N, gpg_solver_settings=solver_settings)

    # Select the rewards
    r1 = world.thesis_reward(1, 6.5, 0.001)[0]
    r2 = world.thesis_reward(1, 5., 1.)[0]

    # Set the rewards
    world.set_reward(id1, r1)
    world.set_reward(id2, r2)

    # Add the 'humans' to the corresponding vehicles
    world.add_human(id1, id2, r2)
    world.add_human(id2, id1, r1)


    # Add the common and the boundary constraints
    world.add_boundary_constraint(id1)
    world.add_boundary_constraint(id2)
    world.set_collision_avoidance_mode(collision_mode, 0.5)

    return world


def world_merging_scenario(solver_settings, learning_settings, human_behaviour='inattentive', human_belief='courteous'):
    """ Defines the set-up of a merging scenario """
    Ts = 0.25
    N = 12
    collision_mode = 'rectangle'

    # Initialize the world with a highway with 2 lanes
    world = World()
    world.set_nb_lanes(2)

    # Initialize the vehicles
    world.Ts = Ts
    dyn_2d = dynamics.CarDynamics(world.Ts)
    dyn_1d = dynamics.CarDynamicsLongitudinal(world.Ts)

    id1 = world.add_vehicle('GPGOptimizerCar', dyn_2d, [4., 3., 0., 5.], N, gpg_solver_settings=solver_settings,
                            online_learning_settings=learning_settings)
    id2 = world.add_vehicle('GPGOptimizerCar', dyn_1d, [0., 0., 0., 5.], N, gpg_solver_settings=solver_settings)
    id3 = world.add_vehicle('UserControlledCar', dyn_1d, [8.5, 0., 0., 5.], N)
    world.cars[id3].fix_control([0.0])
    id4 = world.add_vehicle('UserControlledCar', dyn_1d, [62., 3.4, 0., 0], 1)
    world.cars[id4].fix_control([0.0])

    # Select the rewards
    p = cs.SX.sym('param', 1, 1)
    r1, r_p1, p1 = world.thesis_reward(0.5, 5., 0.5)
    r_p2 = world.thesis_reward(0., 5., additional_reward=p * world.cars[id2].traj.quadratic_following_reward(4.5, world.cars[id3]))[0]

    # Set the rewards
    world.set_reward(id1, r1)
    if human_behaviour == 'courteous':
        world.set_reward(id2, r_p2, params=p, param_values=[0.1])
    else:
        world.set_reward(id2, r_p2, params=p, param_values=[50])

    # Add the 'humans' and the 'obstacles' to the corresponding vehicles
    if human_belief == 'courteous':
        world.add_human(id1, id2, r_p2, params=p, param_values=[0.1])
    else:
        world.add_human(id1, id2, r_p2, params=p, param_values=[50])

    world.add_obstacle(id1, id3)
    world.add_obstacle(id1, id4)
    world.add_human(id2, id1, r1)
    world.add_obstacle(id2, id3)
    world.add_obstacle(id2, id4)

    # Add the common and the boundary constraints
    world.add_boundary_constraint(id1)
    world.set_collision_avoidance_mode(collision_mode, 1)
    # world.cars[0].add_h(collision.rectangle_formulation_inequality_constraints(world.cars[0], world.cars[1], 1))
    # world.cars[0].add_h(collision.rectangle_formulation_inequality_constraints(world.cars[0], world.cars[2], 1))
    # world.cars[0].add_h(collision.rectangle_formulation_inequality_constraints(world.cars[0], world.cars[3], 1))
    # world.cars[1].add_h(collision.rectangle_formulation_inequality_constraints(world.cars[0], world.cars[1], 1))
    # world.cars[1].add_h(collision.rectangle_formulation_inequality_constraints(world.cars[0], world.cars[2], 1))
    # world.cars[1].add_h(collision.rectangle_formulation_inequality_constraints(world.cars[0], world.cars[3], 1))
    return world


def world_merging_scenario_new(solver_settings, learning_settings, human_behaviour='inattentive', human_belief='courteous'):
    """ Defines the set-up of a merging scenario """
    Ts = 0.25
    N = 12
    collision_mode = 'rectangle'

    # Initialize the world with a highway with 2 lanes
    world = World()
    world.set_nb_lanes(2, length_list=[1000, 50])

    # Initialize the vehicles
    world.Ts = Ts
    dyn_2d = dynamics.CarDynamics(world.Ts)
    dyn_1d = dynamics.CarDynamicsLongitudinal(world.Ts)

    id1 = world.add_vehicle('GPGOptimizerCar', dyn_2d, [4., 3., 0., 5.], N, gpg_solver_settings=solver_settings,
                            online_learning_settings=learning_settings)
    id2 = world.add_vehicle('GPGOptimizerCar', dyn_1d, [0., 0., 0., 5.], N, gpg_solver_settings=solver_settings)
    id3 = world.add_vehicle('UserControlledCar', dyn_1d, [8.5, 0., 0., 5.], N)
    world.cars[id3].fix_control([0.0])

    # Select the rewards
    p = cs.SX.sym('param', 1, 1)
    r1, r_p1, p1 = world.thesis_reward(0.5, 5., 0.5)
    r_p2 = world.thesis_reward(0., 5., additional_reward=p * world.cars[id2].traj.quadratic_following_reward(4.5, world.cars[id3]))[0]

    # Set the rewards
    world.set_reward(id1, r1)
    if human_behaviour == 'courteous':
        world.set_reward(id2, r_p2, params=p, param_values=[0.1])
    else:
        world.set_reward(id2, r_p2, params=p, param_values=[50])

    # Add the 'humans' and the 'obstacles' to the corresponding vehicles
    if human_belief == 'courteous':
        world.add_human(id1, id2, r_p2, params=p, param_values=[0.1])
    else:
        world.add_human(id1, id2, r_p2, params=p, param_values=[50])

    world.add_obstacle(id1, id3)
    world.add_human(id2, id1, r1)
    world.add_obstacle(id2, id3)

    # Add the common and the boundary constraints
    world.add_boundary_constraint(id1)
    world.set_collision_avoidance_mode(collision_mode, 1)
    world.cars[id1].add_player_specific_g(id1, world.highway.right_lane_constraint(world.cars[id1]))
    return world
