import numpy as np
import casadi.casadi as cs
import customopengen as og
from penalty import PenaltyUpdater


class GPG_solver(object):
    """
    A class used to solve GPG formulations using a Gauss-Seidel algorithm with an additional quadratic penalty method
    """

    def __init__(self, identifier, f_dict, u_dict, x0_dict, lbx_dict, ubx_dict, g, h, player_g_dict, player_h_dict,
                 v, theta_dict, p_dict, symbolics, dyn_dict, solver_settings, x_dict=None, g_dyn_dict=None):
        """
        Parameters
        ----------
        identifier : int
            the identifier of the vehicle
        f_dict : dict
            the rewards of the vehicles
        u_dict : dict
            the control variables of the vehicles
        x0_dict : dict
            the initial states of the vehicles
        lbx_dict : dict
            the lower bounds on the optimization variables of the players in the GPG
        ubx_dict : dict
            the upper bounds on the optimization variables of the players in the GPG
        g : list
            the shared equality constraints
        h : list
            the shared inequality constraints
        player_g_dict : dict
            the player-specific equality constraints
        player_h_dict : dict
            the player-specific inequality constraints
        v : cs.SX or cs.MX
            the additional shared optimization variables
        theta_dict : dict
            the parameters of the optimal control problems of the human players
        p_dict : dict
            the parameters of the obstacles in the GPG or any additional parameters
        symbolics: cs.SX or cs.MX attribute
            allows to generate additional SX or MX symbolics
        dyn_dict : dict
            the Dynamics objects of the vehicles in the GPG
        solver_settings : GPGSolverSettings object
            the solver settings for the GPG
        x_dict : dict, optional
            the state variables of the players in the GPG, only for multiple shooting
        g_dyn_dict : dict, optional
            the dynamics constraints of the vehicles (multiple shooting implementation)
        """
        assert (solver_settings.solver in ['qpoases', 'ipopt', 'OpEn'])
        if x_dict is None:
            self.x_dict = u_dict
        else:
            self.x_dict = {i: cs.vertcat(u_dict.get(i), np.asarray(x_dict.get(i)).flatten())
                           for i in set(u_dict).union(x_dict)}
        self.id_list = f_dict.keys()
        if g_dyn_dict is None:
            g_dyn_dict = {}
            for i in self.id_list:
                g_dyn_dict[i] = []
        self.init = True
        self.solver = solver_settings.solver
        self.id = identifier
        self.dyn_dict = dyn_dict
        self.nb_inner_iterations = 0
        self.nb_outer_iterations = 0
        self.cost = 0
        self.stage_cost = 0
        self.constraint_violation = 0
        self.f_dict = f_dict
        self.lbx_dict = lbx_dict
        self.ubx_dict = ubx_dict
        self.x_numeric_dict = {}
        self.lam_x_numeric = None
        self.lam_g_numeric = None
        self.nb_v_dict = {}
        self.nb_u_dict = {}
        self.nb_x_dict = {}
        self.x_index_dict = {}
        self.nb_player_constraints_dict = {}
        self.constraints_index_dict = {}
        self.ubg_dict = {}
        self.x0_dict = x0_dict
        self.max_outer_iterations = solver_settings.max_outer_iterations
        self.max_inner_iterations = solver_settings.max_inner_iterations
        self.game_tolerance = solver_settings.game_tolerance
        self.nb_v = v.size()[0]

        x_index = 0
        constraint_index = 0
        for i in self.id_list:
            self.nb_u_dict[i] = u_dict[i].size()[0]
            self.nb_x_dict[i] = self.x_dict[i].size()[0]
            self.x_index_dict[i] = np.s_[x_index: x_index + self.nb_x_dict[i]]
            x_index += self.nb_x_dict[i]
            self.nb_player_constraints_dict[i] = len(g_dyn_dict[i]) + len(player_g_dict[i]) + len(player_h_dict[i])
            self.ubg_dict[i] = [0] * (len(g_dyn_dict[i]) + len(player_g_dict[i])) + [float("inf")] * len(player_h_dict[i])
            self.constraints_index_dict[i] = np.s_[constraint_index: constraint_index + self.nb_player_constraints_dict[i]]
            constraint_index += self.nb_player_constraints_dict[i]

        self.ubg_common = [0] * len(g) + [float("inf")] * len(h)

        self.cpu_time = 0

        all_parameters = [cs.vertcat(*[x0_dict[key] for key in self.id_list],
                                     *[self.x_dict[key] for key in self.id_list],
                                     *[theta_dict[key] for key in theta_dict.keys()],
                                     *[p_dict[key] for key in p_dict.keys()], v)]

        # Create function to evaluate the cost function of each player
        self.cost_function = cs.Function('f', all_parameters, [-f_dict[self.id]])

        self.v_numeric = [0] * self.nb_v
        self.x_numeric = [0] * sum(self.nb_x_dict.values())

        if solver_settings.rebuild_solver:
            # Initialize solver for 'OpEn'
            # Set settings for PANOC
            solver_config = og.config.SolverConfiguration() \
                .with_tolerance(solver_settings.solver_tolerance) \
                .with_delta_tolerance(solver_settings.panoc_delta_tolerance) \
                .with_max_inner_iterations(solver_settings.panoc_max_inner_iterations) \
                .with_max_outer_iterations(solver_settings.panoc_max_outer_iterations) \
                .with_initial_penalty(solver_settings.panoc_initial_penalty)
                #.with_max_duration_micros(0.05*1e-6)

            # Concatenate all variables for ALM solver
            lbx = [lb for key in self.id_list for lb in lbx_dict[key]]
            ubx = [ub for key in self.id_list for ub in ubx_dict[key]]
            lbg = [0] * (sum(self.nb_player_constraints_dict.values()) + len(self.ubg_common))
            ubg = [ub for key in self.id_list for ub in self.ubg_dict[key]] + self.ubg_common
            bounds_x = og.constraints.Rectangle(lbx, ubx)
            bounds_g = og.constraints.Rectangle(lbg, ubg)
            f = sum(f_dict.values())
            player_constraints = []
            for key in self.id_list:
                player_constraints += g_dyn_dict[key] + player_g_dict[key] + player_h_dict[key]

            # Create problem for current player and build the optimizer
            x = cs.vertcat(*[self.x_dict[key] for key in self.id_list], v)
            problem_parameters = cs.vertcat(*[x0_dict[key] for key in self.id_list],
                                            *[theta_dict[key] for key in theta_dict.keys()],
                                            *[p_dict[key] for key in p_dict.keys()])
            self.problem = \
                og.builder.Problem(x, problem_parameters, -f) \
                .with_constraints(bounds_x) \
                .with_aug_lagrangian_constraints(cs.vertcat(*player_constraints, *g, *h), bounds_g)

            tcp_config = og.config.TcpServerConfiguration('127.0.0.1', 8000 + self.id)
            build_config = og.config.BuildConfiguration() \
                .with_build_directory("og_builds") \
                .with_tcp_interface_config(tcp_config)
            build_config.with_build_mode(solver_settings.build_mode)\
                .with_build_directory("og_builds/" + solver_settings.directory_name)
            meta = og.config.OptimizerMeta() \
                .with_optimizer_name("solver_" + str(self.id))
            builder = og.builder.OpEnOptimizerBuilder(self.problem, meta, build_config, solver_config)
            builder.build()

        print("start controller TCP servers")
        # Start TCP server for calling optimizer
        self.solver = og.tcp.OptimizerTcpManager("og_builds/" + solver_settings.directory_name
                                                 + "/solver_" + str(self.id),
                                                 debug=solver_settings.debug_rust_code)
        self.solver.start()
        self.solver.ping()
        print("TCP servers initialized")

    def reset(self):
        # Reset x and v numeric
        self.init = True
        self.v_numeric = [0] * self.nb_v
        self.x_numeric = [0] * sum(self.nb_x_dict.values())

    def shift_vector_control_inputs(self, x0_numeric_dict):
        """ Shift the optimization variables by one time step

        Parameters
        ----------
        x0_numeric_dict : dict
            the numeric values of the new initial states for the players of the GPG
        """
        # Test whether this is the very first iteration of the GPG
        if self.init:
            # Initialize state and control variables
            self.init = False
            # for i in self.id_list:
            #     if self.nb_x_dict[i] > self.nb_u_dict[i]:
            #         nu = self.dyn_dict[i].nu
            #         nx = self.dyn_dict[i].nx
            #         self.x_numeric_dict[i][self.nb_u_dict[i]:self.nb_u_dict[i] + nx] = cs.DM(
            #             self.dyn_dict[i](x0_numeric_dict[i], [0] * nu))
            #         for j in range(1, self.nb_u_dict[i] // nu):
            #             self.x_numeric_dict[i][self.nb_u_dict[i] + j * nx:self.nb_u_dict[i] + (j + 1) * nx] = cs.DM(
            #                 self.dyn_dict[i](
            #                     self.x_numeric_dict[i][self.nb_u_dict[i] + (j - 1) * nx:self.nb_u_dict[i] + j * nx],
            #                     [0] * nu))
        else:
            # Shift state and control variables
            for i in self.id_list:
                nu = self.dyn_dict[i].nu
                nx = self.dyn_dict[i].nx
                if self.nb_u_dict[i] > nu:
                    self.x_numeric[self.x_index_dict[i]][0:self.nb_u_dict[i] - nu] =\
                        self.x_numeric[self.x_index_dict[i]][nu:self.nb_u_dict[i]]
                self.x_numeric[self.x_index_dict[i]][self.nb_u_dict[i] - nu:self.nb_u_dict[i]] = [0] * nu
                # if self.nb_x_dict[i] > self.nb_u_dict[i]:
                #     if self.nb_x_dict[i] - self.nb_u_dict[i] > nx:
                #         self.x_numeric_dict[i][self.nb_u_dict[i]:self.nb_x_dict[i] - nx] = self.x_numeric_dict[i][
                #                                                                            self.nb_u_dict[i] + nx:
                #                                                                            self.nb_x_dict[i]]
                #     self.x_numeric_dict[i][self.nb_x_dict[i] - nx:self.nb_x_dict[i]] = cs.DM(
                #         self.dyn_dict[i](self.x_numeric_dict[i][self.nb_x_dict[i] - nx:self.nb_x_dict[i]],
                #                          self.x_numeric_dict[i][self.nb_u_dict[i] - nu:self.nb_u_dict[i]]))
        return

    def minimize(self, x0_numeric_dict, theta_numeric_dict, p_numeric_dict):
        """ Solves the GPG and returns a generalized Nash equilibrium

        Parameters
        ----------
        x0_numeric_dict : dict
            the numeric values of the new initial states for the players of the GPG
        theta_numeric_dict : dict
            the numeric values of the current estimates of the parameters of the human players in the GPG
        p_numeric_dict : dict
            the numeric values for the parameters of the optimization problems
        """
        # Initialize parameters, i.e. overhead costs
        self.cpu_time = 0
        self.shift_vector_control_inputs(x0_numeric_dict)
        self.nb_inner_iterations = 0
        self.nb_outer_iterations = 0

        self.minimize_problem(x0_numeric_dict, theta_numeric_dict, p_numeric_dict)

        p_current = cs.vertcat(*[x0_numeric_dict[key] for key in self.id_list],
                               *[self.x_numeric],
                               *[theta_numeric_dict[key] for key in theta_numeric_dict],
                               *[p_numeric_dict[key] for key in p_numeric_dict],
                               *[self.v_numeric])

        # Evaluate the cost and the constraint violation of the obtained Nash equilibrium
        self.cost = self.cost_function(cs.vertcat(p_current))

        x_numeric_dict = {key: self.x_numeric[self.x_index_dict[key]] for key in self.id_list}
        lam_x_numeric_dict = {key: self.lam_x_numeric[self.x_index_dict[key]] for key in self.id_list}
        lam_g_numeric_dict = {key: cs.vertcat(self.lam_g_numeric[self.constraints_index_dict[key]],
                                   self.lam_g_numeric[sum(self.nb_player_constraints_dict.values()):]) for key in self.id_list}
        print('GNEP:        ID: ' + str(self.id) + ',   constraint violation: ' + str(self.constraint_violation))
        return x_numeric_dict, self.v_numeric, lam_x_numeric_dict, lam_g_numeric_dict, cs.DM()

    def minimize_problem(self, x0_numeric_dict, theta_numeric_dict, p_numeric_dict):
        """ Minimizes the optimization problem of a single player

        Parameters
        ----------
        id : int
            the identifier of the player
        x0_numeric_dict : dict
            the numeric values of the initial states of the players of the GPG
        theta_numeric_dict : dict
            the numeric values of the current estimates of the parameters of the human players in the GPG
        p_numeric_dict : dict
            the numeric values for the parameters of the optimization problems
        """
        # Solve the optimization problem
        x_old = self.x_numeric + self.v_numeric
        solution = self.solver.call(cs.vertcat(*[x0_numeric_dict[key] for key in self.id_list],
                                               *[theta_numeric_dict[key] for key in theta_numeric_dict],
                                               *[p_numeric_dict[key] for key in p_numeric_dict]).toarray(True),
                                    initial_guess=x_old)
        if solution.is_ok():
            self.cpu_time = solution['solve_time_ms'] / 1000
            x_opt = solution['solution']
            self.lam_x_numeric = cs.DM([0] * len(x_opt))
            self.lam_g_numeric = cs.DM(solution['lagrange_multipliers'])
            self.nb_inner_iterations = solution['num_inner_iterations']
            self.nb_outer_iterations = solution['num_outer_iterations']
            self.constraint_violation = solution['f1_infeasibility']

            # Set the numeric dicts based on the obtained solution
            self.x_numeric = x_opt[0:sum(self.nb_x_dict.values())]
            if self.nb_v > 0:
                self.v_numeric = x_opt[sum(self.nb_x_dict.values()):]
            if solution['exit_status'] != 'Converged':
                print(solution['exit_status'])
                print(solution['num_outer_iterations'])
                print(solution['num_inner_iterations'])
        else:
            print(solution['message'])
        return
