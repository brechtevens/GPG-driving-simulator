import numpy as np
import casadi.casadi as cs
import customopengen as og
from penalty import PenaltyUpdater


class GPG_solver(object):
    """
    A class used to solve GPG formulations using a Gauss-Seidel algorithm with an additional quadratic penalty method
    """

    def __init__(self, identifier, f_dict, u_dict, x0_dict, lbx_dict, ubx_dict, g, h, player_g_dict, player_h_dict,
                 v, theta_dict, p_dict, symbolics, penalty_parameters, dyn_dict, solver_settings,
                 x_dict=None, g_dyn_dict=None):
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
        penalty_parameters : [cs.SX, cs.SX] or [cs.MX, cs.MX]
            the penalty parameters of the GPG
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
        if solver_settings.sorted:
            self.id_list = sorted(f_dict.keys())
        if g_dyn_dict is None:
            g_dyn_dict = {}
            for i in self.id_list:
                g_dyn_dict[i] = []
        self.init = True
        self.solver = solver_settings.solver
        self.warm_start = solver_settings.warm_start
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
        self.lam_x_numeric_dict = {}
        self.lam_g_numeric_dict = {}
        self.nb_v_dict = {}
        self.nb_u_dict = {}
        self.nb_x_dict = {}
        self.nb_g_dict = {}
        self.ubg_dict = {}
        self.x0_dict = x0_dict
        self.max_outer_iterations = solver_settings.max_outer_iterations
        self.max_inner_iterations = solver_settings.max_inner_iterations
        self.game_tolerance = solver_settings.game_tolerance
        self.penalty_norm = solver_settings.penalty_norm
        self.nb_v = v.size()[0]
        self.nb_g = len(g)

        for i in self.id_list:
            self.nb_u_dict[i] = u_dict[i].size()[0]
            self.nb_x_dict[i] = self.x_dict[i].size()[0]
            self.ubg_dict[i] = [0] * len(g) + [0] * len(g_dyn_dict[i]) + \
                               [float("inf")] * len(h) + [float("inf")] * len(player_h_dict[i])

        self.problem_dict = {}
        self.solver_dict = {}
        self.cpu_time = 0

        all_parameters = [cs.vertcat(*[x0_dict[key] for key in self.id_list],
                                     *[self.x_dict[key] for key in self.id_list],
                                     *[theta_dict[key] for key in theta_dict.keys()],
                                     *[p_dict[key] for key in p_dict.keys()], v)]

        self.penalty_handler = PenaltyUpdater(penalty_parameters)
        self.penalty_handler.generate_constraint_function(all_parameters)
        self.player_specific_constraint_violation = cs.Function('constraint_test', all_parameters,
                                                                [cs.vertcat(*[constraint for i in self.id_list for constraint in player_g_dict[i]],
                                                                            *[cs.fmin(0.0, constraint) for i in self.id_list for constraint in player_h_dict[i]])])

        # Create function to evaluate the cost function of each player
        self.cost_function = cs.Function('f', all_parameters,
                                         [cs.substitute(-f_dict[self.id], penalty_parameters.all, symbolics.zeros(penalty_parameters.nb_all,1))])

        # Initialize solvers
        if self.solver == 'ipopt' or self.solver == 'qpoases':
            # Initialize solver for 'ipopt' or 'qpoases'
            self.v_numeric = cs.DM.zeros(self.nb_v)
            for i in self.id_list:
                self.x_numeric_dict[i] = cs.DM.zeros(self.nb_x_dict[i])
                self.lam_x_numeric_dict[i] = cs.DM.zeros(self.nb_x_dict[i] + self.nb_v)
                self.lam_g_numeric_dict[i] = cs.DM.zeros(len(self.ubg_dict[i]))
                x = cs.vertcat(self.x_dict[i], v)
                x_old = symbolics.sym('x_old_' + str(i), x.size())
                problem_parameters = cs.vertcat(*[x0_dict[key] for key in self.id_list],
                                                *[self.x_dict[key] for key in self.id_list if key != i],
                                                *[theta_dict[key] for key in theta_dict.keys()],
                                                *[p_dict[key] for key in p_dict.keys()], x_old, penalty_parameters.all)
                self.problem_dict[i] = {'x': x, 'f': -f_dict[i] + solver_settings.regularization * cs.sumsqr(x - x_old),
                                        'p': problem_parameters, 'g': cs.vertcat(*g, *g_dyn_dict[i], *h, *player_h_dict[i])}
                if self.solver == 'ipopt':
                    self.solver_dict[i] = cs.nlpsol('Solver_' + str(i), 'ipopt', self.problem_dict[i],
                                                    {'verbose_init': False, 'print_time': False, 'ipopt':
                                                        {'print_level': 0, 'tol': solver_settings.solver_tolerance}})
                else:
                    self.solver_dict[i] = cs.qpsol('Solver_' + str(i), 'qpoases', self.problem_dict[i],
                                                   {'verbose': False, 'jit': True, 'printLevel': 'low'})
        else:
            bounds_dict = {}
            self.v_numeric = [0] * self.nb_v
            for i in self.id_list:
                self.x_numeric_dict[i] = [0] * self.nb_x_dict[i]

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

                for i in self.id_list:
                    bounds_dict[i] = og.constraints.Rectangle(lbx_dict[i], ubx_dict[i])

                    # Create problem for current player and build the optimizer
                    x = cs.vertcat(u_dict[i], v)
                    x_old = symbolics.sym('x_old_' + str(i), x.size())
                    problem_parameters = cs.vertcat(*[x0_dict[key] for key in self.id_list],
                                                    *[self.x_dict[key] for key in self.id_list if key != i],
                                                    *[theta_dict[key] for key in theta_dict.keys()],
                                                    *[p_dict[key] for key in p_dict.keys()], x_old,
                                                    penalty_parameters.all)
                    self.problem_dict[i] = \
                        og.builder.Problem(x, problem_parameters, -f_dict[i]
                                           + solver_settings.regularization * cs.sumsqr(x - x_old)) \
                        .with_constraints(bounds_dict[i]) \
                        .with_penalty_constraints(cs.vertcat(*g, *g_dyn_dict[i], *player_g_dict[i],
                                                             *[cs.fmin(0.0, constraint) for constraint in player_h_dict[i]]))\
                        .with_aug_lagrangian_constraints(cs.vertcat(*h), og.constraints.Rectangle([0]*len(h), [float('inf')]*len(h)))
                                                            #*[cs.fmin(0.0, constraint) for constraint in h],
                    tcp_config = og.config.TcpServerConfiguration('127.0.0.1', 8000 + 10 * self.id + i)
                    build_config = og.config.BuildConfiguration() \
                        .with_build_directory("og_builds") \
                        .with_tcp_interface_config(tcp_config)
                    build_config.with_build_mode(solver_settings.build_mode)\
                        .with_build_directory("og_builds/" + solver_settings.directory_name)
                    meta = og.config.OptimizerMeta() \
                        .with_optimizer_name("solver_" + str(self.id) + "_" + str(i))
                    builder = og.builder.OpEnOptimizerBuilder(self.problem_dict[i], meta, build_config, solver_config)
                    builder.build()

            print("start controller TCP servers")
            # Start TCP servers for calling optimizers
            for i in self.id_list:
                self.solver_dict[i] = og.tcp.OptimizerTcpManager("og_builds/" + solver_settings.directory_name
                                                                 + "/solver_" + str(self.id) + "_" + str(i),
                                                                 debug=solver_settings.debug_rust_code)
                self.solver_dict[i].start()
                self.solver_dict[i].ping()
            print("TCP servers initialized")

    def player_specific_constraint_violation_norm(self, p_current):
        if self.penalty_norm == 'norm_inf':
            return cs.mmax(cs.fabs(self.player_specific_constraint_violation(p_current)))
        elif self.penalty_norm == 'norm_2':
            return cs.norm_2(self.player_specific_constraint_violation(p_current))

    def reset(self):
        # Reset initial penalty parameter values
        self.penalty_handler.reset_penalty_parameters()

        # Reset x and v numeric
        self.init = True
        if self.solver == 'ipopt' or self.solver == 'qpoases':
            self.v_numeric = cs.DM.zeros(self.nb_v)
            for i in self.id_list:
                self.x_numeric_dict[i] = cs.DM.zeros(self.nb_x_dict[i])
                self.lam_x_numeric_dict[i] = cs.DM.zeros(self.nb_x_dict[i] + self.nb_v)
                self.lam_g_numeric_dict[i] = cs.DM.zeros(len(self.ubg_dict[i]))
        else:
            self.v_numeric = [0] * self.nb_v
            for i in self.id_list:
                self.x_numeric_dict[i] = [0] * self.nb_x_dict[i]

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
            for i in self.id_list:
                if self.nb_x_dict[i] > self.nb_u_dict[i]:
                    nu = self.dyn_dict[i].nu
                    nx = self.dyn_dict[i].nx
                    self.x_numeric_dict[i][self.nb_u_dict[i]:self.nb_u_dict[i] + nx] = cs.DM(
                        self.dyn_dict[i](x0_numeric_dict[i], [0] * nu))
                    for j in range(1, self.nb_u_dict[i] // nu):
                        self.x_numeric_dict[i][self.nb_u_dict[i] + j * nx:self.nb_u_dict[i] + (j + 1) * nx] = cs.DM(
                            self.dyn_dict[i](
                                self.x_numeric_dict[i][self.nb_u_dict[i] + (j - 1) * nx:self.nb_u_dict[i] + j * nx],
                                [0] * nu))
        else:
            # Shift state and control variables
            for i in self.id_list:
                nu = self.dyn_dict[i].nu
                nx = self.dyn_dict[i].nx
                if self.nb_u_dict[i] > nu:
                    self.x_numeric_dict[i][0:self.nb_u_dict[i] - nu] = self.x_numeric_dict[i][nu:self.nb_u_dict[i]]
                self.x_numeric_dict[i][self.nb_u_dict[i] - nu:self.nb_u_dict[i]] = [0] * nu
                if self.nb_x_dict[i] > self.nb_u_dict[i]:
                    if self.nb_x_dict[i] - self.nb_u_dict[i] > nx:
                        self.x_numeric_dict[i][self.nb_u_dict[i]:self.nb_x_dict[i] - nx] = self.x_numeric_dict[i][
                                                                                           self.nb_u_dict[i] + nx:
                                                                                           self.nb_x_dict[i]]
                    self.x_numeric_dict[i][self.nb_x_dict[i] - nx:self.nb_x_dict[i]] = cs.DM(
                        self.dyn_dict[i](self.x_numeric_dict[i][self.nb_x_dict[i] - nx:self.nb_x_dict[i]],
                                         self.x_numeric_dict[i][self.nb_u_dict[i] - nu:self.nb_u_dict[i]]))

        # Warm starting of penalty parameters
        if self.warm_start:
            self.penalty_handler.shift_penalty_parameters()
        else:
            self.penalty_handler.reset_penalty_parameters()
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
        delta_players = {self.id: float('inf')}
        p_current = cs.vertcat(*[x0_numeric_dict[key] for key in self.id_list],
                               *[self.x_numeric_dict[key] for key in self.id_list],
                               *[theta_numeric_dict[key] for key in theta_numeric_dict],
                               *[p_numeric_dict[key] for key in p_numeric_dict],
                               *[self.v_numeric])

        # Main loop
        while self.nb_outer_iterations == 0 or \
                not self.penalty_handler.terminated and self.nb_outer_iterations < self.max_outer_iterations:
            self.penalty_handler.update_penalty_parameters()
            self.nb_inner_iterations = 0
            while self.nb_inner_iterations < self.max_inner_iterations and \
                    max(delta_players.values()) > self.game_tolerance:
                for id in self.id_list:
                    delta_players[id] = self.minimize_player(id, x0_numeric_dict, theta_numeric_dict, p_numeric_dict)
                self.nb_inner_iterations += 1
            p_current = cs.vertcat(*[x0_numeric_dict[key] for key in self.id_list],
                                   *[self.x_numeric_dict[key] for key in self.id_list],
                                   *[theta_numeric_dict[key] for key in theta_numeric_dict],
                                   *[p_numeric_dict[key] for key in p_numeric_dict],
                                   *[self.v_numeric])
            self.nb_outer_iterations += 1
            # print('constraint violation: ' + str(self.penalty_handler.constraint_violation(p_current)))
            self.penalty_handler.update_index_set(p_current)
            # print('index_set: ' + str(self.penalty_handler.index_set))
            # print('penalty_parameters: ' + str(self.penalty_handler.values))
            delta_players = {self.id: float('inf')}

        # Evaluate the cost and the constraint violation of the obtained Nash equilibrium
        self.cost = self.cost_function(cs.vertcat(p_current))
        self.constraint_violation = cs.fmax(self.penalty_handler.constraint_violation(p_current),
                                            self.player_specific_constraint_violation_norm(p_current))
        print('GNEP:        ID: ' + str(self.id) + ',   nb_inner_iterations: ' + str(
            self.nb_inner_iterations) + ',   nb_outer_iterations: ' + str(
            self.nb_outer_iterations) + ',   constraint violation: ' + str(self.constraint_violation))
        return self.x_numeric_dict, self.v_numeric, self.lam_x_numeric_dict, self.lam_g_numeric_dict,\
            self.penalty_handler.values

    def minimize_player(self, id, x0_numeric_dict, theta_numeric_dict, p_numeric_dict):
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
        if self.solver in ['qpoases', 'ipopt']:
            x_old = cs.vertcat(self.x_numeric_dict[id], self.v_numeric)
            solution = self.solver_dict[id](x0=x_old, p=cs.vertcat(*[x0_numeric_dict[key] for key in self.id_list],
                                                                   *[self.x_numeric_dict[key] for key in self.id_list if
                                                                     key != id],
                                                                   *[theta_numeric_dict[key] for key in
                                                                     theta_numeric_dict],
                                                                   *[p_numeric_dict[key] for key in p_numeric_dict],
                                                                   x_old, self.penalty_handler.values),
                                            lbx=self.lbx_dict[id], ubx=self.ubx_dict[id], lbg=0, ubg=self.ubg_dict[id],
                                            lam_x0=self.lam_x_numeric_dict[id], lam_g0=self.lam_g_numeric_dict[id])
            x_opt = solution['x']
            lam_x_opt = solution['lam_x']
            lam_g_opt = solution['lam_g']
            self.lam_x_numeric_dict[id] = lam_x_opt
            self.lam_g_numeric_dict[id] = lam_g_opt
        else:
            x0_list = []
            x_list = []
            for key in self.id_list:
                x0_list.extend(x0_numeric_dict[key])
                if key != id:
                    x_list.extend(self.x_numeric_dict[key])
            x_old = self.x_numeric_dict[id] + self.v_numeric
            solution = self.solver_dict[id].call(cs.vertcat(*[x0_numeric_dict[key] for key in self.id_list],
                                                            *[self.x_numeric_dict[key] for key in self.id_list if
                                                              key != id],
                                                            *[theta_numeric_dict[key] for key in theta_numeric_dict],
                                                            *[p_numeric_dict[key] for key in p_numeric_dict], x_old,
                                                            self.penalty_handler.values).toarray(True), initial_guess=x_old)
            if solution.is_ok():
                self.cpu_time += solution['solve_time_ms'] / 1000
                x_opt = solution['solution']
                if solution['exit_status'] != 'Converged':
                    print(solution['exit_status'])
                    print(solution['num_outer_iterations'])
                    print(solution['num_inner_iterations'])
            else:
                print(solution['message'])
                return
            lam_x_opt = [0] * len(x_opt)
            lam_g_opt = solution['lagrange_multipliers']
            self.lam_x_numeric_dict[id] = cs.DM(lam_x_opt)
            self.lam_g_numeric_dict[id] = cs.DM(lam_g_opt)

        # Evaluate the difference between the new and the previous solution
        delta = np.linalg.norm(np.array(x_old) - np.array(x_opt), ord=np.inf)

        # Set the numeric dicts based on the obtained solution
        self.x_numeric_dict[id] = x_opt[0:self.nb_x_dict[id]]
        if self.nb_v > 0:
            self.v_numeric = x_opt[self.nb_x_dict[id]:]
        return delta
