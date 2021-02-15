import numpy as np
import casadi.casadi as cs
import customopengen as og


class online_learning_solver(object):
    """
    A class used to set-up and solve the proposed online learning methodology
    """

    def __init__(self, ego_id, f_dict, u_dict, x0_dict, lbx_dict, ubx_dict, g, h, player_g, player_h, penalty_parameters,
                 v, theta_dict, p_dict, symbolics, nu, settings, x_dict=None, g_dyn_dict=None):
        """
        Parameters
        ----------
        ego_id : int
            the id of the ego vehicle
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
        player_h : list
            the player-specific inequality constraints
        penalty_parameters : [cs.SX, cs.SX] or [cs.MX, cs.MX]
            the penalty parameters of the GPG
        v : cs.SX or cs.MX
            the additional shared optimization variables
        theta_dict : dict
            the parameters of the optimal control problems of the human players
        p_dict : dict
            the parameters of the obstacles in the GPG or any additional parameters
        symbolics: cs.SX or cs.MX attribute
            allows to generate additional SX or MX symbolics
        nu : int
            the number of inputs of the human players
        settings : OnlineLearningSettings object
            the settings for the online learning methodology
        x_dict : dict, optional
            the state variables of the players in the GPG, only for multiple shooting
        g_dyn_dict : dict, optional
            the dynamics constraints of the vehicles (multiple shooting implementation)
        """
        # Do not generate observer if nb_observations == 0
        self.cpu_time = 0
        if settings.nb_observations == 0:
            return

        # Setup the optimization variables of the problems, dependent on single or multiple shooting
        if x_dict is None:
            self.x_dict = u_dict
        else:
            self.x_dict = {i: cs.vertcat(u_dict.get(i), np.asarray(x_dict.get(i)).flatten())
                           for i in set(u_dict).union(x_dict)}

        # Get the ids of all humans with uncertain parameters
        self.id_list = theta_dict.keys()

        # Initialize empty dynamics dictionary if None is given, i.e. single shooting
        if g_dyn_dict is None:
            g_dyn_dict = {}
            for i in self.id_list:
                g_dyn_dict[i] = []

        # Store variables
        self.id = ego_id
        self.f_dict = f_dict
        self.x0_dict = x0_dict
        self.theta_dict = theta_dict
        self.nu = nu
        self.settings = settings
        self.symbolics = symbolics

        # Initialize various dictionaries to store information about the solver
        self.nb_x_dict = {}
        self.nb_mus_dict = {}
        self.nb_lambdas_dict = {}
        self.ubg_dict = {}
        self.lbx_dict = {}
        self.ubx_dict = {}
        self.nb_v = v.size()[0]
        self.parametric_bounds_index_dict = {}

        # Initialize other variables
        self.x = symbolics()
        try:
            penalty_parameters = penalty_parameters.all
        except:
            penalty_parameters = symbolics()

        def is_casadi(var):
            if isinstance(var, cs.SX) or isinstance(var, cs.MX):
                return True
            return False

        def proper_bounds(lower_bound, upper_bound):
            if is_casadi(lower_bound) or is_casadi(upper_bound):
                return True
            if not (lower_bound == -float('inf') and upper_bound == float('inf')):
                return True
            return False

        # Gather the required information for the solver
        for id in self.id_list:
            self.nb_x_dict[id] = self.x_dict[id].size()[0]
            nb_mus_bounds = 0
            for lower_bound, upper_bound in zip(lbx_dict[id], ubx_dict[id]):
                if proper_bounds(lower_bound, upper_bound):
                    nb_mus_bounds += 1
            self.nb_mus_dict[id] = (len(h) + len(player_h[id]) + nb_mus_bounds) * settings.nb_observations
            self.nb_lambdas_dict[id] = (len(g) + len(g_dyn_dict[id])) * settings.nb_observations
            self.ubg_dict[id] = [0] * self.nb_lambdas_dict[id] + [float("inf")] * self.nb_mus_dict[id] + [0] * \
                self.nb_mus_dict[id]

            self.parametric_bounds_index_dict[id] = []
            proper_bounds_index = 0
            lbx_temp_list = lbx_dict[id].copy()
            ubx_temp_list = ubx_dict[id].copy()
            for index, bounds in enumerate(zip(lbx_temp_list, ubx_temp_list)):
                lower_bound, upper_bound = bounds
                if is_casadi(lower_bound):
                    lbx_temp_list[index] = -float('inf')
                if is_casadi(upper_bound):
                    ubx_temp_list[index] = float('inf')
                if proper_bounds(lower_bound, upper_bound):
                    if is_casadi(lower_bound) or is_casadi(upper_bound):
                        self.parametric_bounds_index_dict[id].append(proper_bounds_index)
                    proper_bounds_index += 1
            self.lbx_dict[id] = [0] * theta_dict[id].size()[0] + lbx_temp_list[self.nu:] * settings.nb_observations + \
                                [-float('inf')] * self.nb_lambdas_dict[id] + [0] * self.nb_mus_dict[id]
            self.ubx_dict[id] = [float('inf')] * theta_dict[id].size()[0] + ubx_temp_list[self.nu:] * settings.nb_observations + \
                                [float('inf')] * (self.nb_lambdas_dict[id] + self.nb_mus_dict[id])

        # Initialize dictionaries for optimization variables, online learning problems and their solvers
        self.observation_variables = {}
        self.observation_problem_dict = {}
        self.observation_solver_dict = {}

        # Initialize blocks of variables
        self.block_indices = {}
        self.block_keys = {}

        # Set-up the symbolics, Lagrangian, constraints etc for the online learning methodology
        for id in self.id_list:
            # The cost function of the observer
            V_full = symbolics()
            # The parameter vector of the observer
            p_full = symbolics()
            # The penalty vector of the observer
            penalty_full = symbolics()
            # The vector containing the optimization variables (v denotes the 'additional' optimization variables)
            x_v_full = symbolics()
            # The full vector of equality constraints for the observer
            g_full = symbolics()
            # The full vector of inequality constraints for the observer
            h_full = symbolics()
            # The full vector of inequality constraints for the observer corresponding to parametric rectangle bounds
            h_full_bounds = symbolics()
            # The full vector of Lagrangian parameters mu for the observer
            mu_full = symbolics()
            # The full vector of mu times h for the observer
            mu_times_h_full = symbolics()
            # The full vector of mu times h for the observer
            mu_times_h_full_bounds = symbolics()
            # The full vector of Lagrangian parameters lambda for the observer
            lambda_full = symbolics()

            # Make lists for x0, x and the parameters
            x0_list = [x0_dict[key] for key in x0_dict]
            x_list = [self.x_dict[key] for key in self.x_dict]
            parameters_list = [p_dict[key] for key in p_dict]

            # Iterate over the number of previous observations used to update the parameter
            for t in range(settings.nb_observations):
                # Make CasADi symbolics for optimization variables and parameters at this time stamp
                x0_step = symbolics()
                x_step = symbolics()
                v_step = symbolics()
                parameters_step = symbolics()
                p_step = symbolics()
                u_step = symbolics()
                lambda_step = symbolics.sym('lambda_step_' + str(t), len(g) + len(g_dyn_dict[id]), 1)
                mu_step = symbolics.sym('mu_step_' + str(t), len(h) + len(player_h[id]), 1)

                # Initialize the box constraints
                def get_bound(lower, upper, var):
                    try:
                        if lower == -float('inf'):
                            return upper - var
                    except RuntimeError:
                        pass
                    try:
                        if upper == float('inf'):
                            return var - lower
                    except RuntimeError:
                        pass
                    return (upper - var) * (var - lower)

                mu_step_bounds = symbolics.sym('', 0, 1)
                h_step_bounds = symbolics.sym('', 0, 1)
                for index, bounds in enumerate(zip(lbx_dict[id], ubx_dict[id])):
                    lower_bound, upper_bound = bounds
                    if proper_bounds(lower_bound, upper_bound):
                        mu_step_bounds = cs.vertcat(mu_step_bounds, symbolics.sym(
                            'mu_step_bounds_' + str(t) + '_' + str(index), 1, 1))
                        variable = self.x_dict[id][index] if index < self.nb_x_dict[id] else v[index - self.nb_x_dict[id]]
                        bound = get_bound(lower_bound, upper_bound, variable)
                        h_step_bounds = cs.vertcat(h_step_bounds, bound)

                # Set-up Lagrangian at the current time stamp
                cost = -self.f_dict[id]
                if len(g) + len(g_dyn_dict[id]) > 0:
                    cost -= cs.dot(lambda_step, cs.vertcat(*g, *g_dyn_dict[id]))
                if len(h) + len(player_h[id]) > 0:
                    cost -= cs.dot(mu_step, cs.vertcat(*h, *player_h[id]))
                cost -= cs.sum1(mu_step_bounds * h_step_bounds)
                lagrangian = cs.jacobian(cost, cs.vertcat(self.x_dict[id], v))

                # Set-up x0 variables
                for i in x0_dict:
                    sym = symbolics.sym('x0_' + str(i) + '_step_' + str(t), x0_dict[i].shape)
                    x0_step = cs.vertcat(x0_step, sym)
                    p_step = cs.vertcat(p_step, sym)

                # Set-up x variables
                for i in self.x_dict:
                    if i != id:
                        sym = symbolics.sym('x_' + str(i) + '_step_' + str(t), self.x_dict[i].shape)
                        x_step = cs.vertcat(x_step, sym)
                        p_step = cs.vertcat(p_step, sym)
                    else:
                        # For the ego player, the control variables for the first time step are observed
                        sym_p = symbolics.sym('x_' + str(i) + '_step_' + str(t), self.nu, 1)
                        sym_u = symbolics.sym('x_' + str(i) + '_step_' + str(t), self.x_dict[i].shape[0] - self.nu, 1)
                        x_step = cs.vertcat(x_step, sym_p, sym_u)
                        p_step = cs.vertcat(p_step, sym_p)
                        u_step = sym_u

                # Set-up parameters
                for i in p_dict:
                    sym = symbolics.sym('x0_' + str(i) + '_step_' + str(t), p_dict[i].shape)
                    parameters_step = cs.vertcat(parameters_step, sym)
                    p_step = cs.vertcat(p_step, sym)

                # Set-up penalty parameters
                penalty_step = symbolics.sym('rho_step_' + str(t), penalty_parameters.shape[0])

                # Set-up additional optimization variables
                v_step = cs.vertcat(v_step, symbolics.sym('v_step_' + str(t), v.shape))

                # Substitute Lagrangian and constraints with new optimization variables
                V_step = cs.substitute(lagrangian,
                                       cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                       cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))
                g_step = cs.substitute(cs.vertcat(*g, *g_dyn_dict[id]),
                                       cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                       cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))
                h_step = cs.substitute(cs.vertcat(*h, *player_h[id]),
                                       cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                       cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))
                h_step_bounds = cs.substitute(h_step_bounds,
                                              cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                              cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))

                # Add new variables to the full vectors storing this information
                x_v_full = cs.vertcat(x_v_full, u_step, v_step)
                V_full = cs.horzcat(V_full, V_step)
                g_full = cs.vertcat(g_full, g_step)
                h_full = cs.vertcat(h_full, h_step)
                h_full_bounds = cs.vertcat(h_full_bounds, h_step_bounds[self.parametric_bounds_index_dict[id]])
                p_full = cs.vertcat(p_full, p_step)
                penalty_full = cs.vertcat(penalty_full, penalty_step)
                mu_full = cs.vertcat(mu_full, mu_step, mu_step_bounds)
                if h_step.shape[0] > 0:
                    mu_times_h_full = cs.vertcat(mu_times_h_full, mu_step * h_step)
                if h_step_bounds.shape[0] > 0:
                    mu_times_h_full_bounds = cs.vertcat(mu_times_h_full_bounds, mu_step_bounds * h_step_bounds)
                lambda_full = cs.vertcat(lambda_full, lambda_step)

            # Add regularization term
            self.two_norm_lagrangian = cs.sumsqr(V_full)
            current_estimate_parameters = symbolics.sym('current_estimate_parameters', theta_dict[id].shape)
            try:
                regularization = symbolics.sym('reg', len(settings.regularization))
            except:
                regularization = symbolics.sym('reg')
            V_full = cs.sumsqr(V_full) + cs.sum1(regularization * (theta_dict[id] - current_estimate_parameters) ** 2)

            # Set up parameter vector
            p_full = cs.vertcat(p_full, penalty_full, current_estimate_parameters, regularization)

            # Store all required information in a single dictionary
            self.observation_variables[id] = {'theta': theta_dict[id], 'x_v': x_v_full, 'lambda': lambda_full,
                                              'mu': mu_full, 'f': V_full, 'p': p_full, 'g': g_full, 'h': h_full,
                                              'h_bounds': h_full_bounds, 'mu_times_h': mu_times_h_full,
                                              'mu_times_h_bounds': mu_times_h_full_bounds}

            # Set up the optimization problem
            self.setup_solver()

    def reset(self):
        return

    def setup_solver(self):
        """ Builds the solver for the online learning methodology for each human player with unknown parameters """
        print("setup observer")
        for id in self.id_list:
            self.observation_variables[id]['x'] = cs.vertcat(self.theta_dict[id], self.observation_variables[id]['x_v'],
                                self.observation_variables[id]['lambda'], self.observation_variables[id]['mu'])
            keys = ['theta', 'x_v', 'lambda', 'mu']
            current_index = 0
            self.block_indices[id] = {}
            self.block_keys[id] = []
            for key in keys:
                sx = self.observation_variables[id][key]
                if sx.shape[0] > 0:
                    new_index = current_index + sx.shape[0]
                    self.block_indices[id][key] = (current_index, new_index)
                    self.block_keys[id].append(key)
                    current_index = new_index

            # set-up solver for online learning methodology
            if self.settings.solver == 'ipopt':
                g = cs.vertcat(self.observation_variables[id]['g'], self.observation_variables[id]['h'],
                               self.observation_variables[id]['h_bounds'], self.observation_variables[id]['mu_times_h'],
                               self.observation_variables[id]['mu_times_h_bounds'])
                self.observation_problem_dict[id] = {'x': self.observation_variables[id]['x'],
                                                     'f': self.observation_variables[id]['f'],
                                                     'p': self.observation_variables[id]['p'],
                                                     'g': g}
                self.observation_solver_dict[id] = cs.nlpsol('Observer_' + str(id), 'ipopt',
                                                             self.observation_problem_dict[id],
                                                             {'verbose_init': False, 'print_time': False,
                                                              'ipopt': {'print_level': 3,
                                                                        'acceptable_tol': self.settings.ipopt_acceptable_tolerance,
                                                                        'tol': self.settings.ipopt_tolerance,
                                                                        'mu_strategy': 'adaptive',
                                                                        'nlp_scaling_method': 'gradient-based',
                                                                        'max_soc': 4}})
            else:
                self.primal_factor = self.settings.open_delta_tolerance / self.settings.open_delta_tolerance_primal_feas
                self.bounds_factor = self.settings.open_delta_tolerance / self.settings.open_delta_tolerance_bounds
                self.complementarity_factor = self.settings.open_delta_tolerance / self.settings.open_delta_tolerance_complementarity
                self.complementarity_bounds_factor = self.settings.open_delta_tolerance / self.settings.open_delta_tolerance_complementarity_bounds
                if self.settings.rebuild_solver:
                    # Initialize solver for 'OpEn'
                    factor = self.symbolics.sym('factor', 4)
                    g = cs.vertcat(factor[0]*self.observation_variables[id]['g'],
                                   factor[0]*self.observation_variables[id]['h'],
                                   factor[1]*self.observation_variables[id]['h_bounds'],
                                   factor[2]*self.observation_variables[id]['mu_times_h'],
                                   factor[3]*self.observation_variables[id]['mu_times_h_bounds'])
                    # Set settings for PANOC
                    solver_config = og.config.SolverConfiguration() \
                        .with_initial_tolerance(self.settings.open_initial_tolerance) \
                        .with_tolerance(self.settings.open_tolerance) \
                        .with_delta_tolerance(self.settings.open_delta_tolerance) \
                        .with_initial_penalty(self.settings.open_initial_penalty) \
                        .with_penalty_weight_update_factor(self.settings.open_penalty_weight_update_factor) \
                        .with_max_inner_iterations(self.settings.max_inner_iterations) \
                        .with_max_outer_iterations(self.settings.max_outer_iterations) \
                        .with_max_duration_micros(100000)

                    # Remove constraints which are independent from the optimization variables
                    valid_constraint_indices = []
                    for constraint_index, constraint in enumerate(g.elements()):
                        if cs.depends_on(constraint, self.observation_variables[id]['x']):
                            valid_constraint_indices.append(constraint_index)

                    bounds_x = og.constraints.Rectangle(self.lbx_dict[id], self.ubx_dict[id])
                    ubg_list = [self.ubg_dict[id][j] for j in valid_constraint_indices]
                    bounds_g = og.constraints.Rectangle([0]*len(ubg_list), ubg_list)

                    # Create problem for current player and build the optimizer
                    self.observation_problem_dict[id] = \
                        og.builder.Problem(self.observation_variables[id]['x'],
                                           cs.vertcat(self.observation_variables[id]['p'], factor),
                                           self.observation_variables[id]['f']) \
                            .with_constraints(bounds_x) \
                            .with_aug_lagrangian_constraints(g[valid_constraint_indices], bounds_g)
                            # .with_penalty_constraints(cs.vertcat(
                            # *[cs.fmin(0.0, constraint) if self.ubg_dict[id][valid_constraint_indices[j]] == float("inf") else constraint for
                            #   j, constraint in enumerate(g[valid_constraint_indices].elements())]))


                    tcp_config = og.config.TcpServerConfiguration('127.0.0.1', 7000 + 10 * self.id + id)
                    build_config = og.config.BuildConfiguration() \
                        .with_build_directory("og_builds/" + self.settings.directory_name) \
                        .with_build_mode(self.settings.build_mode) \
                        .with_tcp_interface_config(tcp_config)
                        # .with_build_python_bindings()

                    meta = og.config.OptimizerMeta() \
                        .with_optimizer_name("learning_" + str(self.id) + "_" + str(id))
                    builder = og.builder.OpEnOptimizerBuilder(self.observation_problem_dict[id], meta, build_config,
                                                              solver_config)
                    builder.build()

            # Start TCP servers for calling optimizers
            self.observation_solver_dict[id] = og.tcp.OptimizerTcpManager("og_builds/" + self.settings.directory_name
                                                                          + "/learning_" + str(self.id) + "_" + str(id))
            self.observation_solver_dict[id].start()
            self.observation_solver_dict[id].ping()

            # Connect to python bindings
            # import sys
            # sys.path.insert(1, "./og_builds/one_dimensional_gpg/learning_0_1")
            # import learning_0_1
            # solver = learning_0_1.solver()

    def observe(self, current_estimate_parameters_dict, observations):
        """ Initializes the bounds for the players in the GPG

        Parameters
        ----------
        current_estimate_parameters_dict : dict
            the current estimate for the parameters of the human players
        observations : deque object
            contains the required information for initializing and warm-starting the online learning methodology
        """
        if observations.maxlen == 0 or len(observations) < observations.maxlen:
            return current_estimate_parameters_dict
        new_estimate_parameters_dict = {}

        # Set-up optimization variables and parameter lists in the correct form based on the given observation object
        for id in self.id_list:
            p_numeric = []
            penalty_numeric = []
            x_v0_numeric = []
            lam_numeric_g = []
            lam_numeric_bounds = []
            for observation in observations:
                p_step = []
                u_step = 0
                lam_x_step = 0
                lam_g_step = 0
                if len(observation) == 8:
                    for i, value in observation[0].items():
                        p_step = cs.vertcat(p_step, value)
                    for i, value in observation[1].items():
                        if i != id:
                            p_step = cs.vertcat(p_step, value)
                        else:
                            p_step = cs.vertcat(p_step, observation[3][i][:self.nu])
                            u_step = value[self.nu:]
                            lam_x_step = observation[4][i]
                            lam_g_step = observation[5][i]
                    for i, value in observation[6].items():
                        p_step = cs.vertcat(p_step, value)
                    v_step = observation[2]
                    penalty_step = observation[7]

                    p_numeric = cs.vertcat(p_numeric, p_step)
                    penalty_numeric = cs.vertcat(penalty_numeric, penalty_step)
                    x_v0_numeric = cs.vertcat(x_v0_numeric, u_step, v_step)
                    lam_numeric_g = cs.vertcat(lam_numeric_g, lam_g_step)
                    # indices = [index for index in range(lam_x_step.shape[0]) if #TODO Why was this code here?
                    #            index not in self.parametric_bounds_index_dict[id]]
                    # lam_numeric_bounds = cs.vertcat(lam_numeric_bounds, lam_x_step[indices])
                    lam_numeric_bounds = cs.vertcat(lam_numeric_bounds, lam_x_step)
                else:
                    raise Exception('Observation format is incorrect')
            if cs.vertcat(lam_numeric_g, lam_numeric_bounds).size()[0] == 0:
                lam_numeric_g = cs.DM.zeros(self.nb_mus_dict[id], 1)
                lam_numeric_bounds = cs.DM.zeros(0, 1)
            p_numeric = cs.vertcat(p_numeric, penalty_numeric, current_estimate_parameters_dict[id], self.settings.regularization)
            x = cs.vertcat(current_estimate_parameters_dict[id], x_v0_numeric, lam_numeric_g, lam_numeric_bounds)
            # Solve online learning optimization problem
            if self.settings.solver == 'ipopt':
                solution = self.observation_solver_dict[id](x0=x,
                                                            p=p_numeric,
                                                            lbx=self.lbx_dict[id],
                                                            ubx=self.ubx_dict[id],
                                                            lbg=0,
                                                            ubg=self.ubg_dict[id])
                x = solution['x']
            else:
                solution = self.observation_solver_dict[id].call(
                    cs.vertcat(p_numeric, self.primal_factor, self.bounds_factor,
                               self.complementarity_factor, self.complementarity_bounds_factor).toarray(True),
                    initial_guess=x.toarray(True))
                try:
                    if solution['exit_status'] != 'Converged':
                        print(solution['exit_status'])
                except:
                    print(solution)
                    print(solution.get().message)
                self.cpu_time = solution['solve_time_ms']/1000
                x = solution['solution']

            print('Online learning:  2-norm of Lagrangian: ' + str(
                cs.substitute(self.two_norm_lagrangian,
                              cs.vertcat(self.observation_variables[id]['x'], self.observation_variables[id]['p']),
                              cs.vertcat(x, p_numeric))))
            a, b = self.block_indices[id]['theta']
            new_estimate_parameters_dict[id] = x[a:b]
            print('New estimate parameters = ' + str(new_estimate_parameters_dict[id]))
        return new_estimate_parameters_dict