import car

class PygletVisualizationSettings(object):
    """
        A class used to represent the logger settings

        Attributes
        ----------
        magnify : float
            the magnification factor
        height_ratio : float
            the ratio width/height of the pyglet window
        width_factor : float
            the magnification factor of the width, factor 1 equals 1000 pixels
        camera_offset : list
            offset of the camera center
    """
    def __init__(self):
        self.magnify = 0.5
        self.height_ratio = 5
        self.width_factor = 1
        self.camera_offset = [10., 0]
        self.show_live_data = False
        self.live_data_box_position = (10, 10)
        self.live_data_border_size = 10
        self.id_main_car = 0


def default_data_visualization_windows(world):
    id_list = [vehicle.id for vehicle in world.cars]
    id_list_gpg = [vehicle.id for vehicle in world.cars if isinstance(vehicle, car.GPGOptimizerCar)]
    default = {
        'velocity': id_list,
        'acceleration': id_list,
        'steering angle': id_list,
        'cost': id_list_gpg,
        'potential': [],
        'effective constraint violation': id_list_gpg,
        'belief': [0, 1]
    }
    return default


class LoggerSettings(object):
    """
        A class used to represent the logger settings

        Attributes
        ----------
        nb_iterations_experiment : int
            the number of iterations of the experiment to log
        name_experiment : str
            the name of the experiment
        save_video : boolean
            determines whether a video of the current experiment should be saved
        only_save_statistics : boolean
            determines whether only the computation statistics of the current experiment should be saved
        statistics_index : int
            the index of the resulting statistics file
    """
    def __init__(self, name_experiment):
        self.nb_iterations_experiment = 150
        self.name_experiment = name_experiment
        self.save_video = True
        self.only_save_statistics = False
        self.statistics_index = None


class GPGSolverSettings(object):
    """
        A class used to represent the gpg solver settings

        Attributes
        ----------
        shooting_mode : str
            the shooting mode for the optimal control problems of the GNEPOptimizerCars, i.e. 'single' or 'multiple'
        constraint_mode : str
            the constraint mode for the common constraints of the GPG formulations, i.e. 'hard' or 'soft'
        sorted : boolean
            determines whether the ids of the vehicles in the decomposition method for solving the GPG are sorted
        warm_start : boolean
            determines whether penalty parameters are warm started
        regularization : float
            the regularization parameter tau for the Gauss-Seidel best-response algorithm
        initial_penalty : int
            the initial value of the penalty parameters
        penalty_update_factor : int
            the update factor alpha for the penalty parameters
        max_outer_iterations : int
            the maximum amount of outer iterations for the quadratic penalty method
        penalty_eta : float
            the allowed constraint violation for the shared constraints
        solver : str
            the used solver for the optimal control problems of the GNEPOptimizerCars, i.e. 'ipopt' or 'OpEn'
        solver_tolerance : float
            the tolerance of the applied solver
        max_inner_iterations : int
            the maximum number of iterations for the Gauss-Seidel best-response algorithm
        game_tolerance : float
            the stopping criterion for the Gauss-Seidel outer iterations, delta_solution < game_tolerance
        build_mode : str
            build mode for Open: 'debug' = fast compilation, 'release' = fast runtime
        rebuild_solver : boolean
            determines whether the solver needs to be recompiled, set False for successive experiments with same code
        directory_name : str
            name of the subdirectory in og_builds for storing the compiled code
        debug_rust_code : boolean
            whether the log statements from Rust should be printed
    """
    def __init__(self, directory_name="default"):
        self.shooting_mode = 'single'
        self.constraint_mode = 'soft'
        self.penalty_method = 'quadratic'
        self.sorted = True
        self.warm_start = False
        self.regularization = 0.
        self.initial_penalty = 1
        self.penalty_update_factor = 5
        self.penalty_norm = 'norm_inf'
        self.penalty_update_rule = 'individual'
        self.max_outer_iterations = 10
        self.penalty_eta = 1e-2
        self.solver = 'OpEn'
        self.solver_tolerance = 1e-4                # OpEn default = 1e-4, ipopt default 1e-8
        self.max_inner_iterations = 50
        self.game_tolerance = 1e-2
        self.build_mode = 'release'
        self.rebuild_solver = True
        self.directory_name = directory_name
        self.panoc_max_outer_iterations = 10        # OpEn default 10
        self.panoc_max_inner_iterations = 500       # OpEn default 500
        self.panoc_initial_penalty = 1              # OpEn default 1
        self.panoc_delta_tolerance = 1e-4           # OpEn default 1e-4
        self.debug_rust_code = False
        self.use_gauss_seidel = False


class OnlineLearningSettings(object):
    """
        A class used to represent the online learning settings

        Attributes
        ----------
        regularization : float
            the regularization parameter of the online learning methodology
        nb_observations
            the amount of previous observations used to update the estimate
        ipopt_tolerance : float
            the tolerance of the ipopt solver
        ipopt_acceptable_tolerance : float
            the tolerance of the ipopt solver
        open_tolerance : float
            the tolerance of the OpEn solver
        open_initial_tolerance : float
            the initial tolerance of the OpEn solver
        build_mode : str
            build mode for Open: 'debug' = fast compilation, 'release' = fast runtime
        rebuild_solver : boolean
            determines whether the OpEn solver needs to be recompiled, False for successive experiments with same code
        directory_name : str
            name of the subdirectory in og_builds for storing the compiled code
    """
    def __init__(self, directory_name="default"):
        self.regularization = 1
        self.nb_observations = 0
        self.solver = 'OpEn'
        self.ipopt_tolerance = 1e-3 # ipopt default 1e-8
        self.ipopt_acceptable_tolerance = 1e-1
        self.open_tolerance = 1e-6 # OpEn default = 1e-5
        self.open_initial_tolerance = 1e-4
        self.open_delta_tolerance = 1e-2
        self.open_delta_tolerance_primal_feas = 1e-2
        self.open_delta_tolerance_bounds = 1e-4
        self.open_delta_tolerance_complementarity = 1e-3
        self.open_delta_tolerance_complementarity_bounds = 1e-3
        self.open_initial_penalty = 1
        self.open_penalty_weight_update_factor = 2
        self.build_mode = 'release'  # Open build mode: 'debug' = fast compilation, 'release' = fast runtime
        self.rebuild_solver = True
        self.directory_name = directory_name
        self.max_outer_iterations = 20
        self.max_inner_iterations = 1e5
        self.based_on_original_gpg = False

