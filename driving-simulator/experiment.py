import settings
import world


class Experiment(object):
    """
    A class used to represent experiments

    Attributes
    ----------
    name_experiment : str
        the name of the world of the experiment
    experiment_variant : str, optional
        the variant of the experiment
    pyglet_visualization_settings : PygletVisualizationSettings object
        the visualization settings for the pyglet window
    data_visualization_windows : dict
        the data visualization windows, shown as pop-up matplotlib plots
    logger_settings: LoggerSettings object
        the settings for the data logger of the experiment
    """
    def __init__(self, name_experiment, experiment_variant=None):
        # Initialize variables
        self.name_experiment = name_experiment
        self.experiment_variant = experiment_variant
        self.world = None
        self.data_visualization_windows = None

        # Initialize default settings
        if experiment_variant is None:
            directory_name = name_experiment
        else:
            directory_name = name_experiment
            for i, param in enumerate(experiment_variant):
                if i == 0:
                    directory_name += '/' + param
                else:
                    directory_name += '-' + param
        self.solver_settings = settings.GPGSolverSettings(name_experiment)
        self.learning_settings = settings.OnlineLearningSettings(name_experiment)
        self.logger_settings = settings.LoggerSettings(directory_name)
        self.pyglet_visualization_settings = settings.PygletVisualizationSettings()

        # Look for presets and build world
        self.preset_settings()

    def preset_settings(self):
        if self.name_experiment == 'one_dimensional_gpg':
            ## Solver settings
            # solver_settings.solver = 'ipopt'
            self.solver_settings.constraint_mode = 'hard'
            self.solver_settings.initial_penalty = 10 #exact 0.1, quadratic 10
            self.solver_settings.penalty_update_factor = 2 #exact and quadratic 2
            self.solver_settings.rebuild_solver = False
            self.solver_settings.game_tolerance = 1e-3
            self.solver_settings.penalty_method = 'quadratic'
            self.solver_settings.penalty_eta = 1e-3
            # self.solver_settings.panoc_initial_penalty = 0.1
            # self.solver_settings.panoc_delta_tolerance = 1e-2
            self.solver_settings.max_outer_iterations = 20
            self.solver_settings.max_inner_iterations = 100

            self.solver_tolerance = 1e-5
            self.panoc_delta_tolerance = 1e-5

            ## Learning settings
            if self.experiment_variant[0] == "learning":
                self.learning_settings.nb_observations = 1
                self.learning_settings.regularization = [1e3, 1e1, 1e5, 1e5]
                self.learning_settings.rebuild_solver = True
                self.learning_settings.based_on_original_gpg = True
                self.learning_settings.open_tolerance = 1e-6
                self.learning_settings.open_initial_tolerance = 1e-4
                self.learning_settings.open_delta_tolerance = 1e-4
                self.learning_settings.open_delta_tolerance_primal_feas = 1e-4
                self.learning_settings.open_delta_tolerance_bounds = 1e-5
                self.learning_settings.open_delta_tolerance_complementarity = 1e-6
                self.learning_settings.open_delta_tolerance_complementarity_bounds = 1e-7

        elif self.name_experiment == 'merging_scenario':
            ## Logger settings
            self.logger_settings.nb_iterations_experiment = 60

            ## Visualisation settings
            self.pyglet_visualization_settings.magnify = 1.0
            self.pyglet_visualization_settings.camera_offset = [20, 1.5]
            self.pyglet_visualization_settings.width_factor = 2000/1250
            self.pyglet_visualization_settings.height_ratio = 8
            self.pyglet_visualization_settings.id_main_car = 2

            ## Solver settings
            self.solver_settings.initial_penalty = 10
            self.solver_settings.penalty_update_factor = 1.5
            self.solver_settings.penalty_eta = 0.1
            self.solver_settings.warm_start = True
            self.solver_settings.rebuild_solver = False
            self.solver_settings.game_tolerance = 1e-2
            self.solver_settings.panoc_delta_tolerance = 1e-1
            self.solver_settings.max_inner_iterations = 50
            self.solver_settings.max_outer_iterations = 20
            self.solver_settings.panoc_max_inner_iterations = 1e3

            self.solver_settings.constraint_mode = 'hard'

            ## Learning settings
            if self.experiment_variant[0] == "learning":
                self.learning_settings.nb_observations = 3
                self.learning_settings.regularization = 1
                self.learning_settings.rebuild_solver = False
                self.learning_settings.open_tolerance = 1e-5  # OpEn default = 1e-5
                self.learning_settings.open_initial_tolerance = 1e-4
                self.learning_settings.open_delta_tolerance_primal_feas = 1e-1
                self.learning_settings.open_delta_tolerance_bounds = 1e-4
                self.learning_settings.open_delta_tolerance_complementarity = 1e-3
                self.learning_settings.open_delta_tolerance_complementarity_bounds = 1e-3
        return

    def build_world(self):
        try:
            self.world = getattr(world, 'world_' + self.name_experiment)(self.solver_settings, self.learning_settings,
                                                                         *self.experiment_variant[1:])
            self.data_visualization_windows = settings.default_data_visualization_windows(self.world)
        except:
            pass
        return

    def set_data_visualization_window(self, window, id_list):
        """ Adds a visualization for the current run

        See plot_trajectories.py for more info

        Parameters
        ----------
        window : str
            name of the visualized variable
        id_list : list
            list of ids relevant for visualizing the variable
        """
        self.data_visualization_windows[window] = id_list

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['world']
        del odict['data_visualization_windows']
        return odict

    def __setstate__(self, state):
        self.__dict__ = state
        self.build_world()
