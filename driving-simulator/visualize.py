import pyglet
from pyglet.window import key
import pyglet.gl as gl
import pyglet.graphics as graphics
import numpy as np
import time
import math
import logger

import car
import matplotlib.cm
import visualize_data
import os
import datetime
import multiprocessing as mp

# MULTIPROCESSING HACKS

_func = None


def worker_init(func):
    global _func
    _func = func


def worker(x):
    return _func(x)


class Visualizer(object):
    """
    A class used to visualize the behaviour of the vehicles over time in the traffic scenario
    """
    def __init__(self, experiment, fullscreen=False):
        """
        Parameters
        ----------
        experiment: Experiment object
            the experiment which should be handled
        fullscreen : boolean, optional
            determines whether the window should be fullscreen
        """
        # Set-up window settings
        self.width_ratio = experiment.pyglet_visualization_settings.width_factor
        self.height_ratio = experiment.pyglet_visualization_settings.height_ratio
        self.width = int(1250 * self.width_ratio)
        self.height = self.width // self.height_ratio
        self.window = pyglet.window.Window(self.width, self.height, fullscreen=fullscreen)
        # gl.glScalef(1.0, 1.0 * self.height_ratio, 0.0)
        self.window.on_draw = self.on_draw
        self.keys = key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.window.on_key_press = self.on_key_press
        self.window.on_mouse_drag = self.on_mouse_drag
        self.window.on_mouse_scroll = self.on_mouse_scroll
        self.magnify = experiment.pyglet_visualization_settings.magnify
        self.camera_center = None
        self.camera_offset = experiment.pyglet_visualization_settings.camera_offset
        self.show_live_data = experiment.pyglet_visualization_settings.show_live_data
        self.live_data_border_size = experiment.pyglet_visualization_settings.live_data_border_size
        self.live_data_box_position = experiment.pyglet_visualization_settings.live_data_box_position

        # Set-up pyglet variables
        self.iters = 1000  # todo hard-coded
        self.event_loop = pyglet.app.EventLoop()
        self.grass = pyglet.resource.texture('images/grass.png')
        self.paused = False

        # Initialize variables for the lanes and for the cars along with their positions
        self.cars = [c for c in experiment.world.cars]
        self.roads = [r for r in experiment.world.roads]
        self.Ts = experiment.world.Ts
        self.visible_cars = []
        self.anim_x = {}
        self.prev_x = {}
        self.main_car = None

        # Initialize variables for visualizing cars and bounding boxes
        self.show_trajectory_mode = 1
        self.show_bounding_box = True

        # Initialize heatmap variables
        self.heatmap_x1 = None
        self.heatmap_x0 = None
        self.heat = None
        self.heatmap = None
        self.heatmap_valid = False
        self.heatmap_show = False
        self.cm = matplotlib.cm.jet
        self.heatmap_size = (64, 64)

        # Settings for visualization windows when pressing 'ESC' button
        self.data_visualization_windows = experiment.data_visualization_windows

        # Setting for saving the experiment
        self.logger = logger.Logger(experiment)

        # Current iteration of the world
        self.current_iteration = 0

        # Define colors
        self.colors_dict = {'red': [1., 0., 0.], 'yellow': [1., 1., 0.], 'purple': [0., 0.5, 0.5],
                            'white': [1., 1., 1.], 'orange': [1., 0.5, 0.], 'gray': [0.2, 0.2, 0.2],
                            'blue': [0., 0.7, 1.]}

        # Initialize required variables for live data visualization
        self.live_data_shown = ['x', 'y', 'angle', 'velocity', 'acceleration', 'steering angle', 'stage cost', 'cost',
                                'effective constraint violation', 'planned constraint violation']
        [x_min, x_max, y_min, y_max] = self.initialize_live_data_window()
        self.text_box_background = pyglet.sprite.Sprite(pyglet.image.load('images/gray_box.png'), 0, 0)
        self.text_box_background.scale_x = (x_max - x_min) / 150
        self.text_box_background.scale_y = (y_max - y_min) / 200
        self.text_box_background.position = (x_min, y_min)
        self.text_box_background.opacity = 150

        # Set main car and heatmap
        self.main_car = experiment.world.cars[experiment.pyglet_visualization_settings.id_main_car]
        if isinstance(self.main_car, car.GPGOptimizerCar):
            self.set_heat(experiment.world.cars[experiment.pyglet_visualization_settings.id_main_car].reward)

        def centered_image(filename):
            """ Returns a centered image from a given image file

            Parameters
            ----------
            filename : str
                the location of the given image
            """
            img = pyglet.resource.image(filename)
            img.anchor_x = img.width/2.
            img.anchor_y = img.height/2.
            return img

        def car_sprite(color, scale=4./600):
            """ Returns the sprite of a car

            Parameters
            ----------
            color : str
                the color of the car for the sprite
            scale : float
                the scale of the image, i.e. (length in meters) / (number of pixels)
            """
            sprite = pyglet.sprite.Sprite(centered_image('images/car-{}.png'.format(color)), subpixel=True)
            sprite.scale = scale
            return sprite

        # Initialize sprites
        self.sprites = {c: car_sprite(c) for c in ['red', 'yellow', 'purple', 'white', 'orange', 'gray', 'blue']}

    def initialize_live_data_window(self):
        x_min = self.live_data_box_position[0]
        x_max = x_min + 2*self.live_data_border_size
        y_max = self.window.height - self.live_data_box_position[1]
        y_min = self.window.height - 2*self.live_data_border_size - 20*len(self.live_data_shown) - 20

        def make_label(obj, name, nb_shifted):
            setattr(obj, 'label_' + name,
                    pyglet.text.Label(
                        name + ':  N/A',
                        font_name='Georgia',
                        font_size=12,
                        x=x_min + self.live_data_border_size,
                        y=y_max - self.live_data_border_size - 20*nb_shifted,
                        anchor_x='left', anchor_y='top',
                        color=(0, 0, 0, 255)
                    ))
        for shift, item in enumerate(self.live_data_shown):
            make_label(self, item, shift)
            x_max = max(x_max, x_min + 2*self.live_data_border_size + getattr(self, 'label_' + item).content_width)
        return [x_min, x_max, y_min, y_max]

    def kill_all(self):
        """ Kills the TCP servers of the OpEn optimizers of the GPGOptimizerCars, if any """
        for car in reversed(self.cars):
            try:
                for i in car.optimizer.id_list:
                    car.optimizer.solver_dict[i].kill()
            except:
                pass
            try:
                for i in car.optimizer.id_list:
                    car.observer.observation_solver_dict[i].kill()
            except:
                pass

    def save_screenshot(self, folder_name='screenshots', index=None):
        """ Saves a screenshot

        Parameters
        ----------
        folder_name : str
            the location of the folder to save the screenshot
        index : int
            the index of the image
        """
        if index is None:
            time_info = datetime.datetime.now()
            index = time_info.month*1e8 + time_info.day*1e6 + time_info.hour*1e4 + time_info.minute*1e2 +\
                time_info.second
        # Capture image from pyglet
        pyglet.image.get_buffer_manager().get_color_buffer().save(folder_name + '/screenshot-%d.png' % index)

    def on_key_press(self, symbol, *args):
        """ Defines what should be performed when a key is pressed

        Parameters
        ----------
        symbol : key attribute
            the pressed key
        """
        # the ESC button stops the experiment, kills all TCP servers and visualizes the requested information
        if symbol == key.ESCAPE:
            self.event_loop.exit()
            self.kill_all()
            if self.data_visualization_windows is not None:
                visualize_data.plot(self.logger.history, self.data_visualization_windows)

        # the P button saves a screenshot
        if symbol == key.P:
            self.save_screenshot()

        # the SPACE button pauses and unpauses the game
        if symbol == key.SPACE:
            self.paused = not self.paused

        # the H button can be used to show a heatmap
        if symbol == key.H:
            self.heatmap_show = not self.heatmap_show
            if self.heatmap_show:
                self.heatmap_valid = False

        # the T button can be used to change the visualization of the trajectories
        if symbol == key.T:
            self.show_trajectory_mode = (self.show_trajectory_mode+1)%4

        # the B button can be used to show bounding boxes of vehicles
        if symbol == key.B:
            self.show_bounding_box = not self.show_bounding_box

        # the L button can be used to show or hide the live data
        if symbol == key.L:
            self.show_live_data = not self.show_live_data

        # the K button can be used to kill all processes and close the window
        if symbol == key.K:
            self.event_loop.exit()
            self.kill_all()
            self.window.close()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.camera_center is None:
            self.camera_center = self.center()
        self.camera_center[0] -= (dx/self.width)*80*self.magnify
        self.camera_center[1] -= (dy/self.height)*(80/self.height_ratio)*self.magnify
        self.heatmap_valid = False

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.magnify *= (1 - 0.1*scroll_y)
        self.heatmap_valid = False

    def control_loop(self, _=None):
        if self.paused:
            return
        if self.iters is not None and self.current_iteration >= self.iters:
            self.event_loop.exit()
            return
        print('__________NEW GAME ITERATION__________')
        steer = 0.
        gas = 0.

        # Control the UserControlledCars using the arrow keys
        if self.keys[key.UP]:
            gas += 1.
        if self.keys[key.DOWN]:
            gas -= 1.
        if self.keys[key.LEFT]:
            steer += 0.2
        if self.keys[key.RIGHT]:
            steer -= 0.2

        # Set heatmap false again
        self.heatmap_valid = False

        # Update previous x variable
        for vehicle in self.cars:
            self.prev_x[vehicle] = vehicle.x

        # Calculate control actions for each vehicle
        for vehicle in reversed(self.cars):
            vehicle.control(steer, gas)

        # Update iteration
        self.current_iteration += 1

        # Log the data from the cars
        self.logger.log_data(self.cars)
        exit_status = self.logger.write_data_to_files(self.current_iteration)

        # Let vehicles observe actions of other drivers
        for vehicle in self.cars:
            if isinstance(vehicle, car.GPGOptimizerCar):
                vehicle.observe()

        # Move cars
        for vehicle in self.cars:
            vehicle.move()

        # Start animation loop
        self.animation_loop()

        # Stop experiment when logger has completed
        if exit_status == 1:
            self.event_loop.exit()

    def center(self):
        """ Returns the 'center' for the camera """
        if self.main_car is None:
            return np.asarray([0., 0.])
        elif self.camera_center is not None:
            return np.asarray(self.camera_center[0:2])
        else:
            return [self.anim_x[self.main_car][0] + self.camera_offset[0], self.camera_offset[1]]

    def camera_vision_vertices(self):
        o = self.center()
        return [o[0]-40*self.magnify, o[0]+40*self.magnify,
                o[1]-40*self.magnify/self.height_ratio, o[1]+40*self.magnify/self.height_ratio]

    def camera(self):
        """ Sets camera """
        gl.glOrtho(*self.camera_vision_vertices(), -1., 1.)

    def set_heat(self, f):
        """ Sets heatmap function """
        def val(p, other):
            return f([p[0], p[1], 0, 0], [0, 0], other)
        self.heat = val

    def draw_heatmap(self):
        """ Draws the heatmap """
        if not self.heatmap_show:
            return
        if not self.heatmap_valid:
            o = self.center()
            x0 = o - np.asarray([40., 40. / self.height_ratio]) * self.magnify
            x1 = o + np.asarray([40., 40. / self.height_ratio]) * self.magnify
            self.heatmap_x0 = x0
            self.heatmap_x1 = x1
            values = self.compute_heatmap_values(x0, x1)
            self.heatmap = self.values_to_img(values).get_texture()
            self.heatmap_valid = True
        gl.glClearColor(1., 1., 1., 1.)
        gl.glEnable(self.heatmap.target)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindTexture(self.heatmap.target, self.heatmap.id)
        gl.glEnable(gl.GL_BLEND)
        graphics.draw(4, gl.GL_QUADS,
                      ('v2f', (self.heatmap_x0[0], self.heatmap_x0[1], self.heatmap_x1[0], self.heatmap_x0[1],
                               self.heatmap_x1[0], self.heatmap_x1[1], self.heatmap_x0[0], self.heatmap_x1[1])),
                      ('t2f', (0., 0., 1., 0., 1., 1., 0., 1.)), )
        gl.glDisable(self.heatmap.target)

    def compute_heatmap_values(self, x0, x1):
        x_range = np.linspace(x0[0], x1[0], self.heatmap_size[0])
        y_range = np.linspace(x0[1], x1[1], self.heatmap_size[1])

        x_grid, y_grid = np.meshgrid(x_range, y_range)
        positions = np.vstack((x_grid.ravel(), y_grid.ravel()))

        values = np.zeros(self.heatmap_size)
        state_other = {}
        for i in self.cars[0].humans:
            state_other[i] = self.cars[i].x
        for i in self.cars[0].obstacles:
            state_other[i] = self.cars[i].x

        func = lambda pt: self.heat(pt, state_other)

        #with mp.Pool(None, initializer=worker_init, initargs=(func,)) as p:
        #worker_init(func)
        #with mp.Pool() as p:
        #    values = p.map(worker, positions.T)
        #values = np.reshape(values, self.heatmap_size)

        for i, x in enumerate(np.linspace(x0[0], x1[0], self.heatmap_size[0])):
            for j, y in enumerate(np.linspace(x0[1], x1[1], self.heatmap_size[1])):
                values[j, i] += np.float(func([x, y]))
        return values

    def values_to_img(self, values):
        values = (values-np.min(values))/(np.max(values)-np.min(values)+1e-6)
        values = self.cm(values)
        values[:, :, 3] = 0.7
        values = (values*255).astype(np.uint8)
        img = pyglet.image.ImageData(self.heatmap_size[0], self.heatmap_size[1], 'RGBA', values.tobytes())
        return img

    def animation_loop(self, _=None):
        """ Performs the animation loop, i.e. updates anim_x for the vehicles """
        for car in self.cars:
            self.anim_x[car] = car.center_x(car.x)
        return

    def draw_lane_surface(self, lane):
        """ Draws the surface of the given lane

        Parameters
        ----------
        lane : Lane object
            the given lane
        """
        gl.glColor3f(0.3, 0.3, 0.3)
        graphics.draw(4, gl.GL_QUAD_STRIP,
                      ('v2f', np.hstack([lane.p-lane.m*lane.length-0.55*lane.w*lane.n,
                                         lane.p-lane.m*lane.length+0.55*lane.w*lane.n,
                                         lane.p+lane.m*lane.length-0.55*lane.w*lane.n,
                                         lane.p+lane.m*lane.length+0.55*lane.w*lane.n])))

    def draw_simple_lane_lines(self, lane):
        """ Draws simple white lines for the given lane

        Parameters
        ----------
        lane : Lane object
            the given lane
        """
        gl.glColor3f(1., 1., 1.)
        gl.glLineWidth(1 * self.width_ratio / self.magnify)
        graphics.draw(2, gl.GL_LINES,
                      ('v2f', np.hstack([lane.p - lane.m*lane.length-0.5*lane.w*lane.n,
                                         lane.p + lane.m*lane.length-0.5*lane.w*lane.n])))
        graphics.draw(2, gl.GL_LINES,
                      ('v2f', np.hstack([lane.p + lane.m*lane.length+0.5*lane.w*lane.n,
                                         lane.p - lane.m*lane.length+0.5*lane.w*lane.n])))

    def draw_road(self, road):
        """ Draws the given road

        Parameters
        ----------
        road : Road object
            the given road
        """
        for lane in road.lanes:
            self.draw_lane_surface(lane)
        gl.glLineWidth(1 * self.width_ratio / self.magnify)

        def left_line(lane, k):
            return [np.hstack([lane.p - lane.m * lane.length + 0.5 * lane.w * lane.n,
                        lane.p + lane.m * lane.length + 0.5 * lane.w * lane.n]),
                    np.hstack([lane.p + lane.m * lane.length + 0.5 * lane.w * lane.n,
                               lane.p + lane.m * road.lanes[k-1].length + 0.5 * lane.w * lane.n])]

        def right_line(lane, k):
            return [np.hstack([lane.p - lane.m * lane.length - 0.5 * lane.w * lane.n,
                        lane.p + lane.m * lane.length - 0.5 * lane.w * lane.n]),
                    np.hstack([lane.p + lane.m * lane.length - 0.5 * lane.w * lane.n,
                               lane.p + lane.m * road.lanes[k-1].length - 0.5 * lane.w * lane.n])]

        def end_line(lane):
            return np.hstack([lane.p + lane.m * lane.length - 0.5 * lane.w * lane.n,
                               lane.p + lane.m * lane.length + 0.5 * lane.w * lane.n])

        if len(road.lanes) == 1:
            self.draw_simple_lane_lines(road.lanes[0])
        else:
            for k, lane in enumerate(road.lanes):
                if k == 0:
                    gl.glColor3f(1., 1., 0.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', right_line(lane, k)[0]))
                    graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
                elif k == len(road.lanes)-1:
                    gl.glLineStipple(int(36 / self.magnify), 0x1111)
                    gl.glEnable(gl.GL_LINE_STIPPLE)
                    gl.glColor3f(1., 1., 1.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', right_line(lane, k)[0]))
                    gl.glDisable(gl.GL_LINE_STIPPLE)

                    #Hard coded parking slots for merging experiment
                    W = 58
                    for i in range(1):
                        graphics.draw(4, gl.GL_LINE_LOOP, ('v2f',
                           np.hstack(
                               [lane.p + lane.m * (
                                           W + 6 * i) - 0.5 * lane.w * lane.n,
                                lane.p + lane.m * (
                                            W + 6 * i) + 0.5 * lane.w * lane.n,
                                lane.p + lane.m * (
                                        W + 6 * (i + 1)) + 0.5 * lane.w * lane.n,
                                lane.p + lane.m * (
                                        W + 6 * (i + 1)) - 0.5 * lane.w * lane.n])
                           ))

                    gl.glColor3f(1., 1., 0.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', right_line(lane, k)[1]))
                    graphics.draw(2, gl.GL_LINES, ('v2f', left_line(lane, k)[0]))
                    graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
                else:
                    gl.glLineStipple(int(36 / self.magnify), 0x1111)
                    gl.glEnable(gl.GL_LINE_STIPPLE)
                    gl.glColor3f(1., 1., 1.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', left_line(lane)[0]))
                    graphics.draw(2, gl.GL_LINES, ('v2f', right_line(lane)[0]))
                    gl.glDisable(gl.GL_LINE_STIPPLE)
                    gl.glColor3f(1., 1., 0.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', left_line(lane)[1]))
                    graphics.draw(2, gl.GL_LINES, ('v2f', right_line(lane)[1]))
                    graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
        gl.glColor3f(1., 1., 1.)


    def draw_car(self, x, color='yellow', opacity=255):
        """ Draws a car with the given color at the given position

        Parameters
        ----------
        x : CasADi MX
            the given state vector for the vehicle [x, y, angle, ...]
        color : str, optional
            the color of the sprite
        opacity : int, optional
            the opacity of the sprite
        """
        sprite = self.sprites[color]
        sprite.x, sprite.y = x[0], x[1]
        sprite.rotation = -x[2]*180./math.pi
        sprite.opacity = opacity
        sprite.draw()

    def draw_bounding_box_car(self, x, car, color='yellow'):
        """ Draws the rectangular bounding box around a car

        Parameters
        ----------
        x : CasADi MX
            the given state vector for the vehicle [x, y, angle, ...]
        car : Car object
            the regarded car
        color : str, optional
            the color of the bounding box
        """
        if self.show_bounding_box:
            gl.glColor3f(self.colors_dict[color][0], self.colors_dict[color][1], self.colors_dict[color][2])
            gl.glLineWidth(1 * self.width_ratio / self.magnify)
            gl.glBegin(gl.GL_LINE_LOOP)
            length = car.lf + car.lr
            gl.glVertex2f(x[0] + length / 2. * np.cos(x[2]) - car.width / 2. * np.sin(x[2]), x[1] + length / 2. * np.sin(x[2]) + car.width / 2. * np.cos(x[2]))
            gl.glVertex2f(x[0] + length / 2. * np.cos(x[2]) + car.width / 2. * np.sin(x[2]), x[1] + length / 2. * np.sin(x[2]) - car.width / 2. * np.cos(x[2]))
            gl.glVertex2f(x[0] - length / 2. * np.cos(x[2]) + car.width / 2. * np.sin(x[2]), x[1] - length / 2. * np.sin(x[2]) - car.width / 2. * np.cos(x[2]))
            gl.glVertex2f(x[0] - length / 2. * np.cos(x[2]) - car.width / 2. * np.sin(x[2]), x[1] - length / 2. * np.sin(x[2]) + car.width / 2. * np.cos(x[2]))
            gl.glEnd()
            gl.glColor3f(1., 1., 1.)

    def draw_trajectory(self, traj, car, color):
        if self.show_trajectory_mode == 0:
            return
        if self.show_trajectory_mode == 1:
            self.draw_trajectory_line(traj, color)
        if self.show_trajectory_mode == 2:
            self.draw_trajectory_faded(traj, car, color)
        if self.show_trajectory_mode == 3:
            self.draw_past_trajectory_faded(car, color)

    def draw_trajectory_line(self, traj, color):
        """ Draws the given trajectory

        Parameters
        ----------
        traj : Trajectory object
            the given trajectory
        color : str, optional
            the color of the trajectory
        """
        if traj.N > 1:
            trajectory = traj.get_future_trajectory()
            gl.glColor3f(self.colors_dict[color][0], self.colors_dict[color][1], self.colors_dict[color][2])
            gl.glLineWidth(1 * self.width_ratio / self.magnify)

            # Draw a line strip
            gl.glLineStipple(5, 0x5555)
            gl.glEnable(gl.GL_LINE_STIPPLE)
            gl.glBegin(gl.GL_LINE_STRIP)
            gl.glVertex2f(traj.x0[0], traj.x0[1])
            for state in trajectory:
                gl.glVertex2f(state[0], state[1])
            gl.glEnd()
            gl.glDisable(gl.GL_LINE_STIPPLE)

            # Draw nodes at each sampling time
            gl.glPointSize(4 * self.width_ratio / self.magnify)
            gl.glEnable(gl.GL_POINT_SMOOTH)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glBegin(gl.GL_POINTS)
            gl.glVertex2f(traj.x0[0], traj.x0[1])
            for state in trajectory:
                gl.glVertex2f(state[0], state[1])
            gl.glEnd()
            gl.glColor3f(1., 1., 1.)

    def draw_trajectory_faded(self, traj, car, color):
        """ Draws the given trajectory

        Parameters
        ----------
        traj : Trajectory object
            the given trajectory
        color : str, optional
            the color of the trajectory
        """
        if traj.N > 1:
            trajectory = traj.get_future_trajectory()

            sprite = self.sprites[color]
            opacity_list = np.linspace(100,0,len(trajectory)+2, dtype = int)[1:-1][::-1]

            for index, state in enumerate(trajectory[::-1]):
                center_x = car.center_x(state)
                sprite.x, sprite.y = center_x[0], center_x[1]
                sprite.rotation = -center_x[2] * 180. / math.pi
                sprite.opacity = opacity_list[index]
                sprite.draw()

    def draw_past_trajectory_faded(self, car, color):
        """ Draws the given trajectory

        Parameters
        ----------
        traj : Trajectory object
            the given trajectory
        color : str, optional
            the color of the trajectory
        """
        faded_factor = 2
        trajectory = [[self.logger.history['x'][car.id][faded_factor*(-1-i)],
                       self.logger.history['y'][car.id][faded_factor*(-1-i)],
                       self.logger.history['angle'][car.id][faded_factor*(-1-i)],
                       self.logger.history['velocity'][car.id][faded_factor*(-1-i)]]
                      for i in range(min(self.current_iteration//2, 5))]
        sprite = self.sprites[color]
        opacity_list = np.linspace(150, 0, len(trajectory) + 1, dtype=int)[:-1][::-1]

        for index, state in enumerate(trajectory[::-1]):
            center_x = car.center_x(state)
            sprite.x, sprite.y = center_x[0], center_x[1]
            sprite.rotation = -center_x[2] * 180. / math.pi
            sprite.opacity = opacity_list[index]
            sprite.draw()

    def on_draw(self):
        """ Draws all objects and the background on the window """
        self.window.clear()
        gl.glColor3f(1., 1., 1.)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        self.camera()

        # Draw grass
        gl.glEnable(self.grass.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(self.grass.target, self.grass.id)
        [x_min, x_max, y_min, y_max] = self.camera_vision_vertices()
        x_repeats = math.ceil(self.width/128.)
        y_repeats = math.ceil(self.height/128.)
        graphics.draw(4, gl.GL_QUADS,
            ('v2f', (x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max)),
                      ('t2f', (0., 0., x_repeats, 0., x_repeats, y_repeats, 0., y_repeats)),)
        gl.glDisable(self.grass.target)

        # Draw lanes
        for road in self.roads:
            self.draw_road(road)

        # Draw heatmap
        if self.heat is not None:
            self.draw_heatmap()

        # Draw cars
        for vehicle in self.cars:
            if vehicle != self.main_car:
                self.draw_trajectory(vehicle.traj, vehicle, vehicle.color)
        self.draw_trajectory(self.main_car.traj, self.main_car, self.main_car.color)

        for vehicle in self.cars:
            self.draw_car(self.anim_x[vehicle], vehicle.color)
            self.draw_bounding_box_car(self.anim_x[vehicle], vehicle, vehicle.color)

        gl.glPopMatrix()

        # Draw extra information about Speed and Headway distance
        self.draw_live_data_window()

        # Save image if save_on_draw
        if self.logger.save_on_draw:
            if not self.logger.generate_video:
                video_path = 'experiments/' + self.logger.settings.name_experiment + '/video'
                self.save_screenshot(folder_name=video_path, index=self.current_iteration)
                self.logger.save_on_draw = False
            else:
                # Generate and save video
                video_path = 'experiments/' + self.logger.settings.name_experiment + '/video'
                os.system("C:/ffmpeg/bin/ffmpeg.exe -r " + str(1/self.Ts) + " -i ./" + video_path +
                          "/screenshot-%01d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p ./" + video_path + "/video.mp4")
                time.sleep(0.1)
                self.logger.generate_video = False

    def draw_live_data_window(self):
        if self.show_live_data:
            self.text_box_background.draw()
            for item in self.live_data_shown:
                try:
                    setattr(getattr(self, 'label_' + item), 'text', item + ': %.2f'%self.logger.history[item][self.main_car.id][-1])
                except:
                    setattr(getattr(self, 'label_' + item), 'text', item + ': N/A')
                getattr(self, 'label_' + item).draw()

    def reset(self):
        """ Resets the variables of the visualizer """
        self.paused = False
        self.current_iteration = 0
        self.logger.reset(self.cars)
        for car in self.cars:
            car.reset()
        for car in self.cars:
            self.anim_x[car] = car.center_x(car.x)
            self.prev_x[car] = car.center_x(car.x)

    def run(self):
        """ Resets the visualized and runs the event loop """
        self.reset()
        #pyglet.clock.schedule_interval(self.control_loop, self.Ts)
        pyglet.clock.schedule(self.control_loop)
        self.event_loop.run()

