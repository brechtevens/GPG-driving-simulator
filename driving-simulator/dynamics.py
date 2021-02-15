import casadi as cs
import numpy as np

class Dynamics(object):
    """
    A class used to represent the general dynamics of a vehicle

    Attributes
    ----------
    nx : int
        the number of states
    nu : int
        the number of inputs
    dt : float
        the sampling time
    bounds : list
        the control bounds of the vehicle
    lr : float
        the length between the mass center and the rear end
    lf : float
        the length between the mass center and the front end
    width : float
        the width of the vehicle
    f : function
        the dynamics of the vehicle
    """
    def __init__(self, nx, nu, f, dt=0.25, bounds=None, lr=0, lf=0, width=0, use_rk4=True):
        self.nx = nx
        self.nu = nu
        self.dt = dt
        self.bounds = bounds
        self.lr = lr
        self.lf = lf
        self.width = width
        self.f = f
        if use_rk4:
            def f_discrete(x,u):
                result = []
                k1 = f(x, u)
                k2 = f(x + k1 * (self.dt / 2), u)
                k3 = f(x + k2 * (self.dt / 2), u)
                k4 = f(x + k3 * self.dt, u)
                for i in range(self.nx):
                    result.append(x[i] + (1/6) * self.dt * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]))
                return result
        else:
            def f_discrete(x,u):
                result = []
                list = f(x,u)
                for i in range(self.nx):
                    result.append(x[i] + self.dt * list[i])
                return result
        self.f_discrete = f_discrete

    def __call__(self, x, u):
        return self.f_discrete(x, u)


class CarDynamics(Dynamics):
    """
    A class used to represent the dynamics of a vehicle using the kinematic bicycle model
    """
    def __init__(self, dt=0.25, bounds=None, friction=0., lr=4, lf=0, width=2):
        """
        Parameters
        ----------
        dt : float, optional
            the sampling time
        bounds : list, optional
            the control bounds of the vehicle
        friction : float, optional
            the friction of the kinematic vehicle model
        lr : float, optional
            the length between the mass center and the rear end
        lf : float, optional
            the length between the mass center and the front end
        width : float, optional
            the width of the vehicle
        """
        self.friction = friction
        if bounds is None:
            bounds = [[-2., -0.5], [2., 0.5]]

        def f(x, u):
            beta = cs.arctan(lr / (lf + lr) * cs.tan(u[1]))
            return np.array([x[3] * cs.cos(x[2] + beta), x[3] * cs.sin(x[2] + beta), (x[3] / lr) * cs.sin(beta),
                             u[0] - x[3] * friction])
        Dynamics.__init__(self, 4, 2, f, dt, bounds, lr, lf, width)

    def __getstate__(self):
        return self.dt, self.bounds, self.friction, self.lr, self.lf, self.width

    def __setstate__(self, dt, bounds, friction, lr, lf, width):
        def f(x, u):
            beta = cs.arctan(lr / (lf + lr) * cs.tan(u[1]))
            return np.array([x[3] * cs.cos(x[2] + beta), x[3] * cs.sin(x[2] + beta), (x[3] / lr) * cs.sin(beta),
                             u[0] - x[3] * friction])
        Dynamics.__init__(self, 4, 2, f, dt, bounds, lr, lf, width)


class CarDynamicsLongitudinal(Dynamics):
    """
    A class used to represent the dynamics of a vehicle using the one-dimensional, longitudinal model
    """
    def __init__(self, dt=0.25, bounds=None, friction=0., lr=4, lf=0, width=2):
        """
        Parameters
        ----------
        dt : float, optional
            the sampling time
        bounds : list, optional
            the control bounds of the vehicle
        friction : float, optional
            the friction of the kinematic vehicle model
        lr : float, optional
            the length between the mass center and the rear end
        lf : float, optional
            the length between the mass center and the front end
        width : float, optional
            the width of the vehicle
        """
        self.friction = friction
        if bounds is None:
            bounds = [[-2.], [2.]]

        def f(x, u):
            return np.array([x[3]*cs.cos(x[2]), x[3]*cs.sin(x[2]), 0, u[0]-x[3]*friction])
        Dynamics.__init__(self, 4, 1, f, dt, bounds, lr, lf, width)

    def __getstate__(self):
        return self.dt, self.bounds, self.friction, self.lr, self.lf, self.width

    def __setstate__(self, dt, bounds, friction, lr, lf, width):
        def f(x, u):
            return np.array([x[3]*cs.cos(x[2]), x[3]*cs.sin(x[2]), 0, u[0]-x[3]*friction])
        Dynamics.__init__(self, 4, 1, f, dt, bounds, lr, lf, width)

