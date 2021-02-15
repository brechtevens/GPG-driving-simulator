import numpy as np
import casadi.casadi as cs
import feature


class Lane(object):
    pass


class StraightLane(Lane):
    """
    A class used to represent a straight lane

    Attributes
    ----------
    p : list
        the first point on the center line of the lane
    q : list
        the second point on the center line of the lane
    w : float
        the width of the lane
    m : list
        the vector along the straight lane
    n : list
        the normal vector on the straight lane
    """
    def __init__(self, p, q, w):
        """
        Parameters
        ----------
        p : list
            the first point on the center line of the lane
        q : list
            the second point on the center line of the lane
        w : float
            the width of the lane
        """
        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w
        self.m = (self.q-self.p)/np.linalg.norm(self.q-self.p)
        self.n = np.asarray([-self.m[1], self.m[0]])
        self.length = 1000

    def shifted(self, m):
        """ Returns a new lane with position shifted by m times the width of the lane relatively

        Parameters
        ----------
        m : int
            the amount of times to shift the lane
        """
        return StraightLane(self.p+self.n*self.w*m, self.q+self.n*self.w*m, self.w)

    def dist2(self, x):
        """ Returns the squared distance of a point p from the center line of the lane

        Parameters
        ----------
        x : list
            the position of the point p
        """
        r = (x[0] - self.p[0]) * self.n[0] + (x[1] - self.p[1]) * self.n[1]
        return r * r

    def quadratic(self):
        """ Returns a quadratic cost feature penalizing deviations from the center line of the lane """
        @feature.feature
        def f(x, u, x_other):
            return self.dist2(x)
        return f

    def linear(self, id_other_vehicle=None):
        """ Returns a linear cost feature penalizing driving along the normal vector """
        if id_other_vehicle is None:
            @feature.feature
            def f(x, u, x_other):
                return (x[0] - (self.p - self.n * self.w / 2)[0]) * self.n[0] + \
                       (x[1] - (self.p - self.n * self.w / 2)[1]) * self.n[1]
        else:
            @feature.feature
            def f(x, u, x_other):
                return (x_other[id_other_vehicle][0]-(self.p - self.n * self.w / 2)[0])*self.n[0] + \
                       (x_other[id_other_vehicle][1]-(self.p - self.n * self.w / 2)[1])*self.n[1]
        return f

    def get_edges(self):
        """ Returns a point on an edge of the lane and the corresponding nearest point on the other edge of the lane """
        return self.p - self.n * self.w / 2, self.p + self.n * self.w / 2

    def gaussian(self, width=0.5):
        """ Returns a gaussian cost feature penalizing deviations from the center line of the lane

        Parameters
        ----------
        width : float
            the width of the gaussian
        """
        @feature.feature
        def f(x, u, x_other):
            return cs.exp(-0.5*self.dist2(x)/(width**2*self.w*self.w/4.))
        return f
