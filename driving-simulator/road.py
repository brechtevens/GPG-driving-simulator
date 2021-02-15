import casadi.casadi as cs
import feature
import lane
import constraints


class Highway:
    """
    A class used to represent a highway with multiple lanes

    Attributes
    ----------
    lanes : list
        the list of Lane objects of the highway
    n : list
        the normal vector on the highway direction
    """
    def __init__(self, p, q, w, nb_lanes, length_list=None):
        """
        Parameters
        ----------
        p : list
            the first point on the center line of the 'first' lane
        q : list
            the second point on the center line of the 'first' lane
        w : float
            the width of each lane
        nb_lanes : int
            the number of lanes of the highway
        length_list : list
            the lengths of the different lanes
        """
        center_lane = lane.StraightLane(p, q, w)
        self.lanes = [center_lane]
        for n in range(1, nb_lanes):
            self.lanes += [center_lane.shifted(n)]
        self.n = self.lanes[0].n
        if length_list is not None:
            assert(len(length_list) == nb_lanes)
            for i, L in enumerate(length_list):
                self.lanes[i].length = L

    def get_lanes(self):
        """ Return the number of lanes of the highway """
        return self.lanes

    def gaussian(self, width=0.5):
        """ Returns a gaussian cost feature penalizing deviations from the center line of the lane

        Parameters
        ----------
        width : float
            the width of the gaussian
        """
        @feature.feature
        def f(x, u, x_other):
            return self.lanes[0].gaussian(width)(x, u, x_other)
        return f

    def quadratic(self):
        """ Returns a quadratic cost feature penalizing deviations from the center line of the lane """
        @feature.feature
        def f(x, u, x_other):
            return self.lanes[0].quadratic()(x, u, x_other)
        return f

    def linear(self):
        """ Returns a linear cost feature penalizing driving along the normal vector """
        @feature.feature
        def f(x, u, x_other):
            return self.lanes[0].linear()(x, u, x_other)
        return f

    def boundary_constraint(self, car1, *args):
        """ Returns the 8 inequality boundary constraints ensuring the car remains withing the boundaries of the highway

        Parameters
        ----------
        car1 : Car object
            the ego vehicle
        """
        @constraints.stageconstraints
        def h(x, u=None):
            edge1 = self.lanes[0].get_edges()[0]
            edge2 = self.lanes[-1].get_edges()[1]
            vehicle_corners = car1.corners(x[car1.id])
            h = []
            for corner in vehicle_corners:
                h.append((corner[0] - edge1[0]) * self.lanes[0].n[0] + (corner[1] - edge1[1]) * self.lanes[0].n[1])
                h.append(- (corner[0] - edge2[0]) * self.lanes[-1].n[0] - (corner[1] - edge2[1]) * self.lanes[-1].n[1])
            return h
        h.length = 8
        return h

    def right_lane_constraint(self, car):
        """ Returns the equality constraint ensuring the car remains withing the boundaries of lane zero

        Parameters
        ----------
        car : Car object
            the ego vehicle
        """
        @constraints.stageconstraints
        def g(x, u=None):
            upper_edge = self.lanes[0].get_edges()[1]
            vehicle_corners = car.corners(x[car.id])
            n = self.lanes[0].n
            m = self.lanes[0].m
            g = []
            for corner in vehicle_corners:
                h1 = - (corner[0] - upper_edge[0]) * n[0] - (corner[1] - upper_edge[1]) * n[1]
                h2 = - (corner[0] - self.lanes[1].length) * m[0] - (corner[1] - upper_edge[1]) * m[1]
                g.append(cs.fmin(h1, 0) * cs.fmin(h2, 0))
                # g.append(cs.fmax(cs.fmin(h1, 0), cs.fmin(h2, 0)))
            return g
        g.length = 4
        return g

    def aligned(self, factor=1.):
        """ Returns a quadratic cost feature penalizing deviations from driving along the direction of the highway

        Parameters
        ----------
        factor : float
            the cost feature importance
        """
        @feature.feature
        def f(x, u, x_other):
            return - factor * (x[2] - cs.arctan2(-self.n[0], self.n[1]))**2
        return f
