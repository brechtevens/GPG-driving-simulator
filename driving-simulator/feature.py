import casadi.casadi as cs


class Feature(object):
    """
    A class used to represent cost function features
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        return self.f(*args)

    def __add__(self, r):
        return Feature(lambda *args: self(*args)+r(*args))

    def __radd__(self, r):
        return Feature(lambda *args: r(*args)+self(*args))

    def __mul__(self, r):
        return Feature(lambda *args: self(*args)*r)

    def __rmul__(self, r):
        return Feature(lambda *args: r*self(*args))

    def __pos__(self, r):
        return self

    def __neg__(self):
        return Feature(lambda *args: -self(*args))

    def __sub__(self, r):
        return Feature(lambda *args: self(*args)-r(*args))

    def __rsub__(self, r):
        return Feature(lambda *args: r(*args)-self(*args))


def feature(f):
    """ Decorator function """
    return Feature(f)


def speed(s=1.):
    """ Returns a quadratic cost feature penalizing deviations from the desired speed """
    @feature
    def f(x, u, x_other):
        return -(x[3]-s)*(x[3]-s)
    return f


def control():
    """ Returns a quadratic cost feature penalizing control actions """
    @feature
    def f(x, u, x_other):
        return -cs.sumsqr(u)
    return f


def headway(front, lr_front, lf_back):
    """ Returns a cost feature penalizing driving to close in a one-dimensional scenario

    Parameters
    ----------
    front : boolean
        indicates whether the ego vehicle is the front or the rear vehicle
    lr_front : float
        the distance between the center of mass and the rear end of the front vehicle
    lf_back : float
        the distance between the center of mass and the front end of the rear vehicle
    """
    @feature
    def f(x, u, x_other):
        if front:
            d_x = x[0]-x_other[0]-lr_front-lf_back
        else:
            d_x = x_other[0]-x[0]-lr_front-lf_back
        return d_x
    return f


def gaussian(id_other, height=3., width=1.):
    """ Returns a gaussian cost feature rewarding distance between the ego and the target vehicle

    Parameters
    ----------
    id_other : int
        the id of the target vehicle
    height : float
        the size of the gaussian in the longitudinal direction
    width : float
        the size of the gaussian in the lateral direction
    """
    @feature
    def f(x, u, x_other):
        d = (x_other[id_other][0]-x[0], x_other[id_other][1]-x[1])
        theta = x_other[id_other][2]
        dh = cs.cos(theta)*d[0]+cs.sin(theta)*d[1]
        dw = -cs.sin(theta)*d[0]+cs.cos(theta)*d[1]
        return -cs.exp(-0.5*(dh*dh/(height*height)+dw*dw/(width*width)))
    return f
