from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import splrep, splev
from scipy.misc import derivative
import multiprocessing

import numpy as np


class Surface_2D:
    def __init__(self , x, y, data,
                 boundary=None,
                 relative=False):

        self.relative = relative
        # Interpolate function

        if boundary is not None:
            self.boundary = [[np.min(x), np.max(x)], [np.min(y), np.max(y)]]
        else:
            self.boundary = boundary

        self.inter_funct = RectBivariateSpline(x, y, data,
                                               bbox=[self.boundary[0][0], self.boundary[0][1], self.boundary[1][0], self.boundary[1][1]])

        self.bounds_length = [self.boundary[0][1]-self.boundary[0][0],
                              self.boundary[1][1]-self.boundary[1][0]]

#        self.bounds_length = [np.max(x) - np.min(x), np.max(y) - np.min(y)]

    # Set periodic boundary conditions
    def interaction_function_periodic(self, coor):
        return self.inter_funct(np.mod(coor[0] - self.boundary[0][0], self.bounds_length[0]) + self.boundary[0][0],
                                 np.mod(coor[1] - self.boundary[1][0], self.bounds_length[1]) + self.boundary[1][0])

    # Total function
    def total_potential(self, coor2):
        coor = []
        if self.relative:
            coor[0] = coor2[1] - coor2[0]
            coor[1] = coor2[2] - coor2[1]
        else:
            coor = coor2

        return np.sum(self.interaction_function_periodic(coor))

    # Calculate the derivatives
    def partial_derivative(self, func, point, var=0):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(args)
        return derivative(wraps, point[var], dx=1e-6)

    def partial_derivatives_mat(self, point):
        return np.array([self.partial_derivative(self.interaction_function_periodic, point, var=i)[0]
                         for i in range(len(point))])


class Surface_1D:
    def __init__(self, x, data,
                 relative=False,
                 boundary=None):

        self.relative = relative

        if boundary is not None:
            self.boundary = [[np.min(x), np.max(x)]]
        else:
            self.boundary = boundary

        # Interpolate function
        self.inter_funct = splrep(x, data, xb=self.boundary[0][0], xe=self.boundary[0][1])

        self.bonund_length = [np.max(x) - np.min(x)]
        self.bonund_length = [self.boundary[0][1] - self.boundary[0][0]]

    # Set periodic boundary conditions
    def interaction_function_periodic(self, coor):
        return splev(np.mod(coor[0], self.bonund_length[0]), self.inter_funct)
        # return self.inter_funct(np.mod(coor[0], self.bonund_length[0]))

    def potential_data_fast(self, vector):
        if self.relative:
            x = vector[1] - vector[0]
        else:
            x = vector[0]

        pot = self.interaction_function_periodic(x)
        return pot

    # Total function
    def total_potential(self, coor):
        return np.sum(self.interaction_function_periodic(coor))

    # Calculate the derivatives
    def partial_derivative(self, func, point, var=0):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(args)
        return derivative(wraps, point[var], dx=1e-6)

    def partial_derivatives_mat(self, point):
        return np.array(self.partial_derivative(self.interaction_function_periodic, point, var=0))


class Surface_2D_analytic:
    def __init__(self, analitic_function, relative=False):
        self.analitic_function = analitic_function
        self.relative = relative

    def interaction_function_periodic(self, coor):
        if self.relative:
            x = coor[2]-coor[1]
            y = coor[1]-coor[0]
            return self.analitic_function(x, y)
        else:
            return self.analitic_function(*coor)

    # Total function
    def total_potential(self, coor):
        return np.sum(self.interaction_function_periodic(coor))

    # Calculate the derivatives
    def partial_derivative(self, func, point, var=0):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(args)
        return derivative(wraps, point[var], dx=1e-6)

    def partial_derivatives_mat(self, point):
        return np.array([self.partial_derivative(self.interaction_function_periodic, point, var=i)
                         for i in range(len(point))])


class Surface_1D_analytic:
    def __init__(self, analitic_function, relative=False, offset=0):
        self.analitic_function = analitic_function
        self.relative = relative
        self.position = offset

    def potential_data_fast(self, coor):

        if self.relative:
            x = coor[0] - coor[1]
        else:
            x = coor[0] - self.position

        return self.analitic_function(x)

    # Total function
    def total_potential(self, coor):
        return np.sum(self.potential_data_fast(coor))

    # Calculate the derivatives
    def partial_derivative(self, func, point, var=0):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(args)
        return derivative(wraps, point[var], dx=1e-6)

    def partial_derivatives_mat(self, point):
        return np.array([self.partial_derivative(self.potential_data_fast, point, var=i)
                         for i in range(len(point))])


# Good one
class Potential:
    def __init__(self,
                 complete_interaction):

        self.complete_interaction = complete_interaction

    # Total function
    def total_potential(self, coor):
        coor = np.array(coor)
        potential = 0.0
        for i, pot_func in enumerate(self.complete_interaction):
            potential += pot_func['function'].total_potential(coor[pot_func['coordinates']])
        return potential

    def partial_derivatives(self, point):
        point = np.array(point, dtype=float)
        partial_derivative = np.zeros_like(point)
        for pot_func in self.complete_interaction:
            # print pot_func['function'].partial_derivatives_mat(point[pot_func['coordinates']]).T.flatten()

            partial_derivative[pot_func['coordinates']] += pot_func['function'].partial_derivatives_mat(point[pot_func['coordinates']]).T.flatten()

        return partial_derivative


#    Parallel version  (very slow for now)
# --------------------------------------------

def worker_potential(i, function, coordinates):
    return {i: function.total_potential(coordinates)}


def worker_derivative(i, function, point, coordinates):
    return {i: {'coordinates': coordinates,
                'value': function.partial_derivatives_mat(point)}}


# Good one
class Potential_par:
    def __init__(self,
                 complete_interaction):
        self.complete_interaction = complete_interaction

    # Total function
    def total_potential(self, coor):

        potentials = {}

        def log_result(result):
            potentials.update(result)

        pool = multiprocessing.Pool(processes=max(multiprocessing.cpu_count()-1, 1))
        for i, pot_func in enumerate(self.complete_interaction):
            pool.apply_async(worker_potential,
                             args=(i, pot_func['function'], coor[pot_func['coordinates']]),
                             callback=log_result)
        pool.close()
        pool.join()

        potential = np.sum(potentials.values())

        return potential


    def partial_derivatives(self, point):

        derivatives = {}
        def log_result(result):
            derivatives.update(result)

        pool = multiprocessing.Pool(processes=max(multiprocessing.cpu_count()-1, 1))

        for i, pot_func in enumerate(self.complete_interaction):
            pool.apply_async(worker_derivative,
            args=(i, pot_func['function'], point[pot_func['coordinates']], pot_func['coordinates']),
            callback=log_result)
        pool.close()
        pool.join()


        partial_derivative = np.zeros_like(point)
        for pair in derivatives.values():
            partial_derivative[pair['coordinates']] += pair['value'].T.flatten()

        return partial_derivative

        partial_derivative = np.zeros_like(point)

        for pot_func in self.complete_interaction:
            partial_derivative[pot_func['coordinates']] += pot_func['function'].partial_derivatives_mat(point[pot_func['coordinates']])

        return partial_derivative
