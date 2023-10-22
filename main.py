import numpy
import scipy

import numerical_static_meniscus as static

import lubrication_area


def get_lubrication_startvals(bo, pressure_diff):
    r_arr, z_arr = static.numerical_static_meniscus(bo, pressure_diff, bound=0.02)
    y = 1 - r_arr[-1]
    dy = - (1 / z_arr[-1][1])
    ddy = - static.meniscus_deriv(bo, pressure_diff)(r_arr[-1], z_arr[-1]) / ((z_arr[-1][1]) ** 3)
    return [y, dy, ddy]


def get_left_static_startvals
