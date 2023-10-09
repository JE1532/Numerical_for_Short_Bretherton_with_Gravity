import numpy
import matplotlib.pyplot as plt
import time

from euler_integration import forward_euler_integration


def numerical_static_meniscus(bond_number, pressure_diff, r_0=0.01):
    y_0 = numpy.array([- (pressure_diff / 4) * (r_0 ** 2), - (1 / 2) * pressure_diff * r_0])
    sec_deriv_coeff = lambda x, y:  (1 + (y[1] ** 2)) ** (3 / 2)
    sec_deriv_internal = lambda x, y: - bond_number * y[0] - (y[1] / (x * numpy.sqrt(1 + y[1] ** 2))) - pressure_diff
    sec_deriv = lambda x, y: sec_deriv_coeff(x, y) * sec_deriv_internal(x, y)
    vec_deriv = lambda x, y: numpy.array([y[1], sec_deriv(x, y)])
    delta = 0.00001
    x_arr, y_arr = forward_euler_integration(y_0, vec_deriv, delta, x_0=r_0, bounds=[0, 1])
    return x_arr, y_arr


if __name__ == "__main__":
    x_arr, y_arr = numerical_static_meniscus(0.843, 2.3)
    y_arr = [y[0] for y in y_arr]
    print(y_arr[-1])
    # deviation from unit circle
    #print(max([y_arr[i] - (-1 + numpy.sqrt(1 - (x_arr[i]) ** 2)) for i in range(len(x_arr))]))
    plt.axes().set_aspect('equal')
    plt.plot(x_arr, y_arr)
    plt.show()
