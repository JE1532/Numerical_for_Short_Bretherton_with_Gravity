import numpy
import matplotlib.pyplot as plt
import scipy

from euler_integration import forward_euler_integration


if __name__ == "__main__":
    numpy.seterr(all='raise')


def meniscus_deriv(bond_number, pressure_diff, right=True):
    sec_deriv_coeff = lambda x, y: (1 + (y[1] ** 2)) ** (3 / 2)
    sign = 1 if right else -1
    sec_deriv_internal = lambda x, y: - sign * bond_number * y[0] - (y[1] / (x * numpy.sqrt(1 + y[1] ** 2))) - sign * pressure_diff
    sec_deriv = lambda x, y: sec_deriv_coeff(x, y) * sec_deriv_internal(x, y)
    vec_deriv = lambda x, y: numpy.array([y[1], sec_deriv(x, y)])
    return vec_deriv


def numerical_static_meniscus(bond_number, pressure_diff, r_0=0.01, right=True, bound=1):
    y_0 = numpy.array([- (pressure_diff / 4) * (r_0 ** 2), - (1 / 2) * pressure_diff * r_0])
    vec_deriv = meniscus_deriv(bond_number, pressure_diff, right=right)
    delta = 0.001
    x_arr, y_arr = forward_euler_integration(y_0, vec_deriv, delta, x_0=r_0, bounds=[0, bound])
    return x_arr, y_arr





def plot_conv_graph(bo_range=3, pressure_diff_bound=10, bo_delta=0.1, pressure_delta=0.1, right=True):
    bo = bo_delta
    pressure_diff = pressure_delta
    bo_arr = []
    pressure_diff_arr = []
    while bo < bo_range:
        while pressure_diff < pressure_diff_bound:
            try:
                numerical_static_meniscus(bo, pressure_diff, right=right)
                pressure_diff += pressure_delta
            except FloatingPointError:
                bo_arr.append(bo)
                pressure_diff_arr.append(pressure_diff)
                pressure_diff = pressure_delta
                break
        bo += bo_delta
    plt.plot(bo_arr, pressure_diff_arr)
    plt.show()


def get_max_pressure_diff(bond_number, delta=0.001, right=True):
    pressure_diff = delta
    while True:
        try:
            numerical_static_meniscus(bond_number, pressure_diff, right=right)
        except FloatingPointError:
            break
        pressure_diff += delta
    return pressure_diff - delta


def get_tangent_meniscus_curvature(bond_number, right=True):
    def cost_function(pressure_diff):
        x_arr, y_arr = numerical_static_meniscus(bond_number, pressure_diff, right=right)
        print(pressure_diff)
        return 1 / (y_arr[-1][1] ** 2)

    max_pressure_diff = 0 if bond_number == 0.843 else get_max_pressure_diff(bond_number, right=right)
    optimal_pressure_diff = (2.37 if right else 1.7818) if bond_number == 0.843 else scipy.optimize.minimize_scalar(cost_function, bounds=[0.1, max_pressure_diff]).x
    print('optimal pressure found')
    x_arr, y_arr = numerical_static_meniscus(bond_number, optimal_pressure_diff, right=right, bound=0.99)
    last_val_deriv = meniscus_deriv(bond_number, optimal_pressure_diff, right=right)(x_arr[-1], y_arr[-1])
    #plt.plot(x_arr, [y[0] for y in y_arr])
    #plt.show()

    curvature = last_val_deriv[1] / (last_val_deriv[0] ** 3)

    return curvature


if __name__ == "__main__":
    x_arr, y_arr = numerical_static_meniscus(0.843, 2.37, bound=0.99)
    print(1 / y_arr[-1][1])
    print(get_tangent_meniscus_curvature(0.843))
    y_arr = [y[0] for y in y_arr]
    x_arr, y_arr = numerical_static_meniscus(0.843, 1.7818, bound=0.99, right=False)
    print(1 / y_arr[-1][1])
    # deviation from unit circle
    #print(max([y_arr[i] - (-1 + numpy.sqrt(1 - (x_arr[i]) ** 2)) for i in range(len(x_arr))]))
    '''plt.axes().set_aspect('equal')
    plt.plot(x_arr, [y[0] for y in y_arr])
    plt.show()
    plot_conv_graph(bo_range=5, pressure_diff_bound=100)
    '''
    print(get_tangent_meniscus_curvature(0.843, right=True))
