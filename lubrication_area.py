import numpy
import scipy
from euler_integration import forward_euler_integration
import numerical_static_meniscus
import matplotlib.pyplot as plt


numpy.seterr(all='raise')


third_deriv = lambda x, y, ca_bo_ratio: (y[0] ** 3 - (3 / 2) * ca_bo_ratio) / (y[0] ** 3)


def numerical_lubrication_area(min_d, ca_bo_ratio):
    bond_number = lambda c, right: eval_lub_profile(min_d, ca_bo_ratio, c, right=right)[0]
    func_to_minimize = lambda c: abs(bond_number(c, True) - bond_number(c, False))

    # calculating optimal c
    c_val = scipy.optimize.minimize_scalar(func_to_minimize, bounds=[10 * (2 / 3) * (ca_bo_ratio ** (1 / 6)), 5]).x


    # calculating lubrication region profile right and left of x=0
    r_bo, r_x_arr, r_y_arr = eval_lub_profile(min_d, ca_bo_ratio, c_val, True)
    l_bo, l_x_arr, l_y_arr = eval_lub_profile(min_d, ca_bo_ratio, c_val, False)

    # bond number match error
    match_err = func_to_minimize(c_val)

    # total result arrays
    x_arr = list(reversed(l_x_arr)) + r_x_arr
    y_arr = list(reversed(l_y_arr)) + r_y_arr
    return r_bo, x_arr, y_arr, match_err, c_val


def eval_lub_profile(min_d, ca_bo_ratio, c, right=True):
    x_0 = 0
    y_0 = numpy.array([min_d, 0, c])
    vec_deriv = lambda x, y: numpy.array([y[1],
                                        y[2],
                                        (y[0] ** 3 - (3 / 2) * ca_bo_ratio) / (y[0] ** 3)
                                        ])
    delta = 0.00001 * (1 if right else -1)
    if right:
        conv = 3
    else:
        conv = 3
    x_arr, y_arr = forward_euler_integration(y_0, vec_deriv, delta, x_0, conv=conv)

    # matching a static meniscus to calculate bo, pressure-diff
    minimizable = lambda arr: meniscus_fit_parameter(y_arr[-1], arr[0], arr[1], right=right)
    bo, pressure_diff = scipy.optimize.minimize(minimizable, [1, 1], bounds=[(0.001, 10), (0.01, 2)]).x
    #print(bo)
    # return resulting lubrication region and bond number
    return bo, x_arr, y_arr


def meniscus_fit_parameter(last_eta_val, bo, pressure_diff, right=True):
    #print(bo)
    #evaluating meniscus
    try:
        x_arr, y_arr = numerical_static_meniscus.numerical_static_meniscus(bo, pressure_diff, right=right, bound=1 - last_eta_val[0])
    except FloatingPointError:
        return 100


    try:
        y_deriv = numerical_static_meniscus.meniscus_deriv(bo, pressure_diff, right=right)(x_arr[-1], y_arr[-1])
    except FloatingPointError:
        return 100


    if y_deriv[0] == 0 or y_deriv[1] == 0 or bo == 0:
        print('k')
        print(bo)
        print(y_deriv)
        return 100


    # [d(eta) / d(xi), d^2 (eta) / d(xi) ^ 2 ]
    men_eta_deriv = numpy.array([
        - (1 / y_deriv[0]) * (bo ** (- 1 / 3)),
        - (y_deriv[1] / (y_deriv[0] ** 3)) * (bo ** (- 2 / 3))])

    # relative deviations of derivative from meniscus values
    first_deriv_dev = abs(men_eta_deriv[0] - last_eta_val[1]) / min(abs(men_eta_deriv[0]), abs(last_eta_val[1]))
    second_deriv_dev = abs(men_eta_deriv[1] - last_eta_val[2]) / min(abs(men_eta_deriv[1]), abs(last_eta_val[2]))

    # binding maximum deviation
    return max(first_deriv_dev, second_deriv_dev)


if __name__ == "__main__":
    ca_bo_ratio = 10 ** (-9)
    min_d = 0.001
    #bo, x_arr, y_arr, match_err, c_val = numerical_lubrication_area(min_d, ca_bo_ratio)


    bo = 0.316227766016838
    optimal_c = 3.8717744549702986
    r_bo, r_x_arr, r_y_arr = eval_lub_profile(min_d, ca_bo_ratio, optimal_c, right=True)
    l_bo, l_x_arr, l_y_arr = eval_lub_profile(min_d, ca_bo_ratio, optimal_c, right=False)
    print(r_bo, l_bo)
    x_arr = list(reversed(l_x_arr)) + r_x_arr
    y_arr = list(reversed(l_y_arr)) + r_y_arr
    x_arr = [(bo ** (1 / 3)) * x for x in x_arr]
    #bo, x_arr, y_arr = eval_lub_profile(0.01, 10 ** (-6), 0.1, right=False)
    '''print(f"Bond number = {bo}\nOptimal c_val = {c_val}"
          f"\nMatch error = {match_err}"
          f"\nLast y value = {y_arr[-1]}"
          f"\nInitial y value = {y_arr[0]}"
          f"\nLubrication area length = {x_arr[-1] - x_arr[0]}"
          f"\nInitial third derivative = {third_deriv(x_arr[-1], y_arr[-1], ca_bo_ratio)}")
    '''
    plt.axes().set_aspect('equal')
    plt.plot(x_arr, [y[0] for y in y_arr])
    plt.show()