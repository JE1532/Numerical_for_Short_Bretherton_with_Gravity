import numpy
import scipy
from euler_integration import forward_euler_integration
import matplotlib.pyplot as plt


def numerical_lubrication_area(min_d, ca_bo_ratio):
    c = 0
    bond_number = lambda c, right: eval_lub_profile(min_d, ca_bo_ratio, c, right=right)[0]
    func_to_minimize = lambda c: abs(bond_number(c, True) - bond_number(c, False))

    # calculating optimal c
    c_val = scipy.optimize.minimize_scalar(func_to_minimize, bounds=[0, 10000]).x

    # calculating lubrication region profile right and left of x=0
    r_bo, r_x_arr, r_y_arr = eval_lub_profile(min_d, ca_bo_ratio, c_val, True)
    l_bo, l_x_arr, l_y_arr = eval_lub_profile(min_d, ca_bo_ratio, c_val, False)

    # bond number match error
    match_err = func_to_minimize(c_val)
    if match_err >= 0.01 * max(l_bo, r_bo):
        raise Exception('Could not match bond numbers.')

    # total result arrays
    x_arr = l_x_arr + r_x_arr
    y_arr = l_y_arr + r_y_arr
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
    bo = vec_deriv(x_arr[-1], y_arr[-1])[-1]  # bond number
    return bo, x_arr, y_arr


if __name__ == "__main__":
    bo, x_arr, y_arr, match_err, c_val = numerical_lubrication_area(0.00001, 0.0000001)
    print(f"Bond number = {bo}\nOptimal c_val = {c_val}\nMatch error = {match_err}\nLast y value = {y_arr[-1]}\nInitial y value = {y_arr[0]}\nLubrication area length = {x_arr[-1] - x_arr[0]}")
    plt.axes().set_aspect('equal')
    plt.plot(x_arr, [y[0] for y in y_arr])
    plt.show()