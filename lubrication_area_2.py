import numpy
import scipy.optimize
import matplotlib.axes as ax
import matplotlib.pyplot as plt

import numerical_static_meniscus
import euler_integration

numpy.seterr(all='raise')

iter_count = 0

def numerical_solution(bond_number, eta_0):

    def get_normalization_const(sec_der_0, right=True):
        # getting normalization const for equation deriv^3(y) = (y^3 - 1) / (y^3)
        print(sec_der_0)
        y_0 = numpy.array([eta_0, 0, sec_der_0])
        x_arr, y_arr = euler_integration.forward_euler_integration(y_0,
                                                                   lambda x, y:
                                                                   numpy.array([
                                                                       y[1],
                                                                       y[2],
                                                                       ((eta_0) ** 3 - 1) / (eta_0 ** 3)
                                                                   ]),
                                                                   0.0001 * (1 if right else -1),
                                                                   conv=3,
                                                                   )
        # integrating equation until third derivative converges
        print('numerical integration done!')

        lub_sec_deriv = y_arr[-1][2]

        static_sec_deriv = numerical_static_meniscus.get_tangent_meniscus_curvature(bond_number, right=right)

        print('static calculation done!')

        if (static_sec_deriv > 0 and lub_sec_deriv < 0) or (static_sec_deriv < 0 and lub_sec_deriv > 0):
            print('wow!!!!!!!!!!!!!!!')

        return static_sec_deriv / lub_sec_deriv


    def cost_function(sec_der_0):
        #iter_count += 1
        #print(iter_count)
        print('yo')

        return (get_normalization_const(sec_der_0, right=True) - get_normalization_const(sec_der_0, right=False)) ** 2

    sec_der_0 = scipy.optimize.minimize_scalar(cost_function, bounds=[0, 100]).x
    print(sec_der_0)
    right_norm_const = get_normalization_const(sec_der_0, right=True)
    left_norm_const = get_normalization_const(sec_der_0, right=False)
    match_err = abs(right_norm_const - left_norm_const) / min(right_norm_const, left_norm_const)
    ca_bo_ratio = (right_norm_const / (bond_number ** (2 / 3))) ** 9
    return ca_bo_ratio, match_err


def plot_cost_graph(bond_number, eta_0):
    def get_normalization_const(sec_der_0, right=True):
        print(sec_der_0)
        y_0 = numpy.array([eta_0, 0, sec_der_0])
        x_arr, y_arr = euler_integration.forward_euler_integration(y_0,
                                                                   lambda x, y:
                                                                   numpy.array([
                                                                       y[1],
                                                                       y[2],
                                                                       ((eta_0) ** 3 - 1) / (eta_0 ** 3)
                                                                   ]),
                                                                   0.0001 * (1 if right else -1),
                                                                   conv=3,
                                                                   )
        print('numerical integration done!')

        lub_sec_deriv = y_arr[-1][2]

        static_sec_deriv = numerical_static_meniscus.get_tangent_meniscus_curvature(bond_number, right=right)

        print('static calculation done!')

        return static_sec_deriv / lub_sec_deriv


    def cost_function(sec_der_0):
        #iter_count += 1
        #print(iter_count)
        print('yo')

        right_norm_coeff = get_normalization_const(sec_der_0, right=True)
        left_norm_coeff = get_normalization_const(sec_der_0, right=False)

        return ((right_norm_coeff - left_norm_coeff) / min(right_norm_coeff, left_norm_coeff)) ** 2

    x_array = [i for i in range(2100, 3001, 100)]
    y_array = [cost_function(x_array[i]) for i in range(0, 10)]
    plt.plot(x_array, y_array)
    plt.show()


if __name__ == "__main__":
    plt.yscale('linear')
    eta_0 = float(input('Enter eta_0: '))
    bond_number = float(input('Enter bond number: '))
    #plot_cost_graph(bond_number, eta_0)
    ca_bond_ratio, match_err = numerical_solution(bond_number, eta_0)

    print(
        f"""ca_bond_ratio = {ca_bond_ratio}
Match error = {match_err}"""
    )
