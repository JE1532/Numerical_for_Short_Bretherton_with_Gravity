import numpy

def forward_euler_integration(y_0, vec_deriv, delta, conv=None):
    x_arr = [0]
    y_arr = [y_0]
    conv = len(y_0) - 1 if not conv else conv
    get_conv_derivative = lambda y_vals: numpy.array([y[conv] for y in y_vals])
    while len(y_arr) < 100 or numpy.std(get_conv_derivative(y_arr[-100:])) >= 0.01 * numpy.avg(get_conv_derivative(y_arr[-100:])):
        x_arr.append(x_arr[-1] + delta)
        delta_y = vec_deriv(y_arr[-1]) * delta
        y_arr.append(y_arr[-1] + delta_y)
    return x_arr, y_arr
