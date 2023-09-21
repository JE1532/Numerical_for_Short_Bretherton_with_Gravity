import numpy

def forward_euler_integration(y_0, vec_deriv, delta):
    x_arr = [0]
    y_arr = [y_0]
    get_second_deriv = lambda y_vals: numpy.array([y[2] for y in y_vals])
    while len(y_arr) < 100 or numpy.std(get_second_deriv(y_arr[-100:])) >= 0.01 * numpy.avg(get_second_deriv(y_arr[-100:][2])):
        x_arr.append(x_arr[-1] + delta)
        delta_y = vec_deriv(y_arr[-1]) * delta
        y_arr.append(y_arr[-1] + delta_y)
    return x_arr, y_arr



