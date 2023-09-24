import numpy

def forward_euler_integration(y_0, vec_deriv, delta, conv=None, bounds=None):
    if bounds == None and conv == None:
        raise Exception("Improper use - need to define end condition for integration: either conv != None or bounds != None")
    x_arr = [0]
    y_arr = [y_0]
    get_conv_derivative = lambda y_vals: numpy.array([y[conv] for y in y_vals])
    conv_condition = lambda y_arr: numpy.std(get_conv_derivative(y_arr[-100:])) >= 0.01 * numpy.avg(get_conv_derivative(y_arr[-100:]))
    counds_condition = lambda x_arr: x_arr[-1] + delta > bounds[1] or x_arr[-1] + delta < bounds[1]
    while len(y_arr) < 100 or (not (conv and conv_condition(y_arr)) and not (bounds and counds_condition(x_arr))):
        x_arr.append(x_arr[-1] + delta)
        delta_y = vec_deriv(y_arr[-1]) * delta
        y_arr.append(y_arr[-1] + delta_y)
    return x_arr, y_arr
