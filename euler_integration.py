import numpy

def forward_euler_integration(y_0, vec_deriv, delta, x_0=0, conv=None, bounds=None):
    if bounds == None and conv == None:
        raise Exception("Improper use - need to define end condition for integration: either conv != None or bounds != None")
    x_arr = [x_0]
    y_arr = [y_0]
    get_conv_derivative = lambda x_vals, y_vals: numpy.array([y[conv] for y in y_vals]) if conv < len(y_vals[-1]) else numpy.array([vec_deriv(x_vals[i], y_vals[i])[-1] for i in range(len(y_vals))])
    conv_condition = lambda x_arr, y_arr: numpy.std(get_conv_derivative(x_arr[-100:], y_arr[-100:])) >= 0.01 * numpy.average(get_conv_derivative(x_arr[-100:], y_arr[-100:]))
    bounds_condition = lambda x_arr: x_arr[-1] + delta < bounds[1] and x_arr[-1] + delta > bounds[0]
    i = 0
    while len(y_arr) < 100 or ((not conv or conv_condition(x_arr, y_arr)) and (not bounds or bounds_condition(x_arr))):
        i += 1
        #print(i)
        x_arr.append(x_arr[-1] + delta)
        delta_y = vec_deriv(x_arr[-1], y_arr[-1]) * delta
        y_arr.append(y_arr[-1] + delta_y)
    return x_arr, y_arr


if __name__ == "__main__":
    x_arr, y_arr = forward_euler_integration(numpy.array([0, 0, 2]), lambda x, y: numpy.array([y[1], y[2], 0]), 0.0001, bounds=[0, 10])
    print(y_arr[60000])
