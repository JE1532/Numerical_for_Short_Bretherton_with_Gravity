import numpy

def forward_euler_integration(y_0, vec_deriv, delta, x_0=0, conv=None, bounds=None):

    def calc_delta(delta, y_der):
        sign = 1 if delta > 0 else -1
        der_delta = abs(delta / min(y for y in y_der if y != 0))
        if abs(der_delta) < abs(delta):
            return sign * der_delta
        return delta

    if bounds == None and conv == None:
        raise Exception("Improper use - need to define end condition for integration: either conv != None or bounds != None")
    x_arr = [x_0]
    y_arr = [y_0]
    get_conv_derivative = lambda x_vals, y_vals: numpy.array([y[conv] for y in y_vals]) if conv < len(y_vals[-1]) else numpy.array([vec_deriv(x_vals[i], y_vals[i])[-1] for i in range(len(y_vals))])
    conv_condition = lambda x_arr, y_arr: numpy.std(get_conv_derivative(x_arr[-100:], y_arr[-100:])) >= 0.001 * numpy.average(get_conv_derivative(x_arr[-100:], y_arr[-100:]))
    bounds_condition = lambda x_arr, delta: x_arr[-1] + delta < bounds[1] and x_arr[-1] + delta > bounds[0]
    i = 0
    y_der = vec_deriv(x_arr[-1], y_arr[-1])
    ev_delta = calc_delta(delta, y_der)
    while len(y_arr) < 100 or ((not conv or conv_condition(x_arr, y_arr)) and (not bounds or bounds_condition(x_arr, ev_delta))):
        i += 1
        #print(i)
        x_arr.append(x_arr[-1] + ev_delta)
        delta_y = y_der * ev_delta
        y_arr.append(y_arr[-1] + delta_y)
        y_der = vec_deriv(x_arr[-1], y_arr[-1])
        ev_delta = calc_delta(delta, y_der)
    return x_arr, y_arr


if __name__ == "__main__":
    x_arr, y_arr = forward_euler_integration(numpy.array([0, 0, 2]), lambda x, y: numpy.array([y[1], y[2], 0]), 0.0001, bounds=[0, 10])
    i = 0
    while x_arr[i] < 6:
        i += 1
    print(y_arr[i])
