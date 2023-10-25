def generate_intervals(interval_start,interval_end,n_int):
    """Generate a list of intervals between interval_start and interval_end with n_int steps

    :param interval_start: _description_
    :type interval_start: _type_
    :param interval_end: _description_
    :type interval_end: _type_
    :param n_int: _description_
    :type n_int: _type_
    :return: _description_
    :rtype: _type_
    """

    # Calculate the step size
    step = (interval_end - interval_start) / n_int

    # Generate a list of intervals
    intervals = [[interval_start + step * i, interval_start + step * (i + 1)] for i in range(n_int)]

    return intervals