import numpy as np
from scipy.optimize import curve_fit

def t1_fitting(t_values, y_values):
    """
    Fit a T1 relaxation curve to given time and signal intensity values.
    
    Parameters:
    t_values (numpy.ndarray): Time values (TI times).
    y_values (numpy.ndarray): Signal intensities.
    
    Returns:
    dict: A dictionary containing the fitting results and error information if any.
    """
    if not (t_values.size and y_values.size):
        return {'error_status': 1, 'error_str': 'No input data'}

    if t_values.size != y_values.size:
        return {'error_status': 2, 'error_str': 'Mismatch between TI and SI series lengths'}

    # Sort values by time
    sorted_indices = t_values.argsort()
    x = t_values[sorted_indices]
    y = y_values[sorted_indices]

    guess_nums = np.array([100, 200, 500, 800, 1000, 1500, 2000, 2500])
    error_nums = np.zeros_like(guess_nums)
    result_list = []
    
    # Try fitting with different initial guesses for T1
    for guess_num in guess_nums:
        guess = [y.max(), 2 * y.max(), guess_num]
        fit_result = run_t1_fit(x, y, guess, abs_fit=True)
        error_nums = fit_result['residue']
        result_list.append(fit_result)

    # Find the result with minimum error
    best_initial_fit = result_list[np.argmin(error_nums)]
    
    # Refine the fit around the point of minimum signal intensity
    index_ylow = np.argmin(y)
    refined_results = []
    refined_indices = []
    refined_errors = []
    
    for offset in [-1, 0, 1]:
        index = index_ylow + offset
        if index < 0 or index >= len(y):
            refined_errors.append(1e8)
            refined_results.append(best_initial_fit)
        else:
            flipped_y = y.copy()
            flipped_y[:index + 1] *= -1
            guess = [best_initial_fit['A'], best_initial_fit['B'], best_initial_fit['T1_star']]
            fit_result = run_t1_fit(x, flipped_y, guess, abs_fit=False)
            refined_errors.append(fit_result['residue'])
            refined_indices.append(index)
            refined_results.append(fit_result)

    best_refined_fit = refined_results[np.argmin(refined_errors)]
    return best_refined_fit
    
def run_t1_fit(x, y, initial_guess, abs_fit):
    """
    Helper function to perform the curve fitting.
    
    Parameters:
    x (numpy.ndarray): Time values.
    y (numpy.ndarray): Signal intensities.
    initial_guess (list): Initial guess for the fitting parameters.
    abs_fit (bool): Whether to use the absolute value for fitting.
    
    Returns:
    dict: A dictionary containing the fitting result.
    """
    smooth_x = np.linspace(x[0], x[-1], 1000)
    exp_func = lambda x, A, B, t: abs(A - B * np.exp(-x / t)) if abs_fit else (A - B * np.exp(-x / t))
    params, _ = curve_fit(exp_func, x, y, p0=initial_guess)
    A, B, t = params
    smooth_y = exp_func(smooth_x, A, B, t)
    T1 = t * (B / A - 1)
    y_fitted = exp_func(x, A, B, t)
    residue = np.sum(np.abs(y_fitted - y))
    
    return {
        'A': A,
        'B': B,
        'T1_star': t,
        'T1': T1,
        't_val_org': x,
        'y_val_org': y,
        't_val_fit': smooth_x,
        'y_val_fit': smooth_y,
        'residue': residue,
        'error_status': 0,
        'error_str': 'OK!',
    }
