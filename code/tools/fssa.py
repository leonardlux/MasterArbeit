import numpy as np
from scipy.optimize import minimize

def vectorized_weight(xs, ys, errs):
    """
    Vectorized weight function used in collapse objective.
    """
    # Apply sliding windows of size 3
    xs_win = np.lib.stride_tricks.sliding_window_view(xs, 3)
    ys_win = np.lib.stride_tricks.sliding_window_view(ys, 3)
    errs_win = np.lib.stride_tricks.sliding_window_view(errs, 3)

    x0, x1, x2 = xs_win[:, 0], xs_win[:, 1], xs_win[:, 2]
    y0, y1, y2 = ys_win[:, 0], ys_win[:, 1], ys_win[:, 2]
    e0, e1, e2 = errs_win[:, 0], errs_win[:, 1], errs_win[:, 2]

    y_bar = ((x2 - x1) * y0 - (x0 - x1) * y2) / (x2 - x0)
    delta_sq = (e1**2 + (e0 * (x2 - x1) / (x2 - x0))**2 + (e2 * (x0 - x1) / (x2 - x0))**2)
    return ((y1 - y_bar) ** 2) / delta_sq

def collapse_objective(params, xs_list, ys_list, errs_list, Ls, derivative=0):
    """
    Objective function for scaling collapse. Optimizes for critical parameters (pc, nu).
    """
    pc, inv_nu = params
    xs_all, ys_all, errs_all = [], [], []

    # Perform the scaling transformation
    for xs, ys, errs, L in zip(xs_list, ys_list, errs_list, Ls):
        x_scaled = (xs - pc) * L ** inv_nu
        if derivative == 1:
            ys /= L ** inv_nu
        xs_all.append(x_scaled)
        ys_all.append(ys)
        errs_all.append(errs)

    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    errs = np.concatenate(errs_all)

    # Sorting the data by xs
    order = np.argsort(xs)
    xs, ys, errs = xs[order], ys[order], errs[order]

    # Calculate the vectorized weight function 
    weights = vectorized_weight(xs, ys, errs)
    return np.mean(weights)

def compute_critical_exponents(xs_list, ys_list, errs_list, Ls, guess_xc, guess_nu, derivative=0):
    """
    Perform the scaling collapse fit to find critical exponents pc and nu.
    """
    x0 = [guess_xc, 1 / guess_nu]
    result = minimize(
        collapse_objective, 
        x0, 
        args=(xs_list, ys_list, errs_list, Ls, derivative),
        method="Nelder-Mead", 
        tol=1e-5,
        )
    return result

def compute_error_bar(xs_list, ys_list, errs_list, Ls, guess_xc, guess_nu, derivative=0, n_samples=250):
    """
    Bootstrap error bars for pc and inv_nu by resampling the data.
    """
    pcs, inv_nus = np.zeros(n_samples), np.zeros(n_samples)

    for i in range(n_samples):
        # Resample ys with normal distribution based on the error bars
        ys_sampled = [np.random.normal(ys, errs) for ys, errs in zip(ys_list, errs_list)]
        result = compute_critical_exponents(xs_list, ys_sampled, errs_list, Ls, guess_xc, guess_nu, derivative)
        pcs[i], inv_nus[i] = result.x

    xc_err = np.percentile(pcs, 97.5) - np.percentile(pcs, 2.5)
    inv_nu_err = np.percentile(inv_nus, 97.5) - np.percentile(inv_nus, 2.5)

    return xc_err, inv_nu_err
