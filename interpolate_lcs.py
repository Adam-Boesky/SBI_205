import george
import numpy as np

from models.lightcurves import LightCurve
from scipy import optimize


def ugrizy_to_numbers(filters: np.ndarray) -> np.ndarray:
    '''Helper function to convert ugrizy strings to numbers.'''
    numberfied = np.zeros(shape=len(filters))  # u
    numberfied[filters == 'g'] = 1
    numberfied[filters == 'r'] = 2
    numberfied[filters == 'i'] = 3
    numberfied[filters == 'z'] = 4
    numberfied[filters == 'y'] = 5

    return numberfied


def interpolate_lc(lc: LightCurve, nfilts: int = 6, abs_lim_mag: float = 24.0):
    """Interpolate the lightcurves using a Matern 3-2 Gaussian Process."""
    # Grab the necessary values
    times = lc.times
    abs_mags = lc.mags
    filters = ugrizy_to_numbers(lc.filters)
    abs_mags_err = (1 / lc.snrs) * lc.mags
    
    # Center the data and stack with filter encoding
    gp_mags = abs_mags - abs_lim_mag
    times = times - np.min(times)
    stacked_data = np.vstack([times, filters]).T

    # Set up the Gaussian kernel
    x_pred = np.zeros((100*nfilts, 2))
    kernel = np.var(gp_mags) * george.kernels.Matern32Kernel([10, 2], ndim=2)
    gp = george.GP(kernel)
    gp.compute(stacked_data, abs_mags_err)

    # Set the objective function and its gradient
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(gp_mags)
    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(gp_mags)

    # Fit our GP to the data
    try:
        result = optimize.minimize(neg_ln_like,
                                         gp.get_parameter_vector(),
                                         jac=grad_neg_ln_like)

        gp.set_parameter_vector(result.x)

    except:
        gp.set_parameter_vector([1,10,2])

    # Predict using the GP
    x_pred[:,0] = np.repeat(np.linspace(0.1,100,100),6)
    x_pred[:,1] = np.tile(np.linspace(0,5,6),100)
    pred, pred_var = gp.predict(gp_mags, x_pred, return_var=True)
    pred = pred + abs_lim_mag  # Un-center

    # interpolated_lc = LightCurve(lc.times, lc.mags, lc.filters, lc.snrs,
    #                              lc.texp, lc.tpeak, lc.rmag, lc.redshift,
    #                              lc.theta, np.repeat(np.linspace(0.1,100,100),6),
    #                              pred, pred_var)
    return lc.set_interped(np.repeat(np.linspace(0.1,100,100),6), pred, pred_var)


def apply_quality_cut(lcs: np.ndarray) -> np.ndarray:
    """Apply a quality cut on the given light curves"""
    lcs_cut = []
    for lc in lcs:

        # Get some helpful values
        num_bad = np.sum(np.isinf(lc.mags) + np.isnan(lc.mags))     # number of bad bands
        prop_bad = num_bad / len(lc.mags)                           # propotion of bad bands
        good_mask = np.isfinite(lc.mags)                      # mask for the good bands

        # Fill in the arrays with the quality bands
        if len(lc.times[good_mask]) > 0.5 * len(lc.times[good_mask]) and len(lc.times) > 50:  # >50% of the bands are not infinite and we have more than 50 data points
            if (lc.times[good_mask][-1] - lc.times[good_mask][0] > 70) & (prop_bad < 0.7):  # We have at least 70 days of observation and the proportion of bad bands is <70%

                # Add all of the good stuff to our good cut
                lcs_cut.append(LightCurve(lc.times[good_mask], lc.mags[good_mask], lc.filters[good_mask], lc.snrs[good_mask], lc.texp, lc.tpeak, lc.rmag, lc.redshift, lc.theta))

    print(f'Original number of lcs is {len(lcs)}, quality cut kept {len(lcs_cut)}')
    
    return lcs_cut


def interpolate_lcs():

    # Import the LCs
    lcs = np.load('full_lcs.npz', allow_pickle=True)['lcs']

    # Apply a quality cut
    lcs_cut = apply_quality_cut(lcs)

    # Interpolate each of the lightcurves
    interpolated_lcs = []
    for i_lc, lc in enumerate(lcs_cut):
        if i_lc % 1000 == 0:
            print(f'Interpolated {i_lc} / {len(lcs_cut)}')
        interpolated_lcs.append(interpolate_lc(lc))

    # Save
    np.savez('full_lcs_interped.npz', lcs=interpolated_lcs)


if __name__=='__main__':
    interpolate_lcs()
