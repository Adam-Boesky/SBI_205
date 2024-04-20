import numpy as np

class LightCurve():
    """
    A basic transient model
    """

    def __init__(self, times, mags, filters, snrs, texp, tpeak, rmag, redshift, theta, times_interped = None, mags_interped = None, magerrs_interped = None):
        """
        Parameters:
        ----------
        ...

        """
        self.times = times
        self.mags = mags
        self.filters = filters
        self.snrs = snrs
        self.texp = texp
        self.tpeak = tpeak
        self.rmag = rmag
        self.redshift = redshift
        self.theta = theta

        # The interpolated data
        self.times_interped = times_interped
        self.mags_interped = mags_interped
        self.magerrs_interped = magerrs_interped


    def set_interped(self, interped_times: np.ndarray, interped_mags: np.ndarray, interped_errs: np.ndarray):
        """Set the interpolated data"""
        if (interped_times.shape[0] != interped_mags.shape[0]) or (interped_mags.shape[0] != interped_errs.shape[0]):
            raise ValueError('The dimensions of the interpolated times, mags, and errors are not all the same.')
        self.times_interped = interped_times
        self.mags_interped = interped_mags
        self.magerrs_interped = interped_errs

        return self
