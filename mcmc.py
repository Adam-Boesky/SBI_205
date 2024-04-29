"""MCMC sampling"""
import pickle
import torch
import emcee
import numpy as np
import torch.distributions as dist

from scipy import stats
from multiprocessing import Pool
from sbi.utils import process_prior
from interpolate_lcs import ugrizy_to_numbers
from models.magnetar_model import gen_magnetar_model
from torch.distributions import Distribution, constraints, Normal
from astropy.cosmology import WMAP9 as cosmo

from models.distributions import TruncatedNormal
from models.custom_sklearn import StandardScaler, train_test_split

import sys
sys.path.append('/Users/adamboesky/Research/SBI_205/models')


# Fixed parameters in our MCMC (these were fixed by prof. Villar in simulation)
mns = 1.40000000e+00
thetapb = 1.57079633e+00
texp = 0.00000000e+00
kappa = 1.12600000e-01
kappagamma = 1.00000000e-01
tfloor = 6.00000000e+03
filt_grid = np.tile(np.linspace(0,5,6),100).astype(int)
time_grid = np.repeat(np.linspace(0.1,100,100),6)


MCMC_PRIOR, NUM_PARAMS, _ = process_prior([dist.Uniform(low=torch.tensor([0.7]), high=torch.tensor([20.0])),    # pspin
                                           dist.Uniform(low=torch.tensor([-2.0]), high=torch.tensor([1.0])),    # bmag
                                           dist.Uniform(low=torch.tensor([-1.0]), high=torch.tensor([1.3])),    # mej
                                           dist.Uniform(low=torch.tensor([1.0]), high=torch.tensor([30.0])),    # vej  (in 1000s)
                                           dist.Exponential(torch.tensor([0.1])),                               # -1 * texp (mean is 27)
                                           dist.Uniform(low=torch.tensor([-10.0]), high=torch.tensor([0.4]))    #dist.Exponential(torch.tensor([1.0]))               # noise
                                        ])


def log_likelihood(xs, y, yerr, filts, theta, z, dist_const):
    """Chi squared likelihood of the SLSN model."""
    y_pred = torch.tensor(gen_magnetar_model(xs + theta[4],
                                torch.tensor([theta[0], 10**theta[1], mns, thetapb, texp, kappa, kappagamma, 10**theta[2], theta[3]*1000, tfloor]),
                                filt=filts,
                                redshift=z,
                                dist_const=dist_const))
    y_pred = np.nan_to_num(y_pred, nan=np.inf)
    sigma_sq = yerr**2 + (10.0**theta[5])**2  # adding a white noise term for underestimated noise
    chi_sq = -0.5 * torch.nansum((y - y_pred) ** 2 / sigma_sq + np.log(2 * np.pi * sigma_sq))
    return chi_sq


def log_probability(theta, ts, y, yerr, filts, z, dist_const):
    """Log probability."""
    lp = MCMC_PRIOR.log_prob(torch.tensor(theta))
    if not np.isfinite(lp) or theta[-2] < 0:
        return -np.inf

    ll = log_likelihood(ts, y, yerr, filts, theta, z, dist_const)
    return (lp + ll).detach().numpy()


def run_mcmc():


    ### LOAD THE DATA###
    print('Importing data!')

    # Import the LCs
    lcs = np.load('data/full_lcs_interped.npz', allow_pickle=True)['lcs']
    lcs = np.array([lc for lc in lcs if np.mean(lc.snrs) > 3])
    for lc in lcs:  # adjust the t explosion
        lc.theta[-1] -= min(lc.times)
        lc.theta[-1] *= -1
    lcs.shape

    # with open('data/full_encoded_lcs.pkl', 'rb') as f:
    #     encoded_lcs, lcs = pickle.load(f)

    # # Filter for snr > 3
    # encoded_lcs     = np.array([e_lc for e_lc, lc in zip(encoded_lcs, lcs) if np.mean(lc.snrs) > 3])
    # lcs             = np.array([lc   for e_lc, lc in zip(encoded_lcs, lcs) if np.mean(lc.snrs) > 3])
    lcs_train, lcs = train_test_split(lcs, random_state=22, test_size=0.2)
    predictor_mask = np.array([ True,  True, False, False, False, False, False,  True,  True, False,  True]) # Mask for theta that only gets what we actually care about
    all_predictor_labels = np.array(['pspin', 'bfield', 'mns', 'thetapb', 'texp', 'kappa', 'kappagamma', 'mej', 'vej', 'tfloor', 'texplosion'])
    predictor_labels = all_predictor_labels[predictor_mask]
    print(f'theta = {predictor_labels}')

    # Tensorize the data
    ts = [np.array(lc.times) - np.min(lc.times) for lc in lcs]
    thetas = torch.tensor([lc.theta for lc in lcs])[:, predictor_mask]
    filters = [ugrizy_to_numbers(lc.filters).astype(int) for lc in lcs]
    zs = torch.tensor([lc.redshift for lc in lcs]).reshape(-1, 1)
    lumdists = cosmo.luminosity_distance(zs)
    dist_consts = torch.tensor(np.log10(4. * np.pi * (lumdists.cgs.value) ** 2).reshape(-1, 1))
    ys = [torch.tensor(lc.mags) for lc in lcs]
    yerrs = [torch.tensor((1/lc.snrs) * lc.mags) for lc in lcs]


    nwalkers = 50
    p0 = MCMC_PRIOR.sample(sample_shape=torch.Size([50,]))
    dtype = [("log_prior", float), ("mean", float)]



    ### RUN MCMC ###
    print('Running MCMC')
    # Multithreaded MCMC
    n_lcs = len(lcs)
    for i in range(231, n_lcs):
        with Pool() as pool:

            # Initialize the sampler
            sampler = emcee.EnsembleSampler(nwalkers, NUM_PARAMS, log_probability, args=(ts[i], ys[i], yerrs[i], filters[i], zs[i], dist_consts[i]), blobs_dtype=dtype, pool=pool)

            # Run 100 burn in steps :)
            print(f'Started burn {i+1} / {n_lcs}')
            burn_results = sampler.run_mcmc(p0, 100, progress=True)
            print(f'Finished burn {i+1} / {n_lcs}')
            sampler.reset()

            # Do the final sampling
            print(f'Started final sampling {i+1} / {n_lcs}')
            final_results = sampler.run_mcmc(burn_results, 6000, progress=True)
            print(f'Finished final sampling {i+1} / {n_lcs}')

        # Save the data
        with open(f'/Users/adamboesky/Research/SBI_205/data/mcmc_results_final/mcmc_results{i}.pkl', 'wb') as f:
            pickle.dump((final_results, sampler), f)


if __name__=='__main__':
    run_mcmc()
