"""Train our SBI model."""
import sys
from sbi import inference as Inference
from sbi import utils as sbiutils
from sbi.utils import process_prior

import torch
import pickle
import numpy as np
import torch.distributions as dist

from models.custom_sklearn import StandardScaler, train_test_split
from models.distributions import TruncatedNormal, LogUniform

if torch.cuda.is_available(): device = 'mps'
else: DEVICE = 'cpu'
print(f'device is {DEVICE}')

sys.path.append('/Users/adamboesky/Research/SBI_205/models')


def train_and_store_sbi():

    # Load our data
    print('Loading data!!!')
    with open('data/full_encoded_lcs.pkl', 'rb') as f:
        encoded_lcs, lcs = pickle.load(f)

    # Split into test and train
    print('Preprocessing data!!!')
    predictor_mask = np.array([ True,  True, False, False, False, False, False,  True,  True, False,  True])  # mask for theta that only gets what we actually care about
    all_predictor_labels = np.array(['pspin', 'bfield', 'mns', 'thetapb', 'texp', 'kappa', 'kappagamma', 'mej', 'vej', 'tfloor', 'texplosion'])
    predictor_labels = all_predictor_labels[predictor_mask]
    print(f'theta = {predictor_labels}')
    X_encoded = np.array([np.concatenate((lc_encoding, [lc.redshift])) for lc_encoding, lc in zip(encoded_lcs, lcs)])
    y = np.array([np.array(lc.theta)[predictor_mask] for lc in lcs])

    # Preprocess the data
    y[:, -1] *= -1  # multiply the explosion time by negative because they're all negative
    y[:, 1] = np.log10(y[:, 1])
    y[:, 2] = np.log10(y[:, 2])
    y_means = np.mean(y, axis=0)
    y_stds = np.std(y, axis=0)
    y_norm = y
    y_norm[:, :-1] -= y_means[:-1]
    y_norm[:, :-1] /= y_stds[:-1]

    # Array of the prior distribution parameters
    PD_params = [[np.min(y_norm[:, 0]), np.max(y_norm[:, 0])],
                [np.min(y_norm[:, 1]), np.max(y_norm[:, 1])],
                [np.min(y_norm[:, 2]), np.max(y_norm[:, 2])],
                [np.min(y_norm[:, 3]), np.max(y_norm[:, 3])],
                [1/y_means[-1]]]

    # Need to round them to be compatible with SBI
    print('Setting up priors!!!')
    print('Prior parameters:')
    for i, row in enumerate(PD_params):
        for j, val in enumerate(row):
            PD_params[i][j] = float(round(val, 4))

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_norm, random_state=22, test_size=0.2)
    # Hyperparameters
    nhidden = 20
    nblocks = 5

    # Initialize your prior
    my_prior, num_params, prior_returns_numpy = process_prior([dist.Uniform(low=torch.tensor([PD_params[0][0]]), high=torch.tensor([PD_params[0][1]])),        # pspin
                        dist.Uniform(torch.tensor([PD_params[1][0]]), torch.tensor([PD_params[1][1]])),                                                    # bmag
                        dist.Uniform(torch.tensor([PD_params[2][0]]), torch.tensor([PD_params[2][1]])),                                                      # mej
                        dist.Uniform(torch.tensor([PD_params[3][0]]), torch.tensor([PD_params[3][1]])),      # vej
                        dist.Exponential(torch.tensor([PD_params[4][0]]))                                                                            # -1 * texp (mean is 1/25)
                        ])

    # Flow!
    print('Training flow!!!')
    anpe = Inference.SNPE(
        prior=my_prior,
        density_estimator=sbiutils.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks),
        device=DEVICE
    )
    anpe.append_simulations(
        torch.as_tensor(y_train.astype(np.float32)),
        torch.as_tensor(X_train.astype(np.float32)))
    p_x_y_estimator = anpe.train(stop_after_epochs=1000)

    # Get the posterior
    hatp_x_y = anpe.build_posterior(p_x_y_estimator)

    # Save the model and posterior
    print('Saving model and data!')
    with open('data/sbi_results0.pkl', 'wb') as f:
        pickle.dump((hatp_x_y, p_x_y_estimator), f)


if __name__=='__main__':
    train_and_store_sbi()
