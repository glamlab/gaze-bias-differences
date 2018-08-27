import glam
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse


def check_convergence(summary, parameters=['v', 's', 'tau'],
                      n_eff_required=100, gelman_rubin_criterion=0.05):
    """
    Checks model convergence based on
    number of effective samples
    and Gelman Rubin statistics
    from pymc3 model summary table.
    """
    parameters = [parameter + '__0_0' for parameter in parameters]

    enough_eff_samples = np.all(summary.loc[parameters]['n_eff'] > n_eff_required)
    good_gelman = np.all(np.abs(summary.loc[parameters]['Rhat'] - 1.0) < gelman_rubin_criterion)

    if not enough_eff_samples or not good_gelman:
        return False
    else:
        return True


def fitModel(model, relevant_parameters=['v', 's', 'tau'],
             n_tuning_initial=1000, n_tuning_increase=1000,
             seed_start=10, seed_increment=1, n_tries_max=1,
             n_advi=200000, fallback='Metropolis',
             progressbar=True):
    """
    Keep fitting a given GLAM model until convergence diagnosed.
    Then fall back to ADVI.
    """
    converged = False

    n_tuning = n_tuning_initial
    seed = seed_start
    n_tries = 0

    while (not converged) and (n_tries < n_tries_max):
        np.random.seed(seed)
        model.fit(method='NUTS', tune=n_tuning, progressbar=progressbar)
        summary = pm.summary(model.trace[0])

        converged = check_convergence(summary, parameters=relevant_parameters)
        seed += seed_increment
        n_tuning += n_tuning_increase
        n_tries += 1

    if not converged:
        if fallback is 'ADVI':
            print("Falling back to ADVI...")
            model.fit(method='ADVI', n_advi=n_advi)
        elif fallback is 'Metropolis':
            print("Falling back to Metropolis...")
            model.fit(method='Metropolis', n_samples=10000)

    return model


def recoverSubject(data, subject, estimates, n_repeats=1, n_tries=1, overwrite=False, progressbar=True):
    """
    Perform parameter recovery for a single subject.
    1) Predict using GLAM with given estimates
    2) Fit GLAM
    3) Save estimates
    """

    print("Processing subject {}...".format(subject))

    # Subset data
    subject_data = data[data['subject'] == subject].copy()
    n_items = subject_data['n_items'].values[0]
    if n_items == 2:
        subject_data = subject_data.drop(['item_value_2', 'gaze_2'], axis=1)
    subject_data['subject'] = 0

    if (overwrite) or (not os.path.isfile(os.path.join('results', 'parameter_recovery', 'recovered_estimates_{}_ins.csv'.format(subject)))):

        parameters = ['v', 's', 'tau', 'gamma']
        # Set up model, supply it with parameter estimates
        generating = glam.GLAM(subject_data, drift='multiplicative')
        generating.make_model('individual', gamma_bounds=(-10, 1), t0_val=0)
        estimates_dict = {parameter: estimates.loc[parameter + '__0_0', 'MAP'] for parameter in parameters}
        estimates_dict['t0'] = 0
        estimates_dict['subject'] = 0
        generating.estimates = pd.DataFrame(estimates_dict, index=np.zeros(1))
        generating.predict(n_repeats=n_repeats)
        generated = generating.prediction
        recovering = glam.GLAM(generated)
        recovering.make_model('individual', gamma_bounds=(-10, 1), t0_val=0)
        recovering = fitModel(recovering, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar)
        summary = pm.summary(recovering.trace[0])
        for parameter in parameters:
            summary.loc[parameter + '__0_0', 'MAP'] = recovering.estimates[parameter].values[0]
            summary.loc[parameter + '__0_0', 'generating'] = estimates_dict[parameter]
        summary.to_csv(os.path.join('results', 'parameter_recovery', 'recovered_estimates_{}_multiplicative_ins.csv'.format(subject)))

        pm.trace_to_dataframe(recovering.trace[0]).to_csv(os.path.join('results', 'traces', 'parameter_recovery', 'trace_{}_parameter_recovery_ins.csv'.format(subject)))
        pm.traceplot(recovering.trace[0])
        plt.savefig(os.path.join('results', 'traces', 'parameter_recovery', 'plots', 'traceplot_{}_parameter_recovery_ins.png'.format(subject)))
        plt.close()

    else:
        print("Previous recovery results found (Subject {}). Skipping...".format(subject))

    return


def recoverGLAM(first=0, last=-1, n_repeats=1, n_tries=1, overwrite=False, progressbar=True):
    data = pd.read_csv(os.path.join('data', 'data_aggregate.csv'))
    for subject in data['subject'].unique()[first:last]:
        estimates = pd.read_csv(os.path.join('results', 'estimates', 'in_sample', 'multiplicative', 'estimates_{}_multiplicative_ins.csv'.format(subject)),
                                index_col=0)
        recoverSubject(data, subject, estimates, n_repeats=n_repeats, n_tries=n_tries, overwrite=overwrite, progressbar=progressbar)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", default=False, action="store_true",
                        help="Overwrite previous results.")
    parser.add_argument("--silent", default=False, action="store_true",
                        help="Run without progressbar.")
    parser.add_argument("--n-tries", default=1, type=int,
                        help="Number of tries for NUTS fitting, before falling back to fallback method.")
    parser.add_argument("--first", default=0, type=int,
                        help="First subject index to use.")
    parser.add_argument("--last", default=-1, type=int,
                        help="Last subject index to use.")
    parser.add_argument("--n-prediction-repeats", default=1, type=int,
                        help="Number of trial repetitions in prediction.")
    args = parser.parse_args()

    recoverGLAM(first=args.first, last=args.last,
                overwrite=args.overwrite, progressbar=(not args.silent),
                n_repeats=args.n_prediction_repeats, n_tries=args.n_tries)
