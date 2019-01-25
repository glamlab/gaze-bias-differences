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


def fitPredictOOS(data, subject, n_repeats=50, n_tries=1, overwrite=False, progressbar=True):
    """
    Perform fitting of additive & no bias GLAM variants
    on even numbered trials, exchange data and
    predict for odd numbered trials
    """

    print("Processing subject {}...".format(subject))

    # Subset data
    subject_data = data[data['subject'] == subject].copy()
    n_items = subject_data['n_items'].values[0]
    if n_items == 2:
        subject_data = subject_data.drop(['item_value_2', 'gaze_2'], axis=1)
    subject_data['subject'] = 0

    # split into even and odd trials
    even = subject_data[(subject_data['trial'] % 2) == 0].copy().reset_index(drop=True)
    odd = subject_data[(subject_data['trial'] % 2) == 1].copy().reset_index(drop=True)

    # # Additive
    # if (overwrite) or (not os.path.isfile(os.path.join('results', 'estimates', 'out_of_sample', 'additive', 'estimates_{}_additive_oos.csv'.format(subject)))):
    #     print('\tS{}: Additive'.format(subject))

    #     parameters = ['v', 's', 'tau', 'gamma']

    #     additive = glam.GLAM(even, drift='additive')
    #     additive.make_model('individual', gamma_bounds=(-100, 100), t0_val=0)

    #     additive = fitModel(additive, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar)
    #     summary = pm.summary(additive.trace[0])
    #     for parameter in parameters:
    #         summary.loc[parameter + '__0_0', 'MAP'] = additive.estimates[parameter].values[0]
    #     summary.to_csv(os.path.join('results', 'estimates', 'out_of_sample', 'additive', 'estimates_{}_additive_oos.csv'.format(subject)))

    #     additive_model = additive.model[0]
    #     additive_model.name = 'additive'
    #     additive_trace = additive.trace[0]
    #     pm.trace_to_dataframe(additive_trace).to_csv(os.path.join('results', 'traces', 'out_of_sample', 'additive', 'trace_{}_additive_oos.csv'.format(subject)))
    #     pm.traceplot(additive_trace)
    #     plt.savefig(os.path.join('results', 'traces', 'out_of_sample', 'additive', 'plots', 'traceplot_{}_additive_oos.png'.format(subject)))
    #     plt.close()

    #     # out of sample prediction
    #     additive.exchange_data(odd)
    #     additive.predict(n_repeats=n_repeats)
    #     additive.prediction['subject'] = subject
    #     additive.prediction.to_csv(os.path.join('results', 'predictions', 'out_of_sample', 'additive', 'prediction_{}_additive_oos.csv'.format(subject)))
    # else:
    #     print("Previous estimates found for additive model (Subject {}). Skipping...".format(subject))

    # GLAM
    if (overwrite) or (not os.path.isfile(os.path.join('results', 'estimates', 'out_of_sample', 'GLAM_nofirst', 'estimates_{}_GLAM_nofirst_oos.csv'.format(subject)))):
        print('\tS{}: GLAM'.format(subject))

        parameters = ['v', 's', 'tau', 'gamma']

        GLAM = glam.GLAM(even, drift='multiplicative')
        GLAM.make_model('individual', gamma_bounds=(-10, 1), t0_val=0)

        GLAM = fitModel(GLAM, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar)
        summary = pm.summary(GLAM.trace[0])
        for parameter in parameters:
            summary.loc[parameter + '__0_0', 'MAP'] = GLAM.estimates[parameter].values[0]
        summary.to_csv(os.path.join('results', 'estimates', 'out_of_sample', 'GLAM_nofirst', 'estimates_{}_GLAM_nofirst_oos.csv'.format(subject)))

        GLAM_model = GLAM.model[0]
        GLAM_model.name = 'GLAM'
        GLAM_trace = GLAM.trace[0]
        pm.trace_to_dataframe(GLAM_trace).to_csv(os.path.join('results', 'traces', 'out_of_sample', 'GLAM_nofirst', 'trace_{}_GLAM_oos.csv'.format(subject)))
        pm.traceplot(GLAM_trace)
        plt.savefig(os.path.join('results', 'traces', 'out_of_sample', 'GLAM_nofirst', 'plots', 'traceplot_{}_GLAM_oos.png'.format(subject)))
        plt.close()

        # out of sample prediction
        GLAM.exchange_data(odd)
        GLAM.predict(n_repeats=n_repeats)
        GLAM.prediction['subject'] = subject

        GLAM.prediction.to_csv(os.path.join('results', 'predictions', 'out_of_sample', 'GLAM_nofirst', 'prediction_{}_GLAM_oos.csv'.format(subject)))
    else:
        print("Previous estimates found for GLAM model (Subject {}). Skipping...".format(subject))

    # # No-Bias
    # if (overwrite) or (not os.path.isfile(os.path.join('results', 'estimates', 'out_of_sample', 'nobias', 'estimates_{}_nobias_oos.csv'.format(subject)))):
    #     print('\tS{}: No Bias'.format(subject))

    #     parameters = ['v', 's', 'tau']

    #     nobias = glam.GLAM(even, drift='additive')
    #     nobias.make_model('individual', gamma_val=0.0, t0_val=0)

    #     nobias = fitModel(nobias, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar)
    #     summary = pm.summary(nobias.trace[0])
    #     for parameter in parameters:
    #         summary.loc[parameter + '__0_0', 'MAP'] = nobias.estimates[parameter].values[0]
    #     summary.to_csv(os.path.join('results', 'estimates', 'out_of_sample', 'nobias', 'estimates_{}_nobias_oos.csv'.format(subject)))

    #     nobias_model = nobias.model[0]
    #     nobias_model.name = 'nobias'
    #     nobias_trace = nobias.trace[0]
    #     pm.trace_to_dataframe(nobias_trace).to_csv(os.path.join('results', 'traces', 'out_of_sample', 'nobias', 'trace_{}_nobias_oos.csv'.format(subject)))
    #     pm.traceplot(nobias_trace)
    #     plt.savefig(os.path.join('results', 'traces', 'out_of_sample', 'nobias', 'plots', 'traceplot_{}_nobias_oos.png'.format(subject)))
    #     plt.close()

    #     # out of sample prediction
    #     nobias.exchange_data(odd)
    #     nobias.predict(n_repeats=n_repeats)
    #     nobias.prediction['subject'] = subject

    #     nobias.prediction.to_csv(os.path.join('results', 'predictions', 'out_of_sample', 'nobias', 'prediction_{}_nobias_oos.csv'.format(subject)))
    # else:
    #     print("Previous estimates found for no-bias model (Subject {}). Skipping...".format(subject))

    return


def fitSubjects(first=0, last=-1, n_repeats=50, n_tries=1, overwrite=False, progressbar=True):
    data = pd.read_csv(os.path.join('data', 'data_nofirst.csv'))
    for subject in data['subject'].unique()[first:last]:
        fitPredictOOS(data, subject, n_repeats=n_repeats, n_tries=n_tries, overwrite=overwrite, progressbar=progressbar)


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
    parser.add_argument("--n-prediction-repeats", default=50, type=int,
                        help="Number of trial repetitions in prediction.")
    args = parser.parse_args()

    fitSubjects(first=args.first, last=args.last,
                overwrite=args.overwrite, progressbar=(not args.silent),
                n_repeats=args.n_prediction_repeats, n_tries=args.n_tries)
