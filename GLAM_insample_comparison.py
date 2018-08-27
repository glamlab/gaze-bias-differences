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


def fitCompare(data, subject, n_tries=1, overwrite=False, progressbar=True):
    """
    Perform fitting of GLAM variants and
    WAIC model comparisons for a single subject
    1) Additive vs. No Bias
    2) Additive vs. Multiplicative vs. No Bias
    3) Multiplicative vs. No Bias
    4) Additive vs. Multiplicative
    """

    print("Processing subject {}...".format(subject))

    # Subset data
    subject_data = data[data['subject'] == subject].copy()
    n_items = subject_data['n_items'].values[0]
    if n_items == 2:
        subject_data = subject_data.drop(['item_value_2', 'gaze_2'], axis=1)
    subject_data['subject'] = 0

    # Additive
    print('\tS{}: Additive'.format(subject))

    parameters = ['v', 's', 'tau', 'gamma']

    additive = glam.GLAM(subject_data, drift='additive')
    additive.make_model('individual', gamma_bounds=(-10, 20), t0_val=0)

    additive = fitModel(additive, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar)
    summary = pm.summary(additive.trace[0])
    for parameter in parameters:
            summary.loc[parameter + '__0_0', 'MAP'] = additive.estimates[parameter].values[0]
    summary.to_csv(os.path.join('results', 'estimates', 'in_sample', 'additive', 'estimates_{}_additive_ins.csv'.format(subject)))

    additive_model = additive.model[0]
    additive_model.name = 'additive'
    additive_trace = additive.trace[0]
    pm.trace_to_dataframe(additive_trace).to_csv(os.path.join('results', 'traces', 'in_sample', 'additive', 'trace_{}_additive_ins.csv'.format(subject)))
    pm.traceplot(additive_trace)
    plt.savefig(os.path.join('results', 'traces', 'in_sample', 'additive', 'plots', 'traceplot_{}_additive_ins.png'.format(subject)))
    plt.close()

    # Multiplicative
    print('\tS{}: Multiplicative'.format(subject))

    parameters = ['v', 's', 'tau', 'gamma']

    multiplicative = glam.GLAM(subject_data, drift='multiplicative')
    multiplicative.make_model('individual', gamma_bounds=(-10, 1), t0_val=0)

    multiplicative = fitModel(multiplicative, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar)
    summary = pm.summary(multiplicative.trace[0])
    for parameter in parameters:
            summary.loc[parameter + '__0_0', 'MAP'] = multiplicative.estimates[parameter].values[0]
    summary.to_csv(os.path.join('results', 'estimates', 'in_sample', 'multiplicative', 'estimates_{}_multiplicative_ins.csv'.format(subject)))

    multiplicative_model = multiplicative.model[0]
    multiplicative_model.name = 'multiplicative'
    multiplicative_trace = multiplicative.trace[0]
    pm.trace_to_dataframe(multiplicative_trace).to_csv(os.path.join('results', 'traces', 'in_sample', 'multiplicative', 'trace_{}_multiplicative_ins.csv'.format(subject)))
    pm.traceplot(multiplicative_trace)
    plt.savefig(os.path.join('results', 'traces', 'in_sample', 'multiplicative', 'plots', 'traceplot_{}_multiplicative_ins.png'.format(subject)))
    plt.close()

    # No-Bias
    print('\tS{}: No Bias'.format(subject))

    parameters = ['v', 's', 'tau']

    nobias = glam.GLAM(subject_data, drift='nobias')
    nobias.make_model('individual', gamma_val=1.0, t0_val=0)

    nobias = fitModel(nobias, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar)
    summary = pm.summary(nobias.trace[0])
    for parameter in parameters:
            summary.loc[parameter + '__0_0', 'MAP'] = nobias.estimates[parameter].values[0]
    summary.to_csv(os.path.join('results', 'estimates', 'in_sample', 'nobias', 'estimates_{}_nobias_ins.csv'.format(subject)))

    nobias_model = nobias.model[0]
    nobias_model.name = 'nobias'
    nobias_trace = nobias.trace[0]
    pm.trace_to_dataframe(nobias_trace).to_csv(os.path.join('results', 'traces', 'in_sample', 'nobias', 'trace_{}_nobias_ins.csv'.format(subject)))
    pm.traceplot(nobias_trace)
    plt.savefig(os.path.join('results', 'traces', 'in_sample', 'nobias', 'plots', 'traceplot_{}_nobias.png'.format(subject)))
    plt.close()

    # Comparison Multiplicative vs Additive vs NoBias
    try:
        waic_df = pm.compare({multiplicative_model: multiplicative_trace,
                              additive_model: additive_trace,
                              nobias_model: nobias_trace},
                             ic='WAIC')
        waic_df.to_csv(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative_vs_nobias', 'additive_vs_multiplicative_vs_nobias_{}_waic.csv'.format(subject)))
        pm.compareplot(waic_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative_vs_nobias', 'plots', 'additive_vs_multiplicative_vs_nobias_{}_waic.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in WAIC comparison for subject {}'.format(subject))

    try:
        loo_df = pm.compare({multiplicative_model: multiplicative_trace,
                             additive_model: additive_trace,
                             nobias_model: nobias_trace},
                            ic='LOO')
        loo_df.to_csv(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative_vs_nobias', 'additive_vs_multplicative_vs_nobias_{}_loo.csv'.format(subject)))
        pm.compareplot(loo_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative_vs_nobias', 'plots', 'additive_vs_multplicative_vs_nobias_{}_loo.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in LOO comparison for subject {}'.format(subject))

    # Comparison Additive vs NoBias
    try:
        waic_df = pm.compare({additive_model: additive_trace,
                              nobias_model: nobias_trace},
                             ic='WAIC')
        waic_df.to_csv(os.path.join('results', 'model_comparison', 'additive_vs_nobias', 'additive_vs_nobias_{}_waic.csv'.format(subject)))
        pm.compareplot(waic_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'additive_vs_nobias', 'plots', 'additive_vs_nobias_{}_waic.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in WAIC comparison for subject {}'.format(subject))

    try:
        loo_df = pm.compare({additive_model: additive_trace,
                             nobias_model: nobias_trace},
                            ic='LOO')
        loo_df.to_csv(os.path.join('results', 'model_comparison', 'additive_vs_nobias', 'additive_vs_nobias_{}_loo.csv'.format(subject)))
        pm.compareplot(loo_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'additive_vs_nobias', 'plots', 'additive_vs_nobias_{}_loo.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in LOO comparison for subject {}'.format(subject))

    # Comparison Multiplicative vs NoBias
    try:
        waic_df = pm.compare({multiplicative_model: multiplicative_trace,
                              nobias_model: nobias_trace},
                             ic='WAIC')
        waic_df.to_csv(os.path.join('results', 'model_comparison', 'multiplicative_vs_nobias', 'multiplicative_vs_nobias_{}_waic.csv'.format(subject)))
        pm.compareplot(waic_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'multiplicative_vs_nobias', 'plots', 'multiplicative_vs_nobias_{}_waic.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in WAIC comparison for subject {}'.format(subject))

    try:
        loo_df = pm.compare({multiplicative_model: multiplicative_trace,
                             nobias_model: nobias_trace},
                            ic='LOO')
        loo_df.to_csv(os.path.join('results', 'model_comparison', 'multiplicative_vs_nobias', 'multiplicative_vs_nobias_{}_loo.csv'.format(subject)))
        pm.compareplot(loo_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'multiplicative_vs_nobias', 'plots', 'multiplicative_vs_nobias_{}_loo.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in LOO comparison for subject {}'.format(subject))

    # Comparison Additive vs Multiplicative
    try:
        waic_df = pm.compare({additive_model: additive_trace,
                              multiplicative_model: multiplicative_trace},
                             ic='WAIC')
        waic_df.to_csv(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative', 'additive_vs_multiplicative_{}_waic.csv'.format(subject)))
        pm.compareplot(waic_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative', 'plots', 'additive_vs_multiplicative_{}_waic.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in WAIC comparison for subject {}'.format(subject))

    try:
        loo_df = pm.compare({additive_model: additive_trace,
                             multiplicative_model: multiplicative_trace},
                            ic='LOO')
        loo_df.to_csv(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative', 'additive_vs_multiplicative_{}_loo.csv'.format(subject)))
        pm.compareplot(loo_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative', 'plots', 'additive_vs_multiplicative_{}_loo.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in LOO comparison for subject {}'.format(subject))

    return True


def fitSubjects(first=0, last=-1, n_tries=1, overwrite=False, progressbar=True):
    data = pd.read_csv(os.path.join('data', 'data_aggregate.csv'))
    for subject in data['subject'].unique()[first:last]:
        filename = os.path.join('results', 'model_comparison', 'additive_vs_multiplicative_vs_nobias', 'additive_vs_multiplicative_vs_nobias_{}_waic.csv'.format(subject))
        previous_results_present = os.path.isfile(filename)

        if previous_results_present:
            if not overwrite:
                print("Found existing model comparison results for Subject {}. Skipping all estimation and comparison...".format(subject))
            else:
                fitCompare(data, subject, n_tries=-1, overwrite=overwrite, progressbar=progressbar)
        else:
            fitCompare(data, subject, n_tries=n_tries, overwrite=overwrite, progressbar=progressbar)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", default=False, action="store_true",
                        help="Overwrite previous results.")
    parser.add_argument("--silent", default=False, action="store_true",
                        help="Run without progressbar.")
    parser.add_argument("--n-tries", default=1, type=int,
                        help="Number of tries for NUTS fitting, before falling back to ADVI.")
    parser.add_argument("--first", default=0, type=int,
                        help="First subject index to use.")
    parser.add_argument("--last", default=-1, type=int,
                        help="Last subject index to use.")
    args = parser.parse_args()

    fitSubjects(first=args.first, last=args.last,
                overwrite=args.overwrite, progressbar=(not args.silent),
                n_tries=args.n_tries)
