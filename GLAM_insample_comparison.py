import glam
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import errno


def make_sure_path_exists(path):
    """
    Used to check or create existing folder structure for results.
    https://stackoverflow.com/a/5032238
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

            
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


# def fitModel(model, relevant_parameters=['v', 's', 'tau'],
#              n_tuning_initial=1000, n_tuning_increase=1000,
#              seed_start=10, seed_increment=1, n_tries_max=1,
#              n_advi=200000, fallback='Metropolis', use_fallback=False,
#              progressbar=True):
#     """
#     Keep fitting a given GLAM model until convergence diagnosed.
#     Then fall back to fallback method.
#     """
#     converged = False

#     n_tuning = n_tuning_initial
#     seed = seed_start
#     n_tries = 0

#     if not use_fallback:
#         while (not converged) and (n_tries < n_tries_max):
#             np.random.seed(seed)
#             model.fit(method='NUTS', tune=n_tuning, progressbar=progressbar)
#             summary = pm.summary(model.trace[0])

#             converged = check_convergence(summary, parameters=relevant_parameters)
#             seed += seed_increment
#             n_tuning += n_tuning_increase
#             n_tries += 1

#     if (not converged) or (use_fallback):
#         use_fallback = True
#         if fallback is 'ADVI':
#             print("Falling back to ADVI...")
#             model.fit(method='ADVI', n_advi=n_advi)
#         elif fallback is 'Metropolis':
#             print("Falling back to Metropolis...")
#             model.fit(method='Metropolis', n_samples=10000)

#     if converged:
#         use_fallback = False

#     return model, use_fallback


def fit_indModel(data, subject,
                 drift, parameters=['v', 's', 'tau', 'gamma'], gamma_bounds=(-10, 1), gamma_val=None, t0_val=0,
                 model_name='GLAM',
                 n_tuning_initial=1000, n_tuning_increase=1000,
                 seed_start=10, seed_increment=1, n_tries_max=1,
                 n_advi=200000, fallback='Metropolis',
                 progressbar=True):

    # make model
    model = glam.GLAM(data, drift=drift)
    model.make_model('individual', gamma_bounds=gamma_bounds, gamma_val=gamma_val, t0_val=t0_val)

    # fit model
    converged = False

    n_tuning = n_tuning_initial
    seed = seed_start
    n_tries = 0

    while (not converged) and (n_tries < n_tries_max):
        np.random.seed(seed)
        model.fit(method='NUTS', tune=n_tuning, progressbar=progressbar)
        summary = pm.summary(model.trace[0])

        # check convergence
        converged = check_convergence(summary, parameters=parameters)

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

    # save results
    summary = pm.summary(model.trace[0])
    for parameter in parameters:
            summary.loc[parameter + '__0_0', 'MAP'] = model.estimates[parameter].values[0]
    path = os.path.join('results', 'estimates', 'in_sample', model_name)
    make_sure_path_exists(path)
    summary.to_csv(os.path.join(path, 'estimates_{}_{}_ins.csv'.format(subject, model_name)))

    model.model[0].name = model_name
    model_trace = model.trace[0]
    path = os.path.join('results', 'traces', 'in_sample', model_name)
    make_sure_path_exists(path)
    pm.trace_to_dataframe(model_trace).to_csv(os.path.join(path, 'trace_{}_{}_ins.csv'.format(subject, model_name)))
    pm.traceplot(model_trace)
    path = os.path.join(path, 'plots')
    make_sure_path_exists(path)
    plt.savefig(os.path.join(path, 'traceplot_{}_{}_ins.png'.format(subject, model_name)))
    plt.close()

    return model, converged


def fitCompare(data, subject, n_tries=1, overwrite=False, progressbar=True):
    """
    Perform fitting of GLAM variants and
    WAIC model comparisons for a single subject
    1) Multiplicative vs Additive
    3) Multiplicative vs No Bias
    4) Multiplicative vs Additive vs No Bias
    """

    print("Processing subject {}...".format(subject))

    # Subset data
    subject_data = data[data['subject'] == subject].copy()
    n_items = subject_data['n_items'].values[0]
    if n_items == 2:
        subject_data = subject_data.drop(['item_value_2', 'gaze_2'], axis=1)
    subject_data['subject'] = 0

    # model specifiations
    model_names = ('GLAM',
                   'additive',
                   'nobias')
    drifts = ('multiplicative',
              'additive',
              'multiplicative')
    parameter_sets = (['v', 's', 'tau', 'gamma'],
                      ['v', 's', 'tau', 'gamma'],
                      ['v', 's', 'tau'])
    gamma_bounds = ((-10, 1),
                    (-100, 100),
                    (-10, 1))
    gamma_vals = (None, None, 1.0)

    # fit models
    converged_models = np.ones(len(model_names))
    models = len(model_names) * [None]
    for i, (model_name, drift, parameters, gamma_bound, gamma_val) in enumerate(zip(model_names,
                                                                                    drifts,
                                                                                    parameter_sets,
                                                                                    gamma_bounds,
                                                                                    gamma_vals)):
        print('\tS{}: {}'.format(subject, model_name))
        model, is_converged = fit_indModel(subject_data, subject,
                                           drift=drift,
                                           parameters=parameters,
                                           gamma_bounds=gamma_bound, gamma_val=gamma_val,
                                           t0_val=0,
                                           model_name=model_name)
        models[i] = model
        converged_models[i] = np.int(is_converged)
        if not is_converged:
            break

    # re-sample all converged models, if any model did not converge
    if np.any(converged_models == 0):
        for i in np.where(converged_models == 1)[0]:
            print('\tRe-sampling S{}: {}'.format(subject, model_name))
            model, is_converged = fit_indModel(subject_data, subject,
                                               drift=drifts[i],
                                               parameters=parameter_sets[i],
                                               gamma_bounds=gamma_bounds[i], gamma_val=gamma_vals[i],
                                               t0_val=0,
                                               model_name=model_names[i],
                                               n_tries_max=0)
            models[i] = model

    # un-pack models
    if np.any(models == None):
        raise ValueError('Model {} not sampled.'.format(model_names[models==None]))
    multiplicative, additive, nobias = models

    # Multiplicative
    # print('\tS{}: Multiplicative'.format(subject))

    # parameters = ['v', 's', 'tau', 'gamma']

    # multiplicative = glam.GLAM(subject_data, drift='multiplicative')
    # multiplicative.make_model('individual', gamma_bounds=(-10, 1), t0_val=0)

    # multiplicative = fitModel(multiplicative, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar, use_fallback=use_fallback)

    # summary = pm.summary(multiplicative.trace[0])
    # for parameter in parameters:
    #         summary.loc[parameter + '__0_0', 'MAP'] = multiplicative.estimates[parameter].values[0]
    # summary.to_csv(os.path.join('results', 'estimates', 'in_sample', 'multiplicative', 'estimates_{}_multiplicative_ins.csv'.format(subject)))

    # multiplicative_model = multiplicative.model[0]
    # multiplicative_model.name = 'multiplicative'
    # multiplicative_trace = multiplicative.trace[0]
    # pm.trace_to_dataframe(multiplicative_trace).to_csv(os.path.join('results', 'traces', 'in_sample', 'multiplicative', 'trace_{}_multiplicative_ins.csv'.format(subject)))
    # pm.traceplot(multiplicative_trace)
    # plt.savefig(os.path.join('results', 'traces', 'in_sample', 'multiplicative', 'plots', 'traceplot_{}_multiplicative_ins.png'.format(subject)))
    # plt.close()

    # Additive
    # print('\tS{}: Additive'.format(subject))

    # parameters = ['v', 's', 'tau', 'gamma']

    # additive = glam.GLAM(subject_data, drift='additive')
    # additive.make_model('individual', gamma_bounds=(-10, 20), t0_val=0)

    # additive, use_fallback = fitModel(additive, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar, use_fallback=use_fallback)
    # summary = pm.summary(additive.trace[0])
    # for parameter in parameters:
    #         summary.loc[parameter + '__0_0', 'MAP'] = additive.estimates[parameter].values[0]
    # summary.to_csv(os.path.join('results', 'estimates', 'in_sample', 'additive', 'estimates_{}_additive_ins.csv'.format(subject)))

    # additive_model = additive.model[0]
    # additive_model.name = 'additive'
    # additive_trace = additive.trace[0]
    # pm.trace_to_dataframe(additive_trace).to_csv(os.path.join('results', 'traces', 'in_sample', 'additive', 'trace_{}_additive_ins.csv'.format(subject)))
    # pm.traceplot(additive_trace)
    # plt.savefig(os.path.join('results', 'traces', 'in_sample', 'additive', 'plots', 'traceplot_{}_additive_ins.png'.format(subject)))
    # plt.close()

    # No Leak
    # print('\tS{}: No Leak'.format(subject))

    # parameters = ['v', 's', 'tau', 'gamma']

    # noleak = glam.GLAM(subject_data, drift='multiplicative')
    # noleak.make_model('individual', gamma_bounds=(0, 1), t0_val=0)

    # noleak, use_fallback = fitModel(noleak, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar, use_fallback=use_fallback)
    # summary = pm.summary(noleak.trace[0])
    # for parameter in parameters:
    #         summary.loc[parameter + '__0_0', 'MAP'] = noleak.estimates[parameter].values[0]
    # summary.to_csv(os.path.join('results', 'estimates', 'in_sample', 'noleak', 'estimates_{}_noleak_ins.csv'.format(subject)))

    # noleak_model = noleak.model[0]
    # noleak_model.name = 'noleak'
    # noleak_trace = noleak.trace[0]
    # pm.trace_to_dataframe(noleak_trace).to_csv(os.path.join('results', 'traces', 'in_sample', 'noleak', 'trace_{}_noleak_ins.csv'.format(subject)))
    # pm.traceplot(noleak_trace)
    # plt.savefig(os.path.join('results', 'traces', 'in_sample', 'noleak', 'plots', 'traceplot_{}_noleak_ins.png'.format(subject)))
    # plt.close()

    # No-Bias
    # print('\tS{}: No Bias'.format(subject))

    # parameters = ['v', 's', 'tau']

    # nobias = glam.GLAM(subject_data, drift='nobias')
    # nobias.make_model('individual', gamma_val=1.0, t0_val=0)

    # nobias, use_fallback = fitModel(nobias, relevant_parameters=parameters, n_tries_max=n_tries, progressbar=progressbar, use_fallback=use_fallback)
    # summary = pm.summary(nobias.trace[0])
    # for parameter in parameters:
    #         summary.loc[parameter + '__0_0', 'MAP'] = nobias.estimates[parameter].values[0]
    # summary.to_csv(os.path.join('results', 'estimates', 'in_sample', 'nobias', 'estimates_{}_nobias_ins.csv'.format(subject)))

    # nobias_model = nobias.model[0]
    # nobias_model.name = 'nobias'
    # nobias_trace = nobias.trace[0]
    # pm.trace_to_dataframe(nobias_trace).to_csv(os.path.join('results', 'traces', 'in_sample', 'nobias', 'trace_{}_nobias_ins.csv'.format(subject)))
    # pm.traceplot(nobias_trace)
    # plt.savefig(os.path.join('results', 'traces', 'in_sample', 'nobias', 'plots', 'traceplot_{}_nobias_ins.png'.format(subject)))
    # plt.close()

    # Individual Model Comparisons
    # 1) Multiplicative vs Additive
    try:
        waic_df = pm.compare({additive.model[0]: additive.trace[0],
                              multiplicative.model[0]: multiplicative.trace[0]},
                             ic='WAIC')
        path = os.path.join('results', 'model_comparison', 'additive_vs_multiplicative')
        make_sure_path_exists(path)
        make_sure_path_exists(path+'/plots/')
        waic_df.to_csv(os.path.join(path, 'additive_vs_multiplicative_{}_waic.csv'.format(subject)))
        pm.compareplot(waic_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative', 'plots', 'additive_vs_multiplicative_{}_waic.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in WAIC comparison for subject {}'.format(subject))

    # 2) Multiplicative vs No Bias
    try:
        waic_df = pm.compare({multiplicative.model[0]: multiplicative.trace[0],
                              nobias.model[0]: nobias.trace[0]},
                             ic='WAIC')
        path = os.path.join('results', 'model_comparison', 'multiplicative_vs_nobias')
        make_sure_path_exists(path)
        make_sure_path_exists(path+'/plots/')
        waic_df.to_csv(os.path.join(path, 'multiplicative_vs_nobias_{}_waic.csv'.format(subject)))
        pm.compareplot(waic_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'multiplicative_vs_nobias', 'plots', 'multiplicative_vs_nobias_{}_waic.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in WAIC comparison for subject {}'.format(subject))

    # 3) Multiplicative vs Additive vs No Bias
    try:
        waic_df = pm.compare({multiplicative.model[0]: multiplicative.trace[0],
                              additive.model[0]: additive.trace[0],
                              nobias.model[0]: nobias.trace[0]},
                             ic='WAIC')
        path = os.path.join('results', 'model_comparison', 'additive_vs_multiplicative_vs_nobias')
        make_sure_path_exists(path)
        make_sure_path_exists(path+'/plots/')
        waic_df.to_csv(os.path.join(path, 'additive_vs_multiplicative_vs_nobias_{}_waic.csv'.format(subject)))
        pm.compareplot(waic_df)
        plt.savefig(os.path.join('results', 'model_comparison', 'additive_vs_multiplicative_vs_nobias', 'plots', 'additive_vs_multiplicative_vs_nobias_{}_waic.png'.format(subject)))
        plt.close()
    except:
        print('  /!\  Error in WAIC comparison for subject {}'.format(subject))  

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
                fitCompare(data, subject, n_tries=n_tries, overwrite=overwrite, progressbar=progressbar)
        else:
            fitCompare(data, subject, n_tries=n_tries, overwrite=overwrite, progressbar=progressbar)


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
    args = parser.parse_args()

    fitSubjects(first=args.first, last=args.last,
                overwrite=args.overwrite, progressbar=(not args.silent),
                n_tries=args.n_tries)
