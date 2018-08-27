#!/usr/bin/python

import pymc3 as pm
import numpy as np


def fit_models(models, method='NUTS', verbose=True, n_samples=2000, n_advi=200000, **kwargs):
    if isinstance(models, pm.model.Model):
        models = [models]
    elif isinstance(models, list) and np.alltrue(np.array([isinstance(model, pm.model.Model)
                                                           for model in models])):
        pass
    else:
        raise ValueError("Models must be list of <pymc3.model.Model> instances.")
    if verbose:
        print('Fitting {} model(s) using {}...'.format(len(models), method))
    traces = []

    for m, model in enumerate(models):
        if verbose:
            print('  Fitting model {} of {}...'.format(m+1, len(models)))
        with model:
            if method == 'NUTS':
                trace = pm.sample(draws=n_samples, cores=1, chains=2, **kwargs)
            elif method == 'Metropolis':
                trace = pm.sample(draws=n_samples, cores=1, chains=2, step=pm.Metropolis(), **kwargs)
            elif method == 'ADVI':
                vi_est = pm.fit(n=n_advi, **kwargs)
                trace = vi_est.sample(n_samples)
            traces.append(trace)

    return traces
