import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("axes.spines", top=False, right=False)


def add_difficulty(df, n_bins=10):

    # infer number of items
    value_cols = ([col for col in df.columns
                   if col.startswith('item_value_')])

    values = df[value_cols].values
    values_sorted = np.sort(values, axis=1)
    difficulty = values_sorted[:, -1] - np.mean(values_sorted[:, :-1], axis=1)

    n_bins = np.min([np.unique(difficulty).size, n_bins])
    bins = np.linspace(np.min(difficulty), np.max(difficulty), n_bins)
    bins = np.round(bins, 2)
    difficulty_binned = pd.cut(difficulty, bins)
    df['difficulty'] = bins[difficulty_binned.codes]

    return df.copy()


def plot_rt_by_difficulty(data,
                          predictions=None,
                          ax=None,
                          xlims=(1.5, 8.5),
                          xlabel_skip=2,
                          fontsize=7,
                          prediction_labels=None,
                          prediction_colors=None,
                          prediction_markers=None,
                          prediction_ls=None,
                          prediction_alphas=None,
                          prediction_lws=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    if prediction_lws is None:
        predlws = [1 for i in range(len(dataframe)-1)]

    if prediction_colors is None:
        predcolors = ['C{}'.format(i) for i in range(len(dataframe)-1)]

    if prediction_ls is None:
        predls = ['-' for i in range(len(dataframe)-1)]

    if prediction_markers is None:
        predmarkers = ['o' for i in range(len(dataframe)-1)]

    if prediction_alphas is None:
        predalphas = [0.75 for i in range(len(dataframe)-1)]

    add_labels = False
    if (prediction_labels is not None):
        if len(prediction_labels) == len(predictions):
            add_labels = True
        else:
            raise ValueError(
                'Number of prediction labels does not match number of prediction datasets.')

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_difficulty(df)

        # Compute summary statistics
        subject_means = df.groupby(['subject', 'difficulty']).rt.mean()
        means = subject_means.groupby(
            'difficulty').mean()  # [xlims[0]:xlims[1]]
        sems = subject_means.groupby('difficulty').sem()  # [xlims[0]:xlims[1]]

        # x = np.arange(len(means))
        x = means.index.values

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            ax.bar(x, means,
                   linewidth=1, edgecolor='k', facecolor='w',
                   width=0.5)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color='k')

        else:  # plot predictions
            if add_labels:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1], color=prediction_colors[i-1],
                        ls=prediction_ls[i-1], label=prediction_labels[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1])
            else:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1],
                        color=prediction_colors[i-1], ls=prediction_ls[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1])

    ylim = np.mean(np.concatenate([a['rt'].ravel()
                                   for a in [data]+predictions]))
    ax.set_ylim(0, ylim*2)
    ax.set_xlabel('Max. value –\nmean value others', fontsize=fontsize)
    ax.set_ylabel('Response time (ms)', fontsize=fontsize)
    if xlims is not None:
        ax.set_xlim(xlims)
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(means.index.values[::xlabel_skip])
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize, frameon=False)


def add_value_minus_mean_others(df, n_bins=10):

    # infer number of items
    value_cols = ([col for col in df.columns
                   if col.startswith('item_value_')])
    n_items = len(value_cols)

    values = df[value_cols].values
    values_minus_mean_others = np.zeros_like(values)

    for t in np.arange(values.shape[0]):
        for i in np.arange(n_items):
            values_minus_mean_others[t, i] = values[t, i] - \
                np.mean(values[t, np.arange(n_items) != i])

    n_bins = np.min([np.unique(values_minus_mean_others.ravel()).size, n_bins])
    bins = np.linspace(np.min(values_minus_mean_others.ravel()), np.max(
        values_minus_mean_others.ravel()), n_bins)
    bins = np.round(bins, 2)
    values_minus_mean_others_binned = pd.cut(
        values_minus_mean_others.ravel(), bins)
    values_minus_mean_others_binned = bins[values_minus_mean_others_binned.codes]
    values_minus_mean_others_binned = values_minus_mean_others_binned.reshape(
        values_minus_mean_others.shape)

    for i in np.arange(n_items):
        df['value_minus_mean_others_{}'.format(
            i)] = values_minus_mean_others_binned[:, i]

    return df.copy()


def plot_pchoose_by_value_minus_mean_others(data,
                                            predictions=None,
                                            ax=None,
                                            xlims=[-5, 5],
                                            xlabel_skip=2,
                                            xlabel_start=1,
                                            fontsize=7,
                                            prediction_labels=None,
                                            prediction_colors=None,
                                            prediction_lws=None,
                                            prediction_ls=None,
                                            prediction_alphas=None,
                                            prediction_markers=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    n_items = len([col for col in data.columns
                   if col.startswith('item_value_')])

    if prediction_lws is None:
        predlws = [1 for i in range(len(dataframe)-1)]

    if prediction_colors is None:
        predcolors = ['C{}'.format(i) for i in range(len(dataframe)-1)]

    if prediction_ls is None:
        predls = ['-' for i in range(len(dataframe)-1)]

    if prediction_markers is None:
        predmarkers = ['o' for i in range(len(dataframe)-1)]

    if prediction_alphas is None:
        predalphas = [0.75 for i in range(len(dataframe)-1)]

    add_labels = False
    if (prediction_labels is not None):
        if len(prediction_labels) == len(predictions):
            add_labels = True
        else:
            raise ValueError(
                'Number of prediction labels does not match number of prediction datasets.')

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_value_minus_mean_others(df)

        # create temporary dataframe
        subjects = df['subject'].values
        value_minus_mean_others = df[['value_minus_mean_others_{}'.format(ii)
                                      for ii in range(n_items)]].values
        is_choice = np.zeros_like(value_minus_mean_others)
        is_choice[np.arange(is_choice.shape[0]),
                  df['choice'].values.astype(np.int)] = 1

        df_tmp = pd.DataFrame({'subject': np.repeat(subjects, n_items),
                               'value_minus_mean_others': value_minus_mean_others.ravel(),
                               'is_choice': is_choice.ravel()})

        # Compute summary statistics
        subject_means = df_tmp.groupby(
            ['subject', 'value_minus_mean_others']).is_choice.mean()
        means = subject_means.groupby('value_minus_mean_others').mean()
        sems = subject_means.groupby('value_minus_mean_others').sem()

        # x = np.arange(len(means))
        x = means.index

        # subset
        means = means[x <= np.max(xlims)]
        sems = sems[x <= np.max(xlims)]
        x = x[x <= np.max(xlims)]

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            ax.bar(x, means,
                   linewidth=1, edgecolor='k', facecolor='w',
                   width=1.)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color='k')

        else:  # plot predictions
            if add_labels:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1],
                        alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])
            else:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1],
                        alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])

    ax.axhline(1 / n_items, linestyle='--', color='k', linewidth=1, alpha=0.75)

    ax.set_xlabel('Item value –\nmean value others', fontsize=fontsize)
    ax.set_ylabel('P(choose item)', fontsize=fontsize)
    ax.set_ylim(-0.05, 1.05)
    if xlims is not None:
        ax.set_xlim(xlims)
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize, frameon=False)


def add_gaze_advantage(df, n_bins=8):

    # infer number of items
    gaze_cols = ([col for col in df.columns
                  if col.startswith('gaze_')])
    n_items = len(gaze_cols)

    gaze = df[gaze_cols].values
    gaze_advantage = np.zeros_like(gaze)
    for t in np.arange(gaze.shape[0]):
        for i in range(n_items):
            gaze_advantage[t, i] = gaze[t, i] - \
                np.mean(gaze[t, np.arange(n_items) != i])

    gaze_bins = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    for i in range(n_items):
        df['gaze_advantage_{}'.format(i)] = gaze_advantage[:, i]
        df['gaze_advantage_binned_{}'.format(i)] = pd.cut(df['gaze_advantage_{}'.format(i)],
                                                          bins=gaze_bins,
                                                          include_lowest=True,
                                                          labels=gaze_bins[:-1])

    return df.copy()


def compute_corrected_choice(df):

    # recode choice
    n_items = len([c for c in df.columns if c.startswith(
        'gaze_') and not ('advantage' in c)])
    is_choice = np.zeros((df.shape[0], n_items))
    is_choice[np.arange(is_choice.shape[0]),
              df['choice'].values.astype(np.int)] = 1

    if n_items > 2:
        values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
        value_range_others = np.zeros_like(is_choice)
        for t in range(value_range_others.shape[0]):
            for i in range(n_items):
                value_range_others[t, i] = values[t, np.arange(
                    n_items) != i].max() - values[t, np.arange(n_items) != i].min()

    # relative value
    df = add_value_minus_mean_others(df)
    relative_values = df[['value_minus_mean_others_{}'.format(i)
                          for i in range(n_items)]].values

    df_tmp = pd.DataFrame({"subject": np.repeat(df['subject'].values, n_items),
                           "relative_value": relative_values.ravel(),
                           "is_choice": is_choice.ravel()})
    if n_items > 2:
        df_tmp['value_range_others'] = value_range_others.ravel()

    # place in dataframe
    data_out = []
    for s, subject in enumerate(df['subject'].unique()):
        subject_data_tmp = df_tmp[df_tmp['subject'] == subject].copy()
        if n_items > 2:
            X = subject_data_tmp[['relative_value', 'value_range_others']]
            X = sm.add_constant(X)
            y = subject_data_tmp['is_choice']
        else:
            X = subject_data_tmp[['relative_value']]
            # exclude every second entry, bc 2-item case is symmetrical
            X = sm.add_constant(X)[::2]
            y = subject_data_tmp['is_choice'].values[::2]

        logit = sm.Logit(y, X)
        result = logit.fit(disp=0)
        predicted_pchoice = result.predict(X)

        subject_data_tmp['corrected_choice'] = (
            subject_data_tmp['is_choice'] - predicted_pchoice)
        data_out.append(subject_data_tmp)

    data_out = pd.concat(data_out)

    return data_out.copy()


def plot_corp_by_gaze_advantage(data, 
                                predictions=None,
                                ax=None,
                                n_bins=8,
                                xlabel_skip=2,
                                fontsize=7,
                                xlims=None,
                                prediction_labels=None,
                                prediction_colors=None,
                                prediction_lws=None,
                                prediction_ls=None,
                                prediction_alphas=None,
                                prediction_markers=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    if prediction_lws is None:
        predlws = [1 for i in range(len(dataframe)-1)]

    if prediction_colors is None:
        predcolors = ['C{}'.format(i) for i in range(len(dataframe)-1)]

    if prediction_ls is None:
        predls = ['-' for i in range(len(dataframe)-1)]

    if prediction_markers is None:
        predmarkers = ['o' for i in range(len(dataframe)-1)]

    if prediction_alphas is None:
        predalphas = [0.75 for i in range(len(dataframe)-1)]

    add_labels = False
    if (prediction_labels is not None):
        if len(prediction_labels) == len(predictions):
            add_labels = True
        else:
            raise ValueError(
                'Number of prediction labels does not match number of prediction datasets.')

    n_items = len([c for c in data.columns if c.startswith('gaze_')])
    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        # gaze advantage
        df = add_gaze_advantage(df, n_bins=n_bins)
        gaze_advantages = df[['gaze_advantage_binned_{}'.format(
            i) for i in range(n_items)]].values
        # corrected choice
        corrected_choice_data = compute_corrected_choice(df)
        corrected_choice_data['gaze_advantage_binned'] = gaze_advantages.ravel(
        )

        # Compute summary statistics
        subject_means = corrected_choice_data.groupby(
            ['subject', 'gaze_advantage_binned']).corrected_choice.mean()
        means = subject_means.groupby('gaze_advantage_binned').mean()
        sems = subject_means.groupby('gaze_advantage_binned').sem()
        x = means.index.values

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            ax.bar(x, means,
                   linewidth=1, edgecolor='k', facecolor='w',
                   width=0.125)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color='k')

        else:  # plot predictions
            if add_labels:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1],
                        alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])
            else:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1],
                        alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])

    ax.set_xlabel('Item gaze –\nmean gaze others', fontsize=fontsize)
    ax.set_ylabel('Corrected\nP(choose item)', fontsize=fontsize)
    ax.set_xticks([-1, -.5, 0, .5, 1.])
    ax.set_xticklabels([-1, -.5, 0, .5, 1.], fontsize=fontsize)
    ax.set_ylim(-1.05, 1.05)
    if xlims is not None:
        ax.set_xlim(xlims)
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize, frameon=False)
