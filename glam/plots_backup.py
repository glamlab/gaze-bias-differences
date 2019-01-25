import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from seaborn import despine
plt.rc("axes.spines", top=False, right=False)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table


def plot_fit(data, predictions, prediction_labels=None):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    plot_rt_by_difficulty(data, predictions,
                          xlims=(1.5, 6.5), xlabel_skip=2,
                          ax=axs[0],
                          prediction_labels=prediction_labels)
    plot_pleft_by_left_minus_mean_others(data, predictions,
                                         xlabel_skip=4, xlims=[-6, 6.5], xlabel_start=0,
                                         ax=axs[1])
    plot_pleft_by_left_gaze_advantage(data, predictions,
                                      ax=axs[2])
    plot_corpleft_by_left_gaze_advantage(data, predictions,
                                         ax=axs[3])

    # Labels
    for label, ax in zip(list('ABCD'), axs.ravel()):
        ax.text(-0.15, 1.175, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    fig.tight_layout()

    return fig, axs


def add_difficulty(df, n_bins=10):
    """
    Compute trial difficulties and add to DataFrame.

    Maximum value - mean other values.
    In the binary case, this reduces to abs(v0 - v1).

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for item_value_i
    """

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


def plot_rt_by_difficulty(data, predictions=None, ax=None, xlims=(1.5, 8.5), xlabel_skip=2, prediction_labels=None, fontsize=8, prediction_colors=None, prediction_markers=None, prediction_ls=None, prediction_alphas=None, prediction_lws=None):
    """
    Plot SI1 Data with model predictions
    a) RT by difficulty

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
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
            raise ValueError('Number of prediction labels does not match number of prediction datasets.')

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_difficulty(df)

        # Compute summary statistics
        subject_means = df.groupby(['subject', 'difficulty']).rt.mean()
        means = subject_means.groupby('difficulty').mean()#[xlims[0]:xlims[1]]
        sems = subject_means.groupby('difficulty').sem()#[xlims[0]:xlims[1]]

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
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1], color=prediction_colors[i-1], ls=prediction_ls[i-1], label=prediction_labels[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1])
            else:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1], color=prediction_colors[i-1], ls=prediction_ls[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1])

    ylim = np.mean(np.concatenate([a['rt'].ravel() for a in [data]+predictions]))
    ax.set_ylim(0, ylim*2)
    ax.set_xlabel('Max. value –\nmean value others', fontsize=fontsize)
    ax.set_ylabel('Response time (ms)', fontsize=fontsize)
    if xlims is not None:
        ax.set_xlim(xlims)
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(means.index.values[::xlabel_skip])
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize, frameon=False)

    # despine()


# def add_left_minus_mean_others(df, n_bins=10):
#     """
#     Compute relative value of left item and add to DataFrame.

#     Left rating – mean other ratings
#     In the binary case, this reduces to v0 - v1.

#     Parameters
#     ----------
#     df :      <pandas DataFrame>
#               Trial wise DataFrame containing columns for item_value_i
#     """

#     # infer number of items
#     value_cols = ([col for col in df.columns
#                    if col.startswith('item_value_')])

#     values = df[value_cols].values
#     left_minus_mean_others = values[:, 0] - np.mean(values[:, 1:], axis=1)

#     n_bins = np.min([np.unique(left_minus_mean_others).size, n_bins])
#     bins = np.linspace(np.min(left_minus_mean_others), np.max(left_minus_mean_others), n_bins)
#     bins = np.round(bins, 2)
#     left_minus_mean_others_binned = pd.cut(left_minus_mean_others, bins)
#     df['left_minus_mean_others'] = bins[left_minus_mean_others_binned.codes]

#     return df.copy()


# def plot_pleft_by_left_minus_mean_others(data, predictions=None, ax=None, xlims=[-5, 5], xlabel_skip=2, xlabel_start=1, prediction_labels=None):
#     """
#     Plot SI1 Data with model predictions
#     b) P(left chosen) by left rating minus mean other rating

#     Parameters
#     ----------
#     data: <pandas DataFrame>

#     predictions: <pandas DataFrame> or <list of pandas DataFrames>

#     ax: <matplotlib.axes>

#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(4, 3))

#     if predictions is None:
#         dataframes = [data]
#     elif isinstance(predictions, list):
#         dataframes = [data] + predictions
#     else:
#         dataframes = [data] + [predictions]

#     n_items = len([col for col in data.columns
#                    if col.startswith('item_value_')])

#     add_labels = False
#     if (prediction_labels is not None):
#         if len(prediction_labels) == len(predictions):
#             add_labels = True
#         else:
#             raise ValueError('Number of prediction labels does not match number of prediction datasets.')

#     for i, dataframe in enumerate(dataframes):

#         df = dataframe.copy()

#         # Compute relevant variables
#         df = add_left_minus_mean_others(df)
#         df['left_chosen'] = df['choice'] == 0

#         # Compute summary statistics
#         subject_means = df.groupby(
#             ['subject', 'left_minus_mean_others']).left_chosen.mean()
#         means = subject_means.groupby('left_minus_mean_others').mean()#[
#             #xlims[0]:xlims[1]]
#         sems = subject_means.groupby('left_minus_mean_others').sem()#[
#             #xlims[0]:xlims[1]]

#         # x = np.arange(len(means))
#         x = means.index

#         predicted = False if i == 0 else True

#         if not predicted:  # plot underlying data
#             ax.plot(x, means, color='k', linewidth=1, ls='--')
#             ax.vlines(x, means - sems, means + sems,
#                       linewidth=1, color='k')

#         else:  # plot predictions
#             if add_labels:
#                 ax.plot(x, means, '--', markerfacecolor='none', label=prediction_labels[i-1])
#             else:
#                 ax.plot(x, means, '--', markerfacecolor='none')

#     ax.axhline(1 / n_items, linestyle='--', color='k', linewidth=1, alpha=0.75)

#     ax.set_xlabel('Left rating – mean other ratings')
#     ax.set_ylabel('P(left chosen)')
#     ax.set_ylim(-0.05, 1.05)
#     #ax.set_xticks(x[xlabel_start::xlabel_skip])
#     #ax.set_xticklabels(means.index.values[xlabel_start::xlabel_skip])
#     if add_labels:
#         ax.legend(loc='upper left', fontsize=12)

#     # despine()


def add_value_minus_mean_others(df, n_bins=10):
    """
    """

    # infer number of items
    value_cols = ([col for col in df.columns
                   if col.startswith('item_value_')])
    n_items = len(value_cols)

    values = df[value_cols].values
    values_minus_mean_others = np.zeros_like(values)

    for t in np.arange(values.shape[0]):
        for i in np.arange(n_items):
            values_minus_mean_others[t, i] = values[t, i] - np.mean(values[t, np.arange(n_items)!=i])

    n_bins = np.min([np.unique(values_minus_mean_others.ravel()).size, n_bins])
    bins = np.linspace(np.min(values_minus_mean_others.ravel()), np.max(values_minus_mean_others.ravel()), n_bins)
    bins = np.round(bins, 2)
    values_minus_mean_others_binned = pd.cut(values_minus_mean_others.ravel(), bins)
    values_minus_mean_others_binned = bins[values_minus_mean_others_binned.codes]
    values_minus_mean_others_binned = values_minus_mean_others_binned.reshape(values_minus_mean_others.shape)

    for i in np.arange(n_items):
        df['value_minus_mean_others_{}'.format(i)] = values_minus_mean_others_binned[:,i]

    return df.copy()


def plot_pchoose_by_value_minus_mean_others(data, predictions=None, ax=None, xlims=[-5, 5], xlabel_skip=2, xlabel_start=1, prediction_labels=None, fontsize=8, prediction_colors=None, prediction_lws=None, prediction_ls=None, prediction_alphas=None, prediction_markers=None):
    """

    """
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
            raise ValueError('Number of prediction labels does not match number of prediction datasets.')

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_value_minus_mean_others(df)

        # create temporary dataframe
        subjects = df['subject'].values
        value_minus_mean_others = df[['value_minus_mean_others_{}'.format(ii)
                                        for ii in range(n_items)]].values
        is_choice = np.zeros_like(value_minus_mean_others)
        is_choice[np.arange(is_choice.shape[0]), df['choice'].values.astype(np.int)] = 1

        df_tmp = pd.DataFrame({'subject': np.repeat(subjects, n_items),
                               'value_minus_mean_others': value_minus_mean_others.ravel(),
                               'is_choice': is_choice.ravel()})

        # Compute summary statistics
        subject_means = df_tmp.groupby(
            ['subject', 'value_minus_mean_others']).is_choice.mean()
        means = subject_means.groupby('value_minus_mean_others').mean()#[
            #xlims[0]:xlims[1]]
        sems = subject_means.groupby('value_minus_mean_others').sem()#[
            #xlims[0]:xlims[1]]

        # x = np.arange(len(means))
        x = means.index

        # subset
        means = means[x<=np.max(xlims)]
        sems = sems[x<=np.max(xlims)]
        x = x[x<=np.max(xlims)]

        predicted = False if i == 0 else True

        if not predicted:  # plot underlying data
            ax.bar(x, means,
                   linewidth=1, edgecolor='k', facecolor='w',
                   width=1.)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color='k')

        else:  # plot predictions
            if add_labels:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])
            else:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])

    ax.axhline(1 / n_items, linestyle='--', color='k', linewidth=1, alpha=0.75)

    ax.set_xlabel('Item value –\nmean value others', fontsize=fontsize)
    ax.set_ylabel('P(choose item)', fontsize=fontsize)
    ax.set_ylim(-0.05, 1.05)
    if xlims is not None:
        ax.set_xlim(xlims)
    #ax.set_xticks(x[xlabel_start::xlabel_skip])
    #ax.set_xticklabels(means.index.values[xlabel_start::xlabel_skip])
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize, frameon=False)


# def add_left_gaze_advantage(df):
#     """
#     Compute gaze advantage of left item and add to DataFrame.

#     Left relative gaze – mean other relative gaze
#     In the binary case, this reduces to g0 - g1.

#     Parameters
#     ----------
#     df :      <pandas DataFrame>
#               Trial wise DataFrame containing columns for gaze_i
#     """

#     # infer number of items
#     gaze_cols = ([col for col in df.columns
#                   if col.startswith('gaze_')])

#     gaze = df[gaze_cols].values
#     left_gaze_advantage = gaze[:, 0] - np.mean(gaze[:, 1:], axis=1)

#     df['left_gaze_advantage'] = left_gaze_advantage

#     return df.copy()


# def plot_pleft_by_left_gaze_advantage(data, predictions=None, ax=None, n_bins=8, xlabel_skip=2, prediction_labels=None):
#     """
#     Plot SI1 Data with model predictions
#     c) P(left chosen) by left gaze minus mean other gaze

#     x-axis label indicate left bound of interval.

#     Parameters
#     ----------
#     data: <pandas DataFrame>

#     predictions: <pandas DataFrame> or <list of pandas DataFrames>

#     ax: <matplotlib.axes>

#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(4, 3))

#     if predictions is None:
#         dataframes = [data]
#     elif isinstance(predictions, list):
#         dataframes = [data] + predictions
#     else:
#         dataframes = [data] + [predictions]

#     add_labels = False
#     if (prediction_labels is not None):
#         if len(prediction_labels) == len(predictions):
#             add_labels = True
#         else:
#             raise ValueError('Number of prediction labels does not match number of prediction datasets.')

#     for i, dataframe in enumerate(dataframes):

#         df = dataframe.copy()

#         # Compute relevant variables
#         df = add_left_gaze_advantage(df)
#         bins = np.linspace(-1, 1, n_bins+1)
#         df['left_gaze_advantage_bin'] = pd.cut(df['left_gaze_advantage'],
#                                                bins=bins, include_lowest=True,
#                                                labels=bins[:-1])
#         df['left_chosen'] = df['choice'] == 0

#         # Compute summary statistics
#         subject_means = df.groupby(
#             ['subject', 'left_gaze_advantage_bin']).left_chosen.mean()
#         means = subject_means.groupby('left_gaze_advantage_bin').mean()
#         sems = subject_means.groupby('left_gaze_advantage_bin').sem()

#         x = np.arange(len(means))

#         predicted = False if i == 0 else True

#         if not predicted:  # plot underlying data
#             ax.bar(x, means,
#                    linewidth=1, edgecolor='k', facecolor='w',
#                    width=0.5)
#             ax.vlines(x, means - sems, means + sems,
#                       linewidth=1, color='k')

#         else:  # plot predictions
#             if add_labels:
#                 ax.plot(x, means, '--o', markerfacecolor='none', label=prediction_labels[i-1])
#             else:
#                 ax.plot(x, means, '--o', markerfacecolor='none')

#     ax.set_xlabel('Left gaze – mean other gaze')
#     ax.set_ylabel('P(left chosen)')
#     ax.set_ylim(-0.05, 1.05)
#     ax.set_xticks(x[::xlabel_skip])
#     ax.set_xticklabels(means.index.values[::xlabel_skip])
#     if add_labels:
#         ax.legend(loc='upper left', fontsize=12)

#     # despine()


# def add_left_relative_value(df):
#     """
#     Compute relative value of left item.

#     Left item value – mean other item values
#     In the binary case, this reduces to v0 - v1.

#     Parameters
#     ----------
#     df :      <pandas DataFrame>
#               Trial wise DataFrame containing columns for gaze_i
#     """

#     # infer number of items
#     # relative value left
#     value_cols = ([col for col in df.columns
#                    if col.startswith('item_value_')])
#     values = df[value_cols].values
#     relative_value_left = values[:, 0] - np.mean(values[:, 1:])
#     df['left_relative_value'] = relative_value_left

#     return df.copy()


# def add_corrected_choice_left(df):
#     """
#     Compute corrected choice left

#     Corrected choice ~ (choice==left) - p(choice==left | left relative item value)

#     Parameters
#     ----------
#     df :      <pandas DataFrame>
#               Trial wise DataFrame containing columns for gaze_i
#     """

#     # recode choice
#     df['left_chosen'] = df['choice'].values == 0

#     # left relative value
#     df = add_left_relative_value(df)

#     # compute p(choice==left|left relative value)
#     subject_value_psychometric = df.groupby(
#         ['subject', 'left_relative_value']).left_chosen.mean()
#     # place in dataframe
#     for s, subject in enumerate(df['subject'].unique()):
#         subject_df = df[df['subject'] == subject].copy()
#         df.loc[df['subject'] == subject, 'p_choice_left_given_value'] = subject_value_psychometric[
#             subject][subject_df['left_relative_value'].values].values

#     # compute corrected choice left
#     df['corrected_choice_left'] = df['left_chosen'] - \
#         df['p_choice_left_given_value']

#     return df.copy()


# def plot_corpleft_by_left_gaze_advantage(data, predictions=None, ax=None, n_bins=8, xlabel_skip=2, prediction_labels=None,  prediction_colors=None, prediction_lws=None, prediction_ls=None, prediction_alphas=None, prediction_markers=None):
#     """
#     Plot SI1 Data with model predictions
#     c) Corrected P(choice==left) by left gaze minus mean other gaze
#     Corrected P(choice==left) ~ P(choice==left | left final gaze adv.) - P(choice==left | left relative value)

#     Parameters
#     ----------
#     data: <pandas DataFrame>

#     predictions: <pandas DataFrame> or <list of pandas DataFrames>

#     ax: <matplotlib.axes>

#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(4, 3))

#     if predictions is None:
#         dataframes = [data]
#     elif isinstance(predictions, list):
#         dataframes = [data] + predictions
#     else:
#         dataframes = [data] + [predictions]
        
#     if prediction_lws is None:
#         predlws = [1 for i in range(len(dataframe)-1)]

#     if prediction_colors is None:
#         predcolors = ['C{}'.format(i) for i in range(len(dataframe)-1)]
        
#     if prediction_ls is None:
#         predls = ['-' for i in range(len(dataframe)-1)]
        
#     if prediction_markers is None:
#         predmarkers = ['o' for i in range(len(dataframe)-1)]
        
#     if prediction_alphas is None:
#         predalphas = [0.75 for i in range(len(dataframe)-1)]

#     add_labels = False
#     if (prediction_labels is not None):
#         if len(prediction_labels) == len(predictions):
#             add_labels = True
#         else:
#             raise ValueError('Number of prediction labels does not match number of prediction datasets.')

#     for i, dataframe in enumerate(dataframes):

#         df = dataframe.copy()

#         # Compute relevant variables
#         # recode choice
#         df['left_chosen'] = df['choice'].values == 0
#         # left final gaze advantage
#         df = add_left_gaze_advantage(df)
#         gaze_bins = np.linspace(-1, 1, n_bins+1)
#         df['left_gaze_advantage_bin'] = pd.cut(df['left_gaze_advantage'],
#                                                bins=gaze_bins, include_lowest=True,
#                                                labels=gaze_bins[:-1])
#         df['left_chosen'] = df['choice'] == 0
#         # corrected choice
#         df = add_corrected_choice_left(df)

#         # Compute summary statistics
#         subject_means = df.groupby(
#             ['subject', 'left_gaze_advantage_bin']).corrected_choice_left.mean()
#         means = subject_means.groupby('left_gaze_advantage_bin').mean()
#         sems = subject_means.groupby('left_gaze_advantage_bin').sem()
#         x = np.arange(len(means))

#         predicted = False if i == 0 else True

#         if not predicted:  # plot underlying data
#             ax.bar(x, means,
#                    linewidth=1, edgecolor='k', facecolor='w',
#                    width=0.5)
#             ax.vlines(x, means - sems, means + sems,
#                       linewidth=1, color='k')

#         else:  # plot predictions
#             if add_labels:
#                 ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])
#             else:
#                 ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])

#     ax.set_xlabel('Left gaze – mean other gaze')
#     ax.set_ylabel('Corrected P(left chosen)')
#     ax.set_xticks(x[::xlabel_skip])
#     ax.set_xticklabels(means.index.values[::xlabel_skip])
#     ax.set_ylim(-1.05, 1.05)
#     if add_labels:
#         ax.legend(loc='upper left', fontsize=12)

#     # despine()


def add_gaze_advantage(df, n_bins=8):
    """
    Compute gaze advantage of left item and add to DataFrame.

    Left relative gaze – mean other relative gaze
    In the binary case, this reduces to g0 - g1.

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for gaze_i
    """

    # infer number of items
    gaze_cols = ([col for col in df.columns
                  if col.startswith('gaze_')])
    n_items = len(gaze_cols)

    gaze = df[gaze_cols].values
    gaze_advantage = np.zeros_like(gaze)
    for t in np.arange(gaze.shape[0]):
        for i in range(n_items):
            gaze_advantage[t, i] = gaze[t, i] - np.mean(gaze[t, np.arange(n_items)!=i])

    gaze_bins = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    for i in range(n_items):
        df['gaze_advantage_{}'.format(i)] = gaze_advantage[:,i]
        df['gaze_advantage_binned_{}'.format(i)] = pd.cut(df['gaze_advantage_{}'.format(i)],
                                                          bins=gaze_bins,
                                                          include_lowest=True,
                                                          labels=gaze_bins[:-1])

    return df.copy()


def compute_corrected_choice(df):
    """
    Compute corrected choice left

    Corrected choice ~ (choice==left) - p(choice==left | left relative item value)

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for gaze_i
    """

    # recode choice
    n_items = len([c for c in df.columns if c.startswith('gaze_') and not ('advantage' in c)])
    is_choice = np.zeros((df.shape[0], n_items))
    is_choice[np.arange(is_choice.shape[0]), df['choice'].values.astype(np.int)] = 1

    if n_items > 2:
        values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
        value_range_others = np.zeros_like(is_choice)
        for t in range(value_range_others.shape[0]):
            for i in range(n_items):
                value_range_others[t,i] = values[t,np.arange(n_items)!=i].max() - values[t,np.arange(n_items)!=i].min()
    
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
        # subject_df = df[df['subject'] == subject].copy()
        # df.loc[df['subject'] == subject, 'p_choice_given_value'] = subject_value_psychometric[
        #     subject][subject_df['relative_value'].values].values
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

        subject_data_tmp['corrected_choice'] = (subject_data_tmp['is_choice'] - predicted_pchoice)
        data_out.append(subject_data_tmp)

    data_out = pd.concat(data_out)

    return data_out.copy()


def plot_corp_by_gaze_advantage(data, predictions=None, ax=None, n_bins=8, xlabel_skip=2, prediction_labels=None, fontsize=8, xlims=None, prediction_colors=None, prediction_lws=None, prediction_ls=None, prediction_alphas=None, prediction_markers=None):
    """

    """
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
            raise ValueError('Number of prediction labels does not match number of prediction datasets.')

    n_items = len([c for c in data.columns if c.startswith('gaze_')])
    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        # gaze advantage
        df = add_gaze_advantage(df, n_bins=n_bins)
        gaze_advantages = df[['gaze_advantage_binned_{}'.format(i) for i in range(n_items)]].values
        # corrected choice
        corrected_choice_data = compute_corrected_choice(df)
        corrected_choice_data['gaze_advantage_binned'] = gaze_advantages.ravel()

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
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])
            else:
                ax.plot(x, means, marker=prediction_markers[i-1], markerfacecolor=prediction_colors[i-1], alpha=prediction_alphas[i-1], lw=prediction_lws[i-1], ls=prediction_ls[i-1], color=prediction_colors[i-1])

    ax.set_xlabel('Item gaze –\nmean gaze others', fontsize=fontsize)
    ax.set_ylabel('Corrected\nP(choose item)', fontsize=fontsize)
    ax.set_xticks([-1, -.5, 0, .5, 1.])
    ax.set_xticklabels([-1, -.5, 0, .5, 1.], fontsize=fontsize)
    ax.set_ylim(-1.05, 1.05)
    if xlims is not None:
        ax.set_xlim(xlims)
    if add_labels:
        ax.legend(loc='upper left', fontsize=fontsize, frameon=False)


def plot_correlation(x, y,
                     xlabel='',
                     ylabel='',
                     title='',
                     ci=0.95,
                     alpha=0.25,
                     color='red',
                     markercolor='black',
                     marker='o',
                     xticks=None,
                     yticks=None,
                     xticklabels=None,
                     yticklabels=None,
                     xlim=None,
                     ylim=None,
                     annotate=True,
                     annotation_pos=(0.1, 0.1),
                     annotation_halign='left',
                     fontsize_title=8,
                     fontsize_axeslabel=8,
                     fontsize_ticklabels=8,
                     fontsize_annotation=8,
                     regression=True,
                     plot_diagonal=False,
                     return_correlation=False,
                     ax=None):

    # Defaults
    if ax is None:
        fig, ax = plt.subplots()

    # Axes, ticks, ...
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=fontsize_ticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=fontsize_ticklabels)

    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticklabels)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Scatter
    ax.scatter(x, y, marker=marker, color=markercolor, alpha=alpha)

    if regression:
        # LM fit
        X = sm.add_constant(x)
        lm = sm.OLS(y, X).fit()
        intercept, slope = lm.params
        table, data, columns = summary_table(lm, alpha=1. - ci)
        predicted, mean_ci_lower, mean_ci_upper = data[:, np.array([2, 4, 5])].T

        xs = np.linspace(*ax.get_xlim(), 100)
        line = ax.plot(xs, intercept + slope * xs,
                       color=color)
        sort_idx = np.argsort(x)
        ax.fill_between(x[sort_idx], mean_ci_lower[sort_idx], mean_ci_upper[sort_idx],
                        color=color, alpha=0.1)

        # Annotation
        tval = lm.tvalues[-1]
        pval = lm.pvalues[-1]
        if pval < 0.0001:
            p_string = r'$P < 0.0001$'
        else:
            p_string = r'$P = {}$'.format(np.round(pval, 4))
        r = np.sign(tval) * np.sqrt(lm.rsquared)
        annotation = (r'$r = {:.2f}$, '.format(r)) + p_string
        if annotate:
            ax.text(*annotation_pos,
                    annotation,
                    verticalalignment='bottom',
                    horizontalalignment=annotation_halign,
                    transform=ax.transAxes,
                    fontsize=fontsize_annotation)

    # Diagonal
    if plot_diagonal:
        ax.plot([0, 1], [0, 1], transform=ax.transAxes,
                color='black', alpha=0.5, zorder=-10, lw=1)

    # Labels
    ax.set_xlabel(xlabel, fontsize=fontsize_axeslabel)
    ax.set_ylabel(ylabel, fontsize=fontsize_axeslabel)
    ax.set_title(title, fontsize=fontsize_title)
    # despine()

    if return_correlation:
        return ax, line, annotation
    else:
        return ax


def add_regression_line(ax, intercept, slope, color='darkgray', **kwargs):
    """
    Adds a regression line to an axis object.
    """

    xs = np.linspace(*ax.get_xlim(), 100)

    ax.plot(xs, intercept + slope * xs, color=color, **kwargs)

    return ax
