from statsmodels.stats.outliers_influence import summary_table
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
plt.rc("axes.spines", top=False, right=False)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def show_img(path):
    img = pltimg.imread(path)
    f, a = plt.subplots()
    a.imshow(img)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['left'].set_visible(False)
    a.set_xticks([])
    a.set_yticks([])
    f.tight_layout()
    plt.show()


def plot_correlation(x, y,
                     xlabel='',
                     ylabel='',
                     title='',
                     ci=0.95,
                     alpha=0.5,
                     size=30,
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
                     fontsize_title=7,
                     fontsize_axeslabel=7,
                     fontsize_ticklabels=7,
                     fontsize_annotation=7,
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

    # Scatter (translucent dots with solid outlines)
    ax.scatter(x, y,
               marker='o',
               color='none',
               edgecolor=markercolor,
               linewidth=0.5,
               s=size)
    ax.scatter(x, y,
               marker='o',
               color=markercolor,
               alpha=alpha,
               linewidth=0,
               s=size)

    if regression:
        # LM fit
        X = sm.add_constant(x)
        lm = sm.OLS(y, X).fit()
        intercept, slope = lm.params
        table, data, columns = summary_table(lm, alpha=1. - ci)
        predicted, mean_ci_lower, mean_ci_upper = data[:, np.array([
                                                                   2, 4, 5])].T

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

    if return_correlation:
        return ax, line, annotation
    else:
        return ax


def add_regression_line(ax, intercept, slope, color='darkgray', **kwargs):

    xs = np.linspace(*ax.get_xlim(), 100)

    ax.plot(xs, intercept + slope * xs, color=color, **kwargs)

    return ax