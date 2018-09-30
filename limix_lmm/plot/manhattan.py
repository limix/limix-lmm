import pylab as pl
import scipy as sp

def plot_manhattan(plt, df, pv_thr=None, colors=None, offset=None, callback=None):
    """
    Utility function to make manhattan plot

    Parameters
    ----------
    plt : pyplot plot
        subplot
    df : pandas.DataFrame
        pandas DataFrame with chrom, pos and pv
    colors : list
        colors to use in the manhattan plot
    offset : float
        offset between in chromosome expressed as fraction of the
        length of the longest chromosome (default is 0.2)
    callback : function
        callback function that takes as input df

    Examples
    --------

    .. doctest::

        >>> from limix_lmm.plot import plot_manhattan
        >>> import scipy as sp
        >>> import pandas as pd
        >>> import pylab as plt
        >>>
        >>> n_chroms = 5
        >>> n_snps_per_chrom = 10000
        >>> chrom = sp.kron(sp.arange(1, n_chroms + 1), sp.ones(n_snps_per_chrom))
        >>> pos = sp.kron(sp.ones(n_chroms), sp.arange(n_snps_per_chrom))
        >>> pv = sp.rand(n_chroms * n_snps_per_chrom)
        >>> df = pd.DataFrame({'chrom': chrom, 'pos': pos, 'pv': pv})
        >>>
        >>> ax = plt.subplot(111)
        >>> plot_manhattan(ax, df)
    """
    if colors is None:
        colors = ['k', 'Gray']
    if offset is None:
        offset = 0.2
    dx = offset * df['pos'].values.max()
    _x = 0
    xticks = []
    for chrom_i in sp.unique(chrom):
        _df = df[df['chrom']==df['chrom_i']]
        if chrom_i%2==0:    color = colors[0]
        else:               color = colors[1]
        pl.plot(_df['pos'] + _x, -sp.log10(_df['pv']), '.', color=color)
        if callback is not None: callback(_df)
        xticks.append(_x + 0.5 * df['_pos'].values.max())
        _x += df['_pos'].values.max() + dx
    plt.set_xticks(xticks)
    plt.set_xticklabels(sp.unique(chrom))
