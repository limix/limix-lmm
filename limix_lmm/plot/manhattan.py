import pylab as pl
import scipy as sp

def plot_manhattan(plt, pv, chrom, pos, colors=None, offset=None):
    """
    Utility function to make manhattan plot

    Parameters
    ----------
    plt : pyplot plot
        subplot
    pv : nd array
        vector of P values
    chrom : nd array
        vector of chromosome
    pos : nd array
        vector of chromosomal positions
    colors : list
        colors to use in the manhattan plot
    offset : float
        offset between in chromosome expressed as fraction of the
        length of the longest chromosome (default is 0.2)
    """
    if colors is None:
        colors = ['k', 'Gray']
    if offset is None:
        offset = 0.2
    dx = offset * pos.max()
    _x = 0
    xticks = []
    for chrom_i in sp.unique(chrom):
        I = chrom==chrom_i
        _pos = pos[I]
        if chrom_i%2==0:    color = colors[0]
        else:               color = colors[1]
        pl.plot(_pos + _x, -sp.log10(pv[I]), '.', color=color)
        xticks.append(_x + 0.5 * _pos.max())
        _x += _pos.max() + dx
    plt.set_xticks(xticks)
    plt.set_xticklabels(sp.unique(chrom))

if __name__=="__main__":

    n_chroms = 5
    n_snps_per_chrom = 10000
    chrom = sp.kron(sp.arange(1, n_chroms + 1), sp.ones(n_snps_per_chrom))
    pos = sp.kron(sp.ones(n_chroms), sp.arange(n_snps_per_chrom))
    pv = sp.rand(n_chroms * n_snps_per_chrom)

    import pdb
    pdb.set_trace()
    plot_manhattan(pv, chrom, pos)
