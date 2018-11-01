import pylab as pl
import scipy as sp


def qqplot(
    plt,
    pv,
    color="k",
    plot_method=None,
    line=False,
    pv_thr=1e-2,
    plot_xyline=False,
    xy_labels=False,
    xyline_color="r",
):
    """
    Utility function to make manhattan plot

    Parameters
    ----------
    plt : pyplot plot
        subplot
    df : pandas.DataFrame
        pandas DataFrame with chrom, pos and pv
    color : color
        colors to use in the manhattan plot (default is black)
    plot_method : function
        function that takes x and y and plots a custom scatter plot
        The default value is None.
    pv_thr : float
        threshold on pvs
    plot_xyline : bool
        if True, the x=y line is plotteed. The default value is False.

    Examples
    --------

    .. doctest::

        >>> from limix_lmm.plot import qqplot
        >>> import scipy as sp
        >>> import pylab as plt
        >>>
        >>> pv1 = sp.rand(10000)
        >>> pv2 = sp.rand(10000)
        >>> pv3 = sp.rand(10000)
        >>>
        >>> ax = plt.subplot(111)
        >>> qqplot(ax, pv1, color='C0')
        >>> qqplot(ax, pv2, color='C1')
        >>> qqplot(ax, pv3, color='C2', plot_xyline=True, xy_labels=True)
    """
    pv1 = pv[pv < pv_thr]
    pvo = -sp.log10(sp.sort(pv1))
    pvt = -sp.log10(sp.linspace(0, pv_thr, pv1.shape[0] + 2)[1:-1])
    if plot_method is not None:
        plot_method(pvt, pvo)
    else:
        pl.plot(pvt, pvo, marker=".k", color=color)

    if plot_xyline:
        xlim1, xlim2 = plt.get_xlim()
        ylim1, ylim2 = plt.get_ylim()
        xlim1 = -sp.log10(pv_thr)
        ylim1 = -sp.log10(pv_thr)
        pl.plot([xlim1, xlim2], [xlim1, xlim2], xyline_color)
        plt.set_xlim(xlim1, xlim2)
        plt.set_ylim(ylim1, ylim2)
    if xy_labels:
        pl.xlabel("Expected P values")
        pl.ylabel("Observed P values")
