def qqplot(
    ax,
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
    ax : pyplot plot
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
        >>> from matplotlib import pyplot as plt
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
    from matplotlib import pyplot as plt
    import scipy as sp

    pv1 = pv[pv < pv_thr]
    pvo = -sp.log10(sp.sort(pv1))
    pvt = -sp.log10(sp.linspace(0, pv_thr, pv1.shape[0] + 2)[1:-1])
    if plot_method is not None:
        plot_method(pvt, pvo)
    else:
        ax.plot(pvt, pvo, marker=".", color=color)

    if plot_xyline:
        xlim1, xlim2 = ax.get_xlim()
        ylim1, ylim2 = ax.get_ylim()
        xlim1 = -sp.log10(pv_thr)
        ylim1 = -sp.log10(pv_thr)
        ax.plot([xlim1, xlim2], [xlim1, xlim2], xyline_color)
        ax.set_xlim(xlim1, xlim2)
        ax.set_ylim(ylim1, ylim2)
    if xy_labels:
        ax.set_xlabel("Expected P values")
        ax.set_ylabel("Observed P values")
