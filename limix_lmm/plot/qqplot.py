import pylab as pl
import scipy as sp
import pdb

def plot_manhattan(plt, pv, pos, chrom, Iflag=None, thr=2, gw_thr=5e-8, rsid=None):
    offset = 0
    colors = ['k', 'Gray']
    for chrom_i in range(1, 23):
        print('   .. chrom %d' % chrom_i)
        Ichrom = chrom==chrom_i
        x = pos[Ichrom] + offset
        y = -sp.log10(pv[Ichrom])
        Iplot = y>thr
        x = x[Iplot]; y = y[Iplot]
        pl.plot(x, y, '.', color=colors[chrom_i % 2])
        if rsid is not None:
            II = y > (-sp.log10(gw_thr))
            if II.sum()>1:
                idxs = sp.where(II)[0]
                for idx in idxs:
                    pl.text(x[idx], y[idx], rsid[idx])
        if Iflag is not None:
            iflag = Iflag[Ichrom][Iplot]
            pl.plot(x[iflag], y[iflag], '.', color='g')
        offset = x.max()
    xlim1, xlim2 = plt.get_xlim()
    _dx = 0.01 * (xlim2-xlim1)
    pl.plot(plt.get_xlim(), -sp.log10(gw_thr) * sp.ones(2), 'r')
    if Iflag is not None:
        thr = 1e-2 / float(Iflag.sum())
        pl.plot(plt.get_xlim(), -sp.log10(thr) * sp.ones(2), 'DarkGreen')
    pl.xlim(-_dx, x.max() + _dx)
    pl.ylabel('-log$_{10}$P')
    pl.xlabel('Genomic position')


def qqplot(plt, pv, thr=1e-2):
    pv1 = pv[pv<thr]
    pvo = -sp.log10(sp.sort(pv1))
    pvt = -sp.log10(sp.linspace(0, thr, pv1.shape[0]+2)[1:-1])
    xlim1 = -sp.log10(thr)
    pl.plot(pvt, pvo, '.k')
    pl.plot([xlim1, 1.05*pvo.max()], [xlim1, 1.05*pvo.max()], 'r')
    plt.set_xlim(xlim1, 1.05*pvo.max())
    pl.xlabel('Expected P values')
    pl.xlabel('Observed P values')
