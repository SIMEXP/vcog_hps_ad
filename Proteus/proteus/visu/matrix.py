__author__ = 'Christian Dansereau'

from matplotlib.pylab import *
import numpy as np
import matplotlib as mpl

def calcul_zero(min,max):
    """
    calculate the position of the zero in the colorbar
    min: minimum value on the scale
    max: maximum value on the scale
    """
    return np.abs(min)/float(np.abs(min) + np.abs(max))

def mat(m, lim=None, cbar=True, show_axis=True,cm=None):
    """
    Plot a matrix with color bar
    m: the actual matrix
    lim: Tuple with the range of limits (min, max) of the color bar
    cbar: (boolean) specify if the color bar should apear in the plot
    show_axis: (boolean) render or not the axis values
    """
    #fig, ax = plt.subplots()
    if cm==None:
        shifted_cmap = get_cmap(m, lim)
    else:
        shifted_cmap = cm    

    if lim == None:
        #shrunk_cmap = shiftedColorMap(orig_cmap, start=0.15, midpoint=0.75, stop=0.85, name='shrunk')
        im2 = plt.imshow(m, interpolation="none", cmap=shifted_cmap)

        #im2 = ax.imshow(m, interpolation="none", cmap=shrunk_cmap)
    else:
        im2 = plt.imshow(m, interpolation="none", cmap=shifted_cmap, vmin=lim[0], vmax=lim[1])

    #plt.colorbar(im2)

    # delete the colorbar if specified
    if cbar:
        plt.colorbar(im2,fraction=0.046, pad=0.04)

        #fig.delaxes(fig.axes[1])
    if show_axis == False:
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

    #cax = ax.imshow(m, interpolation='none', cmap=shifted_cmap)
    #cbar = fig.colorbar(cax, orientation='vertical')
    #mplmatshow(m, cmap=cm.Spectral)
    #cb1 = fig.colorbar.ColorbarBase(ax1, cmap=cmap,norm=norm,orientation='horizontal')
    #show()

def color_bar_horizontal(ax, cmap, lim, nbins=None):
    if nbins!=None:
        bounds = np.linspace(lim[0], lim[1], nbins)
        norm = mpl.colors.Normalize(vmin=lim[0], vmax=lim[1])
        mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds ,orientation='horizontal')
    else:
        norm = mpl.colors.Normalize(vmin=lim[0], vmax=lim[1])
        mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, spacing='proportional' ,orientation='horizontal')

def color_bar_vertical(ax, cmap, lim, nbins=None):
    if nbins!=None:
        bounds = np.linspace(lim[0], lim[1], nbins)
        norm = mpl.colors.Normalize(vmin=lim[0], vmax=lim[1])
        mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds ,orientation='vertical')
    else:
        norm = mpl.colors.Normalize(vmin=lim[0], vmax=lim[1])
        mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, spacing='proportional' ,orientation='vertical')

def get_cmap(data, lim=None):
    # compute center for the color bar
    orig_cmap = cm.hot
    if data.min() >= 0 and (lim==None or lim[0]>=0):
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint=lim[1]-lim[0]/2., name='shifted')
        shifted_cmap = orig_cmap
    else:
        orig_cmap = cm.RdBu_r #cm.coolwarm
        if lim==None:
            shifted_cmap = shiftedColorMap(orig_cmap, midpoint=calcul_zero(data.min(),data.max()), name='shifted')
        else:
            shifted_cmap = shiftedColorMap(orig_cmap, midpoint=calcul_zero(*lim), name='shifted')

    if lim == None:
        pass
    else:
        pass
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint=calcul_zero(*lim), name='shifted')

    if len(np.unique(data))==2:
        shifted_cmap = cm.gray
    return shifted_cmap

def part(m):
    fig, ax = plt.subplots()
    orig_cmap = cm.jet
    #shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.75, name='shifted')
    im2 = ax.imshow(m, interpolation="none", cmap=orig_cmap)
    fig.colorbar(im2)
    show()

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
