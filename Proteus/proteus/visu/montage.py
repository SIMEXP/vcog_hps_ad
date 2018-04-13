__author__ = 'Christian Dansereau'
import numpy as np

def _getspec(vol):
    nx, ny, nz = vol.shape
    nrows = int(np.ceil(np.sqrt(nz)))
    ncolumns = int(np.ceil(nz / (1. * nrows)))
    return nrows, ncolumns, nx, ny, nz


def transform(vol1):
    if vol1.ndim > 3:
        mozaic = []
        for ii in range(vol1.shape[3]):
            mozaic.append(transform(vol1[..., ii]))
        return np.dstack(mozaic)

    vol = np.swapaxes(vol1, 0, 1)
    nrows, ncolumns, nx, ny, nz = _getspec(vol)

    mozaic = np.zeros((nrows * nx, ncolumns * ny))
    indx, indy = np.where(np.ones((nrows, ncolumns)))

    for ii in np.arange(vol.shape[2]):
        # we need to flip the image in the x axis
        mozaic[(indx[ii] * nx):((indx[ii] + 1) * nx), (indy[ii] * ny):((indy[ii] + 1) * ny)] = vol[::-1, :, ii]

    return mozaic


