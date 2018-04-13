__author__ = 'Christian Dansereau'

'''
Tools for registration of 3D volumes
'''

import numpy as np
import numpy.linalg as npl
from scipy import ndimage
from scipy.optimize import fmin_powell
from scipy.ndimage.filters import gaussian_filter


def resample_trans(sv, sv2sw_affine, tv2tw_affine, tv_shape, sw2tw_affine=np.eye(4)):
    # transform
    # start = time.time()
    transform_affine = npl.inv(np.dot(npl.inv(tv2tw_affine), np.dot(sw2tw_affine, sv2sw_affine)))

    # Split an homogeneous transform into its matrix and vector components.
    # The transformation must be represented in homogeneous coordinates.
    # It is split into its linear transformation matrix and translation vector
    # components.

    ndimin = transform_affine.shape[0] - 1
    ndimout = transform_affine.shape[1] - 1
    matrix = transform_affine[0:ndimin, 0:ndimout]
    vector = transform_affine[0:ndimin, ndimout]
    # print matrix,vector

    # interpolation
    new_volume = ndimage.affine_transform(sv, matrix,
                                          offset=vector,
                                          output_shape=tv_shape,
                                          order=1)
    # print(time.time() - start)
    return new_volume

def apply_affine(aff, pts):
    """ Apply affine matrix `aff` to points `pts`
    Returns result of application of `aff` to the *right* of `pts`.  The
    coordinate dimension of `pts` should be the last.
    For the 3D case, `aff` will be shape (4,4) and `pts` will have final axis
    length 3 - maybe it will just be N by 3. The return value is the
    transformed points, in this case::
        res = np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]
        transformed_pts = res.T
    This routine is more general than 3D, in that `aff` can have any shape
    (N,N), and `pts` can have any shape, as long as the last dimension is for
    the coordinates, and is therefore length N-1.
    Parameters
    ----------
    aff : (N, N) array-like
        Homogenous affine, for 3D points, will be 4 by 4. Contrary to first
        appearance, the affine will be applied on the left of `pts`.
    pts : (..., N-1) array-like
        Points, where the last dimension contains the coordinates of each
        point.  For 3D, the last dimension will be length 3.
    Returns
    -------
    transformed_pts : (..., N-1) array
        transformed points
    Examples
    --------
    >>> aff = np.array([[0,2,0,10],[3,0,0,11],[0,0,4,12],[0,0,0,1]])
    >>> pts = np.array([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
    >>> apply_affine(aff, pts) #doctest: +ELLIPSIS
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]]...)
    Just to show that in the simple 3D case, it is equivalent to:
    >>> (np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]).T #doctest: +ELLIPSIS
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]]...)
    But `pts` can be a more complicated shape:
    >>> pts = pts.reshape((2,2,3))
    >>> apply_affine(aff, pts) #doctest: +ELLIPSIS
    array([[[14, 14, 24],
            [16, 17, 28]],
    <BLANKLINE>
           [[20, 23, 36],
            [24, 29, 44]]]...)
    """
    aff = np.asarray(aff)
    pts = np.asarray(pts)
    shape = pts.shape
    pts = pts.reshape((-1, shape[-1]))
    # rzs == rotations, zooms, shears
    rzs = aff[:-1, :-1]
    trans = aff[:-1, -1]
    res = np.dot(pts, rzs.T) + trans[None, :]
    return res.reshape(shape)



def aff_tsf(xt, yt, zt, xr, yr, zr, inv_affine=False):
    A = np.eye(4)
    transf = np.eye(4)
    # translation
    transf[0, 3] = xt
    transf[1, 3] = yt
    transf[2, 3] = zt
    # rotation
    rot_x = np.eye(4)
    rot_y = np.eye(4)
    rot_z = np.eye(4)
    if xr != 0:
        rot_x = np.array([[1., 0., 0., 0.],
                          [0., np.cos(xr), -np.sin(xr), 0.],
                          [0., np.sin(xr), np.cos(xr), 0.],
                          [0., 0., 0., 1.]]).astype(float)
    if yr != 0:
        rot_y = np.array([[np.cos(yr), 0., np.sin(yr), 0.],
                          [0., 1., 0., 0.],
                          [-np.sin(yr), 0., np.cos(yr), 0.],
                          [0., 0., 0., 1.]]).astype(float)
    if zr != 0:
        rot_z = np.array([[np.cos(zr), -np.sin(zr), 0., 0.],
                          [np.sin(zr), np.cos(zr), 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]]).astype(float)

    rot = rot_x.dot(rot_y).dot(rot_z)

    if inv_affine:
        # apply the inverse of the transform in the correct order ref see: http://negativeprobability.blogspot.ca/2011/11/affine-transformations-and-their.html
        rot_inv = npl.inv(rot)
        A[:3, :3] = rot_inv[:3, :3]
        A[:3, 3] = -rot_inv[:3, :3].dot(transf[:3, 3])
        #A = npl.inv(transf.dot(rot))
    else:
        A = transf.dot(rot)
    return A


def _aff_trans(params, *args):
    transf = aff_tsf(*params)
    coreg_vol = resample_trans(args[1], args[2], args[3], args[4], sw2tw_affine=transf)
    return coreg_vol, transf

def transform(vol, params, v2w, inv_affine=False, rotation_unit='deg'):
    if rotation_unit=='deg':
        params = deg2rad(params)
    params = list(params)
    params.append(inv_affine)
    transf = aff_tsf(*params)
    coreg_vol = resample_trans(vol, v2w, v2w, vol.shape, sw2tw_affine=transf)
    return coreg_vol, transf


def _coreg(params, *args):
    mask_ = args[5]
    coreg_vol, _ = _aff_trans(params, *args)

    #coreg_vol = gaussian_filter(coreg_vol, 0.5, 0)
    #score = np.corrcoef(coreg_vol.ravel(), args[0].ravel())
    score = np.corrcoef(coreg_vol[mask_], args[0][mask_])
    # print score
    # print score[0,1]
    # print x,args
    return 1 - score[0, 1]
    # return score

def rad2deg(params):
    params_c = params.copy()
    params_c[3:, ...] = (params_c[3:, ...] / np.pi) * 180.
    return params_c

def deg2rad(params):
    params_c = params.copy()
    params_c[3:, ...] = (params_c[3:, ...] * np.pi) / 180.
    return params_c

def coreg(vols, affine, ref='median'):
    if ref=='median':
        coreg_vols, transf, motion_params = fit(source=vols, v2w_source=affine, target=np.median(vols, 3),
                                                v2w_target=affine, mask=[], verbose=False, stride=2, dowsamp_flag=True)
    elif ref=='first':
        coreg_vols, transf, motion_params = fit(source=vols, v2w_source=affine, target=vols[..., 0],
                                                v2w_target=affine, mask=[], verbose=False, stride=2, dowsamp_flag=True)
    elif ref == 'last':
        coreg_vols, transf, motion_params = fit(source=vols, v2w_source=affine, target=vols[..., -1],
                                                v2w_target=affine, mask=[], verbose=False, stride=2, dowsamp_flag=True)
    elif ref == 'mean':
        coreg_vols, transf, motion_params = fit(source=vols, v2w_source=affine, target=np.mean(vols, 3),
                                                v2w_target=affine, mask=[], verbose=False, stride=2, dowsamp_flag=True)

    return coreg_vols, transf, motion_params

def fit(source, v2w_source, target, v2w_target, mask = [], verbose = False, stride=2,dowsamp_flag=False):
    # TODO add initialization params for each frame based on the precedent param
    # TODO change size of the target matrix for faster evaluation
    coreg_vols    = []
    transfs       = []
    motion_params = []
    nframes = 1

    if mask==[]:
        mask = np.ones_like(target).astype(bool)
    #else:
    #    mask[::stride, :, :] = False
    #    mask[:, ::stride, :] = False
    #    mask[::stride+1, :, :] = False
    #    mask[:, ::stride+1, :] = False

    if len(source.shape) > 3:
        nframes = source.shape[3]

    if dowsamp_flag:
        # dowsample target
        tv2tw_affine = np.copy(v2w_target)  # np.eye(4)
        tv_shape = (np.ceil(np.array(target.shape) / 2.)).astype(int)
        tv2tw_affine[:3, :3] = tv2tw_affine[:3, :3] * 2.

        target_downsamp = resample_trans(target, np.copy(v2w_target), tv2tw_affine, tv_shape)
        #target_downsamp = gaussian_filter(target_downsamp, 0.5, 0)
        #mask_down = np.ones_like(target_downsamp).astype(bool)
        mask_down = resample_trans(mask, np.copy(v2w_target), tv2tw_affine, tv_shape)

    for frame in range(nframes):
        if nframes == 1:
            source_ = source
        else:
            source_ = source[..., frame]


        if dowsamp_flag:
            '''
            # dowsample target
            tv2tw_affine = np.copy(v2w_target)#np.eye(4)
            tv_shape = np.ceil(np.array(target.shape) / 2.)
            tv2tw_affine[:3, :3] = tv2tw_affine[:3, :3] * 2.

            target_downsamp = resample_trans(target, v2w_target, tv2tw_affine, tv_shape)
            #mask = np.ones_like(target_downsamp).astype(bool)
            mask = resample_trans(mask, v2w_target, tv2tw_affine, tv_shape)
            '''
            #source_ = resample_trans(source_, v2w_source, tv2tw_affine, tv_shape)
            #v2w_source = tv2tw_affine
            # Rough estimate
            params = fmin_powell(func=_coreg, x0=np.zeros((1, 6))[0, :],
                                 args=(target_downsamp, source_, v2w_source, tv2tw_affine, target_downsamp.shape, mask_down),
                                 xtol=0.0001, ftol=0.005, disp=verbose)

            # Fine tuned estimate
            #params = fmin_powell(func=_coreg, x0=params,
            #                     args=(target, source_, v2w_source, v2w_target, target.shape, mask),
            #                     xtol=0.0001, ftol=0.005, disp=verbose)
        else:
            params = fmin_powell(func=_coreg, x0=np.zeros((1, 6))[0, :],
                                 args=(target, source_, v2w_source, v2w_target, target.shape, mask),
                                 xtol=0.0001, ftol=0.001, disp=verbose)
        coreg_vol, transf = _aff_trans(params, *(target, source_, v2w_source, v2w_target, target.shape))
        coreg_vols.append(coreg_vol)
        transfs.append(transf)
        motion_params.append(params)
        if nframes == 1:
            motion_params[0] = rad2deg(motion_params[0])
            return coreg_vols[0], transfs[0], motion_params[0]

    motion_params = np.stack(motion_params, axis=1)
    motion_params = rad2deg(motion_params).T
    return np.stack(coreg_vols, axis=3), np.stack(transfs, axis=2), motion_params


def displacement_field(v2w, motion_params, vol_shape, rotation_unit='deg'):
    '''
    Compute the displacement field in the world space (normally in mm)
    '''
    a = np.zeros(vol_shape)
    coord_voxel = np.array(np.where(a == 0)).T

    if rotation_unit=='deg':
        motion_params = deg2rad(motion_params)

    if len(motion_params.shape) == 1:
        flag_multi_entry = False
        N = 1
    else:
        flag_multi_entry = True
        N = motion_params.shape[0]

    # Iterate over all motion entry
    motion_fields = []
    for ii in range(N):
        if flag_multi_entry:
            params = motion_params[ii]
        else:
            params = motion_params

        # get the affine transformations
        motion = aff_tsf(*params)
        v2w_motion = motion.dot(v2w)

        coord_w = apply_affine(v2w, coord_voxel)
        coord_w2 = apply_affine(v2w_motion, coord_voxel)

        diff_coord = coord_w - coord_w2
        motion_field = diff_coord.reshape(vol_shape[0], vol_shape[1], vol_shape[2], 3)
        motion_fields.append(motion_field)

    return np.stack(motion_fields)

