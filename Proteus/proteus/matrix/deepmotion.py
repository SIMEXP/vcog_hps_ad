import numpy.linalg as npl
import numpy as np
from numba import jit
from nibabel.affines import apply_affine
from sklearn import preprocessing

@jit
def extract_cube(vol_target,vol_tdata,affine_ref,initial_position,epi_coord,cube_size):
        # Calculate transformation from ref to target vol
        # calculate the affine transf matrix from EPI to anat
        epi_vox2anat_vox = npl.inv(vol_target.get_affine()).dot(affine_ref)
        # x,y,z
        target_anat = apply_affine(epi_vox2anat_vox,epi_coord)
        # get the cube
        side_size = int(np.floor(cube_size/2.))

        # target spacing
        side_size_targ = (apply_affine(epi_vox2anat_vox,np.array(epi_coord)+side_size) - target_anat)
        side_size_targ = side_size_targ[0]

        x_s,y_s,z_s = vol_target.get_data().shape
        x_interval = np.arange(target_anat[0]-side_size_targ,target_anat[0]+side_size_targ+1,dtype=int)
        y_interval = np.arange(target_anat[1]-side_size_targ,target_anat[1]+side_size_targ+1,dtype=int)
        z_interval = np.arange(target_anat[2]-side_size_targ,target_anat[2]+side_size_targ+1,dtype=int)

        # normalize the data between 0 and 1
        #norm_vol_target = vol_target.get_data()
        norm_vol_target = vol_tdata#avg_frame_norm(vol_target.get_data())
        #norm_vol_target = rescale(norm_vol_target)
        # check if we need padding
        if (x_interval>=0).all() & (x_interval<(x_s)).all() & (y_interval>=0).all() & (y_interval<(y_s)).all() & (z_interval>=0).all() & (z_interval<(z_s)).all():
            small_cube = norm_vol_target[x_interval,...][:,y_interval,...][:,:,z_interval,...]
        else:
            padded_target = np.lib.pad(norm_vol_target, (side_size,side_size), 'constant', constant_values=(0))
            x_interval = np.arange(target_anat[0]-side_size_targ,target_anat[0]+side_size_targ+1,dtype=int)
            y_interval = np.arange(target_anat[1]-side_size_targ,target_anat[1]+side_size_targ+1,dtype=int)
            z_interval = np.arange(target_anat[2]-side_size_targ,target_anat[2]+side_size_targ+1,dtype=int)
            small_cube = padded_target[x_interval+side_size,...][:,y_interval+side_size,...][:,:,z_interval+side_size,...]

        # pad the vols
        #padded_target = np.lib.pad(vol_target.get_data(), (side_size,side_size), 'constant', constant_values=(0))
        #small_cube = padded_target[x_interval+side_size,...][:,y_interval+side_size,...][:,:,z_interval+side_size,...]

        #x_interval = np.arange(target_anat[0]-side_size,target_anat[0]+side_size,dtype=int)
        #y_interval = np.arange(target_anat[1]-side_size,target_anat[1]+side_size,dtype=int)
        #z_interval = np.arange(target_anat[2]-side_size,target_anat[2]+side_size,dtype=int)

        #padded_pos_vol = np.lib.pad(initial_position, (side_size,side_size), 'constant', constant_values=(0))[...,side_size:-side_size]
        #init_pos = padded_pos_vol[x_interval+side_size,...][:,y_interval+side_size,...][:,:,z_interval+side_size,...]
        #init_pos = initial_position[target_anat[0],target_anat[1],target_anat[2],...]
        init_pos = initial_position[epi_coord[0],epi_coord[1],epi_coord[2],...]
        return small_cube, init_pos

@jit
def get_motion_deriv(aff_transforms,tmp_deriv_init):
    world_motion=[]
    point_motion_deriv = []
    tmp_deriv_old = np.copy(tmp_deriv_init)
    for ii in range(aff_transforms.shape[2]):
        world_motion.append(apply_affine(aff_transforms[...,ii],tmp_deriv_old))
    
    # compute the delta
    world_motion = np.array(world_motion)
    deriv_values = world_motion[1:]-world_motion[0:-1]
    
    if len(deriv_values.shape)>3:
        deriv_values = np.squeeze(np.swapaxes(deriv_values[...,np.newaxis],0,5))
    else:
        deriv_values = np.squeeze(np.swapaxes(deriv_values[...,np.newaxis],0,2))
    
    return deriv_values

@jit
def rescale(data,range=(-1,1)):
    scaler = preprocessing.MinMaxScaler(feature_range=range)
    flat_data = scaler.fit_transform(data.flatten())
    #print flat_data.shape
    return np.reshape(flat_data,data.shape)

def trim_outlier(data,pct=0.999):
    cut_data = data.copy()
    nitems = len(cut_data.flatten())*1.
    nbins = np.histogram(np.abs(cut_data.flatten()),1000)
    idx2cut=0

    for i in range(0,1000):
        if (np.sum(nbins[0][0:i])/nitems)>=pct:
            idx2cut = nbins[1][i]
            #print i, nbins[1][i]
            break
    cut_data[data>idx2cut]=idx2cut
    cut_data[data<-idx2cut]=-idx2cut
    return cut_data

def frame_normalize(data,data_mask,norm_avg=False):
    new_data = data.copy()
    new_data[new_data<0]=0
    if norm_avg:
        avg_func = new_data.mean(axis=-1)
        new_data = normalize_brains(new_data,avg_func,data_mask)
        avg_func = normalize_brains(avg_func,avg_func,data_mask)
    else:
        avg_func = new_data.mean(axis=-1)
        new_data = normalize_brains(new_data,avg_func,data_mask)
        avg_func = new_data.mean(axis=-1)

    new_data = new_data - np.repeat(avg_func[...,np.newaxis],new_data.shape[-1],axis=-1)
    return new_data, avg_func

def frame_normalize_old(frame,avg_frame):
    new_frame = frame.copy()
    new_frame[new_frame<0]=0
    new_frame = frame-avg_frame
    new_frame = trim_outlier(new_frame)
    new_frame = rescale(new_frame,(-1,1))
    new_frame = new_frame - new_frame.mean()
    return new_frame

def avg_frame_norm(data):
    new_data = rescale(data,(0,1))
    #new_data = new_data-new_data.mean()
    return new_data

def normalize_brains(target_data,ref_data,mask,newMin=0,newMax=1):
    # Normalize the avg functional map by scaling between 0 and 1 (centering the 50th percentile inside the brain to 1)
    norm_b = target_data.copy()
    dMin = newMin
    dMax = np.percentile(ref_data[mask], 50)
    #dMax = ref_data[mask].mean()
    #print dMax
    if dMax>=900:
        print dMax
        dMax = 900.
    #print dMin,dMax
    norm_b = (norm_b - dMin)*((newMax-newMin)/(dMax-dMin))+newMin
    return norm_b

def qlike_lin_norm(data,newMin=-0.05,newMax=0.05):
    dMin = np.percentile(data, 25)
    dMax = np.percentile(data, 75)
    #print dMin,dMax
    new_data = (data - dMin)*((newMax-newMin)/(dMax-dMin))+newMin
    return new_data

def flip_data(data,motion,flip_x,flip_y,flip_z):
    mat_motion = np.reshape(motion,(motion.shape[0],motion.shape[1]/3,3))
    if flip_x:
        new_data = data[...,::-1,:,:]
        new_motion = mat_motion*np.array([-1,1,1])
    else:
        new_data = data.copy()
        new_motion = mat_motion.copy()

    if flip_y:
        new_data = new_data[...,:,::-1,:]
        new_motion = new_motion*np.array([1,-1,1])

    if flip_z:
        new_data = new_data[...,:,:,::-1]
        new_motion = new_motion*np.array([1,1,-1])

    return new_data, np.reshape(new_motion,(motion.shape[0],motion.shape[1]))

def data_augm(data,motion,y):
    sequence = np.array([[0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]])
    new_data = []
    new_motion = []
    new_y = []
    #print sequence.shape
    for i in range(sequence.shape[0]):
        augm,augm_motion = flip_data(data,motion,sequence[i,0],sequence[i,1],sequence[i,2])
        if i==0:
            new_data = augm
            new_motion = augm_motion
            new_y = y
        else:
            new_data = np.vstack((new_data,augm))
            new_motion = np.vstack((new_motion,augm_motion))
            new_y = np.hstack((new_y,y))

        #print new_data.shape,new_motion.shape
    return [new_data,new_motion],new_y

