# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:09:40 2021

@author: danielm
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

def downsample_and_warp_ten_session(sourcepath = r'\\allen\\programs\\braintv\\workgroups\\cortexmodels\\michaelbu\\Stimuli\\SignalNoise\\arrays\\',
                                    destpath = r'/Users/danielm/Desktop/movies_ten_session/',
                                    MONITOR_DIST_CM=15,
                                    do_plots=True):

    stim = get_filelist(sourcepath,ext_name='.npy')
    
    for s, stim_name in enumerate(stim):   
        
        print(stim_name)
        
        warped_movie = get_warped_and_downsampled_stack(sourcepath,
                                        destpath,
                                        stim_name,
                                        dist_cm=MONITOR_DIST_CM)
        if do_plots:
            plt.figure()
            plt.imshow(warped_movie[0],interpolation='none',cmap='gray')
            plt.title('warped and downsampled')
            plt.show()
        
        warping_mask = get_warping_mask(warped_movie)
        if do_plots:
            plt.figure()
            plt.imshow(warping_mask,interpolation='none',cmap='gray',vmin=0,vmax=1)
            plt.title('warping mask')
            plt.show()
        
        #convert to unwarped version to get stim with monitor boundaries
        unwarped_stack = unwarp_stack(destpath,
                                      str(MONITOR_DIST_CM)+'_'+stim_name)
        if do_plots:
            plt.figure()
            plt.imshow(unwarped_stack[0],interpolation='none',cmap='gray')
            plt.title('unwarped')
            plt.show()

def get_warped_and_downsampled_stack(source_path,destpath,stim_name,dist_cm,new_y=400,new_x=640):
    
    warped_and_downsampled_filepath = destpath+'ds_warped_'+str(dist_cm)+'_'+stim_name
    if os.path.isfile(warped_and_downsampled_filepath):
        warped_and_downsampled_im_stack = np.load(warped_and_downsampled_filepath)
    else:
        im_stack = load_original_movie(source_path,stim_name)
        
        im_stack = crop_movie(im_stack,new_y,new_x)
                      
        warped_im_stack = warp_image_stack(im_stack,dist_cm)

        warped_im_stack = truncate_to_whole_secs(warped_im_stack)

        warped_and_downsampled_im_stack = downsample_image_stack(warped_im_stack)
        
        np.save(warped_and_downsampled_filepath,warped_and_downsampled_im_stack)
    
    return warped_and_downsampled_im_stack

def truncate_to_whole_secs(movie,frame_rate=30):
    print('truncating to whole seconds...')
    movie_length = np.shape(movie)[0]
    new_movie_length = int(np.floor(movie_length/frame_rate) * frame_rate)
    return movie[:new_movie_length]

def load_original_movie(source_path,filename):
    print('loading source movie...')
    return np.load(source_path+filename).astype(np.uint8)

def crop_movie(im_stack,new_y,new_x):
    print('cropping...')
    (num_frames,y_dim,x_dim) = im_stack.shape
    y_min = int(y_dim/2 - new_y/2)
    y_max = int(y_dim/2 + new_y/2)
    x_min = int(x_dim/2 - new_x/2)
    x_max = int(x_dim/2 + new_x/2)
    return im_stack[:,y_min:y_max,x_min:x_max]
 
def warp_image_stack(im_stack,
                     dist_cm=15,
                     mon_x = 1920,#pixels
                     mon_y = 1200#pixels
                     ):
    
    print('warping...')
    (num_images,num_y_samples,num_x_samples) = im_stack.shape 

    sample_points = num_x_samples * num_y_samples
    x_samples = np.linspace(0,mon_x,num_x_samples).astype(int) - mon_x/2
    y_samples = np.linspace(0,mon_y,num_y_samples).astype(int) - mon_y/2
    pixel_coors = np.zeros((sample_points,2))
    for ix,x in enumerate(x_samples):
        for iy,y in enumerate(y_samples):
            pixel_coors[ix*num_y_samples+iy,0] = x
            pixel_coors[ix*num_y_samples+iy,1] = y

    warped_pixel_coors = warp_stimulus_coords(pixel_coors,distance=dist_cm)
    
    image_x = np.linspace(0,mon_x,np.shape(im_stack)[2]).astype(int) - mon_x/2
    image_y = np.linspace(0,mon_y,np.shape(im_stack)[1]).astype(int) - mon_y/2    

    warped_image_stack = np.zeros((num_images,num_y_samples,num_x_samples)).astype(np.uint8)
    for ix in range(num_x_samples):
        for iy in range(num_y_samples):
            
            distance_to_x = np.sqrt(np.square(image_x - warped_pixel_coors[ix*num_y_samples+iy,0]))
            distance_to_y = np.sqrt(np.square(image_y - warped_pixel_coors[ix*num_y_samples+iy,1]))
            
            x_idx = np.argmin(distance_to_x)
            y_idx = np.argmin(distance_to_y)
                
            warped_image_stack[:,iy,ix] = im_stack[:,y_idx,x_idx]
        
    return warped_image_stack

def warp_stimulus_coords(vertices,
                         distance=15.0,
                         mon_height_cm=32.5,
                         mon_width_cm=51.0,
                         mon_res=(1920, 1200),
                         eyepoint=(0.5, 0.5)):
    '''
    For a list of screen vertices, provides a corresponding list of texture coordinates.
    Parameters
    ----------
    vertices: numpy.ndarray
        [[x0,y0], [x1,y1], ...] A set of vertices to  convert to texture positions.
    distance: float
        distance from the monitor in cm.
    mon_height_cm: float
        monitor height in cm
    mon_width_cm: float
        monitor width in cm
    mon_res: tuple
        monitor resolution (x,y)
    eyepoint: tuple
    Returns
    -------
    np.ndarray
        x,y coordinates shaped like the input that describe what pixel coordinates
        are displayed an the input coordinates after warping the stimulus.
    '''

    mon_width_cm = float(mon_width_cm)
    mon_height_cm = float(mon_height_cm)
    distance = float(distance)
    mon_res_x, mon_res_y = float(mon_res[0]), float(mon_res[1])

    vertices = vertices.astype(np.float)

    # from pixels (-1920/2 -> 1920/2) to stimulus space (-0.5->0.5)
    vertices[:, 0] = vertices[:, 0] / mon_res_x
    vertices[:, 1] = vertices[:, 1] / mon_res_y

    x = (vertices[:, 0] + 0.5) * mon_width_cm
    y = (vertices[:, 1] + 0.5) * mon_height_cm

    xEye = eyepoint[0] * mon_width_cm
    yEye = eyepoint[1] * mon_height_cm

    x = x - xEye
    y = y - yEye

    r = np.sqrt(np.square(x) + np.square(y) + np.square(distance))

    azimuth = np.arctan(x / distance)
    altitude = np.arcsin(y / r)
    azimuth[azimuth == 0] = np.finfo(np.float32).eps
    altitude[altitude == 0] = np.finfo(np.float32).eps
    centralAngle = np.arccos(np.cos(altitude) * np.cos(np.abs(azimuth)))
    arcLength = centralAngle * distance
    
    # calculate the texture coordinates
    tx = distance * (1 + x / r) - distance
    ty = distance * (1 + y / r) - distance
    # the texture coordinates (which are now lying on the sphere)
    # need to be remapped back onto the plane of the display.
    # This effectively stretches the coordinates away from the eyepoint.
    theta = np.arctan2(ty, tx)
    tx = arcLength * np.cos(theta)
    ty = arcLength * np.sin(theta)

    u_coords = tx / mon_width_cm
    v_coords = ty / mon_height_cm

    retCoords = np.column_stack((u_coords, v_coords))

    # back to pixels
    retCoords[:, 0] = retCoords[:, 0] * mon_res_x
    retCoords[:, 1] = retCoords[:, 1] * mon_res_y

    return retCoords

def downsample_image_stack(im_stack):
    
    print('downsampling warped movie...')
    
    num_images = np.shape(im_stack)[0]

    sample_zoom = zoom(im_stack[0],0.75)

    zoom_y = np.shape(sample_zoom)[0]
    zoom_x = np.shape(sample_zoom)[1]

    downsampled_stack = np.zeros((num_images,zoom_y,zoom_x)).astype(np.uint8)
    for i in range(num_images):
        downsampled_stack[i,:,:] = zoom(im_stack[i],0.75)
    
    return downsampled_stack

def unwarp_stack(stim_path,filename,mon_x = 1920,mon_y = 1200):
    
    warped_fullpath = stim_path+'ds_warped_'+filename
    unwarped_fullpath = stim_path + 'ds_unwarped_' +filename
    if os.path.isfile(unwarped_fullpath):
        print('loading unwarped movie that was already generated...')
        unwarped_stack = np.load(unwarped_fullpath)
    else:
        print('unwarping movie...')
        warped_stack = np.load(warped_fullpath)
        
        (num_images,num_y_samples,num_x_samples) = warped_stack.shape 

        sample_points = num_x_samples * num_y_samples
        x_samples = np.linspace(0,mon_x,num_x_samples).astype(int) - mon_x/2
        y_samples = np.linspace(0,mon_y,num_y_samples).astype(int) - mon_y/2
        pixel_coors = np.zeros((sample_points,2))
        for ix,x in enumerate(x_samples):
            for iy,y in enumerate(y_samples):
                pixel_coors[ix*num_y_samples+iy,0] = x
                pixel_coors[ix*num_y_samples+iy,1] = y
    
        warped_pixel_coors = warp_stimulus_coords(pixel_coors)
        
        image_x = np.linspace(0,mon_x,num_x_samples).astype(int) - mon_x/2
        image_y = np.linspace(0,mon_y,num_y_samples).astype(int) - mon_y/2    
    
        unwarped_stack = np.zeros((num_images,num_y_samples,num_x_samples)).astype(np.uint8)+127
        for ix in range(num_x_samples):
            for iy in range(num_y_samples):
                
                distance_to_x = np.sqrt(np.square(image_x - warped_pixel_coors[ix*num_y_samples+iy,0]))
                distance_to_y = np.sqrt(np.square(image_y - warped_pixel_coors[ix*num_y_samples+iy,1]))
                
                x_idx = np.argmin(distance_to_x)
                y_idx = np.argmin(distance_to_y)
                    
                unwarped_stack[:,y_idx,x_idx] = warped_stack[:,iy,ix]
        np.save(unwarped_fullpath,unwarped_stack)
    
    return unwarped_stack
     
def get_warping_mask(warped_movie,mon_x = 1920,mon_y = 1200):
    
    (num_frames,num_y_samples,num_x_samples) = warped_movie.shape
    
    sample_points = num_x_samples * num_y_samples
    x_samples = np.linspace(0,mon_x,num_x_samples).astype(int) - mon_x/2
    y_samples = np.linspace(0,mon_y,num_y_samples).astype(int) - mon_y/2
    pixel_coors = np.zeros((sample_points,2))
    for ix,x in enumerate(x_samples):
        for iy,y in enumerate(y_samples):
            pixel_coors[ix*num_y_samples+iy,0] = x
            pixel_coors[ix*num_y_samples+iy,1] = y

    warped_pixel_coors = warp_stimulus_coords(pixel_coors)
    
    image_x = np.linspace(0,mon_x,num_x_samples).astype(int) - mon_x/2
    image_y = np.linspace(0,mon_y,num_y_samples).astype(int) - mon_y/2
    
    warping_mask = np.zeros((num_y_samples,num_x_samples)).astype(np.bool)
    for ix in range(num_x_samples):
        for iy in range(num_y_samples):
            
            distance_to_x = np.sqrt(np.square(image_x - warped_pixel_coors[ix*num_y_samples+iy,0]))
            distance_to_y = np.sqrt(np.square(image_y - warped_pixel_coors[ix*num_y_samples+iy,1]))
            
            x_idx = np.argmin(distance_to_x)
            y_idx = np.argmin(distance_to_y)

            warping_mask[y_idx,x_idx] = True
    
    return warping_mask

def get_filelist(path,ext_name='.npz'):
    
    path_list = os.listdir(path)
    ext_files = []
    
    for file_path in path_list:
        if file_path.find(ext_name) != -1:
            ext_files.append(file_path)

    return ext_files

if __name__=='__main__':  
    downsample_and_warp_ten_session()