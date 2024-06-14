#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:10:51 2023

@author: joshwolpert
"""

# LEM driver file for Chapter 1 of thesis

# Description:
# LEM to simulate evolving landscape in response to various discharge oscillations

# The model evolves according to:
    # Stream power erosion
    # User-defined surface water discharge oscillations

#################### #################### #################### ####################
# Import Modules
#################### #################### #################### ####################

import numpy as np
#from matplotlib import cm
import matplotlib.pyplot as plt
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/joshwolpert/opt/anaconda3/envs/divide_sig/bin/ffmpeg"
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import *
import shutil
import yaml
import pandas as pd 
#from tempfile import mkdtemp
from landlab.io.netcdf import read_netcdf, write_netcdf # For exporting grids to netcdf files
from landlab import RasterModelGrid # The different types of grid objects
from landlab import imshow_grid # For plotting grids
from landlab.components import FlowAccumulator, FastscapeEroder, DepressionFinderAndRouter
from landlab.plot.imshow import imshow_grid, imshow_grid_at_node
from landlab import load_params # This allows us to instantiate components from input files
# ^ See the component tutorial for information on using the load_params Landlab function. It's pretty nifty.
#%matplotlib inline # Tells Jupyter Notebook to display plots here on the page.
from methods import *


#################### #################### #################### ####################
# Define component parameter values
#################### #################### #################### ####################

config_list = ['config.yaml']

for i in range(len(config_list)):

    config = yaml.load(open('/Users/joshwolpert/Desktop/ksnBias_Repo/'+config_list[i],'r'),Loader=yaml.FullLoader)
    
    # Define grid parameters
    dxy = config['grid']['grid_dxy']
    grid_length = config['grid']['grid_length']
    grid_width = config['grid']['grid_width']
    
    # Define fastscape stream power erosion fluvial parameters
    K = float(config['fluvial_ero']['K_sp']) # Not sure why we suddenly need to specify float
    m = config['fluvial_ero']['m_sp']
    n = config['fluvial_ero']['n_sp']
    
    # Define uplift rates
    ic_U = config['uplift']['ic_U']
    ramp_U = config['uplift']['ramp_U']
    
    # Define time variables
    dt = config['time']['dt']
    total_t = config['time']['run_time']
    nt = int(total_t // dt) # Number of iterations in main model loop

    
    #################### #################### #################### ####################
    # Prepare folders for output
    #################### #################### #################### ####################
        
    # Create output folders if they don't already exist
    run_name = config['output_parameters']['run_name']
    dir_parent = config['output_parameters']['output_dir']
    
    path = os.path.join(dir_parent, run_name)
    os.makedirs(path)
    
    dir_frames_path = os.path.join(path, 'output_frames')
    os.mkdir(dir_frames_path)
    
    main_netcdf_path = os.path.join(path, 'output_frames/netcdf')
    os.mkdir(main_netcdf_path)

    # Copy and save the run's config.yaml file to the output path.
    config_path = os.path.join(os.path.abspath(os.curdir), 'config.yaml')
    shutil.copy(config_path, path)

    # Instantiate new grid and give it steady state topography. This ensures all grid fields are of correct types.
    mg = RasterModelGrid((grid_length, grid_width), xy_spacing = dxy)
    z = mg.add_zeros('topographic__elevation', at = "node")
    initial_roughness = np.multiply(np.random.rand(z.size),10) # Force a new random elevation initial condition
    z += initial_roughness

    # Set boundary conditions for grid sides
    bc_right = config['bcs']['bc_right']
    bc_top = config['bcs']['bc_top']
    bc_left = config['bcs']['bc_left']
    bc_bottom = config['bcs']['bc_bottom']
    mg.set_status_at_node_on_edges(right=bc_right, top=bc_top, left=bc_left, bottom=bc_bottom)

    # Make open boundary elevations zero
    mg.at_node['topographic__elevation'][mg.open_boundary_nodes] = 0   
    
    # Flow accumulator
    fr = FlowAccumulator(
        mg,
        'topographic__elevation',
        flow_director = 'D8',
        depression_finder=DepressionFinderAndRouter
        ) # Langston and Tucker (2018) use D8. 
        
    # River erosion
    sp = FastscapeEroder(
        mg,
        K_sp = K,
        m_sp = m,
        n_sp = n,
        discharge_field="surface_water__discharge"
        )
            
    
    #################### #################### #################### ####################
    # Main model loop
    #################### #################### #################### ####################
    
    # Build rock uplift field
    u_field = np.full_like(mg.at_node['topographic__elevation'],ic_U)
    
    # Apply a rock uplift field gradient
    # l=0
    # for q in range(mg.shape[0]):
    #                 u_field[l:l+mg.shape[1]]=np.linspace(7e-5,5.65e-4,200)[q]
    #                 l+=mg.shape[1]
    
    # Build water__unit_flux_in field
    # Apply a runoff field gradient
    # Bottom-Heavy
    l=0
    for q in range(mg.shape[0]):
                    mg.at_node['water__unit_flux_in'][l:l+mg.shape[1]]=np.linspace(2.0,0.5,200)[q]
                    l+=mg.shape[1]
       
    # Top-Heavy
    # l=0
    # for q in range(mg.shape[0]):
    #                 mg.at_node['water__unit_flux_in'][l:l+mg.shape[1]]=np.linspace(0.5,2.0,200)[q]
    #                 l+=mg.shape[1]
    
    # Define climate forcing parameters.
    # 5 Myr: 1.2566e-6
    # 2.5 Myr: 2.5133e-6
    # 1 Myr: 6.2832e-6
    # 500kyr: 1.2566e-5
    # 200kyr: 3.1416e-5
    # 100kyr: 6.2832e-5
    # 40kyr: 1.5708e-4
    # 20kyr: 3.1416e-4
    amplitude = 0.25
    period =   3.1416e-4 
    phase_shift = 0
    
    # Field to store topography before erosion during time steps
    mg.add_zeros('prev_topo', at="node", clobber=True)
    mg.at_node['prev_topo'] += np.copy(mg.at_node['topographic__elevation'])
    
    # Field to store erosion rates during time steps
    mg.add_zeros('erosion_rate', at="node", clobber=True)
    
    # Field to store furthest extent of snow throughout model run
    mg.add_zeros('snow__furthest_extent', at="node", clobber=True)
    
    # Track erosion rate to check for steady state
    erosion_tracker = []  
    
    for i in range(nt):
        
        time = (i+1)*dt
        print(time)
        
        # Set water__unit_flux_in field for time step
        
        # Uniform Amplitude precip changes
        mg.at_node['water__unit_flux_in']=[mg.at_node['water__unit_flux_in'][j]+amplitude*np.sin(period*((i*dt)+phase_shift)) for j in range(len(mg.at_node['water__unit_flux_in']))]
        
        # Uniform fractional precip changes
        #mg.at_node['water__unit_flux_in']=[mg.at_node['water__unit_flux_in'][j]+(mg.at_node['water__unit_flux_in'][j]*(0.25*np.sin(period*((i*dt)+phase_shift)))) for j in range(len(mg.at_node['water__unit_flux_in']))]
        
        
        # Route and accumulate flow with FlowAccumulator D8
        fr.run_one_step()
        
        # Update pre-erosion topography for next time step
        mg.add_zeros('prev_topo', at="node", clobber=True)
        mg.at_node['prev_topo'] += np.copy(mg.at_node['topographic__elevation'])
        
        # Stream power erosion
        sp.run_one_step(dt)
        
        # Measure model erosion rate (diffusion not included)
        erosion_rate = (mg.at_node['prev_topo']-mg.at_node['topographic__elevation'])/dt
        mg.at_node['erosion_rate'] = np.copy(erosion_rate)
        mg.at_node['erosion_rate'][np.isnan(mg.at_node['erosion_rate'])]=0
        erosion_rate[np.where(erosion_rate==0)]=np.nan
        erosion_tracker.append(np.nanmean(erosion_rate[np.where(mg.status_at_node==0)]))
        
        # Uplift nodes and make sure boundaries are correct.
        mg.at_node['topographic__elevation'][mg.core_nodes] += (u_field*dt)[mg.core_nodes]
        mg.set_status_at_node_on_edges(right=bc_right, top=bc_top, left=bc_left, bottom=bc_bottom)
        mg.at_node['topographic__elevation'][mg.open_boundary_nodes] = 0 
                
        # Keep track of time and save frames
        if i*dt > 24000000:
            print ('Time: %d' % time)
            
        
        # Save timestep as netcdf file
            mg.save(main_netcdf_path + '/' + str((i+1)*dt), format = 'netcdf')
            
        # Reset water__unit_flux_in field for next iteration
        # Uniform Amplitude Precip Change
        mg.at_node['water__unit_flux_in']=[mg.at_node['water__unit_flux_in'][j]-amplitude*np.sin(period*((i*dt)+phase_shift)) for j in range(len(mg.at_node['water__unit_flux_in']))]
    
        # Uniform fractional precip changes
        # Apply a runoff field range
        # l=0
        # for q in range(mg.shape[0]):
        #                 mg.at_node['water__unit_flux_in'][l:l+mg.shape[1]]=np.linspace(2.0,0.5,200)[q]
        #                 l+=mg.shape[1]        
    
    np.savetxt(path+"/erosion_tracker.csv",erosion_tracker)
    

