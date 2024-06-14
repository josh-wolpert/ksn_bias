#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:43:34 2024

@author: joshwolpert
"""

# Import modules
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from landlab.utils import get_watershed_outlet, get_watershed_nodes, get_watershed_mask, get_watershed_masks
from landlab.components import DrainageDensity, TrickleDownProfiler, ChannelProfiler, SteepnessFinder,ChiFinder
from landlab.io.netcdf import read_netcdf, write_netcdf # For exporting grids to netcdf files
from landlab import imshow_grid # For plotting grids
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/joshwolpert/opt/anaconda3/envs/divide_sig/bin/ffmpeg" # workaround for error; try to eliminate need in future
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import *
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
import pandas as pd
import csv
import re
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})
from scipy import integrate
import statsmodels.api as sm
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker

#%% Make a video from png frames
def make_video(image_folder):
  """
  Make a video from .png files stored in a folder.
  """

  fps=15
  image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
  image_files.sort(key=lambda f: int(re.sub('\D', '', f)))
  clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
  clip.write_videofile(image_folder+'/my_video.mp4')
  
  
#%% Figure 1: Sensitivity analysis for Equations (7c)

def calc_ksn_bias(m,n):
    # Assign variables
    U = 0.0002 # Uplift; m/yr
    K = 2.5e-11 # Erosional efficiency; Not needed, I think
    T_start = 0 # years
    T_end = 40000 # years
    T = np.linspace(T_start,T_end,400)
    m = m
    n = n
    Qt_amp = 1e5
    Qt_mean = 1e5+1e-5
    Qt = Qt_amp*np.sin(1.5708e-4*(T+10000))+Qt_mean # Discharge; m^3/yr
    
    # Numerically integrate discharge function over climate cycle.
    Qt_integral = integrate.trapezoid((Qt**m),T)
    
    # Calculate Qeff.
    Qeff = ((1/T_end)*Qt_integral)**(1/m)
    
    # Solve for ratio of ksnq ksnqeff using solution in text
    ksnq_bias = ((T_end*(Qt**m))/(Qt_integral))**(1/n)
    
    theta = ksnq_bias*0+m/n
    
    Q_fraction = ((T_end*(Qt**m))/(Qt_integral))
    
    return ksnq_bias,theta,Q_fraction

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
          0.0 and 1.0.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
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
        
    newcmap = mcolors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# Plot predicted versus modeled channel steepness values
fig,ax = plt.subplots(figsize = (6,4))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)

m = 0.3
ns = np.arange(m,m*10,0.01)
Q_min_global = 100 # Dummy value replaced later
Q_max_global = 0 # Dummy value replaced later
ksnqBias_min_global = 100 # Dummy value replaced later
ksnqBias_max_global = 0 # Dummy value replaced later

# Iterate once to get range of ksnq_bias
for i in range(len(ns)):
    
    n = ns[i]

    ksnq_bias,theta,Q_fraction = calc_ksn_bias(m,n)
    
    Q_min_local = np.min(Q_fraction)
    Q_min_global = np.min([Q_min_local,Q_min_global])
    Q_max_local = np.max(Q_fraction)
    Q_max_global = np.max([Q_max_local,Q_max_global])
    
    ksnqBias_min_local = np.min(ksnq_bias)
    ksnqBias_min_global = np.min([ksnqBias_min_local,ksnqBias_min_global])
    ksnqBias_max_local = np.max(ksnq_bias)
    ksnqBias_max_global = np.max([ksnqBias_max_local,ksnqBias_max_global])
    
newcmap = shiftedColorMap(plt.cm.seismic, start=0, midpoint=0.355, stop=1.0, name='shiftedcmap')
    
# Set up color scheme
colormap = newcmap#plt.cm.get_cmap('seismic')
k = 401
colors = plt.cm.seismic(np.linspace(0,1,k))
colors = colormap(np.linspace(0,1,k))

# Iterate again to plot
for i in range(len(ns)):
    
    n = ns[i]

    ksnq_bias,theta,Q_fraction = calc_ksn_bias(m,n)
    
    colors_plot = colors[[int(i) for i in (ksnq_bias/ksnqBias_max_global)*400]]
    
    data1 = plt.scatter(Q_fraction,theta,s = 100,color = colors_plot,marker = 's')

sm = plt.cm.ScalarMappable(cmap=colormap)
sm.set_clim(vmin=ksnqBias_min_global, vmax=ksnqBias_max_global)
plt.colorbar(sm)

ax.grid(False)
ax.set_xlim(np.min(Q_fraction),np.max(Q_fraction))
#ax.set_xlim(0.01,2)
ax.set_ylim(0.1,1)
#ax.set_xticks([],[])
ax.set_facecolor((1, 1, 1))
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()

# Save figure
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/Bias_sensitivity_analysis/m03_test', facecolor='white', edgecolor='none',dpi=500)

#%% Figrue 1: Companion to sensitivity analysis, showing flood distribution and bias through time

m=0.3
T = np.linspace(0,40000,400)
Qt_amp = 1e5
Qt_mean = 1e5+0.1
Qt = Qt_amp*np.sin(1.5708e-4*(T+10000))+Qt_mean 
Qt_func = Qt**m# Denominator inside parentheses in Equation 6 

# Plot predicted versus modeled channel steepness values
fig,ax = plt.subplots(figsize = (6,4))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)

# Discharge curve
data1 = plt.plot(T,Qt_func,color='b',linestyle='-',linewidth=2,alpha=1,zorder=5)
data2 = ax.fill_between(T,Qt_func,Qt_func*0,color='b',alpha = 0.5)

# Discharge proxy items
data3 = plt.plot([0,40000],[Qt[100]**m,Qt[100]**m],color='k',linestyle='--',linewidth=2,zorder=10)
data4 = plt.scatter(1e4,Qt[100]**m,s=100,c='k',zorder=20)
data6 = ax.fill_between([0,40000],[Qt[100]**m,Qt[100]**m],[0,0],color='k',alpha = 0.25)

ax.grid(False)
ax.set_xlim(0,40000)
ax.set_ylim(0,np.max(Qt_func)+0.1*np.max(Qt_func))
#ax.set_yticks([],[])
ax.set_facecolor((1, 1, 1))
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()

# Save figure
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/Bias_sensitivity_analysis/Qcurve_m03A1e5_axon', facecolor='white', edgecolor='none',dpi=500)


#%% Figure 2: Plot Stream Profiles

# Info about runs to remember
# 5 Myr: 1.2566e-6
# 2.5 Myr: 2.5133e-6
# 1 Myr: 6.2832e-6
# 500kyr: 1.2566e-5
# 200kyr: 3.1416e-5
# 100kyr: 6.2832e-5
# 40kyr: 1.5708e-4
# 20kyr: 3.1416e-4

# Define figure
fig=plt.figure(0,figsize=(10,10))
ax1 = plt.subplot2grid((4,2),(0,0), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((4,2),(1,0), colspan=1, rowspan=1)
ax3 = plt.subplot2grid((4,2),(2,0), colspan=1, rowspan=1)
ax4 = plt.subplot2grid((4,2),(0,1), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((4,2),(1,1), colspan=1, rowspan=1)
ax6 = plt.subplot2grid((4,2),(2,1), colspan=1, rowspan=1)
ax7 = plt.subplot2grid((4,2),(3,0), colspan=2, rowspan=1)


# Load model time step
# 1 and 8
# run_name = ['Run_1','Run_1','Run_1','Run_1',
#             'Run_8','Run_8','Run_8','Run_8']
# 10 and 17
run_name = ['Run_10','Run_10','Run_10','Run_10',
            'Run_17','Run_17','Run_17','Run_17',
            'Run_control']

times = [19980200,19985200,19990200,19995200,
         15000200,16250200,17500200,18755200,
         20000000] # years

# 1 and 8
# outlet_id = [14,14,14,14,
#              36,36,36,36]
# 10 and 17
outlet_id = [11,11,11,11,
             21,21,21,21,
             27]

# 1 and 8
# keys = [(14, 4848),(14, 4848),(14, 4848),(14, 4848),
#         (36, 4947),(36, 4947),(36, 4947),(36, 4947)]
# 10 and 17
keys = [(11, 4851),(11, 4851),(11, 4851),(11, 4851),
        (21, 4896),(21, 4896),(21, 4896),(21, 4896),
        (27, 4947)]

cr = 1/255*np.array([255,0,0])
cb = 1/255*np.array([0,0,255])

colors = [cr,cr,cr,cr,
          cb,cb,cb,cb,
          'k']
linestyles = ['-','--','-.',':',
              '-','--','-.',':',
              '-']
linewidths = [2,2,2,2,
              2,2,2,2,
              2]

zorders = [2,2,2,2,
           2,2,2,2,
           1]
    
# Start of loop
for i in range(len(times)):
            
    mg = read_netcdf('/Volumes/PhD_1/ksn_Bias/model_runs2/'+run_name[i]+'/output_frames/netcdf/'+str(int(times[i]))+'.nc')
    
    # Make sure flow indicator fields are type int
    mg.at_node['flow__link_to_receiver_node'] = mg.at_node['flow__link_to_receiver_node'].astype(int)
    mg.at_node['flow__receiver_node'] = mg.at_node['flow__receiver_node'].astype(int)
    mg.at_node['flow__upstream_node_order'] = mg.at_node['flow__upstream_node_order'].astype(int)
        
    # Extract largest stream from boundary to watershed divide
    profiler = ChannelProfiler(mg, minimum_channel_threshold=1)
    profiler.run_one_step()

    ids = profiler.data_structure[outlet_id[i]][keys[i]]["ids"]
    elevations = mg.at_node['topographic__elevation'][ids]
    distances = profiler.data_structure[outlet_id[i]][keys[i]]["distances"]
    erates = mg.at_node['erosion_rate'][ids]
    
    # Calculate ksn and chi
    sf = SteepnessFinder(mg, min_drainage_area=1)
    sf.calculate_steepnesses()
    ksn = sf.steepness_indices[ids]
    cf = ChiFinder(mg, min_drainage_area=1, reference_concavity=0.5)
    cf.calculate_chi()
    chis = mg.at_node['channel__chi_index'][ids]
    
    # Change drainage area to discharge for ksnQ and Chi-Q
    mg.at_node['drainage_area']=mg.at_node['surface_water__discharge']
    sfq = SteepnessFinder(mg, min_drainage_area=1)
    sfq.calculate_steepnesses()
    ksnq = sfq.steepness_indices[ids]
    cf.calculate_chi()
    chiqs = mg.at_node['channel__chi_index'][ids]

    # Plot
    ax1.plot(distances, elevations, color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i], zorder=zorders[i])
    ax2.plot(chis, elevations, color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i], zorder=zorders[i])
    ax3.plot(chiqs, elevations, color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i], zorder=zorders[i])
    ax4.plot(distances[2:], erates[2:]*1e6, color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i], zorder=zorders[i])
    ax5.plot(distances[2:], ksn[2:], color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i], zorder=zorders[i])
    ax6.plot(distances[2:], ksnq[2:], color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i], zorder=zorders[i])

# Plot erosion rates through time
run10_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_10/erosion_tracker.csv')
run17_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_17/erosion_tracker.csv')
runcontrol_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_Control/erosion_tracker.csv')
x_standard = np.linspace(0,1,101)
y_standard = 0.75*np.sin((2*np.pi)*x_standard)+1
y_10 = np.array(run10_erosion[99900:])/0.0002#np.max(np.array(run10_erosion[99900:]))
x_10 = np.linspace(0,1,y_10.shape[0])
y_17 = np.array(run17_erosion[75000:])/0.0002#np.max(np.array(run17_erosion[75000:]))
x_17 = np.linspace(0,1,y_17.shape[0])
y_control = np.array(runcontrol_erosion[15000:])/0.0002#np.max(np.array(runcontrol_erosion[15000:])) # Final 1 Myr for control
x_control = np.linspace(0,1,y_control.shape[0])

ax7.plot(x_standard,y_standard,color = 1/255*np.array([100,100,100]), linewidth = 1, zorder = 1)
ax7.scatter(x_standard,y_standard,color = 1/255*np.array([100,100,100]),marker = 'o', facecolors='none',zorder = 2)
ax7.plot(x_10,y_10,color = cr, linewidth = 2, linestyle = '-', zorder = 3)
ax7.plot(x_17,y_17,color = cb, linewidth = 2, linestyle = '-', zorder = 3)
ax7.plot(x_control,y_control,color = 'k', linewidth = 2, linestyle = '-', zorder = 2)


ax1.grid(False)
ax1.set_ylim(0,2500)
ax1.set_xlim([0,20000])
#ax1.set_xticks([],[])
#ax1.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Elevation (m)')
ax1.set_facecolor((1, 1, 1))
ax1.spines['bottom'].set_color('0')
ax1.spines['top'].set_color('0')
ax1.spines['right'].set_color('0')
ax1.spines['left'].set_color('0')

ax2.grid(False)
ax2.set_ylim(0,2500)
ax2.set_xlim([0,9])
#ax2.set_xticks([],[])
#ax2.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax2.set_xlabel('Chi')
ax2.set_ylabel('Elevation (m)')
ax2.set_facecolor((1, 1, 1))
ax2.spines['bottom'].set_color('0')
ax2.spines['top'].set_color('0')
ax2.spines['right'].set_color('0')
ax2.spines['left'].set_color('0')

ax3.grid(False)
ax3.set_ylim(0,2500)
ax3.set_xlim([0,15])
#ax3.set_xticks([],[])
#ax3.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax3.set_xlabel('Chi$_{q}$')
ax3.set_ylabel('Elevation (m)')
ax3.set_facecolor((1, 1, 1))
ax3.spines['bottom'].set_color('0')
ax3.spines['top'].set_color('0')
ax3.spines['right'].set_color('0')
ax3.spines['left'].set_color('0')

ax4.grid(False)
ax4.set_ylim(0,500)
ax4.set_xlim([0,20000])
#ax4.set_xticks([],[])
#ax4.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax4.set_xlabel('Distance (m)')
ax4.set_ylabel('Erosion Rate (m My$r^{-1}$)')
ax4.set_facecolor((1, 1, 1))
ax4.spines['bottom'].set_color('0')
ax4.spines['top'].set_color('0')
ax4.spines['right'].set_color('0')
ax4.spines['left'].set_color('0')

ax5.grid(False)
ax5.set_ylim(150,350)
ax5.set_xlim([0,20000])
#ax5.set_xticks([],[])
#ax5.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax5.set_xlabel('Distance (m)')
ax5.set_ylabel('$k_{sn}$ (m)')
ax5.set_facecolor((1, 1, 1))
ax5.spines['bottom'].set_color('0')
ax5.spines['top'].set_color('0')
ax5.spines['right'].set_color('0')
ax5.spines['left'].set_color('0')

ax6.grid(False)
ax6.set_ylim(100,300)
ax6.set_xlim([0,20000])
#ax6.set_xticks([],[])
#ax6.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax6.set_xlabel('Distance (m)')
ax6.set_ylabel('$k_{snQ}$ ($m^{1.5}$ y$r^{-0.5}$)')
ax6.set_facecolor((1, 1, 1))
ax6.spines['bottom'].set_color('0')
ax6.spines['top'].set_color('0')
ax6.spines['right'].set_color('0')
ax6.spines['left'].set_color('0')

ax7.grid(False)
ax.set_ylim(0,1)
ax7.set_xlim([0,1])
#ax7.set_xticks([],[])
#ax7.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax7.set_xlabel('Normalized Time')
ax7.set_ylabel('Normalized Mean Erosion Rate (m My$r^{-1}$)')
ax7.set_facecolor((1, 1, 1))
ax7.spines['bottom'].set_color('0')
ax7.spines['top'].set_color('0')
ax7.spines['right'].set_color('0')
ax7.spines['left'].set_color('0')

# Save figure
plt.tight_layout()
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/stream_profiles/runs10_17.png', facecolor='white', edgecolor='none',dpi=500)
plt.clf()


#%% Figure 2: Measure time to steady state

run1_erosion = np.array(pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_Control_n3_Length5_V2/erosion_tracker.csv'))

U = 200
threshold = 0.99
dt = 1000
wavelength = 20000


for i in range(len(run1_erosion)):
    print(i)
    erosion_mean = np.mean(np.array(run1_erosion)[i:i+(int(wavelength/dt))])
    
    if erosion_mean*1e6 > U*threshold and erosion_mean*1e6 < U+(U*(1-threshold)):
        print('Time to Steady State: ' + str(i*dt))
        break
    else:
        pass


#%% Figure 2: Compute RMSE for model runs with respect to prediction

def calc_ksn_bias(Qt,Ps,DAs):
    # Assign variables
    U = 0.0002 # Uplift; m/yr
    K = 2.5e-11 # Erosional efficiency;
    T_start = 0 # years
    T_end = 2500000 # years
    num_steps = 12500
    T = np.linspace(T_start,T_end,num_steps)
    n = 3.0
    m = 1.5
    Qt = Qt # Discharge time series; m^3/yr
    
    # Numerically integrate discharge function over climate cycle.
    Qt_integral = integrate.trapezoid((Qt**m),T)
    
    # Calculate Qeff.`
    Qeff = ((1/T_end)*Qt_integral)**(1/m)
    
    # Solve for ratios of ksnq and ksnp to ksnqeff using analytical solutions in text
    ksn_bias = ((T_end*(DAs**m))/(Qt_integral))**(1/n)
    ksnp_bias = ((T_end*((Ps*DAs)**m))/(Qt_integral))**(1/n)
    ksnq_bias = ((T_end*(Qt**m))/(Qt_integral))**(1/n)
    
    return ksn_bias,ksnp_bias,ksnq_bias

num_steps = 12500

Qts = np.zeros((num_steps,50,25))
DAs = np.zeros((num_steps,50,25))
Ps = np.zeros((num_steps,50,25))
ksns = np.zeros((num_steps,50,25))
ksnps = np.zeros((num_steps,50,25))
ksnqs = np.zeros((num_steps,50,25))
chis = np.zeros((num_steps,50,25))

run_name = 'Run_36'
#times = np.linspace(15980200,16000000,num_steps)
times = np.linspace(11500200,14000000,num_steps)

# Iterate through time steps and assemble discharge time series
for i in range(len(times)):
    
    print(i)

    mg = read_netcdf('/Volumes/PhD_1/ksn_Bias/model_runs2/'+run_name+'/output_frames/netcdf/'+str(int((times[i])))+'.nc')
        
    Qts[i,:,:] = np.reshape(mg.at_node['surface_water__discharge'],(mg.shape[0],mg.shape[1]))
    DAs[i,:,:] = np.reshape(mg.at_node['drainage_area'],(mg.shape[0],mg.shape[1]))
    Ps[i,:,:] = np.reshape(mg.at_node['surface_water__discharge']/mg.at_node['drainage_area'],(mg.shape[0],mg.shape[1]))

    # Make sure flow indicator fields are type int
    mg.at_node['flow__link_to_receiver_node'] = mg.at_node['flow__link_to_receiver_node'].astype(int)
    mg.at_node['flow__receiver_node'] = mg.at_node['flow__receiver_node'].astype(int)
    mg.at_node['flow__upstream_node_order'] = mg.at_node['flow__upstream_node_order'].astype(int)

    # Calculate Channel Steepness
    sf = SteepnessFinder(mg, min_drainage_area=1e5, reference_concavity=0.5)
    sf.calculate_steepnesses()
    ksns[i,:,:] = np.reshape(sf.steepness_indices,(mg.shape[0],mg.shape[1]))
    
    # Calculate chi as a proxy for upstream distance
    cf = ChiFinder(mg, min_drainage_area=1e5, reference_concavity=0.5)
    cf.calculate_chi()
    chis[i,:,:] = np.reshape(mg.at_node['channel__chi_index'],(mg.shape[0],mg.shape[1])) 
    
    # Calculate ksnq by replacing drainage_area with surface_water__discharge
    da_field = np.copy(mg.at_node['drainage_area'])
    mg.at_node['drainage_area']=mg.at_node['surface_water__discharge']
    # Discharge normalized channel Steepness. Streams initiated at 1e5 m^2 drainage area
    sfq = SteepnessFinder(mg, min_drainage_area=1e5, reference_concavity=0.5)
    sfq.calculate_steepnesses()
    # Return the drainage_area field to drainage area
    mg.at_node['drainage_area']=da_field
    ksnqs[i,:,:] = np.reshape(sfq.steepness_indices,(mg.shape[0],mg.shape[1]))

N = 0 # Track number of measurements
SSE_ksn = 0 # Track sum of squared error
SSE_ksnq = 0 # Track sum of squared error

for j in np.linspace(2,49,48): # You want to avoid rows adjacent to open boundaries
    j = int(j)
    print(j)
    for k in np.linspace(1,23,23):
        k = int(k)
        
        if chis[0,j,k]>0.0:
            
            # Account for nodes where streams don't always occur (these need to be removed)
            if np.sum(np.where(ksnqs[:,j,k]==0))==0:
                
                N += int(len(Qts[:,j,k])) # Update sample size
                
                # Define color based on upstream distance. Just an option.
                c = (1/255)*np.array([(chis[0,j,k]/np.max(chis[0,:,:]))*255,0,0])
                
                # Compare biases predicted by analytical solutions
                ksn_bias,ksnp_bias,ksnq_bias = calc_ksn_bias(Qts[:,j,k],Ps[:,j,k],DAs[:,j,k])
                ksns_true = ksns[:,j,k]
                ksnps_true = ksnps[:,j,k]
                ksnqs_true = ksnqs[:,j,k]
                ksns_predicted = 200*ksn_bias
                ksnps_predicted = 200*ksnp_bias
                ksnqs_predicted = 200*ksnq_bias
                            
                sse_ksn = np.sum((ksns_predicted-ksns_true)**2)
                SSE_ksn += sse_ksn
                sse_ksnq = np.sum((ksnqs_predicted-ksnqs_true)**2)
                SSE_ksnq += sse_ksnq

                print(sse_ksn)
            else:
                pass
           
        else:
            pass

RMSE_ksn = (SSE_ksn/N)**(0.5)
RMSE_ksnq = (SSE_ksnq/N)**(0.5)
        
# Save figure
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/model_validation/Run_2_Validation_ksn', facecolor='white', edgecolor='none',dpi=500)


#%% Figure 2: RMSE Plots with x-axis equal to Period/Response Time
sns.set_style('ticks')

#color_n1 = 1/255*np.array([255,153,51])
#color_n3 = 1/255*np.array([153,51,255])

cmap = plt.cm.get_cmap('magma_r')

# ksn
# Runs with 10x5 km domain and n=1
RMSE_n1 = np.array([1.37,2.41,2.29,2.8,3.93,5.55,10.35,15.25])/200
normPeriod_n1 = np.array([0.001935921,0.003871842,0.009679605,0.01935921,
                          0.048398025,0.096796051,0.241990127,0.483980254])
color_n1 = np.log(np.array([0.02,0.04,0.1,0.2,0.5,1,2.5,5])/0.02)
color_n1 = color_n1/(np.log(5/0.02))
color_n1 = [int(i*(len(cmap.colors)-1)) for i in color_n1]
color_n1 = [cmap.colors[i] for i in color_n1]

# Runs with 10x5 km domain and n=3
RMSE_n3 = np.array([0.19,0.42,1.61,2.78,6.62,10.14,15.09,17.93])/200
normPeriod_n3 = np.array([0.002103049,0.004206099,0.010515247,0.021030494,
                       0.052576236,0.105152471,0.262881178,0.525762355])
color_n3 = np.log(np.array([0.02,0.04,0.1,0.2,0.5,1,2.5,5])/0.02)
color_n3 = color_n3/(np.log(5/0.02))
color_n3 = [int(i*(len(cmap.colors)-1)) for i in color_n3]
color_n3 = [cmap.colors[i] for i in color_n3]

# Runs with 5x2.5 km domain
RMSE_Length5 = np.array([0.32,2.77,18.61])/200
normPeriod_Length5 = np.array([0.00464091,0.023204548,0.290056851])
color_n3_Length5 = np.log(np.array([0.04,0.2,2.5])/0.02)
color_n3_Length5 = color_n3_Length5/(np.log(5/0.02))
color_n3_Length5 = [int(i*(len(cmap.colors)-1)) for i in color_n3_Length5]
color_n3_Length5 = [cmap.colors[i] for i in color_n3_Length5]

# Runs with 20x10 km domain
RMSE_Length20 = np.array([2.83])/200
normPeriod_Length20 = np.array([0.016091399])
color_n3_Length20 = np.log(np.array([0.2])/0.02)
color_n3_Length20 = color_n3_Length20/(np.log(5/0.02))
color_n3_Length20 = [int(i*(len(cmap.colors)-1)) for i in color_n3_Length20]
color_n3_Length20 = [cmap.colors[i] for i in color_n3_Length20]

# Runs with higher K and ksnQeff=100
RMSE_HighK = np.array([0,0.33,2.59,6.64])/100
normPeriod_HighK = np.array([0,0.003940887,0.039408867,0.197044335])
color_n3_HighK = np.log(np.array([1,0.02,0.2,1])/0.02)
color_n3_HighK = color_n3_HighK/(np.log(5/0.02))
color_n3_HighK = [int(i*(len(cmap.colors)-1)) for i in color_n3_HighK]
color_n3_HighK = [cmap.colors[i] for i in color_n3_HighK]

# Runs with 0.125 discharge amplitude and ksnQeff=200
RMSE_LowA = np.array([0.08,3.56,8.47,9.25,9.92])/200
normPeriod_LowA = np.array([0.002103049,0.010515247,0.052576236,0.105152471,0.525762355])
color_n3_LowA = np.log(np.array([0.02,0.1,0.5,1,5])/0.02)
color_n3_LowA = color_n3_LowA/(np.log(5/0.02))
color_n3_LowA = [int(i*(len(cmap.colors)-1)) for i in color_n3_LowA]
color_n3_LowA = [cmap.colors[i] for i in color_n3_LowA]


fig,ax = plt.subplots(figsize = (6,4))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)

plt.scatter(normPeriod_n3,RMSE_n3,s=300,marker='s',c = color_n3,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_n1,RMSE_n1,s=400,marker='H',c = color_n1,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_Length5,RMSE_Length5,s=100,marker='s',c = color_n3_Length5,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_Length20,RMSE_Length20,s=500,marker='s',c = color_n3_Length20,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_HighK[1:],RMSE_HighK[1:],s=300,marker='o',c = color_n3_HighK[1:],linewidth=1,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_LowA,RMSE_LowA,s=400,marker='v',c = color_n3_LowA,edgecolor='k',alpha=0.75)


plt.scatter(np.array([0.002103049,0.525762355]),np.array([0.19,17.93])/200,s=300,marker = 's',edgecolor=[1/255*np.array([255,0,0]),1/255*np.array([0,0,255])],facecolors='none',linewidth=3,alpha=1) # Highlighted data point


ax.grid(True)
#ax.set_xlim(0,5.8)
#ax.set_ylim(0,20)
ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_xticks([],[])
ax.set_facecolor((1, 1, 1))
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()

# Save figure
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/model_validation/Runs2/ksn', facecolor='white', edgecolor='none',dpi=500)



# ksnq
# Runs with 10x5 km domain and n=1
RMSE_n1 = np.array([1.53,2.62,1.82,2.32,4.28,5.61,10.63,14.63])/200
normPeriod_n1 = np.array([0.001935921,0.003871842,0.009679605,0.01935921,
                          0.048398025,0.096796051,0.241990127,0.483980254])
color_n1 = np.log(np.array([0.02,0.04,0.1,0.2,0.5,1,2.5,5])/0.02)
color_n1 = color_n1/(np.log(5/0.02))
color_n1 = [int(i*(len(cmap.colors)-1)) for i in color_n1]
color_n1 = [cmap.colors[i] for i in color_n1]

# Runs with 10x5 km domain and n=3
RMSE_n3 = np.array([0.19,0.46,1.63,3.07,6.96,10.71,15.08,16.64])/200
normPeriod_n3 = np.array([0.002103049,0.004206099,0.010515247,0.021030494,
                       0.052576236,0.105152471,0.262881178,0.525762355])
color_n3 = np.log(np.array([0.02,0.04,0.1,0.2,0.5,1,2.5,5])/0.02)
color_n3 = color_n3/(np.log(5/0.02))
color_n3 = [int(i*(len(cmap.colors)-1)) for i in color_n3]
color_n3 = [cmap.colors[i] for i in color_n3]

# Runs with 5x2.5 km domain
RMSE_Length5 = np.array([0.34,2.91,17.1])/200
normPeriod_Length5 = np.array([0.00464091,0.023204548,0.290056851])
color_n3_Length5 = np.log(np.array([0.04,0.2,2.5])/0.02)
color_n3_Length5 = color_n3_Length5/(np.log(5/0.02))
color_n3_Length5 = [int(i*(len(cmap.colors)-1)) for i in color_n3_Length5]
color_n3_Length5 = [cmap.colors[i] for i in color_n3_Length5]

# Runs with 20x10 km domain
RMSE_Length20 = np.array([3.02])/200
normPeriod_Length20 = np.array([0.016091399])
color_n3_Length20 = np.log(np.array([0.2])/0.02)
color_n3_Length20 = color_n3_Length20/(np.log(5/0.02))
color_n3_Length20 = [int(i*(len(cmap.colors)-1)) for i in color_n3_Length20]
color_n3_Length20 = [cmap.colors[i] for i in color_n3_Length20]

# Runs with higher K and ksnQeff=100
RMSE_HighK = np.array([0,0.3,2.81,6.82])/100
normPeriod_HighK = np.array([0,0.003940887,0.039408867,0.197044335])
color_n3_HighK = np.log(np.array([1,0.02,0.2,1])/0.02)
color_n3_HighK = color_n3_HighK/(np.log(5/0.02))
color_n3_HighK = [int(i*(len(cmap.colors)-1)) for i in color_n3_HighK]
color_n3_HighK = [cmap.colors[i] for i in color_n3_HighK]

# Runs with 0.125 discharge amplitude and ksnQeff=200
RMSE_LowA = np.array([0.08,3.04,7.77,5.72,8.96])/200
normPeriod_LowA = np.array([0.002103049,0.010515247,0.052576236,0.105152471,0.525762355])
color_n3_LowA = np.log(np.array([0.02,0.1,0.5,1,5])/0.02)
color_n3_LowA= color_n3_LowA/(np.log(5/0.02))
color_n3_LowA = [int(i*(len(cmap.colors)-1)) for i in color_n3_LowA]
color_n3_LowA = [cmap.colors[i] for i in color_n3_LowA]

fig,ax = plt.subplots(figsize = (6,4))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)

plt.scatter(normPeriod_n3,RMSE_n3,s=300,marker='s',c = color_n3,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_n1,RMSE_n1,s=400,marker='H',c = color_n1,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_Length5,RMSE_Length5,s=100,marker='s',c = color_n3_Length5,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_Length20,RMSE_Length20,s=500,marker='s',c = color_n3_Length20,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_HighK[1:],RMSE_HighK[1:],s=300,marker='o',c = color_n3_HighK[1:],linewidth=1,edgecolor='k',alpha=0.75)
plt.scatter(normPeriod_LowA,RMSE_LowA,s=400,marker='v',c = color_n3_LowA,edgecolor='k',alpha=0.75)

plt.scatter(np.array([0.002103049,0.525762355]),np.array([0.19,16.64])/200,s=300,marker = 's',edgecolor=[1/255*np.array([255,0,0]),1/255*np.array([0,0,255])],facecolors='none',linewidth=3,alpha=1) # Highlighted data point


ax.grid(True)
#ax.set_xlim(0,5.8)
#ax.set_ylim(0,20)
ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_xticks([],[])
ax.set_facecolor((1, 1, 1))
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()

# Save figure
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/model_validation/Runs2/ksnq', facecolor='white', edgecolor='none',dpi=500)


#%% Figure 3: Time step panels

run_name = 'Run_6_NoDiff_U_P_gradient'
time = 11970000
mg = read_netcdf('/Volumes/PhD_1/ksn_Bias/model_runs/'+run_name+'/output_frames/netcdf/'+str(int(time))+'.nc')

# Make sure flow indicator fields are type int
mg.at_node['flow__link_to_receiver_node'] = mg.at_node['flow__link_to_receiver_node'].astype(int)
mg.at_node['flow__receiver_node'] = mg.at_node['flow__receiver_node'].astype(int)
mg.at_node['flow__upstream_node_order'] = mg.at_node['flow__upstream_node_order'].astype(int)

# Calculate Channel Steepness
sf = SteepnessFinder(mg, min_drainage_area=1e5)
sf.calculate_steepnesses()

# Calculate ksnq by replacing drainage_area with surface_water__discharge
da_field = np.copy(mg.at_node['drainage_area'])
mg.at_node['drainage_area']=mg.at_node['surface_water__discharge']
# Discharge normalized channel Steepness. Streams initiated at 1e5 m^2 drainage area
sfq = SteepnessFinder(mg, min_drainage_area=1e5)
sfq.calculate_steepnesses()
# Return the drainage_area field to drainage area
mg.at_node['drainage_area']=da_field

#Plot channel steepness on elevation and save png
imshow_grid(mg, 
            mg.at_node['topographic__elevation'], 
            at='node',
            allow_colorbar=False, 
            limits=(0,2500), 
            cmap = 'gray')
imshow_grid(mg, 
            sfq.masked_steepness_indices, 
            at='node',
            color_for_closed=None, 
            cmap='jet', 
            plot_name='Time: 0 kyr', 
            limits = (125,275), 
            colorbar_label = '$k_{snq}$ [m]')

# plot trunk indices
#trunk_y = np.round(trunk_indices/80)
#trunk_x = trunk_y%80
#plt.scatter(trunk_x*100,trunk_y*100,s=200)

plt.savefig('/Volumes/PhD_1/ksn_Bias/Figures/Example_models/Scen4_ksnq.png')
plt.clf()


#%% Figure 3: Plot sine curves for 'example_models' figure

# Equal amplitude runoff change
T = np.linspace(0,40000,400)
Qt_amp = 0.25
Qt_mean = 0.25
Qt = Qt_amp*np.sin(1.5708e-4*(T+20000))+Qt_mean


# Plot predicted versus modeled channel steepness values
fig,ax = plt.subplots(figsize = (4,10))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)

# Discharge curve
data2 = plt.plot(T,Qt+0.25,color='k',linestyle='-',linewidth=2,alpha=1,zorder=5)
data4 = plt.plot(T,Qt+0.75,color='k',linestyle='-',linewidth=2,alpha=1,zorder=5)
data6 = plt.plot(T,Qt+1.25,color='k',linestyle='-',linewidth=2,alpha=1,zorder=5)
data8 = plt.plot(T,Qt+1.75,color='k',linestyle='-',linewidth=2,alpha=1,zorder=5)


ax.grid(False)
ax.set_xlim(0,40000)
ax.set_ylim(0,2.26)
ax.set_facecolor((1, 1, 1))
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()

# Save figure
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/Example_Models/sine_EqualAmp', facecolor='white', edgecolor='none',dpi=500)


# Equal fraction runoff change
# Plot predicted versus modeled channel steepness values
fig,ax = plt.subplots(figsize = (4,10))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)

# Discharge curve
# Equal amplitude runoff change. 0.375, 0.75, 1.125, 1.5
T = np.linspace(0,40000,400)
Qt_amp = 0.401
Qt_mean = 0.5
Qt = Qt_amp*np.sin(1.5708e-4*(T+20000))+Qt_mean
data1 = plt.plot(T,Qt,color='k',linestyle='-',linewidth=2,alpha=1,zorder=5)

T = np.linspace(0,40000,400)
Qt_amp = 0.3125
Qt_mean = 1
Qt = Qt_amp*np.sin(1.5708e-4*(T+20000))+Qt_mean
data2 = plt.plot(T,Qt,color='k',linestyle='-',linewidth=2,alpha=1,zorder=5)

T = np.linspace(0,40000,400)
Qt_amp = 0.21875
Qt_mean = 1.5
Qt = Qt_amp*np.sin(1.5708e-4*(T+20000))+Qt_mean
data3 = plt.plot(T,Qt,color='k',linestyle='-',linewidth=2,alpha=1,zorder=5)

T = np.linspace(0,40000,400)
Qt_amp = 0.125
Qt_mean = 2
Qt = Qt_amp*np.sin(1.5708e-4*(T+20000))+Qt_mean
data4 = plt.plot(T,Qt,color='k',linestyle='-',linewidth=2,alpha=1,zorder=5)


ax.grid(False)
ax.set_xlim(0,40000)
ax.set_ylim(0,2.26)
ax.set_facecolor((1, 1, 1))
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()

# Save figure
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/Example_Models/sine_EqualFrac', facecolor='white', edgecolor='none',dpi=500)


#%% Figure S1: Erosion_tracker plots

#n1_color = 1/255*np.array([255,153,51])
#n3_color = 1/255*np.array([153,51,255])

# For alternative figure version
#n1_color = 1/255*np.array([255,0,0])
#n3_color = 1/255*np.array([0,0,255])

n1_color = 1/255*np.array([0,0,0])
n3_color = 1/255*np.array([0,0,0])

time10 = np.linspace(200,20000000,49999)
time12 = np.linspace(200,12000000,59999)
time14 = np.linspace(200,14000000,69999)
time16 = np.linspace(200,14000000,79999)
time20 = np.linspace(200,20000000,99999)
time25 = np.linspace(200,25000000,124999)



run1_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_1/erosion_tracker.csv')
run2_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_2/erosion_tracker.csv')
run3_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_3/erosion_tracker.csv')
run4_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_4/erosion_tracker.csv')
run5_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_5/erosion_tracker.csv')
run6_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_6/erosion_tracker.csv')
run7_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_7/erosion_tracker.csv')
run8_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_8/erosion_tracker.csv')
run10_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_10/erosion_tracker.csv')
run11_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_11/erosion_tracker.csv')
run12_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_12/erosion_tracker.csv')
run13_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_13/erosion_tracker.csv')
run14_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_14/erosion_tracker.csv')
run15_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_15/erosion_tracker.csv')
run16_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_16/erosion_tracker.csv')
run17_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_17/erosion_tracker.csv')
run31_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_31/erosion_tracker.csv')
run33_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_33/erosion_tracker.csv')
run36_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_36/erosion_tracker.csv')
run43_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_43/erosion_tracker.csv')
run50_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_50/erosion_tracker.csv')
run53_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_53/erosion_tracker.csv')
run55_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_55/erosion_tracker.csv')
run60_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_60/erosion_tracker.csv')
run62_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_62/erosion_tracker.csv')
run64_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_64/erosion_tracker.csv')
run65_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_65/erosion_tracker.csv')
run67_erosion = pd.read_csv('/Volumes/PhD_1/ksn_Bias/model_runs2/Run_67/erosion_tracker.csv')

fig=plt.figure(0,figsize=(10,10))
ax1 = plt.subplot2grid((7,4),(0,0), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((7,4),(0,1), colspan=1, rowspan=1)
ax3 = plt.subplot2grid((7,4),(0,2), colspan=1, rowspan=1)
ax4 = plt.subplot2grid((7,4),(0,3), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((7,4),(1,0), colspan=1, rowspan=1)
ax6 = plt.subplot2grid((7,4),(1,1), colspan=1, rowspan=1)
ax7 = plt.subplot2grid((7,4),(1,2), colspan=1, rowspan=1)
ax8 = plt.subplot2grid((7,4),(1,3), colspan=1, rowspan=1)
ax10 = plt.subplot2grid((7,4),(2,0), colspan=1, rowspan=1)
ax11 = plt.subplot2grid((7,4),(2,1), colspan=1, rowspan=1)
ax12 = plt.subplot2grid((7,4),(2,2), colspan=1, rowspan=1)
ax13 = plt.subplot2grid((7,4),(2,3), colspan=1, rowspan=1)
ax14 = plt.subplot2grid((7,4),(3,0), colspan=1, rowspan=1)
ax15 = plt.subplot2grid((7,4),(3,1), colspan=1, rowspan=1)
ax16 = plt.subplot2grid((7,4),(3,2), colspan=1, rowspan=1)
ax17 = plt.subplot2grid((7,4),(3,3), colspan=1, rowspan=1)
ax31 = plt.subplot2grid((7,4),(4,0), colspan=1, rowspan=1)
ax33 = plt.subplot2grid((7,4),(4,1), colspan=1, rowspan=1)
ax36 = plt.subplot2grid((7,4),(4,2), colspan=1, rowspan=1)
ax43 = plt.subplot2grid((7,4),(4,3), colspan=1, rowspan=1)
ax50 = plt.subplot2grid((7,4),(5,0), colspan=1, rowspan=1)
ax53 = plt.subplot2grid((7,4),(5,1), colspan=1, rowspan=1)
ax55 = plt.subplot2grid((7,4),(5,2), colspan=1, rowspan=1)
ax60 = plt.subplot2grid((7,4),(5,3), colspan=1, rowspan=1)
ax62 = plt.subplot2grid((7,4),(6,0), colspan=1, rowspan=1)
ax64 = plt.subplot2grid((7,4),(6,1), colspan=1, rowspan=1)
ax65 = plt.subplot2grid((7,4),(6,2), colspan=1, rowspan=1)
ax67 = plt.subplot2grid((7,4),(6,3), colspan=1, rowspan=1)


ax1.plot(time20,run1_erosion,color=n1_color)
ax1.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax1.grid(False)
ax1.set_ylim(0,0.00031)
ax1.set_xlim([0,2e7])
ax1.set_xticks([],[])
ax1.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_facecolor((1, 1, 1))
ax1.spines['bottom'].set_color('0')
ax1.spines['top'].set_color('0')
ax1.spines['right'].set_color('0')
ax1.spines['left'].set_color('0')

ax2.plot(time20,run2_erosion,color=n1_color)
ax2.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax2.grid(False)
ax2.set_ylim(0,0.00031)
ax2.set_xlim([0,2e7])
ax2.set_xticks([],[])
ax2.set_yticks([],[])
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_facecolor((1, 1, 1))
ax2.spines['bottom'].set_color('0')
ax2.spines['top'].set_color('0')
ax2.spines['right'].set_color('0')
ax2.spines['left'].set_color('0')

ax3.plot(time20,run3_erosion,color=n1_color)
ax3.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax3.grid(False)
ax3.set_ylim(0,0.00031)
ax3.set_xlim([0,2e7])
ax3.set_xticks([],[])
ax3.set_yticks([],[])
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_facecolor((1, 1, 1))
ax3.spines['bottom'].set_color('0')
ax3.spines['top'].set_color('0')
ax3.spines['right'].set_color('0')
ax3.spines['left'].set_color('0')

ax4.plot(time20,run4_erosion,color=n1_color)
ax4.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax4.grid(False)
ax4.set_ylim(0,0.00031)
ax4.set_xlim([0,2e7])
ax4.set_xticks([],[])
ax4.set_yticks([],[])
ax4.set_xlabel('')
ax4.set_ylabel('')
ax4.set_facecolor((1, 1, 1))
ax4.spines['bottom'].set_color('0')
ax4.spines['top'].set_color('0')
ax4.spines['right'].set_color('0')
ax4.spines['left'].set_color('0')

ax5.plot(time20,run5_erosion,color=n1_color)
ax5.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax5.grid(False)
ax5.set_ylim(0,0.00031)
ax5.set_xlim([0,2e7])
ax5.set_xticks([],[])
ax5.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax5.set_xlabel('')
ax5.set_ylabel('')
ax5.set_facecolor((1, 1, 1))
ax5.spines['bottom'].set_color('0')
ax5.spines['top'].set_color('0')
ax5.spines['right'].set_color('0')
ax5.spines['left'].set_color('0')

ax6.plot(time20,run6_erosion,color=n1_color)
ax6.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax6.grid(False)
ax6.set_ylim(0,0.00031)
ax6.set_xlim([0,2e7])
ax6.set_xticks([],[])
ax6.set_yticks([],[])
ax6.set_xlabel('')
ax6.set_ylabel('')
ax6.set_facecolor((1, 1, 1))
ax6.spines['bottom'].set_color('0')
ax6.spines['top'].set_color('0')
ax6.spines['right'].set_color('0')
ax6.spines['left'].set_color('0')

ax7.plot(time20,run7_erosion,color=n1_color)
ax7.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax7.grid(False)
ax7.set_ylim(0,0.00031)
ax7.set_xlim([0,2e7])
ax7.set_xticks([],[])
ax7.set_yticks([],[])
ax7.set_xlabel('')
ax7.set_ylabel('')
ax7.set_facecolor((1, 1, 1))
ax7.spines['bottom'].set_color('0')
ax7.spines['top'].set_color('0')
ax7.spines['right'].set_color('0')
ax7.spines['left'].set_color('0')

ax8.plot(time20,run8_erosion,color=n1_color)
ax8.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax8.grid(False)
ax8.set_ylim(0,0.00031)
ax8.set_xlim([0,2e7])
ax8.set_xticks([],[])
ax8.set_yticks([],[])
ax8.set_xlabel('')
ax8.set_ylabel('')
ax8.set_facecolor((1, 1, 1))
ax8.spines['bottom'].set_color('0')
ax8.spines['top'].set_color('0')
ax8.spines['right'].set_color('0')
ax8.spines['left'].set_color('0')

ax10.plot(time20,run10_erosion,color=n1_color,linewidth = 1)
ax10.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax10.grid(False)
ax10.set_ylim(0,0.00031)
ax10.set_xlim([0,2e7])
ax10.set_xticks([],[])
ax10.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax10.set_xlabel('')
ax10.set_ylabel('')
ax10.set_facecolor((1, 1, 1))
ax10.spines['bottom'].set_color('0')
ax10.spines['top'].set_color('0')
ax10.spines['right'].set_color('0')
ax10.spines['left'].set_color('0')

ax11.plot(time20,run11_erosion,color=n3_color)
ax11.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax11.grid(False)
ax11.set_ylim(0,0.00031)
ax11.set_xlim([0,2e7])
ax11.set_xticks([],[])
ax11.set_yticks([],[])
ax11.set_xlabel('')
ax11.set_ylabel('')
ax11.set_facecolor((1, 1, 1))
ax11.spines['bottom'].set_color('0')
ax11.spines['top'].set_color('0')
ax11.spines['right'].set_color('0')
ax11.spines['left'].set_color('0')

ax12.plot(time20,run12_erosion,color=n3_color)
ax12.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax12.grid(False)
ax12.set_ylim(0,0.00031)
ax12.set_xlim([0,2e7])
ax12.set_xticks([],[])
ax12.set_yticks([],[])
ax12.set_xlabel('')
ax12.set_ylabel('')
ax12.set_facecolor((1, 1, 1))
ax12.spines['bottom'].set_color('0')
ax12.spines['top'].set_color('0')
ax12.spines['right'].set_color('0')
ax12.spines['left'].set_color('0')

ax13.plot(time20,run13_erosion,color=n3_color)
ax13.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax13.grid(False)
ax13.set_ylim(0,0.00031)
ax13.set_xlim([0,2e7])
ax13.set_xticks([],[])
ax13.set_yticks([],[])
ax13.set_xlabel('')
ax13.set_ylabel('')
ax13.set_facecolor((1, 1, 1))
ax13.spines['bottom'].set_color('0')
ax13.spines['top'].set_color('0')
ax13.spines['right'].set_color('0')
ax13.spines['left'].set_color('0')

ax14.plot(time20,run14_erosion,color=n3_color)
ax14.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax14.grid(False)
ax14.set_ylim(0,0.00031)
ax14.set_xlim([0,2e7])
ax14.set_xticks([],[])
ax14.set_yticks([0,0.0001,0.0002,0.0003],['0','100','200','300'])
ax14.set_xlabel('')
ax14.set_ylabel('')
ax14.set_facecolor((1, 1, 1))
ax14.spines['bottom'].set_color('0')
ax14.spines['top'].set_color('0')
ax14.spines['right'].set_color('0')
ax14.spines['left'].set_color('0')

ax15.plot(time20,run15_erosion,color=n3_color)
ax15.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax15.grid(False)
ax15.set_ylim(0,0.00031)
ax15.set_xlim([0,2e7])
ax15.set_xticks([],[])
ax15.set_yticks([],[])
ax15.set_xlabel('')
ax15.set_ylabel('')
ax15.set_facecolor((1, 1, 1))
ax15.spines['bottom'].set_color('0')
ax15.spines['top'].set_color('0')
ax15.spines['right'].set_color('0')
ax15.spines['left'].set_color('0')

ax16.plot(time20,run16_erosion,color=n3_color)
ax16.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax16.grid(False)
ax16.set_ylim(0,0.00031)
ax16.set_xlim([0,2e7])
ax16.set_xticks([],[])
ax16.set_yticks([],[])
ax16.set_xlabel('')
ax16.set_ylabel('')
ax16.set_facecolor((1, 1, 1))
ax16.spines['bottom'].set_color('0')
ax16.spines['top'].set_color('0')
ax16.spines['right'].set_color('0')
ax16.spines['left'].set_color('0')

ax17.plot(time20,run17_erosion,color=n3_color)
ax17.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax17.grid(False)
ax17.set_ylim(0,0.00031)
ax17.set_xlim([0,2e7])
ax17.set_xticks([],[])
ax17.set_yticks([],[])
ax17.set_xlabel('')
ax17.set_ylabel('')
ax17.set_facecolor((1, 1, 1))
ax17.spines['bottom'].set_color('0')
ax17.spines['top'].set_color('0')
ax17.spines['right'].set_color('0')
ax17.spines['left'].set_color('0')

ax31.plot(time14,run31_erosion,color=n3_color)
ax31.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax31.grid(False)
ax31.set_ylim(0,0.00031)
ax31.set_xlim([0,2e7])
ax31.set_xticks([],[])
ax31.set_yticks([],[])
ax31.set_xlabel('')
ax31.set_ylabel('')
ax31.set_facecolor((1, 1, 1))
ax31.spines['bottom'].set_color('0')
ax31.spines['top'].set_color('0')
ax31.spines['right'].set_color('0')
ax31.spines['left'].set_color('0')

ax33.plot(time14,run33_erosion,color=n3_color)
ax33.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax33.grid(False)
ax33.set_ylim(0,0.00031)
ax33.set_xlim([0,2e7])
ax33.set_xticks([],[])
ax33.set_yticks([],[])
ax33.set_xlabel('')
ax33.set_ylabel('')
ax33.set_facecolor((1, 1, 1))
ax33.spines['bottom'].set_color('0')
ax33.spines['top'].set_color('0')
ax33.spines['right'].set_color('0')
ax33.spines['left'].set_color('0')

ax36.plot(time14,run36_erosion,color=n3_color)
ax36.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax36.grid(False)
ax36.set_ylim(0,0.00031)
ax36.set_xlim([0,2e7])
ax36.set_xticks([],[])
ax36.set_yticks([],[])
ax36.set_xlabel('')
ax36.set_ylabel('')
ax36.set_facecolor((1, 1, 1))
ax36.spines['bottom'].set_color('0')
ax36.spines['top'].set_color('0')
ax36.spines['right'].set_color('0')
ax36.spines['left'].set_color('0')

ax43.plot(time25,run43_erosion,color=n3_color)
ax43.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax43.grid(False)
ax43.set_ylim(0,0.00031)
ax43.set_xlim([0,2e7])
ax43.set_xticks([],[])
ax43.set_yticks([],[])
ax43.set_xlabel('')
ax43.set_ylabel('')
ax43.set_facecolor((1, 1, 1))
ax43.spines['bottom'].set_color('0')
ax43.spines['top'].set_color('0')
ax43.spines['right'].set_color('0')
ax43.spines['left'].set_color('0')

ax50.plot(time12,run50_erosion,color=n3_color)
ax50.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax50.grid(False)
ax50.set_ylim(0,0.00031)
ax50.set_xlim([0,2e7])
ax50.set_xticks([],[])
ax50.set_yticks([],[])
ax50.set_xlabel('')
ax50.set_ylabel('')
ax50.set_facecolor((1, 1, 1))
ax50.spines['bottom'].set_color('0')
ax50.spines['top'].set_color('0')
ax50.spines['right'].set_color('0')
ax50.spines['left'].set_color('0')

ax53.plot(time20,run53_erosion,color=n3_color)
ax53.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax53.grid(False)
ax53.set_ylim(0,0.00031)
ax53.set_xlim([0,2e7])
ax53.set_xticks([],[])
ax53.set_yticks([],[])
ax53.set_xlabel('')
ax53.set_ylabel('')
ax53.set_facecolor((1, 1, 1))
ax53.spines['bottom'].set_color('0')
ax53.spines['top'].set_color('0')
ax53.spines['right'].set_color('0')
ax53.spines['left'].set_color('0')

ax55.plot(time10,run55_erosion,color=n3_color)
ax55.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax55.grid(False)
ax55.set_ylim(0,0.00031)
ax55.set_xlim([0,2e7])
ax55.set_xticks([],[])
ax55.set_yticks([],[])
ax55.set_xlabel('')
ax55.set_ylabel('')
ax55.set_facecolor((1, 1, 1))
ax55.spines['bottom'].set_color('0')
ax55.spines['top'].set_color('0')
ax55.spines['right'].set_color('0')
ax55.spines['left'].set_color('0')

ax60.plot(time16,run60_erosion,color=n3_color)
ax60.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax60.grid(False)
ax60.set_ylim(0,0.00031)
ax60.set_xlim([0,2e7])
ax60.set_xticks([],[])
ax60.set_yticks([],[])
ax60.set_xlabel('')
ax60.set_ylabel('')
ax60.set_facecolor((1, 1, 1))
ax60.spines['bottom'].set_color('0')
ax60.spines['top'].set_color('0')
ax60.spines['right'].set_color('0')
ax60.spines['left'].set_color('0')

ax62.plot(time20,run62_erosion,color=n3_color)
ax62.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax62.grid(False)
ax62.set_ylim(0,0.00031)
ax62.set_xlim([0,2e7])
#ax62.set_xticks([0,5e6,1e7,1.5e7,2e7],['0','5','10','15','20'])
ax62.set_yticks([],[])
ax62.set_xlabel('')
ax62.set_ylabel('')
ax62.set_facecolor((1, 1, 1))
ax62.spines['bottom'].set_color('0')
ax62.spines['top'].set_color('0')
ax62.spines['right'].set_color('0')
ax62.spines['left'].set_color('0')

ax64.plot(time20,run64_erosion,color=n3_color)
ax64.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax64.grid(False)
ax64.set_ylim(0,0.00031)
ax64.set_xlim([0,2e7])
#ax64.set_xticks([0,5e6,1e7,1.5e7,2e7],['0','5','10','15','20'])
ax64.set_yticks([],[])
ax64.set_xlabel('')
ax64.set_ylabel('')
ax64.set_facecolor((1, 1, 1))
ax64.spines['bottom'].set_color('0')
ax64.spines['top'].set_color('0')
ax64.spines['right'].set_color('0')
ax64.spines['left'].set_color('0')

ax65.plot(time16,run65_erosion,color=n3_color)
ax65.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax65.grid(False)
ax65.set_ylim(0,0.00031)
ax65.set_xlim([0,2e7])
#ax65.set_xticks([0,5e6,1e7,1.5e7,2e7],['0','5','10','15','20'])
ax65.set_yticks([],[])
ax65.set_xlabel('')
ax65.set_ylabel('')
ax65.set_facecolor((1, 1, 1))
ax65.spines['bottom'].set_color('0')
ax65.spines['top'].set_color('0')
ax65.spines['right'].set_color('0')
ax65.spines['left'].set_color('0')

ax67.plot(time20,run67_erosion,color=n3_color)
ax67.plot([0,2e7],[0.0002,0.0002],color='r',linestyle='--',linewidth=1)
ax67.grid(False)
ax67.set_ylim(0,0.00031)
ax67.set_xlim([0,2e7])
#ax67.set_xticks([0,5e6,1e7,1.5e7,2e7],['0','5','10','15','20'])
ax67.set_yticks([],[])
ax67.set_xlabel('')
ax67.set_ylabel('')
ax67.set_facecolor((1, 1, 1))
ax67.spines['bottom'].set_color('0')
ax67.spines['top'].set_color('0')
ax67.spines['right'].set_color('0')
ax67.spines['left'].set_color('0')


# Save figure
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/erosion_tracker/group1.png', facecolor='white', edgecolor='none',dpi=500)
plt.clf()

  
#%% Figure S2: Predict channel steepnesses at model nodes for a given discharge, 
#   mean upstream precipitation rate, and upstream drainage area
# Run code block below before running this

def calc_ksn_bias(Qt,Ps,DAs):
    # Assign variables
    U = 0.0002 # Uplift; m/yr
    K = 2.5e-11 # Erosional efficiency;
    T_start = 0 # years
    T_end = 200000 # years
    T = np.linspace(T_start,T_end,1000)
    n = 3.0
    m = 1.5
    Qt = Qt # Discharge time series; m^3/yr
    
    # Numerically integrate discharge function over climate cycle.
    Qt_integral = integrate.trapezoid((Qt**m),T)
    
    # Calculate Qeff.
    Qeff = ((1/T_end)*Qt_integral)**(1/m)
    
    # Solve for ratios of ksnq and ksnp to ksnqeff using analytical solutions in text
    ksn_bias = ((T_end*(DAs**m))/(Qt_integral))**(1/n)
    ksnp_bias = ((T_end*((Ps*DAs)**m))/(Qt_integral))**(1/n)
    ksnq_bias = ((T_end*(Qt**m))/(Qt_integral))**(1/n)
    
    return ksn_bias,ksnp_bias,ksnq_bias

# Plot predicted versus modeled channel steepness values
fig,ax = plt.subplots(figsize = (6,4))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)

ksns_true = np.array([0])
ksnps_true = np.array([0])
ksnqs_true = np.array([0])
ksns_predicted = np.array([0])
ksnps_predicted = np.array([0])
ksnqs_predicted = np.array([0])

# data1 = plt.scatter(ksnqs_predicted,ksnqs_true,c='r',edgecolor='k',s=40,zorder=10)

data2 = plt.plot([100,500],[100,500],linewidth=2,c='c',zorder=10)
data3 = plt.plot([100,500],[200,200],linewidth=2,linestyle='--',c='k',zorder=10)

for j in np.linspace(2,79,78): # You want to avoid rows adjacent to open boundaries. (2,199,198)
    j = int(j)
    print(j)
    for k in np.linspace(1,38,38):
        k = int(k)
        
        if chis[0,j,k]>0.0:
            
            # Account for nodes where streams don't always occur (these need to be removed)
            if np.sum(np.where(ksnqs[:,j,k]==0))==0:
                
                # Define color based on upstream distance
                c = (1/255)*np.array([(chis[0,j,k]/np.max(chis[0,:,:]))*255,0,0])
                
                # Node 2. Compare biases predicted by analytical solutions
                ksn_bias,ksnp_bias,ksnq_bias = calc_ksn_bias(Qts[:,j,k],Ps[:,j,k],DAs[:,j,k])
                
                ksns_true = np.append(ksns_true,ksns[:,j,k])
                ksnps_true = np.append(ksnps_true,ksnps[:,j,k])
                ksnqs_true = np.append(ksnqs_true,ksnqs[:,j,k])
                ksns_predicted = np.append(ksns_predicted ,200*ksn_bias)
                ksnps_predicted = np.append(ksnps_predicted,200*ksnp_bias)
                ksnqs_predicted = np.append(ksnqs_predicted,200*ksnq_bias)

            else:
                pass
                
        else:
            pass
        
data1 = plt.scatter(ksns_predicted,ksns_true,color='r',edgecolor='k',s=40,zorder=5,alpha=0.1)
        
ax.grid(False)
ax.set_xlim(100,300)
ax.set_ylim(100,300)
#ax.set_xlim(200,300)
#ax.set_ylim(200,300)
#ax.set_xticks([],[])
ax.set_facecolor((1, 1, 1))
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
plt.tight_layout()
        
# Save figure
fig.savefig('/Volumes/PhD_1/ksn_Bias/Figures/model_validation/Runs2/Run_23_ksn', facecolor='white', edgecolor='none',dpi=500)

  
#%% Make pngs and videos of model time steps from first group of runs

# Load model time step
run_name = 'Run_10'

# 5 Myr: 1.2566e-6 (0 kyr phase shift); 15000200 start
# 2.5 Myr: 2.5133e-6 (0 kyr phase shift); 17500200 start
# 1 Myr: 6.2832e-6 (0 kyr phase shift); 19000200 start
# 500kyr: 1.2566e-5 (0 kyr phase shift); 19500200 start
# 200kyr: 3.1416e-5 (0 kyr phase shift); 19800200 start
# 100kyr: 6.2832e-5 (25000 kyr phase shift); 19900200 start (19900400 for Run_3; missed the first step, no big deal)
# 40kyr: 1.5708e-4 (10000 kyr phase shift); 19960200 start
# 20kyr: 3.1416e-4 (5000 kyr phase shift); 19980200 start
wavenumber = 3.1416e-4
phase_shift = 0

T_start = 19980200 # years
period = 20000000 # years
dt = 200 # years
downscaler = 1 # Need to downscale temporal resolution for videos with large climate wavelength runs. Just too long.
                # Downscale factor:
                    # 1 for 40 and 20 kyr (no downscaling)
                    # 2 for 100 kyr
                    # 4 for 200 kyr
                    # 10 for 500 kyr
                    # 20 for 1 Myr
                    # 50 for 2.5 Myr
                    # 100 for 5 Myr
                    
times = np.linspace(0,period,int(period/(dt*downscaler))+1)

# Start of loop
for i in range(len(times)-1):
    
    time = int(times[i])
    
    mg = read_netcdf('/Volumes/PhD_1/ksn_Bias/model_runs2/'+run_name+'/output_frames/netcdf/'+str(int(T_start+time))+'.nc')
    
    # Make sure flow indicator fields are type int
    mg.at_node['flow__link_to_receiver_node'] = mg.at_node['flow__link_to_receiver_node'].astype(int)
    mg.at_node['flow__receiver_node'] = mg.at_node['flow__receiver_node'].astype(int)
    mg.at_node['flow__upstream_node_order'] = mg.at_node['flow__upstream_node_order'].astype(int)
    
    # Calculate ksn and ksnq with 1e5 threshold for plotting
    sf = SteepnessFinder(mg, min_drainage_area=1e5)
    sf.calculate_steepnesses()
    mg.at_node['drainage_area']=mg.at_node['surface_water__discharge']
    sfq = SteepnessFinder(mg, min_drainage_area=1e5)
    sfq.calculate_steepnesses()
    
    
    my_cmap = plt.cm.viridis
    my_cmap.set_under('k', alpha=0)
    
    
    fig=plt.figure(0,figsize=(8,10))
    ax1 = plt.subplot2grid((5,2),(0,0), colspan=1, rowspan=2)
    ax2 = plt.subplot2grid((5,2),(0,1), colspan=1, rowspan=2)
    ax3 = plt.subplot2grid((5,2),(2,0), colspan=1, rowspan=2)
    ax4 = plt.subplot2grid((5,2),(2,1), colspan=1, rowspan=2)
    ax5 = plt.subplot2grid((5,2),(4,0), colspan=2, rowspan=1)
    
    
    #### Water unit flux in (Surface Runoff Contribution)
    im1 = ax1.imshow(np.flipud(np.reshape(mg.at_node['water__unit_flux_in'],(100,50))),
                           cmap='viridis',
                           vmin=0.25,
                           vmax=2.25)
    cbar1=fig.colorbar(im1, ax=ax1, extend='both', fraction=0.11, pad=0.04)
    cbar1.set_label('Surface Water Runoff (m y$r^{-1}$)', rotation=-90, labelpad = 20)
    ax1.grid(False)
    ax1.set_xticks([0,25,50],["0","2.5","5"])
    ax1.set_yticks([0,25,50,75,100],["10","7.5","5","2.5","0"])
    #ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Distance (km)')
    
    
    #### Erosion Rate
    im2 = ax2.imshow(np.flipud(np.reshape(mg.at_node['erosion_rate'],(100,50))),
                           cmap='viridis',
                           vmin=0.0,
                           vmax=0.0003)
    cbar2=fig.colorbar(im2, ax=ax2, ticks=[0.0, 0.00005, 0.0001, 0.00015, 0.0002,0.00025, 0.0003],
                      extend='both', fraction=0.11, pad=0.04)
    cbar2.ax.set_yticklabels(['0.0', '5E-5', '1E-4','1.5E-4', '2E-4', '2.5E-4', '3E-4'])  # horizontal colorbar
    cbar2.set_label('Erosion Rate (m y$r^{-1}$)', rotation=-90, labelpad = 20)
    #cbar = fig.colorbar(im2, ticks=[-1, 0, 1], orientation='horizontal')
    ax2.grid(False)
    ax2.set_xticks([0,25,50],["0","2.5","5"])
    ax2.set_yticks([0,25,50,75,100],["10","7.5","5","2.5","0"])
    #ax2.set_xlabel('Distance (km)')
    #ax2.set_ylabel('Distance (km)')
    
    
    #### Topographic Elevation and ksn
    im3 = ax3.imshow(np.flipud(np.reshape(mg.at_node['topographic__elevation'],(100,50))),
                            cmap='gray',
                            vmin=0,
                            vmax=1200)
    im3 = ax3.imshow(np.flipud(np.reshape(sf.steepness_indices,(100,50))),
                            cmap=my_cmap,
                            vmin=50,
                            vmax=150,
                            interpolation='none')
    
    cbar3=fig.colorbar(im3, ax=ax3, extend='both', fraction=0.11, pad=0.04)
    cbar3.set_label('$k_{sn}$ (m)', rotation=-90, labelpad = 20)
    #ax3.set_title('Time: ' + str(time) + ' years')
    ax3.grid(False)
    ax3.set_xticks([0,25,50],["0","2.5","5"])
    ax3.set_yticks([0,25,50,75,100],["10","7.5","5","2.5","0"])
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Distance (km)')
    
    
    #### Topographic Elevation and ksnq
    im4 = ax4.imshow(np.flipud(np.reshape(mg.at_node['topographic__elevation'],(100,50))),
                            cmap='gray',
                            vmin=0,
                            vmax=1200)
    im4 = ax4.imshow(np.flipud(np.reshape(sfq.steepness_indices,(100,50))),
                            cmap=my_cmap,
                            vmin=50,
                            vmax=150,
                            interpolation='none')
    
    cbar4=fig.colorbar(im4, ax=ax4, extend='both', fraction=0.11, pad=0.04)
    cbar4.set_label('$k_{snq}$ ($m^{1.5}$ y$r^{-0.5}$)', rotation=-90, labelpad = 20)
    #ax4.set_title('Time: ' + str(time) + ' years')
    ax4.grid(False)
    ax4.set_xticks([0,25,50],["0","2.5","5"])
    ax4.set_yticks([0,25,50,75,100],["10","7.5","5","2.5","0"])
    ax4.set_xlabel('Distance (km)')
    #ax4.set_ylabel('Distance (km)')
    
    
    # Climate cycle curve
    y = 0.25*np.sin(wavenumber*((times)+phase_shift))
    ax5.plot(times,y,color='k')
    ax5.scatter(time,y[i], s=100, c="r")
    ax5.set_xlim(0,period)
    ax5.set_ylim(-0.3,0.3)
    ax5.set_xlabel('Time (kyr)')
    ax5.set_ylabel('Runoff Offset (m y$r^{-1}$)')
    ax5.set_title('Time: ' + str(time) + ' years')
    ax5.grid(False)
    #ax5.set_xticks([0,tmax/2,tmax],["0",str(tmax/2),str(tmax)])
    ax5.set_yticks([-0.25,0,0.25],["-0.25","0","0.25"])
    
    
    # Packing all the plots and displaying them
    plt.tight_layout()
    plt.show()
    
    # Save figure
    fig.savefig('/Volumes/PhD_1/ksn_Bias/model_runs2/'+run_name+'/output_frames/panel' + '/time_' + str(time) + '.png', facecolor='white', edgecolor='none',dpi=300)
    

# Make Video
make_video('/Volumes/PhD_1/ksn_Bias/model_runs2/'+run_name+'/output_frames/panel')
  
  
  
  
  
  
  