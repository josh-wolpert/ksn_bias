# ksn_bias
Code and data associated with 'Channel Steepness Biases and Non-Steady Erosion in Landscapes Evolving with Cyclical Climate Forcings'

main.py
- The landscape evolution model. The model is built with the Landlab software package and evolves according to stream power erosion solved with the Fastscape algorithm, rock uplift, and runoff that can be altered through the ‘water__unit_flux_in’ Landlab model grid field. Output are:

  •	Erosion_tracker.csv – Mean model erosion rates at each time step

  •	netcdf – Folder in ‘output_frames’ folder containing netcdf files of timesteps. The start of when time steps are saved is set under the ‘Keep track of time and save frames’ comment in the main loop.

config.yaml
- Called by main.py and contains parameter values for landscape evolution models.  

figures.py
- Code used to extract data from models and make figures and videos. Each code block has a heading describing its purpose.
