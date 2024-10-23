# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:35:52 2024

@author: shjordan
"""
import geopandas as gpd
import pandas as pd
import numpy as np
#import os
from raster2xyz.raster2xyz import Raster2xyz
from plotly import graph_objects as go
import plotly.offline as pyo
from shapely.geometry import Point
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
#import pykrige.kriging_tools as kt
#from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from sklearn.cluster import KMeans
#import gstools as gs
import webbrowser
import json


#%% Load the input data
# ======================
# Kaweah basin shapefile
# ======================
kaweah = gpd.read_file('./kaweah_GiS/GreaterK_GSA.shp')

# ================================
# shapefile with 2014 water levels
# ================================
wl_gdf = gpd.read_file('./water_levels/rms_upper_mt_gwe.shp')


# ===============================================================================
# Cluster the RMS wells to ensure better spatial distribution of random selection
# NOTE: Currently not using this approach
# ===============================================================================
coords = wl_gdf[['X_Coordina', 'Y_Coordina']].values
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
wl_gdf['cluster'] = kmeans.labels_
p = wl_gdf.plot(column='cluster', cmap='tab20', legend=True)
p.set_xlabel('X')
p.set_ylabel('Y')
p.set_title('Well Clusters')
kaweah.boundary.plot(ax=p,
                     color='k',
                     alpha=0.5)

# ==================
# Gradient transects
# ==================
grad_trans = gpd.read_file('./kaweah_GiS/GradientTransects.shp')
# Grab the start/end points of each transect
grad_trans['points'] = grad_trans.apply(lambda x: [y for y in x['geometry'].coords], axis=1)

# =============================================
# Point data of coordinates of monitoring wells
# Formatted for PyKrige function
# =============================================
points = np.vstack((wl_gdf['X_Coordina'],wl_gdf['Y_Coordina'])).T

# =======================
# Load domestic well data
# =======================
domestic = gpd.read_file('./water_levels/domestic_upper.shp')
# Drop F22 and F14 exceedances
domestic = domestic.loc[domestic['PT_F22']=='Protective Threshold Met']
# Keep wells that met MT or were not yet installed in 2014
domestic = domestic.loc[(domestic['PT_F14']=='Protective Threshold Met') | (domestic['PT_F14']=='Post-2014')]

# =========================================================================
# Convert the 10x_.grd file to a Python readable csv, and load as DataFrame
# =========================================================================
input_raster = "./grid_raster/10x_.grd"
out_csv = "./grid_raster/10x_grid.csv"
rtxyz = Raster2xyz()
rtxyz.translate(input_raster, out_csv)
# Load the grid as a dataframe
grid = pd.read_csv(out_csv)
# Convert the grid to a geoDataFrame of points
grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid['x'],grid['y']))
grid_gdf = grid_gdf.set_crs(domestic.crs)

#%% Initial GiS plot, make sure things look okay
fig,ax = plt.subplots()
kaweah.plot(ax=ax)
grad_trans.plot(ax=ax,
                color='red')
domestic.plot(ax=ax,
              color='darkgreen',
              )

#%% Main functions
# ==================================
# Function for kriging interpolation
# ==================================
def pykrige_interpolation(points,values,grid_x,grid_y,regional=True):
    # Apply regional drift term
    if regional:
        UK = UniversalKriging(points[:,0],
                              points[:,1],
                              values,
                              variogram_model="spherical",
                              drift_terms=['regional_linear']
                              )
        # UK.display_variogram_model()
        z, ss = UK.execute("points", grid_x, grid_y)
    else:
        UK = UniversalKriging(points[:,0],
                              points[:,1],
                              values,
                              variogram_model="gaussian",
                              #drift_terms=['regional_linear'],
                              #nlags=15,
                              )
        UK.display_variogram_model()
        z, ss = UK.execute("points", grid_x, grid_y)
    return z

# ===================================================
# Choose random wells while respecting the clustering
# ===================================================
def choose_clustered_wells(total_wells_to_select,n_clusters=n_clusters):
    # Divide the number of wells evenely across the clusters
    wells_per_cluster = np.ones(n_clusters) * np.floor(total_wells_to_select / n_clusters)
    current_total = np.sum(wells_per_cluster)
    
    # For any remainder, randomly choose clusters and apply it there
    while current_total < total_wells_to_select:
        # Randomly pick a cluster and increase by 1
        cluster_to_adjust = np.random.choice(np.arange(n_clusters))
        wells_per_cluster[cluster_to_adjust] += 1
        current_total += 1
    
    return wells_per_cluster


# ===================================================================
# Function for stochastic analysis
# Will set a random number of randomly selected wells to their MT
# WT surface is then generated, and domestic dry wells are calculated
# ===================================================================
def stochastic_analysis(max_MT=25,all_MT=False):
    # Set a random number of randomly located points equal to their MT
    wl_gdf_rand = wl_gdf.copy()
    n_numbers = np.random.randint(low=1,high=max_MT)
    
    """
    # Select the wells based on clustering to enforce higher spatial distribution
    wells_per_cluster = choose_clustered_wells(n_numbers)
    selected_wells = []
    for cluster_label, wells_in_cluster in wl_gdf_rand.groupby('cluster'):
        n_wells = wells_per_cluster[cluster_label]
        try:
            selected_wells.extend(wells_in_cluster.sample(n=int(n_wells)).index)
        except:
            print(n_numbers,n_wells,wells_in_cluster)
    """
    
    # Use this for purely random, no clusters
    selected_wells = np.random.choice(range(0,len(wl_gdf_rand)), size=n_numbers, replace=False)
    
    # Apply the random set of MTs as the GWE
    wl_gdf_rand.loc[selected_wells,'GWE_F2014'] = wl_gdf_rand.loc[selected_wells,'MT']
    
    # Option to set all elevations equal to thier MT
    if all_MT:
        wl_gdf_rand['GWE_F2014'] = wl_gdf_rand['MT']
    
    # Monitoring well GWE --> from random MT df
    z = wl_gdf_rand['GWE_F2014'].values
    
    # Interpolate points and add elevations to the grid
    interped = pykrige_interpolation(points,z,grid['x'],grid['y'])
    grid_gdf['WL'] = interped

    # Check the gradient of the interpolated surface against the transects
    # How to check the resulting gradients?
    gradient_check = calc_gradient(grid_gdf)
        
    # Spatially join the wells to the grid
    domestic_with_grid = gpd.sjoin_nearest(domestic,grid_gdf)
    
    # Determine if the well has exceeded the Protective Threshold
    domestic_with_grid['exceedance'] = np.zeros(len(domestic_with_grid))
    domestic_with_grid.loc[domestic_with_grid['WL']<=domestic_with_grid['ProThres'],'exceedance'] = 1
    exceeded_wells = domestic_with_grid['exceedance'].sum()
    
    #print('\n\n************************')
    #print(n_numbers)
    #print(exceeded_wells)
    #print('************************\n\n')
    return exceeded_wells, n_numbers, interped, domestic_with_grid, wl_gdf_rand, selected_wells, gradient_check


# ==================================
# Calculate the water table gradient
# ==================================
def calc_gradient(grid_gdf):
    gdf = grid_gdf.copy()
    # ==============================================================
    # Calculate the gradients based on start/end points of transects
    # ==============================================================
    grads = []
    for idx,point_coords in enumerate(grad_trans['points']):
        # Grab the length of the transect
        transect_length = grad_trans['Length_ft'].iloc[idx]
        
        # Grab WL corresponding to start point
        start_point = gpd.GeoDataFrame(geometry=[Point(point_coords[0])])
        wl_start = gpd.sjoin_nearest(start_point,gdf)['WL'].values[0]
        
        # Grab WL corresponding to end point
        end_point = gpd.GeoDataFrame(geometry=[Point(point_coords[1])])
        wl_end = gpd.sjoin_nearest(end_point,gdf)['WL'].values[0]
        
        # Calculate the gradient (slope)
        gradient = (wl_start-wl_end) / transect_length
        grads.append(gradient)
        
    return grads


# ====================================================
# Plot the resulting water level and wells of interest
# ====================================================
def make_plot(interped,domestic_with_grid,wl_gdf_rand,well_list):
    fig = go.Figure()
    Z = interped
    
    # Water table contour
    fig.add_trace(go.Mesh3d(x=grid['x'], y=grid['y'], z=Z, 
                            intensity=Z,
                            #mode='markers', 
                            #marker=dict(size=4,color=interped,colorscale='blues'), 
                            name='Water Table',
                            colorscale='gnbu'
                            
                            )
                  )
    
    
    # Exceeded domestic wells
    fig.add_trace(go.Scatter3d(x=domestic_with_grid.loc[domestic_with_grid['exceedance']==1,'geometry'].x,
                               y=domestic_with_grid.loc[domestic_with_grid['exceedance']==1,'geometry'].y,
                               z=domestic_with_grid.loc[domestic_with_grid['exceedance']==1,'ProThres'],
                               mode='markers', 
                               marker=dict(size=2,color='black'), 
                               name='Exceeded Domestic Wells'
                               )
                  )
    
    """
    # Exceeded domestic wells --> based on Tyler's analysis w/ MT raster
    fig.add_trace(go.Scatter3d(x=domestic_with_grid.loc[domestic_with_grid['PT_MTwo14']=='Protective Threshold Exceeded (well dry)','geometry'].x,
                               y=domestic_with_grid.loc[domestic_with_grid['PT_MTwo14']=='Protective Threshold Exceeded (well dry)','geometry'].y,
                               z=domestic_with_grid.loc[domestic_with_grid['PT_MTwo14']=='Protective Threshold Exceeded (well dry)','ProThres'],
                               mode='markers',
                               marker=dict(size=2,color='red'),
                               name='Exceeded Domestic Wells - Tyler'
                               )
                   )
    """
    """
    # Plot all the domestic wells w/ WSE to compare to contour
    fig.add_trace(go.Scatter3d(x=domestic_with_grid.loc[domestic_with_grid['F14WSE_u']>-1000,'geometry'].x,
                               y=domestic_with_grid.loc[domestic_with_grid['F14WSE_u']>-1000,'geometry'].y,
                               z=domestic_with_grid.loc[domestic_with_grid['F14WSE_u']>-1000,'F14WSE_u'],
                               mode='markers', 
                               marker=dict(size=3,color='black'), 
                               name='MT Wells'
                               )
                  )
    """
    
    # Wells randomly set to MT
    fig.add_trace(go.Scatter3d(x=wl_gdf_rand['X_Coordina'].iloc[well_list],
                               y=wl_gdf_rand['Y_Coordina'].iloc[well_list],
                               z=wl_gdf_rand['GWE_F2014'].iloc[well_list],
                               mode='markers', 
                               marker=dict(size=4,color='blue'), 
                               name='MT Wells'
                               )
                  )
    
    pyo.plot(fig)


# =============================================================================
# Main func
# =============================================================================
def main(its=30,max_MT=30,plot=False):
    # Keep track of number of MT wells and number of exceedances
    results_dict = {}
    gradients_dict = {}
    # Track total number of exceedances for making a heatmap of results
    total_exceedance = np.zeros(len(domestic))
    
    # Create a new surface for each iteration
    for i in tqdm(range(its)):
        exceeded_wells, n_numbers, interped, domestic_with_grid, wl_gdf_rand, well_list, gradient_check = stochastic_analysis(max_MT=max_MT)
        # Summing total exceedances for the heatmap
        total_exceedance += domestic_with_grid['exceedance']
        
        # Save the results
        if n_numbers in results_dict.keys():
            results_dict[n_numbers].append(exceeded_wells)
            gradients_dict[n_numbers].append(gradient_check)
        else:
            results_dict[n_numbers] = [exceeded_wells]
            gradients_dict[n_numbers] = [gradient_check]

    # only run plotting for small numbers of iterations
    if plot:
        make_plot(interped,domestic_with_grid,wl_gdf_rand,well_list)
   
    return results_dict, gradients_dict, total_exceedance


# =============================================================================
# Calculate results for specified number of interations
# If running > ~3 iterations DO NOT set plot to True, it will explode
# Variables:
#    n_its: number of stochastic iterations to run    
#    max_MT: Maximum number of RMS wells to set at MT
#    plot: Specifiy whether or not to create interactive Plotly figure
# =============================================================================
n_its = 4000
max_MT = 50
results_dict, gradients_dict, total_exceedance = main(its=n_its,max_MT=max_MT,plot=False)

# Add the total exceedances to the domestic well shapefile
domestic['total_ex'] = total_exceedance

# %% Plot wells, colored by their total number of exceedances

# Look at the distribution of well exceedances in Space
domestic['cdf'] = (domestic['total_ex'].rank(method='min',pct=True))
m = domestic[['FID_','total_ex','ProThres','geometry','cdf']].explore(column='total_ex',
                                                                      style_kwds={"style_function":lambda x: {"radius":15*x["properties"]["cdf"]}}
                                                                      )
outfp = "temp_map.html"
m.save(outfp)
webbrowser.open(outfp)


#%% Plot the exceedance results in space as a contour
domestic_t = domestic.copy()
#domestic_t = domestic_t.loc[domestic_t['total_ex']>0]

domestic_t['cdf'] = domestic_t['total_ex'].rank(method='min',pct=True)
# domestic['quantile_bin'] = pd.qcut(domestic['total_ex'], 
#                                    q=[0,.1, .80,.82,.85,.9,.95, 1.], 
#                                    labels=False)

points = np.vstack((domestic_t.geometry.x.values,domestic_t.geometry.y.values)).T

# Interpolate the CDF values across the grid, this takes a while...
cdf_interped = pykrige_interpolation(points,
                                     domestic_t['cdf'].values,
                                     grid['x'],
                                     grid['y'],
                                     regional=False)

#%%
grid_gdf_temp = grid_gdf.copy()
grid_gdf_temp['cdf_values'] = cdf_interped

#grid_gdf_temp = grid_gdf_temp[grid_gdf_temp.covered_by(kaweah)]

base = grid_gdf_temp.plot(column='cdf_values',
                          legend=True,
                          cmap='viridis')

kaweah.boundary.plot(ax=base,
                     color='k',
                     )

# %% Analyze the results with a jointplot, showing density with hex's

# Option to load results
with open('./dat_4000_its.json','r') as f:
     results_dict = json.load(f)

# Create DataFrame and Melt into a Seaborn friendly format
df = pd.DataFrame({k:pd.Series(v) for k,v in results_dict.items()})
df_long = df.melt(var_name='MT_wells', value_name='Exceedance')
df_long['MT_wells'] = pd.to_numeric(df_long['MT_wells'], errors='coerce')
df_long = df_long.dropna()
g = sns.jointplot(data=df_long, 
                  x='MT_wells',
                  y='Exceedance', 
                  kind='hex',
                  joint_kws=dict(gridsize=25,
                                 ),
                  marginal_kws=dict(bins=49, fill=True)
                  )

# Get the hexbin object from the Axes to assign a colorbar
ax = g.ax_joint
hb = ax.collections[0]

# Create a new axis for the colorbar on the far right
cbar_ax = g.fig.add_axes([1.02, 0.25, 0.03, 0.5])  # [left, bottom, width, height]

# Add the colorbar to the new axis
cbar = plt.colorbar(hb, cax=cbar_ax, label=f'Normalized Point Density,\n{n_its} Total Iterations')

# Replace colorbar ticks with normalized hexbin count density
counts = hb.get_array()
norm_counts = (counts - counts.min()) / (counts.max() - counts.min())
cbar.set_ticks(np.linspace(counts.min(), counts.max(), 5))
cbar.set_ticklabels(np.linspace(0, 1, 5))

# Calculate the polyfit
c = np.polyfit(df_long['MT_wells'],df_long['Exceedance'],deg=2)

# Option to add a linreg to the plot
g.plot_joint(sns.regplot,
             color="k",
             scatter_kws={'alpha':0},
             order=2,
             line_kws={'label':f'Polynomial Trendline:\ny = {round(c[0],3)}x$^2$ + {round(c[1],2)}x + {round(c[2],2)}'},
             fit_reg=True
             )

# Add line at 60 PT exceedances
g.ax_joint.axhline(60,
                   color='maroon',
                   ls='--',
                   label='60 PT Exceedances')

# Calculate number of MT wells @ 60 exceedances
p = c.copy()
p[2] = p[2] - 60
roots = np.roots(p)
max_mt = roots[np.where(roots>0)]

# Plot the number of MT wells leading to 60 exceedances
g.ax_joint.axvline(max_mt,
                   color='k',
                   ls='--',
                   label=f'MT wells @ 60 Exceedances: {round(max_mt[0],0)}'
                   )

# Plot formatting
g.ax_joint.legend(loc=2,
                  framealpha=1)
g.ax_joint.grid()
g.ax_joint.set_ylabel('Number of PT Exceedances')
g.ax_joint.set_xlabel('Number of RMS Wells at Respective MTs')
g.ax_joint.set_xlim([0,50]) 

#%% PDF hist plot
g = sns.histplot(data=df_long,
                 x='Exceedance',
                 hue='MT_wells',
                 cumulative=True,
                 element='step',
                 fill=False,
                 legend=False,
                 )

#%% Hist plot with mean
MT = 17
g = sns.histplot(data=df_long.loc[df_long['MT_wells']==MT],
                 x='Exceedance',
                 bins=17
                 )

g.axvline(df_long.loc[df_long['MT_wells']==MT,'Exceedance'].mean(),
                       color='maroon',
                       ls='--')

# %% Plot the results of a histogram, showing mean and std of exceedances
mean_results = {}
max_results = []
for key in results_dict:
    mean_results[int(key)] = [np.mean(results_dict[key])]
    max_results.append(np.max(results_dict[key]))
    
std = [np.std(results_dict[key]) for key in results_dict]

results_df = pd.DataFrame(mean_results)

ex_per_RMS = [mean_results[key][0]/int(key) for key in mean_results.keys()]

print(f'Average dry wells per RMS well @ MT: {round(np.mean(ex_per_RMS),2)}')

fig,ax = plt.subplots(figsize=[10,6])
results_df.T.sort_index().plot.bar(ax=ax,
                                   yerr=std,
                                   legend=False,
                                   )

ax.axhline(60,
           color='r',
           ls='--')

ax.set_ylim([0,190])
ax.set_xlabel('Number of RMS Wells Set at MT')
ax.set_ylabel('Number of Domestic Wells Exceeding PT\nMean with Std Errorbars')


#%% Check my MT surface against the MT contour lines

grid_MT = gpd.read_file('./water_levels/MT_krig_grid_Spencer.shp')

dat = gpd.read_file('./water_levels/mt_upper_rev_20240916_contour_20ft.shp')

t = gpd.sjoin_nearest(dat,grid_gdf)
t['WL_diff'] = t['Contour']-t['WL']


#%% Plot the original MT surface, colored by the difference with mine
base = t.plot(column='WL_diff',
              legend=True,
              cmap='tab10')

grid_MT.plot(ax=base,
             zorder=0,
             column='WL',
             alpha=0.3,
             cmap='Blues')

wl_gdf.plot(ax=base,
            color='k',
            markersize=4,
            zorder=10,
            )
