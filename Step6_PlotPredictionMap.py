#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program is used to perform nonergodic ground motion prediction for Japan using the model
developed by Hou et al. (2025) and plot the predictions in a map, as illustrated by Figure 8 
of the paper.

Cite the reference below:
Hou R, Ji K, Liu M, et al. (2025) Nonergodic ground motion modeling for the subduction zone in Japan 
                accounting for spatial variation in source, path, and site effects. 
                Earth Science. (in Chinese)

@author: Ruibin Hou, E-mail: ruibin90816@163.com
"""
# %% Required Packages
# ======================================
#load variables
import os
import sys
import pathlib
import glob
import re           #regular expression package
#arithmetic libraries
import numpy as np
import pandas as pd
from scipy import linalg
from math import radians, degrees, atan2, cos, sin

#geographic coordinates
import pyproj
#plottign libraries
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import ticker
#base map
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#
from geopy.distance import great_circle
from geopy.point import Point
import geopy.distance as geopydist
from geopy.distance import geodesic

#geometry libraries
from shapely.geometry import Point as shp_pt, Polygon as shp_poly
#user-derfined functions
from pyGMMLib import pylib_contour_plots as pycplt
from pyGMMLib import pylib_plots as pyplt
from pyGMMLib import pylib_HZ24model as HZ24
from pyGMMLib import pylib_GP_model as pygp
from pyGMMLib import pylib_cell_dist as cell_dist

import pylib_pred_func as pred_func


plt.rcParams['font.family'] = 'Times New Roman'


#
# --------------------------------------------------------
# input data 
# --------------------------------------------------------
prd_char = '17'
fname_analysis = 'NergJPGMM%s' %prd_char


dir0 = os.getcwd()

df_cellinfo         = pd.read_csv(os.path.join(dir0, 'nonergodic_path', 'Residual%s-UTM_cellinfo.csv' %prd_char))
df_cellatten        = pd.read_csv(os.path.join(dir0, 'nonergodic_path', fname_analysis + '_stan_catten' + '.csv'))

df_posterior_pdf_event   = pd.read_csv(os.path.join(dir0, 'nonergodic_event', fname_analysis + '_stan_posterior' +  '.csv'), index_col=0)
df_posterior_pdf_path    = pd.read_csv(os.path.join(dir0, 'nonergodic_path', fname_analysis + '_stan_posterior' +  '.csv'), index_col=0)
df_posterior_pdf_site    = pd.read_csv(os.path.join(dir0, 'nonergodic_site', fname_analysis + 'Site_stan_posterior' +  '.csv'), index_col=0)

df_coeffs_event     = pd.read_csv(os.path.join(dir0, 'nonergodic_event', fname_analysis + '_stan_coefficients' + '.csv'))
df_coeffs_path      = pd.read_csv(os.path.join(dir0, 'nonergodic_path', fname_analysis + '_stan_coefficients' + '.csv'))
df_coeffs_site      = pd.read_csv(os.path.join(dir0, 'nonergodic_site', fname_analysis + 'Site_stan_coefficients' + '.csv'))

utm_zone = '54'
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+" +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
# --------------------------------------------------------
# setting source, path, and site parameter 
# --------------------------------------------------------

rcd_EQ_Scenario = pd.read_csv(os.path.join(dir0,'Rcds_201103191856E.csv'))

#jisuan fang wei jiao and group by bearing
rcd_EQ_Scenario['bearing'] = rcd_EQ_Scenario.apply(lambda rcd_EQ_Scenario: pred_func.initial_bearing(rcd_EQ_Scenario['ELat'], rcd_EQ_Scenario['ELon'], rcd_EQ_Scenario['staLat'], rcd_EQ_Scenario['staLon']), axis=1)
rcd_EQ_Scenario1 = rcd_EQ_Scenario[(rcd_EQ_Scenario['bearing']>90) & (rcd_EQ_Scenario['bearing']<=225)]
rcd_EQ_Scenario2 = rcd_EQ_Scenario[(rcd_EQ_Scenario['bearing']>225) & (rcd_EQ_Scenario['bearing']<=280)]
rcd_EQ_Scenario3 = rcd_EQ_Scenario[(rcd_EQ_Scenario['bearing']>280) & (rcd_EQ_Scenario['bearing']<=335)]
rcd_EQ_Scenario4 = rcd_EQ_Scenario[(rcd_EQ_Scenario['bearing']>335) | (rcd_EQ_Scenario['bearing']<=90)]



Mw = 5.86
depth = 5.4
Mech = 3
Ts = 0.3
Tp = 0.2

# epicenter = (140., 37.) 
epicenter = (140.572, 36.784) 

# define ploting area
ngridx = 50
ngridy = 50
att_lonlat_win = np.array([[138, 34.5],[143, 39.5]])
x_int = np.linspace(att_lonlat_win[0,0], att_lonlat_win[1,0], ngridx)
y_int = np.linspace(att_lonlat_win[0,1], att_lonlat_win[1,1], ngridy)
X_grid, Y_grid = np.meshgrid(x_int, y_int)

# define ray path
destination_sta_lonlat1 = (139.5387,35.4544)     #  (139.5387,35.4544)
destination_sta_lonlat2 = (138.7523,36.5103)     #  (138.7523,36.5103)
destination_sta_lonlat3 = (139.4772,37.6863)     #  (139.4772,37.6863)
destination_sta_lonlat4 = (140.3907,38.4306)     #  (140.3907,38.4306)
# destination_sta_lonlat1 = (139.8,38.5)
# destination_sta_lonlat2 = (138.5,36.2)
# destination_sta_lonlat3 = (140.5,35.7)


ray_path1 = np.array([epicenter,destination_sta_lonlat1])
ray_path2 = np.array([epicenter,destination_sta_lonlat2])
ray_path3 = np.array([epicenter,destination_sta_lonlat3])
ray_path4 = np.array([epicenter,destination_sta_lonlat4])

# --------------------------------------------------------
# calculate spectral values for a square area 
# --------------------------------------------------------

# ergodic model
spec_NL_grid, spec_L_grid = pred_func.erg_pred_for_square_area(epicenter, X_grid, Y_grid, Mw, Ts, Tp, depth, Mech)

# for non-ergodic model
logy_adj_area = pred_func.nerg_pred_for_square_area(epicenter,df_posterior_pdf_event,df_coeffs_event,
                        df_posterior_pdf_site, df_coeffs_site,
                        df_cellinfo, df_posterior_pdf_path, df_cellatten,
                        depth, utmProj, X_grid, Y_grid, ngridx, ngridy)

# 
spec_L_grid_Nerg = np.exp(np.log(spec_L_grid) + logy_adj_area)


# --------------------------------------------------------
# calculate spectral values for a ray
# --------------------------------------------------------
scaling_factor = -0.3
att_curve = HZ24.attenuation_curve(M = Mw, depth = depth, Ts = Ts, T = Tp, Mech = Mech)


# ergodic model
pred_ray1 = pred_func.erg_pred_for_ray(epicenter, destination_sta_lonlat1, Mw, Ts, Tp, depth, Mech, npts= 80)
dist_vector1 = pred_ray1[:,0]
L_pred_ray1 = pred_ray1[:,1]*np.exp(scaling_factor)
NL_pred_ray1 = pred_ray1[:,2]*np.exp(scaling_factor)

pred_ray2 = pred_func.erg_pred_for_ray(epicenter, destination_sta_lonlat2, Mw, Ts, Tp, depth, Mech, npts= 80)
dist_vector2 = pred_ray2[:,0]
L_pred_ray2  = pred_ray2[:,1]*np.exp(scaling_factor)
NL_pred_ray2 = pred_ray2[:,2]*np.exp(scaling_factor)

pred_ray3 = pred_func.erg_pred_for_ray(epicenter, destination_sta_lonlat3, Mw, Ts, Tp, depth, Mech, npts= 80)
dist_vector3 = pred_ray3[:,0]
L_pred_ray3  = pred_ray3[:,1]*np.exp(scaling_factor)
NL_pred_ray3 = pred_ray3[:,2]*np.exp(scaling_factor)

pred_ray4 = pred_func.erg_pred_for_ray(epicenter, destination_sta_lonlat4, Mw, Ts, Tp, depth, Mech, npts= 80)
dist_vector4 = pred_ray4[:,0]
L_pred_ray4  = pred_ray4[:,1]*np.exp(scaling_factor)
NL_pred_ray4 = pred_ray4[:,2]*np.exp(scaling_factor)

# non-ergodic adjustment
logy_adj_ray1 = pred_func.nerg_pred_for_ray(epicenter,df_posterior_pdf_event,df_coeffs_event,
                    destination_sta_lonlat1, df_posterior_pdf_site, df_coeffs_site,
                    df_cellinfo, df_posterior_pdf_path, df_cellatten,
                    depth, utmProj, npts= 80)

logy_adj_ray2 = pred_func.nerg_pred_for_ray(epicenter,df_posterior_pdf_event,df_coeffs_event,
                    destination_sta_lonlat2, df_posterior_pdf_site, df_coeffs_site,
                    df_cellinfo, df_posterior_pdf_path, df_cellatten,
                    depth, utmProj, npts= 80)

logy_adj_ray3 = pred_func.nerg_pred_for_ray(epicenter,df_posterior_pdf_event,df_coeffs_event,
                    destination_sta_lonlat3, df_posterior_pdf_site, df_coeffs_site,
                    df_cellinfo, df_posterior_pdf_path, df_cellatten,
                    depth, utmProj, npts= 80)

logy_adj_ray4 = pred_func.nerg_pred_for_ray(epicenter,df_posterior_pdf_event,df_coeffs_event,
                    destination_sta_lonlat4, df_posterior_pdf_site, df_coeffs_site,
                    df_cellinfo, df_posterior_pdf_path, df_cellatten,
                    depth, utmProj, npts= 80)

# non-ergodic model
spec_L_ray_Nerg1 = np.exp(np.log(L_pred_ray1) + logy_adj_ray1)
spec_L_ray_Nerg2 = np.exp(np.log(L_pred_ray2) + logy_adj_ray2)
spec_L_ray_Nerg3 = np.exp(np.log(L_pred_ray3) + logy_adj_ray3)
spec_L_ray_Nerg4 = np.exp(np.log(L_pred_ray4) + logy_adj_ray4)


# --------------------------------------------------------
# Plot spectral countour on map
# --------------------------------------------------------

fig = plt.figure(figsize=(7, 3.5))
data_crs = ccrs.PlateCarree()


#
# ergodic attenuation plot

ax = fig.add_subplot(1, 2, 1, projection=data_crs)
fig, ax, data_crs, gl = pycplt.PlotMap( fig = fig, ax=ax, 
                    lon_lims = att_lonlat_win[:,0], lat_lims = att_lonlat_win[:,1], 
                    add_land = False, add_state = False, add_border = True, add_ocean = False,
                    lonLoc = [139,140,141,142], latLoc = [ 35, 36, 37, 38, 39],
                    latLabel = False, lonLabel = False)

log_spec = np.log(spec_L_grid)
cmax = log_spec.max()
cmin = log_spec.min()
clevs = np.linspace(cmin, cmax, 21).tolist() 
clevs_label = clevs[::4]
cs_fill =  ax.contourf(X_grid, Y_grid, log_spec, transform = data_crs, 
                        vmin=cmin, vmax=cmax, levels = clevs, 
                        zorder=13, alpha = 0.8, cmap='summer')
cs_line =  ax.contour(X_grid, Y_grid, log_spec, transform = data_crs, 
                        vmin=cmin, vmax=cmax, levels = clevs, 
                        zorder=14, colors='grey', linewidths=0.5)
plt.clabel(cs_line, inline=True, fontsize=12, fmt="%.2f", levels=clevs_label)

ax.coastlines(resolution='50m', edgecolor='black', zorder=30)

# plot earthquake locations
ax.scatter(epicenter[0], epicenter[1], marker = '*', transform = data_crs, 
                        s = 49, c = 'r', zorder=15,  label='Events')
# ax.text(-0.1,0.98,'(a)',transform = ax.transAxes,fontsize=13, backgroundcolor='white')
# ax.text(0.80,0.92,'Ergodic',transform = ax.transAxes,fontsize=13, backgroundcolor='white')
[x.set_linewidth(0.75) for x in ax.spines.values()]         # 

# plot stations
ax.scatter(destination_sta_lonlat1[0], destination_sta_lonlat1[1], marker = '^', transform = data_crs, 
                        s = 25, c = 'k', zorder=15,  label='Station')
ax.scatter(destination_sta_lonlat2[0], destination_sta_lonlat2[1], marker = '^', transform = data_crs, 
                        s = 25, c = 'k', zorder=15,  label='Station')
ax.scatter(destination_sta_lonlat3[0], destination_sta_lonlat3[1], marker = '^', transform = data_crs, 
                        s = 25, c = 'k', zorder=15,  label='Station')
ax.scatter(destination_sta_lonlat4[0], destination_sta_lonlat4[1], marker = '^', transform = data_crs, 
                        s = 25, c = 'k', zorder=15,  label='Station')

# plot ray_path
ax.plot(ray_path1[:,0], ray_path1[:,1], transform = data_crs, c = 'k', lw=0.75, zorder=15,  label='Ray path1')
ax.plot(ray_path2[:,0], ray_path2[:,1], transform = data_crs, c = 'k', lw=0.75, zorder=15,  label='Ray path2')
ax.plot(ray_path3[:,0], ray_path3[:,1], transform = data_crs, c = 'k', lw=0.75, zorder=15,  label='Ray path3')
ax.plot(ray_path4[:,0], ray_path4[:,1], transform = data_crs, c = 'k', lw=0.75, zorder=15,  label='Ray path4')


# plot colorbar
frmt_clb = '%.2f'
fmt_clb = ticker.FormatStrFormatter(frmt_clb)
cbar_ticks = clevs[0:21:4]
cbar = fig.colorbar(cs_fill, boundaries=clevs, ticks=cbar_ticks, pad=0.12, 
                        orientation="horizontal", format=fmt_clb, shrink=0.7)        

cbar_labels = ['%.4f'%np.exp(c_t) for c_t in cbar_ticks]
cbar.set_ticklabels(cbar_labels)
cbar.ax.tick_params(labelsize=10) 
[x.set_linewidth(0.75) for x in cbar.ax.spines.values()] 



#
# non-ergodic attenuation plot
ax = fig.add_subplot(1, 2, 2, projection=data_crs)
fig, ax, data_crs, gl = pycplt.PlotMap( fig = fig, ax=ax, 
                    lon_lims = att_lonlat_win[:,0], lat_lims = att_lonlat_win[:,1], 
                    add_land = False, add_state = False, add_border = True, add_ocean = False,
                    lonLoc = [139,140,141,142], latLoc = [ 35, 36, 37, 38, 39],
                    latLabel = False, lonLabel = False)

log_spec = np.log(spec_L_grid_Nerg)
cmax = log_spec.max()
cmin = log_spec.min()
clevs = np.linspace(cmin, cmax, 21).tolist() 
clevs_label = clevs[::4]
cs_fill =  ax.contourf(X_grid, Y_grid, log_spec, transform = data_crs, 
                        vmin=cmin, vmax=cmax, levels = clevs, 
                        zorder=13, alpha = 0.8, cmap='summer')
cs_line =  ax.contour(X_grid, Y_grid, log_spec, transform = data_crs, 
                        vmin=cmin, vmax=cmax, levels = clevs, 
                        zorder=14, colors='grey', linewidths=0.5)
plt.clabel(cs_line, inline=True, fontsize=12, fmt="%.2f", levels=clevs_label)

ax.coastlines(resolution='50m', edgecolor='black', zorder=30)

# plot earthquake locations
ax.scatter(epicenter[0], epicenter[1], marker = '*', transform = data_crs, 
                        s = 49, c = 'r', zorder=15,  label='Events')
# ax.text(-0.1,0.98,'(b)',transform = ax.transAxes,fontsize=13, backgroundcolor='white')
# ax.text(0.72,0.92,'Nonergodic',transform = ax.transAxes,fontsize=13, backgroundcolor='white')
[x.set_linewidth(0.75) for x in ax.spines.values()]


# plot ray_path
ax.plot(ray_path1[:,0], ray_path1[:,1], transform = data_crs, c = 'm', lw=0.75, zorder=15,  label='Ray path1')
ax.plot(ray_path2[:,0], ray_path2[:,1], transform = data_crs, c = 'c', lw=0.75, zorder=15,  label='Ray path2')
ax.plot(ray_path3[:,0], ray_path3[:,1], transform = data_crs, c = 'b', lw=0.75, zorder=15,  label='Ray path3')
ax.plot(ray_path4[:,0], ray_path4[:,1], transform = data_crs, c = 'r', lw=0.75, zorder=15,  label='Ray path4')


# plot stations
ax.scatter(rcd_EQ_Scenario1['staLon'], rcd_EQ_Scenario1['staLat'], marker = '^', transform = data_crs, 
                        s = 9, c = 'm', zorder=15,  label='Stations')
ax.scatter(rcd_EQ_Scenario2['staLon'], rcd_EQ_Scenario2['staLat'], marker = '^', transform = data_crs, 
                        s = 9, c = 'c', zorder=15,  label='Stations')
ax.scatter(rcd_EQ_Scenario3['staLon'], rcd_EQ_Scenario3['staLat'], marker = '^', transform = data_crs, 
                        s = 9, c = 'b', zorder=15,  label='Stations')
ax.scatter(rcd_EQ_Scenario4['staLon'], rcd_EQ_Scenario4['staLat'], marker = '^', transform = data_crs, 
                        s = 9, c = 'r', zorder=15,  label='Stations')


# plot colorbar
frmt_clb = '%.2f'
fmt_clb = ticker.FormatStrFormatter(frmt_clb)
cbar_ticks = clevs[0:21:4]
cbar = fig.colorbar(cs_fill, boundaries=clevs, ticks=cbar_ticks, pad=0.12, 
                        orientation="horizontal", format=fmt_clb, shrink=0.7)        

cbar_labels = ['%.4f'%np.exp(c_t) for c_t in cbar_ticks]
cbar.set_ticklabels(cbar_labels)
cbar.ax.tick_params(labelsize=10)
[x.set_linewidth(0.75) for x in cbar.ax.spines.values()] 


# save figure
fig.tight_layout(pad=0.2)
fname_fig = 'JP_Nonerg_prediction_contour'
fig.savefig( os.path.join(dir0, fname_fig + '.eps'))
fig.savefig( os.path.join(dir0, fname_fig + '.png'), dpi=500)


# attenuation curve 


fig, ax = plt.subplots(figsize=(3.5, 2.5))
fig, ax = pyplt.line_plot(att_curve[:,0], att_curve[:,2]*np.exp(scaling_factor), fig=fig, ax=ax, xlabel='Distance(km)', ylabel='Sa(g)', xscale='log', yscale='log', c = 'k', lw = 1.5,label = 'Ergodic')

fig, ax = pyplt.line_plot(dist_vector1, spec_L_ray_Nerg1, fig=fig, ax=ax, xlabel='Distance(km)', ylabel='Sa(g)', xscale='log', yscale='log', c = 'm', lw = 1.5,ls='dashed', label = 'Station1')
fig, ax = pyplt.line_plot(dist_vector2, spec_L_ray_Nerg2, fig=fig, ax=ax, xlabel='Distance(km)', ylabel='Sa(g)', xscale='log', yscale='log', c = 'c', lw = 1.5,ls=(0,(1,1)), label = 'Station2')
fig, ax = pyplt.line_plot(dist_vector3, spec_L_ray_Nerg3, fig=fig, ax=ax, xlabel='Distance(km)', ylabel='Sa(g)', xscale='log', yscale='log', c = 'b', lw = 1.5,ls='dashdot',label = 'Station3')
fig, ax = pyplt.line_plot(dist_vector4, spec_L_ray_Nerg4, fig=fig, ax=ax, xlabel='Distance(km)', ylabel='Sa(g)', xscale='log', yscale='log', c = 'r', lw = 1.5,ls=(0,(4,1)),label = 'Station4')

fig, ax, sc = pyplt.scatter_plot(rcd_EQ_Scenario1['SDist'], rcd_EQ_Scenario1['Sa(g)'], fig=fig, ax=ax, xlabel='Distance(km)', ylabel='Sa(g)', xscale='log', yscale='log', c = 'm', marker = '^', label = 'Records')
fig, ax, sc = pyplt.scatter_plot(rcd_EQ_Scenario2['SDist'], rcd_EQ_Scenario2['Sa(g)'], fig=fig, ax=ax, xlabel='Distance(km)', ylabel='Sa(g)', xscale='log', yscale='log', c = 'c', marker = '^', label = 'Records')
fig, ax, sc = pyplt.scatter_plot(rcd_EQ_Scenario3['SDist'], rcd_EQ_Scenario3['Sa(g)'], fig=fig, ax=ax, xlabel='Distance(km)', ylabel='Sa(g)', xscale='log', yscale='log', c = 'b', marker = '^', label = 'Records')
fig, ax, sc = pyplt.scatter_plot(rcd_EQ_Scenario4['SDist'], rcd_EQ_Scenario4['Sa(g)'], fig=fig, ax=ax, xlabel='Distance(km)', ylabel='Sa(g)', xscale='log', yscale='log', c = 'r', marker = '^', label = 'Records')

# 
ax.legend(loc = 'lower left', fontsize='small', borderaxespad=0, frameon=False)
ax.set_xlim(3, 300)
ax.set_ylim(0.003, 5)

# save figure
fig.tight_layout(pad=0.2)
fname_fig = 'JP_Nonerg_prediction_curve'
fig.savefig( os.path.join(dir0, fname_fig + '.eps'))
fig.savefig( os.path.join(dir0, fname_fig + '.png'), dpi=500)

