#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program includes serveral functions for performing nonergodic ground motion prediction
for Japan using the model developed by Hou et al. (2025) that will published in the Chinese
journal named Earth Science.

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

#
from geopy.distance import great_circle
from geopy.point import Point
import geopy.distance as geopydist
from geopy.distance import geodesic

#geometry libraries
from shapely.geometry import Point as shp_pt, Polygon as shp_poly
#user-derfined functions
from pyGMMLib import pylib_HZ24model as HZ24
from pyGMMLib import pylib_GP_model as pygp
from pyGMMLib import pylib_cell_dist as cell_dist


plt.rcParams['font.family'] = 'Times New Roman'


# --------------------------------------------------------
# user defined functions 
# --------------------------------------------------------
def initial_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing between two points.
    """
    lat1, lat2 = radians(lat1), radians(lat2)
    diff_lon = radians(lon2 - lon1)

    x = sin(diff_lon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(diff_lon)

    bearing = atan2(x, y)
    return (degrees(bearing) + 360) % 360

#
def interpolate_geodesic(lat1, lon1, lat2, lon2, num_points):
    """
    Interpolates along a geodesic line between two points.
    """
    start = (lat1, lon1)
    end = (lat2, lon2)
    total_distance = geodesic(start, end).km
    bearing = initial_bearing(lat1, lon1, lat2, lon2)

    # Interpolate points
    points = [  geodesic(k * total_distance).destination(start, bearing)
                for k in [i / (num_points - 1) for i in range(num_points)]  ]
    int_pts = [(point.latitude, point.longitude) for point in points]
    int_pts_arr = np.array(int_pts)
    int_pts_arr[:, [0, 1]] = int_pts_arr[:, [1, 0]]
    
    return int_pts_arr


#
# functions for ergodic prediction
def erg_pred_for_ray(epicenter, destination_sta_lonlat, Mw, Ts, Tp, depth, Mech, npts= 50):
    
    pts_ray = interpolate_geodesic(epicenter[1], epicenter[0], destination_sta_lonlat[1], destination_sta_lonlat[0], npts+1)
    dist_vector = np.zeros(npts)
    spec_NL_vector = np.zeros(npts)
    spec_L_vector = np.zeros(npts)
    aHZ24GMM = HZ24.GMM_HZ24()
    for i in range(npts):
        obs_pt = ( pts_ray[i+1][1], pts_ray[i+1][0])
        epiDist = geopydist.distance((epicenter[1], epicenter[0]), obs_pt).km
        dist_vector[i] = np.sqrt(epiDist**2 + depth**2)
        aSpec_value = aHZ24GMM( Mw, dist_vector[i], Ts, Tp, depth=depth, Mech=Mech)
        spec_NL_vector[i] = aSpec_value[0]                    # NL site response
        spec_L_vector[i] = aSpec_value[1]                     # linear site response
    
    Sa_pred_ray = np.column_stack((dist_vector,spec_L_vector,spec_NL_vector))
    
    return Sa_pred_ray

#
def erg_pred_for_square_area(epicenter, X_grid, Y_grid, Mw, Ts, Tp, depth, Mech):
    
    dist_grid = np.zeros_like(X_grid)
    spec_NL_grid = np.zeros_like(X_grid)
    spec_L_grid = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            obs_pt = ( Y_grid[i][j], X_grid[i][j])
            epiDist = geopydist.distance((epicenter[1], epicenter[0]), obs_pt).km
            dist_grid[i][j] = np.sqrt(epiDist**2 + depth**2)
            aHZ24GMM = HZ24.GMM_HZ24()
            aSpec_value = aHZ24GMM( Mw, dist_grid[i][j], Ts, Tp, depth=depth, Mech=Mech)
            spec_NL_grid[i][j] = aSpec_value[0]       # NL site response
            spec_L_grid[i][j] = aSpec_value[1]        # linear site response
    
    return spec_NL_grid, spec_L_grid


#
# functions for non-ergodic adjustment terms
def calc_event_adjust_term(coeff_X_eq, df_posterior_pdf_event, df_coeffs_event):
    #
    # hyper-parameters for event terms
    dc_0_evt  = df_posterior_pdf_event.loc['mean','dc_0']
    ell_1e    = df_posterior_pdf_event.loc['mean','ell_1e']           # epistemic uncetainty terms
    omega_1e  = df_posterior_pdf_event.loc['mean','omega_1e']
    
    eq_id_train  = df_coeffs_event['eqid'].values.astype(int)
    eq_id, eq_idx_inv = np.unique(eq_id_train, return_index=True)
    eq_X_train = df_coeffs_event[['eqX', 'eqY']].values
    
    c1e_mu, c1e_sig, c1e_cov= pygp.SampleCoeffs(coeff_X_eq, eq_X_train[eq_idx_inv,:], 
                                            c_data_mu = df_coeffs_event.loc[eq_idx_inv,'dc_1e_mean'].values, 
                                            c_data_sig = df_coeffs_event.loc[eq_idx_inv,'dc_1e_unc'].values,
                                            hyp_ell = ell_1e, hyp_omega = omega_1e)
    
    return dc_0_evt, c1e_mu, c1e_sig, c1e_cov

# 
def calc_site_adjust_term(coeff_X_sta, df_posterior_pdf_site,df_coeffs_site):
    #
    # hyper-parameters for site terms
    dc_0_site = df_posterior_pdf_site.loc['mean','dc_0']
    ell_1as   = df_posterior_pdf_site.loc['mean','ell_1as']
    omega_1as = df_posterior_pdf_site.loc['mean','omega_1as']
    
    sta_id_train = df_coeffs_site['ssn'].values.astype(int)
    sta_id, sta_idx_inv = np.unique(sta_id_train, return_index=True)
    stat_X_train = df_coeffs_site[['staX', 'staY']].values
    
    c1as_mu, c1as_sig, c1as_cov  = pygp.SampleCoeffs(coeff_X_sta, stat_X_train[sta_idx_inv,:], 
                                            c_data_mu = df_coeffs_site.loc[sta_idx_inv,'dc_1as_mean'].values, 
                                            c_data_sig = df_coeffs_site.loc[sta_idx_inv,'dc_1as_unc'].values,
                                            hyp_ell = ell_1as, hyp_omega = omega_1as)
    
    return dc_0_site, c1as_mu, c1as_sig, c1as_cov

# 
def calc_path_adjust_term(df_cellinfo, df_posterior_pdf_path,df_cellatten):
    #
    # hyper-parameters for path terms
    dc_0_path = df_posterior_pdf_path.loc['mean','dc_0']
    ell_ca1   = df_posterior_pdf_path.loc['mean','ell_ca1']
    omega_ca1 = df_posterior_pdf_path.loc['mean','omega_ca1']
    omega_ca2 = df_posterior_pdf_path.loc['mean','omega_ca2']
    mu_ca     = df_cellatten.c_a_erg.values[0]                  # cell attenuation
    pi_ca     = 0.0
    
    cell_X_train= df_cellatten[['cellX','cellY']].values
    
    cell_X      = df_cellinfo[['mptX','mptY']].values 

    ca_mu, ca_sig, ca_cov = pygp.SampleAttenCoeffsNegExp(cell_X, cell_X_train, 
                                            cA_data_mu = df_cellatten['c_ca_mean'].values, 
                                            cA_data_sig = df_cellatten['c_ca_unc'].values,
                                            mu_ca = mu_ca, ell_ca = ell_ca1, omega_ca = omega_ca1, 
                                            pi_ca = pi_ca, sigma_ca=omega_ca2)
    ca_mu[ca_mu>0] = 0.
    dca_mu = ca_mu - mu_ca
    
    
    return dc_0_path, ca_mu, dca_mu, ca_sig, ca_cov


#
# functions for non-ergodic prediction
def nerg_pred_for_ray(epicenter,df_posterior_pdf_event,df_coeffs_event,
                    destination_sta_lonlat, df_posterior_pdf_site, df_coeffs_site,
                    df_cellinfo, df_posterior_pdf_path, df_cellatten,
                    depth, utmProj, npts= 50):
    
    # points in the ray path
    pts_ray = interpolate_geodesic(epicenter[1], epicenter[0], destination_sta_lonlat[1], destination_sta_lonlat[0], npts)
    
    # for c1e
    coeff_X_eq = np.array([utmProj(epicenter[0],epicenter[1])])/1000
    dc_0_evt, c1e_mu, c1e_sig, c1e_cov=calc_event_adjust_term(coeff_X_eq, df_posterior_pdf_event, df_coeffs_event)
    
    # for c1as
    coeff_X_sta = np.array([utmProj(c_lon, c_lat) for c_lon, c_lat in 
                            zip(pts_ray[:,0], pts_ray[:,1])]) / 1000
    dc_0_site, c1as_mu, c1as_sig, c1as_cov = calc_site_adjust_term(coeff_X_sta, df_posterior_pdf_site,df_coeffs_site)
    
    # for c_cA
    # derive cell-specific adjustment att coefficient
    dc_0_path, ca_mu, dca_mu, ca_sig, ca_cov = calc_path_adjust_term(df_cellinfo, df_posterior_pdf_path, df_cellatten)
    
    # compute distance for each path
    flagUTM = True
    n_rcds = coeff_X_sta.shape[0]
    Ray4celldist = np.column_stack(( np.tile(coeff_X_eq,(n_rcds,1)),np.full((n_rcds,1),-1.0*depth),
                                        coeff_X_sta, np.full((n_rcds,1), 0.0) ))
    
    cells4dist  = df_cellinfo.loc[:,['q1X','q1Y','q1Z','q8X','q8Y','q8Z']].values
    
    distancematrix  = np.zeros([len(Ray4celldist), len(cells4dist)])
    for i in range(len(Ray4celldist)):
        pt1 = Ray4celldist[i,(0,1,2)]
        pt2 = Ray4celldist[i,(3,4,5)]
        
        dm = cell_dist.ComputeDistGridCells(pt1, pt2, cells4dist, flagUTM)
        distancematrix[i] = dm
    # derive distance matrix
    ca_adj = np.matmul(distancematrix,dca_mu)
    
    #
    # predicted mean values
    dc_0 = dc_0_evt + dc_0_path + dc_0_site
    logy_adj  = (dc_0 + c1e_mu + c1as_mu + ca_adj)           # how to consider c1bs
    
    return logy_adj

#
def nerg_pred_for_square_area(epicenter,df_posterior_pdf_event,df_coeffs_event,
                    df_posterior_pdf_site, df_coeffs_site,
                    df_cellinfo, df_posterior_pdf_path, df_cellatten,
                    depth, utmProj, X_grid, Y_grid, ngridx, ngridy):
    #
    # for c1e
    coeff_X_eq = np.array([utmProj(epicenter[0],epicenter[1])])/1000
    dc_0_evt, c1e_mu, c1e_sig, c1e_cov=calc_event_adjust_term(coeff_X_eq, df_posterior_pdf_event, df_coeffs_event)
    
    #
    # for c1as
    coeff_lonlat_win = np.vstack([X_grid.T.flatten(), Y_grid.T.flatten()]).T
    coeff_X_sta = np.array([utmProj(c_lon, c_lat) for c_lon, c_lat in 
                            zip(coeff_lonlat_win[:,0], coeff_lonlat_win[:,1])]) / 1000
    dc_0_site, c1as_mu, c1as_sig, c1as_cov = calc_site_adjust_term(coeff_X_sta, df_posterior_pdf_site,df_coeffs_site)
    
    #
    # for c_cA 
    
    # derive cell-specific adjustment att coefficient
    dc_0_path, ca_mu, dca_mu, ca_sig, ca_cov = calc_path_adjust_term(df_cellinfo, df_posterior_pdf_path, df_cellatten)
    
    # compute cell-specific distance for each ray path 
    flagUTM = True
    n_rcds = coeff_X_sta.shape[0]
    Ray4celldist = np.column_stack(( np.tile(coeff_X_eq,(n_rcds,1)),np.full((n_rcds,1),-1.0*depth),
                                        coeff_X_sta, np.full((n_rcds,1), 0.0) ))
    
    cells4dist  = df_cellinfo.loc[:,['q1X','q1Y','q1Z','q8X','q8Y','q8Z']].values
    
    distancematrix  = np.zeros([len(Ray4celldist), len(cells4dist)])
    for i in range(len(Ray4celldist)):
        pt1 = Ray4celldist[i,(0,1,2)]
        pt2 = Ray4celldist[i,(3,4,5)]
        
        dm = cell_dist.ComputeDistGridCells(pt1, pt2, cells4dist, flagUTM)
        distancematrix[i] = dm
    # derive distance matrix
    ca_adj = np.matmul(distancematrix,dca_mu)
    
    #
    # predicted mean values
    dc_0 = dc_0_evt + dc_0_path + dc_0_site
    logy_adj  = (dc_0 + c1e_mu + c1as_mu.reshape(ngridx, ngridy).T + ca_adj.reshape(ngridx, ngridy).T)           # how to consider c1bs
    
    return logy_adj
#




