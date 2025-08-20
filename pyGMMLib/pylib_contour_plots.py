#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:12:38 2019

@author: glavrent
"""

## load libraries
#arithmetic
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
#plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
from matplotlib import ticker
#from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#base map
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
#geometry libraries
from shapely.geometry import Point as shp_pt, Polygon as shp_poly
import pyproj


# plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.linewidth'] = 0.75

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

## Main functions
##---------------------------------------

# Updated PlotContourCAMap function
#----  ----  ----  ----  ----  ----  ----
def PlotContourMap(cont_latlondata, cmin=None,  cmax=None, 
                    flag_grid=True, title=None, cbar_label=None, log_cbar = False, 
                    frmt_clb = '%.2f', cmap = 'viridis', 
                    fig = None, ax = None, 
                    data_crs= ccrs.PlateCarree(), subplot=(1,1,1), 
                    plt_res = '50m', plt_scale = '50m',
                    add_land = True, add_state=True, add_border= True, add_ocean=True,
                    lonLoc=None, latLoc= None, lon_lims=None, lat_lims=None,
                    lonLabel=False, latLabel=False,
                    **kwargs):
    '''
    PlotContourCAMap:
        simplifed function to create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        cont_latlondata (np.array [n1,3]):       contains the latitude, logitude and contour values
                                                 cont_latlondata = [lat, long, data]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    #additional input arguments
    flag_smooth  = kwargs['flag_smooth']   if 'flag_smooth'   in kwargs else False
    sig_smooth   = kwargs['smooth_sig']    if 'smooth_sig'    in kwargs else 0.1
    intrp_method = kwargs['intrp_method']  if 'intrp_method'  in kwargs else 'linear'
        
    plt_res = '50m'
    plt_scale = '50m'

    #number of interpolation points, x & y direction
    ngridx = 500
    ngridy = 500

    #create figure
    if fig is None:
        fig = plt.figure(figsize=(10, 7.6))
        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], projection=data_crs)
    
    #add costal lines
    ax.coastlines(resolution=plt_res, edgecolor='black', zorder=5)
    
    # add basemap
    if add_state:
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                              scale=plt_scale, facecolor='none')
        ax.add_feature(states, edgecolor='black', zorder=3)
    if add_border:
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                               scale=plt_scale, facecolor='none')
        ax.add_feature(borders, edgecolor='black', zorder=4)
    
    if add_ocean:
        oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', facecolor='lightblue', scale=plt_scale)
        ax.add_feature(oceans, zorder=1)
    
    if add_land:
        lands = cfeature.LAND
        ax.add_feature(lands, zorder=1)
    
    #add figure title
    if (not title is None): plt.title(title, fontsize=12)
    
    # linewidth of the frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    
    #project contour data
    x_cont = cont_latlondata[:,1] 
    y_cont = cont_latlondata[:,0]
    
    #interpolation grid
    x_int = np.linspace(x_cont.min(), x_cont.max(), ngridx)
    y_int = np.linspace(y_cont.min(), y_cont.max(), ngridy)
    X_grid, Y_grid = np.meshgrid(x_int, y_int)
    
    #interpolate contour data on grid
    if log_cbar:
        data_cont = np.log(cont_latlondata[:,2])
    else:
        data_cont = cont_latlondata[:,2]
    data_grid = griddata((x_cont, y_cont) , data_cont, (X_grid, Y_grid), method=intrp_method )
    
    # smooth 
    if flag_smooth: 
        data_grid = gaussian_filter(data_grid, sigma=sig_smooth)     
    
    #data colorbar
    cbmin = data_cont.min() if cmin is None else cmin
    cbmax = data_cont.max() if cmax is None else cmax
    clevs = np.linspace(cbmin, cbmax, 41).tolist()    
    
    # interpolated data    
    if 'boundary' in kwargs:
        boundary = kwargs['boundary']
        utm_zone = '54'
        utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+" +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        coeff_X_polyg = np.array([utmProj(c_lon, c_lat) for c_lat, c_lon in 
                          zip(boundary[:,0], boundary[:,1])]) / 1000
        shp_coeff_X_polyg = shp_poly(coeff_X_polyg)
        
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                c_lon = X_grid[i,j]
                c_lat = Y_grid[i,j]
                c_x, c_y=utmProj(c_lon, c_lat)
                shp_c_x = shp_pt((c_x/1000, c_y/1000))
                if not shp_c_x.within(shp_coeff_X_polyg):
                    data_grid[i,j] = np.nan
    
    # plot data
    cs =  ax.contourf(X_grid, Y_grid, data_grid, transform = data_crs, vmin=cmin, vmax=cmax, 
                      levels = clevs, zorder=3, alpha = 0.75, cmap=cmap)
    
    #color bar
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    cbar_ticks = clevs[0:41:8]
    # cax = fig.add_axes([0.8, 0.26, 0.16, 0.02])
    cbbox = inset_axes(ax, '42%', '3.5%', loc='lower left', 
                    bbox_to_anchor=(0.51, 0.12, 1, 1),
                    bbox_transform=ax.transAxes)
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(axis='both', left='off', top='off', right='off', bottom='off', 
                        labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    cbbox.set_facecolor('white')
    cax = inset_axes(cbbox, '100%', '100%', loc = 10)
    cbbox.xaxis.set_major_locator(ticker.NullLocator())
    cbbox.yaxis.set_major_locator(ticker.NullLocator())
    cax.xaxis.set_ticks_position("bottom")
    cax.yaxis.set_major_locator(ticker.NullLocator())
    
    cbar = fig.colorbar(cs, cax=cax, boundaries=clevs, ticks=cbar_ticks, 
                            pad=0.075, orientation="horizontal", 
                            format=fmt_clb, shrink=0.55)
    if log_cbar:
        cbar_labels = [frmt_clb%np.exp(c_t) for c_t in cbar_ticks]
        cbar.set_ticklabels(cbar_labels)

    cbar.ax.tick_params(labelsize=8) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=8, labelpad=-1.0)
    
    
    #grid lines
    if flag_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.4, color='gray', alpha=0.5, 
                          linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xlocator = mticker.FixedLocator(lonLoc)
        gl.ylocator = mticker.FixedLocator(latLoc)
    else:
        gl = None
    #
    if lon_lims is not None:
        ax.set_xlim(lon_lims)
    if lat_lims is not None:
        ax.set_ylim(lat_lims)
    
    # add axis label
    if latLabel:
        ax.text(-0.1, 0.55, 'Latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    if lonLabel:
        ax.text(0.5, -0.12, 'Longitude', va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    
    fig.tight_layout()
    
    return fig, ax, cbar, data_crs, gl

# Base plot function
#----  ----  ----  ----  ----  ----  ----
def PlotMap(lat_lims = None, lon_lims = None, title=None,
            fig = None, ax=None, 
            data_crs = ccrs.PlateCarree(), plt_res = '50m', plt_scale = '50m',
            add_land = True, add_state=True, add_border= True, add_ocean=True,
            flag_grid=False, lonLoc = None, latLoc = None,
            lonLabel=True,latLabel=True):
    '''
    PlotContourCAMap:
        simplifed function to create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        line_latlondata (np.array [n1,3]):       contains the latitude, logitude and contour values
                                                 cont_latlondata = [lat, long, data]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    
    
    #create figure
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    
    #create basemap
    if lon_lims is not None:
        ax.set_xlim(lon_lims)
    if lat_lims is not None:
        ax.set_ylim(lat_lims)

    if add_land:
        lands = cfeature.LAND
        ax.add_feature(lands, zorder=1)
    
    if add_state:
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                                  scale=plt_scale, facecolor='none')
        ax.add_feature(states, edgecolor='black', zorder=3)
    
    if add_border:
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                               scale=plt_scale, facecolor='none')
        ax.add_feature(borders, edgecolor='black', zorder=4)
    
    if add_ocean:
        oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', facecolor='lightblue', scale=plt_scale)
        ax.add_feature(oceans, zorder=6)
    
    ax.coastlines(resolution=plt_res, edgecolor='black', zorder=3)
    
    #add figure title
    if (not title is None): plt.title(title, fontsize=25)
    
    #grid lines
    if flag_grid:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.4, color='gray', alpha=0.5, 
                          linestyle='--',zorder=7)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
        gl.xlocator = mticker.FixedLocator(lonLoc)
        gl.ylocator = mticker.FixedLocator(latLoc)
    else:
        gl = None
    
    # Add ticks with correct CRS
    ax.set_xticks(lonLoc, crs=ccrs.PlateCarree())
    ax.set_yticks(latLoc, crs=ccrs.PlateCarree())

    # Optional: format tick labels
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    
    # 
    if latLabel and lonLabel:
        ax.text(-0.11, 0.55, 'Latitude', va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=12)
        ax.text(0.5, -0.15, 'Longitude', va='bottom', ha='center',
                rotation='horizontal', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=12)
    
    return fig, ax, data_crs, gl


# plot Att coefficient for each cell 
#----  ----  ----  ----  ----  ----  ----
def PlotCellAtt(cell_latlondata, cmin=None, cmax=None, flag_grid=False, 
                    title=None, cbar_label=None, log_cbar = False, frmt_clb = '%.2f',
                    alpha_v = 1.0, cell_size = 50, cmap='seismic', 
                    fig = None, ax = None, data_crs= ccrs.PlateCarree(), 
                    subplot=(1,1,1), plt_res = '50m', plt_scale = '50m',
                    add_land = True, add_state=True, add_border= True, add_ocean=True,
                    lonLoc=None, latLoc= None, lon_lims=None, lat_lims=None,
                    lonLabel = True, latLabel=True):
    '''
    PlotCellsCAMap:
        PlotCellsCAMap function to create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        cell_latlondata (np.array [n1,9]):       contains the latitude, logitude and color values
                                                 cell_latlondata = [lat, long, data]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    
    #create figure
    if fig is None:
        fig = plt.figure(figsize=(10/2.54, 7.6/2.54))
        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], projection=data_crs)
    
    #add costal lines
    ax.coastlines(resolution=plt_res, edgecolor='black', linewidth=1.0, zorder=5)
    
    #basemap
    if add_land:
        lands = cfeature.LAND
        ax.add_feature(lands, zorder=1)
    if add_state:
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                          scale=plt_scale, facecolor='none')
        ax.add_feature(states, edgecolor='black', zorder=3)
    if add_border:
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                           scale=plt_scale, facecolor='none')
        ax.add_feature(borders, edgecolor='black', zorder=4)
    if add_ocean:
        oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', scale=plt_scale, facecolor='lightblue')
        ax.add_feature(oceans, zorder=6)
    
    
    #
    if lon_lims is not None:
        ax.set_xlim(lon_lims)
    if lat_lims is not None:
        ax.set_ylim(lat_lims)
    
    
    #
    # plot data
    cell_xy = cell_latlondata[:,:-1] 
    cell_val = cell_latlondata[:,-1]
    if log_cbar:
        data_cell = np.log(cell_val)
    else:
        data_cell = cell_val
    
    cbmin = data_cell.min() if cmin is None else cmin
    cbmax = data_cell.max() if cmax is None else cmax
    clevs = np.linspace(cbmin, cbmax, 41).tolist()
    
    cmap = cm.Oranges_r
    norm = matplotlib.colors.Normalize(vmin=cbmin, vmax=cbmax)
    
    for i_ce, ce_xy in enumerate(cell_xy):
        data = data_cell[i_ce]
        pts=(data-cbmin)/(cbmax-cbmin)
        c = cmap(pts)
        ax.fill(ce_xy[[1, 3, 5, 7]], ce_xy[[0, 2, 4, 6]], 
                color=c, transform=data_crs, alpha=0.98, linewidth=0.25)
    
    #plot color bar
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    cbar_ticks = clevs[0:41:10]
    
    # cax = fig.add_axes([0.8, 0.26, 0.16, 0.02])
    cbbox = inset_axes(ax, '40%', '3.5%', loc='lower left', 
                    bbox_to_anchor=(0.51, 0.12, 1, 1),
                    bbox_transform=ax.transAxes)
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(axis='both', left='off', top='off', right='off', bottom='off', 
                        labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    cbbox.set_facecolor('white')
    cax = inset_axes(cbbox, '100%', '100%', loc = 10)
    cbbox.xaxis.set_major_locator(ticker.NullLocator())
    cbbox.yaxis.set_major_locator(ticker.NullLocator())
    cax.xaxis.set_ticks_position("bottom")
    cax.yaxis.set_major_locator(ticker.NullLocator())
    
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, cax=cax,
                            boundaries=clevs, ticks=cbar_ticks, 
                            pad=0.075, orientation="horizontal", 
                            format=fmt_clb, shrink=0.5)
    
    if log_cbar:
        cbar_labels = [frmt_clb%np.exp(c_t) for c_t in cbar_ticks]
        cbar.set_ticklabels(cbar_labels,fontsize=8)

    cbar.ax.tick_params(labelsize=8) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=8, labelpad=-1.0)
    
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.75)  # Adjust to your desired thickness
    
    
    # add figure title
    if (not title is None): ax.set_title(title, fontsize=11)
    
    
    #grid lines
    if flag_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.2, color='gray', alpha=0.5, 
                          linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        # gl.xlocator = mticker.FixedLocator(lonLoc)
        # gl.ylocator = mticker.FixedLocator(latLoc)
        
        # Add ticks with correct CRS
        # ax.set_xticks(lonLoc, crs=ccrs.PlateCarree())
        # ax.set_yticks(latLoc, crs=ccrs.PlateCarree())

        # Optional: format tick labels
        # ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        # ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        
    else:
        gl = None
    
    # add axis label
    '''
    if latLabel:
        ax.text(-0.1, 0.55, 'Latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    if lonLabel:
        ax.text(0.5, -0.12, 'Longitude', va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    '''
    
    return fig, ax, cbar, data_crs, gl

# plot map for attribution of cells with scatter points
#----  ----  ----  ----  ----  ----  ----
def PlotCellsMap(cell_latlondata, cmin=None,  cmax=None, flag_grid=False, 
                    title=None, cbar_label=None, log_cbar = False, frmt_clb = '%.2f',
                    alpha_v = 1.0, cell_size = 50, cmap='seismic', 
                    fig = None, ax = None, data_crs= ccrs.PlateCarree(), 
                    subplot=(1,1,1), plt_res = '50m', plt_scale = '50m',
                    add_land = True, add_state=True, add_border= True, add_ocean=True,
                    lonLoc=None, latLoc= None, lon_lims=None, lat_lims=None,
                    lonLabel=True,latLabel=True, resColorBar= False):
    '''
    PlotCellsCAMap:
        PlotCellsCAMap function to create a contour plot of the data in cont_latlondata
        
    Input Arguments:
        cell_latlondata (np.array [n1,3]):       contains the latitude, logitude and color values
                                                 cell_latlondata = [lat, long, data]
        cmin (double-opt):                       lower limit for color levels for contour plot 
        cmax (double-opt):                       upper limit for color levels for contour plot 
        title (str-opt):                         figure title
        cbar_label (str-opt):                    contour plot color bar label
        ptlevs (np.array-opt):                   color levels for points
        pt_label (str-opt):                      points color bar label
        log_cbar (bool-opt):                     if true use log-scale for contour plots
        frmt_clb                                 string format color bar ticks
    
    Output Arguments:
        
    '''
    
    #create figure
    if fig is None:
        fig = plt.figure(figsize=(10, 7.6))
        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], projection=data_crs)
    
    #add costal lines
    ax.coastlines(resolution=plt_res, edgecolor='black', zorder=5)
    
    #basemap
    if add_land:
        lands = cfeature.LAND
        ax.add_feature(lands, zorder=1)
    if add_state:
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                          scale=plt_scale, facecolor='none')
        ax.add_feature(states, edgecolor='black', zorder=3)
    if add_border:
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                           scale=plt_scale, facecolor='none')
        ax.add_feature(borders, edgecolor='black', zorder=4)
    if add_ocean:
        oceans = cfeature.NaturalEarthFeature(category='physical', name='ocean', scale=plt_scale, facecolor='lightblue')
        ax.add_feature(oceans, zorder=6)
    
    # add figure title
    if (not title is None): ax.set_title(title, fontsize=14)
    
    #grid lines
    if flag_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.4, color='gray', alpha=0.5, 
                          linestyle='--',zorder = 7)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
        gl.xlocator = mticker.FixedLocator(lonLoc)
        gl.ylocator = mticker.FixedLocator(latLoc)
    else:
        gl = None
    
    #plot data
    x_cell = cell_latlondata[:,1] 
    y_cell = cell_latlondata[:,0]
    if log_cbar:
        data_cell = np.log(cell_latlondata[:,2])
    else:
        data_cell = cell_latlondata[:,2]
    
    cs =  ax.scatter(x_cell, y_cell, s = cell_size, c = data_cell, 
                        transform = data_crs, 
                        vmin=cmin, vmax=cmax, zorder=100,
                        alpha = alpha_v, cmap=cmap)


    #plot color bar
    cbmin = data_cell.min() if cmin is None else cmin
    cbmax = data_cell.max() if cmax is None else cmax
    clevs = np.linspace(cbmin, cbmax, 41).tolist()    
    if resColorBar:
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=0,vmin=cbmin, vmax=cbmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=cbmin, vmax=cbmax)
    
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    cbar_ticks = clevs[0:41:8]
    # cax = fig.add_axes([0.8, 0.26, 0.16, 0.02])
    cbbox = inset_axes(ax, '35%', '3.5%', loc='lower left', 
                    bbox_to_anchor=(0.6, 0.12, 1, 1),
                    bbox_transform=ax.transAxes)
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(axis='both', left='off', top='off', right='off', bottom='off', 
                        labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    cbbox.set_facecolor('white')
    cax = inset_axes(cbbox, '100%', '100%', loc = 10)
    cbbox.xaxis.set_major_locator(ticker.NullLocator())
    cbbox.yaxis.set_major_locator(ticker.NullLocator())
    cax.xaxis.set_ticks_position("bottom")
    cax.yaxis.set_major_locator(ticker.NullLocator())
    
    cbar = fig.colorbar(cs, cax=cax, boundaries=clevs, ticks=cbar_ticks, 
                            pad=0.075, orientation="horizontal", 
                            format=fmt_clb, shrink=0.5, norm=norm)
    if log_cbar:
        cbar_labels = [frmt_clb%np.exp(c_t) for c_t in cbar_ticks]
        cbar.set_ticklabels(cbar_labels)

    cbar.ax.tick_params(labelsize=10) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=10, labelpad=-1.0)
    
    
    # add axis label
    if latLabel:
        ax.text(-0.1, 0.55, 'Latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    if lonLabel:
        ax.text(0.5, -0.12, 'Longitude', va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    
    #
    if lon_lims is not None:
        ax.set_xlim(lon_lims)
    if lat_lims is not None:
        ax.set_ylim(lat_lims)
    
    return fig, ax, cbar, data_crs, gl

#




