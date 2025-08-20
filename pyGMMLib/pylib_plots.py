#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:00:32 2020

@author: glavrent
"""
# %% Required Packages
# ======================================
#load variables
import os
import sys

#arithmetic libraries
import numpy as np
import pandas as pd

#plottign libraries
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import ticker

#
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.default'] = 'regular'


#
# 
def scatter_plot(data_x, data_y, marker = 'x', size = 8, c = 'b', lw = 0.5,label = None,
                    fig=None, ax=None, xlabel=None, ylabel=None, 
                    cmap=None, vmin=0, vmax=1,
                    aexs_label_fontsize = 10, aexs_label_fontweight = 'bold',
                    xscale=None, xbase=10, yscale=None, ybase=10, text=None, 
                    legend = False, legend_loc = 'lower right'):
    
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize = (5, 3.8))
    
    if cmap is not None:
        sc=ax.scatter(data_x, data_y, marker=marker, s=size, c=c, linewidth=lw, zorder = 100, label = label,cmap=cmap,vmin=vmin,vmax=vmax)
    else:
        sc=ax.scatter(data_x, data_y, marker=marker, s=size, c=c, linewidth=lw, zorder = 100, label = label)
    if xscale is not None: ax.set_xscale(xscale,base = xbase)
    if yscale is not None: ax.set_yscale(yscale,base = ybase)
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=aexs_label_fontsize, fontweight=aexs_label_fontweight)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=aexs_label_fontsize, fontweight=aexs_label_fontweight)
    ax.grid(color = 'lightgray', which = 'both', linestyle = '-', linewidth = 0.3, zorder = 0)
    if text is not None:
        for each in text:
            ax.text(each['loc'][0],each['loc'][1],each['text'], transform=ax.transAxes, fontsize=7, backgroundcolor='white')
    if legend is True: ax.legend(loc = legend_loc)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    return fig, ax, sc
#
# 
def line_plot(data_x, data_y, c = 'k', lw = 1.5,label = None, ls='-',
                    fig=None, ax=None, xlabel=None, ylabel=None,
                    xscale=None, yscale=None, text=None, zorder = 100, 
                    legend = False, legend_loc = 'lower right'):
    
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize = (5, 3.8))
    
    ax.plot(data_x, data_y, c=c, linewidth=lw, zorder = 110, label = label, linestyle = ls)
    if xscale is not None: ax.set_xscale(xscale)
    if yscale is not None: ax.set_yscale(yscale)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.grid(color = 'gray', which = 'both', linestyle = '-', linewidth = 0.25, zorder = 0)
    if text is not None:
        for each in text:
            ax.text(each['loc'][0],each['loc'][1],each['text'], transform=ax.transAxes, fontsize=13, backgroundcolor='white')
    if legend is True: ax.legend(loc = legend_loc)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    return fig, ax
#
# only linear curve is currently supported
def plot_trendline(x, y, logx=False, logy=False, fig=None, ax=None,
                    c = 'red', lw = 1.5, ls='solid', label=None, return_coef = False):
    
    if fig is None and ax is None: 
        fig, ax = plt.subplots(1, 1, 1, figsize = (5, 3.))
        ax.scatter(x, y)
    
    if logx:
        x1 = np.log(x)
    else:
        x1 = list(x)
    
    if logy:
        y1 = np.log(y)
    else:
        y1 = list(y)
    xmin = np.min(x1)
    xmax = np.max(x1)
    
    coef = np.polyfit(x1,y1,1)
    poly1d_fn = np.poly1d(coef)                     # poly1d_fn is now a function which takes in x and returns an estimate for y
    x_pred = np.linspace(xmin,xmax,30)
    y_pred = poly1d_fn(x_pred)
    
    if logx: x_pred = np.exp(x_pred)
    if logy: y_pred = np.exp(y_pred)
    
    ax.plot(x_pred, y_pred, lw= lw, c=c, ls=ls, label=label, zorder=120)
    
    if return_coef:
        return fig, ax, coef
    else:
        return fig, ax 
#
#
# ------------------------------------------------------------------------------
def binned_mean(data_df0, n_bins=15, x_ax='RockSpec01', y_ax='AmpR01', 
                x_in_log=True, y_in_log=True):
    '''
    binning the dataset by PGA
    
    Parameters
    ----------
    data_df0 : pandas.DataFrame
        ampl ratio dataframe
    n_bins : int
        number of bins

    Returns
    -------
    bin_center : list
        center of the bins
    bin_PGA_mean : pandas.series
        logarithm mean of the ampl ratio in each bin
    bin_PGA_std : pandas.series
        logarithm standard deviation of the ampl ratio in each bin
    '''
    # 
    
    data_df = data_df0.copy()
    flag_amp = y_ax
    flag_rockSpec = x_ax
    if x_in_log: 
        data_df=data_df.assign(logPGA=np.log(data_df[flag_rockSpec]))
    else:
        data_df=data_df.assign(logPGA=data_df[flag_rockSpec])
    
    if y_in_log:
        data_df=data_df.assign(logAmp=np.log(data_df[flag_amp]))
    else:
        data_df=data_df.assign(logAmp=data_df[flag_amp])
    #
    binned_PGA_flag, log_PGA_bins = pd.cut(data_df['logPGA'], bins=n_bins, 
                                        retbins=True, labels=np.arange(n_bins))
    data_df=data_df.assign(logPGAFlag=binned_PGA_flag)
    bin_center = data_df.groupby(by=['logPGAFlag'])['logPGA'].mean()
    bin_mean=data_df.groupby(by=['logPGAFlag'])['logAmp'].mean()
    bin_std=data_df.groupby(by=['logPGAFlag'])['logAmp'].std()
    
    if x_in_log:
        bin_center_arr = np.exp(bin_center.to_numpy())
    else:
        bin_center_arr = bin_center.to_numpy()
    
    if y_in_log:
        bin_mean_arr = np.exp(bin_mean.to_numpy())
    else:
        bin_mean_arr = bin_mean.to_numpy()
    
    # drop the additive columns
    data_df.drop(columns=['logPGA','logAmp','logPGAFlag'],inplace=True)
    
    bin_data = np.column_stack((bin_center_arr, bin_mean_arr, bin_std.to_numpy()))
    mask = np.all(np.isnan(bin_data), axis=1)
    bin_data = bin_data[~mask]

    return bin_data
#
#
def plot_mean_errorbar(center, mean, std, fig=None, ax=None, plot_error = True,
                        marker = 's', size = 10, c = 'k', lw = 0.5,
                        errorlw = 1.0, xlabel=None, ylabel=None, xscale=None, yscale=None,
                        ls = '', meanlabel=None,errorlabel=None,capsize=1):
    '''
    
    '''
    
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize = (5, 3.8))
    
    ax.scatter(center, mean, marker=marker, s=size, c=c, linewidth=lw, label = meanlabel,zorder = 1000)
    if plot_error: ax.errorbar(center, mean, yerr=std, capsize=capsize, linewidth=errorlw, c = c, linestyle=ls,zorder = 1000, label=errorlabel)
    if xscale is not None: ax.set_xscale(xscale)
    if yscale is not None: ax.set_yscale(yscale)
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=15)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=15)
    
    return fig, ax
#
