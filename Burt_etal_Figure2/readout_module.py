# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:43:36 2020

@author: Philipp
"""


import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
import warnings

def get_maximum(x, y):
    """
    interpolate maximum
    """
    f = InterpolatedUnivariateSpline(x, y, k=4)
	# get Nullstellen
    cr_pts = f.derivative().roots()
    cr_pts = np.append(cr_pts, (x[0], x[-1]))  # also check the endpoints of the interval
    cr_vals = f(cr_pts)
    max_index = np.argmax(cr_vals)

    max_x = cr_pts[max_index]
    max_y = cr_vals[max_index]

    return max_x, max_y


def get_tau(time, cells):
    """
    get halftime of peak
    """
    cells = cells.array 
    crit = check_criteria(cells)
    
    if crit == True:
    
        peak_idx = np.argmax(cells)
        # get max value
        peak = cells[peak_idx]
        peak_half = peak / 2.
        #print(peak)
        cells = cells[:(peak_idx+1)]
        time = time[:(peak_idx+1)]
        # assert that peak is not at beginning
        if peak_idx <= 3:
            tau = np.nan
        # assert that peak half is in new cell array
        elif np.all(peak_half<cells):
            tau = np.nan
        else:
            f = interpolate.interp1d(cells, time)           
            tau = f(peak_half)
            tau = float(tau)
            
    else:
        tau = np.nan
            
    return tau    



def get_peak_height(time, cells):
    """
    get height of peak
    """
    cells = cells.array
    time = time.array
    crit = check_criteria(cells)

    if crit == True:
        peaktime, peak_val = get_maximum(time, cells)
    else: 
        peak_val = np.nan
        
    return peak_val



def get_peaktime(time, cells):
    """
    get time of peak
    """
    cells = cells.array
    time = time.array
    crit = check_criteria(cells)
    
    if crit == True:
        peaktime, peak_val = get_maximum(time, cells)
    else:
        peaktime = np.nan
    
    return peaktime



def get_duration(time, cells):
    """
    get total time when cells reach given threshold
    """
    cells = cells.array
    time = time.array
    crit = check_criteria(cells)
    thres = 0.001
    if crit == True and (cells > thres).any():
        # get times where cells are > value
        time2 = time[cells > thres]
        #print(time2)
        #use last element 
        dur = time2[-1]

    else:
        dur = np.nan
        
    return dur
        

def get_decay(time, cells):
    """
    get the half-time of decay
    """
    
    cells = cells.array 
    crit = check_criteria(cells)
    cellmax = np.amax(cells)
    cellmin = cells[-1] 
    
    if crit == True:
        peak_id = np.argmax(cells)
        cells = cells[peak_id:]
        time = time[peak_id:]
        
        # make sure there are at least two values in the array
        assert len(cells) > 1
        
        # interpolate to get time unter half of diff between max and arr end is reached
        celldiff = (cellmax - cellmin) / 2
        celldiff = cellmax - celldiff
        f = interpolate.interp1d(cells, time)
        #print(cellmax, cellmin, celldiff)
        tau = f(celldiff)
    else:
        tau = np.nan
    
    return float(tau)


def get_area(time, cells):
    
    cells = cells.array 
    crit = check_criteria(cells)
    
    if crit == True:
        area = np.trapz(cells, time)
    else: 
        area = np.nan
        
    return area


def check_criteria(cells): 
    cellmax = np.amax(cells)
    cellmin = cells[-1]
    last_cells = cells[-10]
    # test first if peak id is at the end of array
    peak_id = np.argmax(cells)
    if peak_id >= len(cells) - 12:
        return False
    
    # check if cells increase and decrease around peak monotonically
    arr_inc = np.diff(cells[(peak_id-10):peak_id]) > 0
    arr_dec = np.diff(cells[peak_id:(peak_id+10)]) < 0
    crit4 = arr_inc.all() and arr_dec.all()

    # check difference between max and endpoint
    crit1 = np.abs(cellmax-cellmin) > 1e-3
    # check that max is higher than endpoint
    crit2 = cellmax > cellmin
    # check that something happens at all
    crit3 = np.std(cells) > 0.001
    # check that last cells are close to 0
    crit5 = (last_cells < 1e-1).all()
    crit5 = np.std(last_cells) < 1e-1

    criteria = [crit1, crit2, crit3, crit4, crit5]
    crit = True if all(criteria) else False

    skip_criteria = False
    if skip_criteria:
        crit = True
        warnings.warn("criteria for readout quality control disabled")
    return crit