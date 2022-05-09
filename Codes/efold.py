import numpy as np
import xarray as xr
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import pickle
mpl.rc('font',size=12,weight='bold') #set default font size and weight for plots
import warnings
warnings.filterwarnings("ignore")
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy.linalg as linalg
from xarray import DataArray
import statsmodels.api as sm

t_max_sample = xr.open_mfdataset("/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_ACCESS1-0_historical_rcp85_r1i1p1_19500101-20101231.nc",chunks='auto',decode_times=True)

lat = t_max_sample.lat.data
lon = t_max_sample.lon.data

print('test1')

multimodel = ["/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_ACCESS1-0_historical_rcp85_r1i1p1_19500101-20101231.nc", 
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_bcc-csm1-1_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_bcc-csm1-1-m_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_BNU-ESM_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_CanESM2_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_CCSM4_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_CESM1-CAM5_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_CNRM-CM5_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_CSIRO-Mk3-6-0_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_GFDL-CM3_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_GFDL-ESM2G_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_GFDL-ESM2M_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_HadGEM2-AO_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_HadGEM2-CC_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_inmcm4_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_IPSL-CM5A-LR_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_IPSL-CM5A-MR_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_MIROC5_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_MIROC-ESM-CHEM_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_MIROC-ESM_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_MPI-ESM-LR_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_MPI-ESM-MR_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_MRI-CGCM3_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_NorESM1-ME_historical_rcp85_r1i1p1_19500101-20101231.nc",
"/export/data/BCCAQv2/tasmax/raw_data/hist/tasmax_day_BCCAQv2_ANUSPLIN300_NorESM1-M_historical_rcp85_r1i1p1_19500101-20101231.nc"]


multimodel_name = ["ACCESS1-0","bcc-csm1-1","bcc-csm1-1-m","BNU-ESM","CanESM2","CCSM4","CESM1-CAM5",
                  "CNRM-CM5","CSIRO-Mk3-6-0","GFDL-CM3","GFDL-ESM2G","GFDL-ESM2M","HadGEM2-AO",
                   "HadGEM2-CC","inmcm4","IPSL-CM5A-LR","IPSL-CM5A-MR","MIROC5","MIROC-ESM-CHEM",
                   "MIROC-ESM","MPI-ESM-LR","MPI-ESM-MR","MRI-CGCM3","NorESM1-ME","NorESM1-M"]

print('test2')


################
# Autocorrelation e-time functions
################

def auto_efold(a_temp):
    ecorr = np.correlate((a_temp-np.mean(a_temp))/np.std(a_temp),(a_temp-np.mean(a_temp))
                         /np.std(a_temp),'same')/len(a_temp)
    a = ecorr[int(len(ecorr)/2)+1] #Find lag 1
    aa = -1/np.log(a) #return e-folding time
    return aa

def auto_efold_xarrray(a_temp,dim="time"):
    auto_efold_xarrray = xr.apply_ufunc(
        auto_efold,
        a_temp,
        input_core_dims=[[dim]],
        #output_dtypes=[a_temp.dtype],
        #exclude_dims=set(("time",)), 
        vectorize=True).rename({"tasmax": "efold"})
        #dask='parallelized')
    return auto_efold_xarrray

print('test3')

################
# Loop over climate data
################

efold_multimodel=[]

for file_name in multimodel:
         tm = xr.open_dataset(file_name)
         print(file_name)
          
         t_temp = tm.where((tm['time.month'] == 6) & (tm['time.year'] == 1981), drop=True)            
         #calculate
         a_efold = auto_efold_xarrray(t_temp.load())            
         #append
         efold_multimodel.append(a_efold)
        
t_mem = xr.concat(efold_multimodel,pd.Index(range(25),name='model'))

#t_mem.to_netcdf(path="/home/gbaggio/efold_1981_January.nc", mode='w')

print('test4')

################
# Figure
################

fig, axs = plt.subplots(5,5, figsize=(20, 25), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .2, wspace= .2)

axs = axs.ravel()

for i in range(len(t_mem['model'])):

    pc = axs[i].pcolormesh(t_mem.lon,t_mem.lat,t_mem['efold'][i,:,:],cmap="Reds",vmin=0,vmax=10)
    axs[i].set_title(multimodel_name[i])
    axs[i].set_ylim([39,81])
    axs[i].set_xlim([-141,-59])
    #axs[i].set_yticks([40,50,60,70,80])
    #axs[i].set_xticks([-140,-120,-100,-80,-60])
    #cax,kw = mpl.colorbar.make_axes(axs[i],location='bottom')#,pad=0.05,shrink=0.7)
    #out=fig.colorbar(pc,cax=cax,extend='both',**kw)    #axs[i].axis("tight")

    if i in np.linspace(0,20,5):
        axs[i].set_yticks([40,50,60,70,80])
    else:
        axs[i].set_yticks([])

    if i in np.linspace(20,24,5):
        axs[i].set_xticks([-140,-120,-100,-80,-60])
    else:
        axs[i].set_xticks([])         
    
fig.suptitle('Multi-Model Ensemble Autocorrelation e-folding time - June, 1981',
             x=0.5,y=0.91,fontsize=18,fontweight="bold")

cax,kw = mpl.colorbar.make_axes(axs,location='bottom',pad=0.03,shrink=0.4)
out=fig.colorbar(pc,cax=cax,extend='both',**kw)

out.set_label("E-folding time (days)", fontweight='bold',fontsize=14)

#fig.tight_layout()
#plt.show()

fig.savefig('efoldJune-1981.png')

print('test5')

################
# Lag 1 calculation
################

# def lag1_numpy(a_temp):
#     ecorr = np.correlate((a_temp-np.mean(a_temp))/np.std(a_temp),(a_temp-np.mean(a_temp))
#                          /np.std(a_temp),'same')/len(a_temp)
#     lag1_numpy = ecorr[int(len(ecorr)/2)+1] #Find lag 1
    
#     return lag1_numpy

# t_lag1_multimodel=[]

# for file_name in multimodel:
#         tm = xr.open_dataset(file_name)
#         t_temp = tm.where((tm['time.year'] > 1949) & (tm['time.year'] < 1952) , drop=True)
#         #calculate
#         lag1 = xr.apply_ufunc(
#             lag1_numpy, # first the function  
#             t_temp.chunk(
#                 {"lat": 200, "lon": 200}
#             ),# now arguments in the order expected by 'interp1_np' # now arguments in the order expected by 'interp1_np' 
#             input_core_dims=[["time"]],  # list with one entry per arg
#             #output_core_dims=[[]],  # returned data has one dimension
#             exclude_dims=set(("time",)),  # dimensions allowed to change size. Must be a set!
#             vectorize=True,  # loop over non-core dims
#             dask="parallelized",
#             output_dtypes=[tm['tasmax'].dtype]).rename({"tasmax": "lag1"})
        
#         #append
#         t_lag1_multimodel.append(lag1)
        
# t_mlm = xr.concat(t_lag1_multimodel,pd.Index(range(3),name='model'))

# print('test3')

# t_mlm = t_mlm.load()

# ####### Fig1 Lag 1

# fig = plt.figure(figsize=(16,12))
# ax = plt.axes(projection=ccrs.PlateCarree())
# #ax.set_global()
# ax.coastlines()
# ax.gridlines(linewidth=1)
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                  linewidth=1, color='darkgrey')
# gl.xlabels_top = False
# gl.ylabels_left = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 15, 'color': 'gray'}
# gl.xlabel_style = {'color': 'black', 'weight': 'bold'}

# # uncomment and complete the line below (see the NAO notebook for a reminder)
# pc = ax.pcolormesh(t_mlm.lon,t_mlm.lat,t_mlm.mean('model'),cmap="Reds")
# cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.7)
# out=fig.colorbar(pc,cax=cax,extend='both',**kw)
# out.set_label('Lag 1',size=14)
# ax.set_title('Multi-Model Ensemble Max. Temperature Lag 1 - January, 1950', 
#              pad = 15,fontweight='bold',size=16)

# fig.savefig('lag1-test1.png')

# print('test9')

# ####### Fig2 Lag 1

# fig, axs = plt.subplots(5,5, figsize=(20, 25), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace = .2, wspace= .2)

# axs = axs.ravel()

# for i in range(len(t_mlm['model'])):

#     pc = axs[i].pcolormesh(t_mlm.lon,t_mlm.lat,t_mlm['lag1'][i,:,:],cmap="Reds",vmin=0.2,vmax=0.85)
#     axs[i].set_title(multimodel_name[i])
#     axs[i].set_ylim([39,81])
#     axs[i].set_xlim([-141,-59])
#     #axs[i].set_yticks([40,50,60,70,80])
#     #axs[i].set_xticks([-140,-120,-100,-80,-60])
#     #cax,kw = mpl.colorbar.make_axes(axs[i],location='bottom')#,pad=0.05,shrink=0.7)
#     #out=fig.colorbar(pc,cax=cax,extend='both',**kw)    #axs[i].axis("tight")

#     if i in np.linspace(0,20,5):
#         axs[i].set_yticks([40,50,60,70,80])
#     else:
#         axs[i].set_yticks([])

#     if i in np.linspace(20,24,5):
#         axs[i].set_xticks([-140,-120,-100,-80,-60])
#     else:
#         axs[i].set_xticks([])       
    
    
# fig.suptitle('Multi-Model Ensemble Max. Temperature Lag 1 - January, 1950',
#              x=0.5,y=0.91,fontsize=18,fontweight="bold")

# cax,kw = mpl.colorbar.make_axes(axs,location='bottom',pad=0.03,shrink=0.4)
# out=fig.colorbar(pc,cax=cax,extend='both',**kw)

# out.set_label("Autocorrelation Lag 1", fontweight='bold',fontsize=14)

# plt.show()

# fig.savefig('lag1-test2.png')

# print('test10')

# t_mean_multimodel=[]
# t_std_multimodel=[]

# for file_name in multimodel:
#         tm = xr.open_mfdataset(file_name,chunks='auto',decode_times=True)
#         #calculate
#         t_mean = tm.where((tm['time.year'] > 1980) & (tm['time.year'] < 2012) , drop=True).groupby('time.season').mean('time')
#         print()
#         #t_std = tm.loc[dict(time=slice('1981', '2010'))].groupby('time.season').std('time')
#         #append
#         t_mean_multimodel.append(t_mean)
#         #t_std_multimodel.append(t_std)

# t_mmm = xr.concat(t_mean_multimodel,pd.Index(range(25),name='model')).rename({"tasmax": "t_mean"})
# #t_msm = xr.concat(t_std_multimodel,pd.Index(range(25),name='model')).rename({"tasmax": "t_std"})

# print('test3')

# t_mmm = t_mmm.load()

# print('test4')

# #### THIS CODE WORKS FOR SEASONAL DATA

# fig, axs = plt.subplots(2,2, figsize=(15, 21), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace = .0, wspace= .2)

# season = ['DFJ','JJA','MAM','SON']

# axs = axs.ravel()

# for i in range(len(t_mmm['season'])):
#     pc = axs[i].pcolormesh(t_mmm.lon,t_mmm.lat,t_mmm['t_mean'][:,i,:,:].mean('model'),cmap="Reds",vmin=-40,vmax=40)
#     axs[i].set_title(season[i])
#     axs[i].set_ylim([39,81])
#     axs[i].set_xlim([-141,-59])
#     #axs[i].set_yticks([40,50,60,70,80])
#     #axs[i].set_xticks([-140,-120,-100,-80,-60])
#     #cax,kw = mpl.colorbar.make_axes(axs[i],location='bottom')#,pad=0.05,shrink=0.7)
#     #out=fig.colorbar(pc,cax=cax,extend='both',**kw)    #axs[i].axis("tight")

#     if i in [0,2]:
#         axs[i].set_yticks([40,50,60,70,80])
#     else:
#         axs[i].set_yticks([])

#     if i in [2,3]:
#         axs[i].set_xticks([-140,-120,-100,-80,-60])
#     else:
#         axs[i].set_xticks([])                

# fig.suptitle('Multi-Model Seasonal Average Max. Temperature ($^{\circ}$C)',
#              x=0.5,y=1.0,fontsize=18,fontweight="bold")

# cax,kw = mpl.colorbar.make_axes(axs,location='bottom',pad=-0.48,shrink=0.4)
# out=fig.colorbar(pc,cax=cax,extend='both',**kw)

# out.set_label("Temperature ($^{\circ}$C)", fontweight='bold',fontsize=14)

# fig.tight_layout()

# fig.savefig('mean2-test.png')

# print('test5')

# t_ensemble = xr.Dataset(
#     data_vars={
#         "T_max_avg": (("model","lat", "lon"), t_mean_multimodel),
#         #"T_max_std": (("model","lat", "lon"), t_std_multimodel),
#     },
#     coords={
#         "model": range(25),
#         "lat": lat,
#         "lon": lon,
#     },
#     attrs = dict(
#         variable="Daily Near-Surface Air Temperature",
#         description="CMIP5 Models Data",
#         units="degC"))

# t_ensemble.to_netcdf(path="/home/gbaggio/meanstd_annual.nc", mode='w')

# print('test41')

# fig = plt.figure(figsize=(16,12))
# ax = plt.axes(projection=ccrs.PlateCarree())
# #ax.set_global()
# ax.coastlines()
# ax.gridlines(linewidth=1)
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                  linewidth=1, color='darkgrey')
# gl.xlabels_top = False
# gl.ylabels_left = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 15, 'color': 'gray'}
# gl.xlabel_style = {'color': 'black', 'weight': 'bold'}

# # uncomment and complete the line below (see the NAO notebook for a reminder)
# pc = ax.pcolormesh(t_mmm.lon,t_mmm.lat,t_mmm.mean('model'),cmap="Reds")
# cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.7)
# out=fig.colorbar(pc,cax=cax,extend='both',**kw)
# out.set_label('Mean Temperature ($^{\circ}$C)',size=14)
# ax.set_title('Multi-Model Ensemble Average Max. Temperature ($^{\circ}$C) - January, 1950', 
#              pad = 15,fontweight='bold',size=16)

# fig.savefig('mean1-test.png')

# print('test5')

# fig, axs = plt.subplots(5,5, figsize=(20, 25), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace = .2, wspace= .2)

# axs = axs.ravel()

# for i in range(len(t_ensemble['model'])):

#     pc = axs[i].pcolormesh(t_mmm.lon,t_mmm.lat,t_mmm[i,:,:],cmap="Reds",vmin=-40,vmax=10)
#     axs[i].set_title("\n\n"+multimodel_name[i])
#     axs[i].set_ylim([39,81])
#     axs[i].set_xlim([-141,-59])
#     #axs[i].set_yticks([40,50,60,70,80])
#     #axs[i].set_xticks([-140,-120,-100,-80,-60])
#     #cax,kw = mpl.colorbar.make_axes(axs[i],location='bottom')#,pad=0.05,shrink=0.7)
#     #out=fig.colorbar(pc,cax=cax,extend='both',**kw)    #axs[i].axis("tight")

#     if i in np.linspace(0,20,5):
#         axs[i].set_yticks([40,50,60,70,80])
#     else:
#         axs[i].set_yticks([])

#     if i in np.linspace(20,24,5):
#         axs[i].set_xticks([-140,-120,-100,-80,-60])
#     else:
#         axs[i].set_xticks([])                
        
# fig.suptitle('Average Max. Temperature ($^{\circ}$C) - January, 1950',
#              x=0.5,y=0.91,fontsize=18,fontweight="bold")

# cax,kw = mpl.colorbar.make_axes(axs,location='bottom',pad=0.03,shrink=0.4)
# out=fig.colorbar(pc,cax=cax,extend='both',**kw)

# out.set_label("Temperature ($^{\circ}$C)", fontweight='bold',fontsize=14)

# plt.show()

# fig.savefig('mean2-test.png')

# print('test6')

################
# Lag 1 calculation
################

# def lag1_numpy(a_temp):
#     ecorr = np.correlate((a_temp-np.mean(a_temp))/np.std(a_temp),(a_temp-np.mean(a_temp))
#                          /np.std(a_temp),'same')/len(a_temp)
#     lag1_numpy = ecorr[int(len(ecorr)/2)+1] #Find lag 1
    
#     return lag1_numpy

# print('test7')

# t_lag1_multimodel=[]

# for file_name in multimodel:
#         tm = xr.open_dataset(file_name)
#         #calculate
#         lag1 = xr.apply_ufunc(
#             lag1_numpy, # first the function  
#             tm.chunk(
#                 {"lat": 100, "lon": 100}
#             ),# now arguments in the order expected by 'interp1_np' # now arguments in the order expected by 'interp1_np' 
#             input_core_dims=[["time"]],  # list with one entry per arg
#             #output_core_dims=[["model"]],  # returned data has one dimension
#             exclude_dims=set(("time",)),  # dimensions allowed to change size. Must be a set!
#             vectorize=True,  # loop over non-core dims
#             dask="parallelized",
#             output_dtypes=[tm['tasmax'].dtype]).rename({"tasmax": "lag1"})
        
#         #append
#         t_lag1_multimodel.append(lag1)
        
# t_mlm = xr.concat(t_lag1_multimodel,pd.Index(range(25),name='model'))

# print('test8')

######## Fig1 Lag 1

# fig = plt.figure(figsize=(16,12))
# ax = plt.axes(projection=ccrs.PlateCarree())
# #ax.set_global()
# ax.coastlines()
# ax.gridlines(linewidth=1)
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                  linewidth=1, color='darkgrey')
# gl.xlabels_top = False
# gl.ylabels_left = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 15, 'color': 'gray'}
# gl.xlabel_style = {'color': 'black', 'weight': 'bold'}

# # uncomment and complete the line below (see the NAO notebook for a reminder)
# pc = ax.pcolormesh(t_mlm.lon,t_mlm.lat,t_mlm.mean('model'),cmap="Reds")
# cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.7)
# out=fig.colorbar(pc,cax=cax,extend='both',**kw)
# out.set_label('Lag 1',size=14)
# ax.set_title('Multi-Model Ensemble Max. Temperature Lag 1 - January, 1950', 
#              pad = 15,fontweight='bold',size=16)

# fig.savefig('lag1-test1.png')

# print('test9')

######## Fig2 Lag 1

# fig, axs = plt.subplots(5,5, figsize=(20, 25), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace = .2, wspace= .2)

# axs = axs.ravel()

# for i in range(len(t_mlm['model'])):

#     pc = axs[i].pcolormesh(t_mlm.lon,t_mlm.lat,t_mlm['lag1'][i,:,:],cmap="Reds",vmin=0.2,vmax=0.85)
#     axs[i].set_title(multimodel_name[i])
#     axs[i].set_ylim([39,81])
#     axs[i].set_xlim([-141,-59])
#     #axs[i].set_yticks([40,50,60,70,80])
#     #axs[i].set_xticks([-140,-120,-100,-80,-60])
#     #cax,kw = mpl.colorbar.make_axes(axs[i],location='bottom')#,pad=0.05,shrink=0.7)
#     #out=fig.colorbar(pc,cax=cax,extend='both',**kw)    #axs[i].axis("tight")

#     if i in np.linspace(0,20,5):
#         axs[i].set_yticks([40,50,60,70,80])
#     else:
#         axs[i].set_yticks([])

#     if i in np.linspace(20,24,5):
#         axs[i].set_xticks([-140,-120,-100,-80,-60])
#     else:
#         axs[i].set_xticks([])       
    
    
# fig.suptitle('Multi-Model Ensemble Max. Temperature Lag 1 - January, 1950',
#              x=0.5,y=0.91,fontsize=18,fontweight="bold")

# cax,kw = mpl.colorbar.make_axes(axs,location='bottom',pad=0.03,shrink=0.4)
# out=fig.colorbar(pc,cax=cax,extend='both',**kw)

# out.set_label("Autocorrelation Lag 1", fontweight='bold',fontsize=14)

# plt.show()

# fig.savefig('lag1-test2.png')

# print('test10')