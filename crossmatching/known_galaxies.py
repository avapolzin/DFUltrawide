import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import cartopy.crs as ccrs
import astropy.constants as c
import astropy.units as u

import matplotlib
matplotlib.rcParams.update({'font.size': 20})


#### Good for searching for satellites of bright galaxies, known galaxies in specific mass regimes, ... ####

### NASA Sloan Atlas -- http://nsatlas.org

NSA_catalog = fits.open('nsa_v0_1_2.fits')
NSA_table = Table(NSA_catalog[1].data)
h=0.7
z_conv = (c.c/(70*(u.km/u.s)/u.Mpc)).decompose().to(u.Mpc).value


### Updated Nearby Galaxies Catalog -- https://ui.adsabs.harvard.edu/abs/2013AJ....145..101K/abstract

LV_ = pd.read_csv('LV_nearbygalaxiescatalog.txt', header = 0, sep = '\s+')
# will assume a mass-to-light of 1 for K-band
LV_coords = SkyCoord(LV_['RA'].values, LV_['Dec'].values, unit = ['hr', 'deg'])



### UW observing status
UWobs = pd.read_csv('https://raw.githubusercontent.com/avapolzin/UWSStatusSearch/main/UWobs.txt', sep = '\t')
coords = SkyCoord(UWobs.RA, UWobs.Dec, unit = ['hr', 'deg'])



N_NSA_obj = []
N_NSA_obj_MW = []

LV_obj = []
LV_obj_MW = []

for i in range(len(UWobs.ID)):
	RA_ = NSA_table[NSA_table['Z'] <= 0.01]['RA'][np.logical_and((abs(coords[i].ra.deg - NSA_table[NSA_table['Z'] <= 0.01]['RA']) <= 1.5), 
		(abs(coords[i].dec.deg - NSA_table[NSA_table['Z'] <= 0.01]['DEC']) <= 1))]
	Dec_ = NSA_table[NSA_table['Z'] <= 0.01]['DEC'][np.logical_and((abs(coords[i].ra.deg - NSA_table[NSA_table['Z'] <= 0.01]['RA']) <= 1.5), 
		(abs(coords[i].dec.deg - NSA_table[NSA_table['Z'] <= 0.01]['DEC']) <= 1))]
	masses_ = h**2 * NSA_table[NSA_table['Z'] <= 0.01]['MASS'][np.logical_and((abs(coords[i].ra.deg - NSA_table[NSA_table['Z'] <= 0.01]['RA']) <= 1.5), 
	(abs(coords[i].dec.deg - NSA_table[NSA_table['Z'] <= 0.01]['DEC']) <= 1))]
	
	N_NSA_obj.append(len(masses_))
	N_NSA_obj_MW.append(len(masses_[masses_ >= 10**10]))

	RA_LV = LV_coords.ra.deg[np.logical_and((abs(coords[i].ra.deg - LV_coords.ra.deg) <= 1.5), (abs(coords[i].dec.deg - LV_coords.dec.deg) <= 1))]
	Dec_LV = LV_coords.dec.deg[np.logical_and((abs(coords[i].ra.deg - LV_coords.ra.deg) <= 1.5), (abs(coords[i].dec.deg - LV_coords.dec.deg) <= 1))]
	loglum_LV = LV_['loglumK'][np.logical_and((abs(coords[i].ra.deg - LV_coords.ra.deg) <= 1.5), (abs(coords[i].dec.deg - LV_coords.dec.deg) <= 1))]
	loglum_LV.reset_index(drop = True)

	LV_obj.append(len(loglum_LV))
	LV_obj_MW.append(len(loglum_LV[loglum_LV.loc[np.isfinite(loglum_LV) & (loglum_LV >= 10)].index]))




	fig, ax = plt.subplots(1, 1, figsize = (12, 8))
	ax.scatter(RA_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values >= 10)], Dec_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values >= 10)], 
		color = 'red', marker = 'X', label = r'$M_* >= 10^{10} \, M_\odot$', s = 150)
	ax.scatter(RA_[masses_ >= 10**10], Dec_[masses_ >= 10**10], color = 'red', marker = 'X', s = 150)

	ax.scatter(RA_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 10) & (loglum_LV.values >= 7)], 
		Dec_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 10) & (loglum_LV.values >= 7)], 
		color = 'k', marker = 'x', label = r'$M_* <= 10^{10} \, M_\odot$ and $M_* >= 10^{7} \, M_\odot$', s = 150)
	ax.scatter(RA_[(masses_ <= 10**10) & (masses_ >= 10**7)], Dec_[(masses_ <= 10**10) & (masses_ >= 10**7)], color = 'k', marker = 'x', s = 150)

	ax.scatter(RA_LV[~np.isfinite(loglum_LV.values)], Dec_LV[~np.isfinite(loglum_LV.values)], 
		color = 'k', label = r'$M_* <= 10^{7} \, M_\odot$', s = 75)
	ax.scatter(RA_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 7)], Dec_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 7)], 
		color = 'k', s = 75)
	ax.scatter(RA_[masses_ <= 10**7], Dec_[masses_ <= 10**7], color = 'k', s = 75)


	ax.legend(loc = 'best', fontsize = 12)
	ax.set_xlim(coords[i].ra.deg - 1.5, coords[i].ra.deg + 1.5)
	ax.set_xlabel('RA (deg)')
	ax.set_ylim(coords[i].dec.deg - 1, coords[i].dec.deg + 1)
	ax.set_ylabel('Dec (deg)')
	ax.invert_xaxis()
	
	ax.set_title(str(UWobs.ID[i]))
	
	fig.savefig('known_galaxies/' + str(UWobs.ID[i]) + '.png', bbox_inches = 'tight')
	plt.close()


fig= plt.figure(figsize=(((max(coords.ra.deg) - min(coords.ra.deg))/(max(coords.dec.deg) - min(coords.dec.deg)))*12, 15))
ax = plt.subplot(projection='mollweide')
ax.grid(color='gray', ls='dashed')
cbar = ax.scatter(coords.ra.radian - np.deg2rad(180), coords.dec.radian, s = 130., c=np.array(N_NSA_obj), cmap = 'inferno')
tick_labels = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
ax.set_xticklabels(tick_labels) 
ax.set_xlabel('Right Ascension (deg)')
ax.set_ylabel('Declination (deg)')
fig.colorbar(cbar, label = r'# NSA objects', pad = 0)
fig.savefig('known_galaxies/NSA_crossmatch.png', bbox_inches = 'tight')
plt.close()

fig= plt.figure(figsize=(((max(coords.ra.deg) - min(coords.ra.deg))/(max(coords.dec.deg) - min(coords.dec.deg)))*12, 15))
ax = plt.subplot(projection='mollweide')
ax.grid(color='gray', ls='dashed')
cbar = ax.scatter(coords.ra.radian - np.deg2rad(180), coords.dec.radian, s = 130., c=np.array(N_NSA_obj_MW), cmap = 'inferno')
tick_labels = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
ax.set_xticklabels(tick_labels) 
ax.set_xlabel('Right Ascension (deg)')
ax.set_ylabel('Declination (deg)')
fig.colorbar(cbar, label = r'# massive ($M_* >= 10^{10} M_\odot$) NSA objects', pad = 0)
fig.savefig('known_galaxies/massive_NSA_crossmatch.png', bbox_inches = 'tight')
plt.close()


fig= plt.figure(figsize=(((max(coords.ra.deg) - min(coords.ra.deg))/(max(coords.dec.deg) - min(coords.dec.deg)))*12, 15))
ax = plt.subplot(projection='mollweide')
ax.grid(color='gray', ls='dashed')
cbar = ax.scatter(coords.ra.radian - np.deg2rad(180), coords.dec.radian, s = 130., c=np.array(LV_obj), cmap = 'inferno')
tick_labels = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
ax.set_xticklabels(tick_labels) 
ax.set_xlabel('Right Ascension (deg)')
ax.set_ylabel('Declination (deg)')
fig.colorbar(cbar, label = r'# Nearby Galaxy Catalog objects', pad = 0)
fig.savefig('known_galaxies/NearbyGalaxyCatalog_crossmatch.png', bbox_inches = 'tight')

fig= plt.figure(figsize=(((max(coords.ra.deg) - min(coords.ra.deg))/(max(coords.dec.deg) - min(coords.dec.deg)))*12, 15))
ax = plt.subplot(projection='mollweide')
ax.grid(color='gray', ls='dashed')
cbar = ax.scatter(coords.ra.radian - np.deg2rad(180), coords.dec.radian, s = 130., c=np.array(LV_obj_MW), cmap = 'inferno')
tick_labels = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
ax.set_xticklabels(tick_labels) 
ax.set_xlabel('Right Ascension (deg)')
ax.set_ylabel('Declination (deg)')
fig.colorbar(cbar, label = r'# massive ($M_* >= 10^{10} M_\odot$) Nearby Galaxy Catalog objects', pad = 0)
fig.savefig('known_galaxies/massive_NearbyGalaxyCatalog_crossmatch.png', bbox_inches = 'tight')

