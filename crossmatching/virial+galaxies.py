import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import shapely.geometry as sg
import shapely.ops as so

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import cartopy.crs as ccrs
import astropy.constants as c
import astropy.units as u

import matplotlib
matplotlib.rcParams.update({'font.size': 20})
from descartes import PolygonPatch


#### Galaxies + virial radii out to 20 Mpc, might be edifying to see relative contribution ####



def get_Rad(mass): #Mowla et al. (2019) -- https://ui.adsabs.harvard.edu/abs/2019ApJ...872L..13M/abstract
	r_p = 8.6*u.kpc
	M_p = 10**10.2 * u.M_sun
	a = 0.17
	b = 0.50
	d = 6   
	
	r_80 = r_p * (mass/M_p)**a * ((1/2)*(1 + (mass/M_p)**d))**((b-a)/d)

	r_vir = r_80/0.047
	return r_vir



### NASA Sloan Atlas -- http://nsatlas.org

NSA_catalog = fits.open('nsa_v0_1_2.fits')
NSA_table = Table(NSA_catalog[1].data)
h=0.7
z_conv = (c.c/(70*(u.km/u.s)/u.Mpc)).decompose().to(u.Mpc).value



### Updated Nearby Galaxies Catalog -- https://ui.adsabs.harvard.edu/abs/2013AJ....145..101K/abstract

LV_ = pd.read_csv('LV_nearbygalaxiescatalog.txt', header = 0, sep = '\s+', comment = '#')
# will assume a mass-to-light of 1 for K-band
LV_coords = SkyCoord(LV_['RA'].values, LV_['Dec'].values, unit = ['hr', 'deg'])



### UW observing status
UWobs = pd.read_csv('https://raw.githubusercontent.com/avapolzin/UWSStatusSearch/main/UWobs.txt', sep = '\t')
coords = SkyCoord(UWobs.RA, UWobs.Dec, unit = ['hr', 'deg'])




for i in range(len(UWobs.ID)):
	field = sg.box(coords[i].ra.deg - 1.5, coords[i].dec.deg - 1, coords[i].ra.deg + 1.5, coords[i].dec.deg + 1)
	virial_ = []
	
	## Nearby Galaxies Catalog	
	RA_LV = LV_coords.ra.deg[np.logical_and((abs(coords[i].ra.deg - LV_coords.ra.deg) <= 20), (abs(coords[i].dec.deg - LV_coords.dec.deg) <= 20))]
	Dec_LV = LV_coords.dec.deg[np.logical_and((abs(coords[i].ra.deg - LV_coords.ra.deg) <= 20), (abs(coords[i].dec.deg - LV_coords.dec.deg) <= 20))]
	dist_LV = LV_['dist_Mpc'][np.logical_and((abs(coords[i].ra.deg - LV_coords.ra.deg) <= 20), (abs(coords[i].dec.deg - LV_coords.dec.deg) <= 20))]
	loglum_LV = LV_['loglumK'][np.logical_and((abs(coords[i].ra.deg - LV_coords.ra.deg) <= 20), (abs(coords[i].dec.deg - LV_coords.dec.deg) <= 20))]
	loglum_LV.reset_index(drop = True)
	
	if len(loglum_LV) > 0:
		for k in range(len(loglum_LV)):
			mass_LV = 10**loglum_LV.values[k]
			ang_rad = (get_Rad(mass_LV*u.Msun)/(dist_LV.values[k]*u.Mpc)).decompose()*u.rad.to(u.deg)
			virial_.append(sg.Point(RA_LV[k], Dec_LV[k]).buffer(ang_rad))
	
	## NASA Sloan ATLAS
	RA_ = NSA_table['RA'][np.logical_and((abs(coords[i].ra.deg - NSA_table['RA']) <= 15), 
	   (abs(coords[i].dec.deg - NSA_table['DEC']) <= 15))]
	Dec_ = NSA_table['DEC'][np.logical_and((abs(coords[i].ra.deg - NSA_table['RA']) <= 15), 
	   (abs(coords[i].dec.deg - NSA_table['DEC']) <= 15))]
	dist_ = z_conv*NSA_table['ZDIST'][np.logical_and((abs(coords[i].ra.deg - NSA_table['RA']) <= 15), 
	   (abs(coords[i].dec.deg - NSA_table['DEC']) <= 15))]
	
	masses_ = h**2 * NSA_table['MASS'][np.logical_and((abs(coords[i].ra.deg - NSA_table['RA']) <= 15), 
	   (abs(coords[i].dec.deg - NSA_table['DEC']) <= 15))]
	
	if len(dist_[dist_ <= 20]) > 0:
		for j in range(len(dist_)):
			if dist_[j] <= 20:
				ang_rad = (get_Rad(masses_[j]*u.Msun)/(dist_[j]*u.Mpc)).decompose()*u.rad.to(u.deg)
				test = sg.Point(RA_[j], Dec_[j]).buffer(ang_rad)
				virial_.append(sg.Point(RA_[j], Dec_[j]).buffer(ang_rad))
	
	area = field.intersection(so.cascaded_union(virial_)).area
	
	fig, ax = plt.subplots(1, 1, figsize = (12, 8))
	for m in virial_:
		ax.add_patch(PolygonPatch(m, fc = 'gray', ec = 'gray', alpha = 0.6, zorder = 0))
	
	ax.scatter(RA_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values >= 10)], Dec_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values >= 10)], 
		color = 'red', marker = 'X', label = r'$M_* >= 10^{10} \, M_\odot$', s = 150)
	ax.scatter(RA_[(dist_ <= 20) & (masses_ >= 10**10)], Dec_[(dist_ <= 20) & (masses_ >= 10**10)], color = 'red', marker = 'X', s = 150)

	ax.scatter(RA_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 10) & (loglum_LV.values >= 7)], 
		Dec_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 10) & (loglum_LV.values >= 7)], 
		color = 'k', marker = 'x', label = r'$M_* <= 10^{10} \, M_\odot$ and $M_* >= 10^{7} \, M_\odot$', s = 150)
	ax.scatter(RA_[(dist_ <= 20) & (masses_ <= 10**10) & (masses_ >= 10**7)], Dec_[(dist_ <= 20) & (masses_ <= 10**10) & (masses_ >= 10**7)], color = 'k', marker = 'x', s = 150)

	ax.scatter(RA_LV[~np.isfinite(loglum_LV.values)], Dec_LV[~np.isfinite(loglum_LV.values)], 
		color = 'k', label = r'$M_* <= 10^{7} \, M_\odot$', s = 75)
	ax.scatter(RA_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 7)], Dec_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 7)], 
		color = 'k', s = 75)
	ax.scatter(RA_[(dist_ <= 20) & (masses_ <= 10**7)], Dec_[(dist_ <= 20) & (masses_ <= 10**7)], color = 'k', s = 75)

	ax.legend(loc = 'best', fontsize = 10)

	ax.set_xlim(coords[i].ra.deg - 1.5, coords[i].ra.deg + 1.5)
	ax.set_xlabel('RA (deg)')
	ax.set_ylim(coords[i].dec.deg - 1, coords[i].dec.deg + 1)
	ax.set_ylabel('Dec (deg)')
	ax.invert_xaxis()
	
	ax.set_title(str(UWobs.ID[i]) + ', %.2f percent covered'%(area/6 * 100))
	
	fig.savefig('galaxies+virial/' + str(UWobs.ID[i]) + '.png', bbox_inches = 'tight')
	plt.close()


