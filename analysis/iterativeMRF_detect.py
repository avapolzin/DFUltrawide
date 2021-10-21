import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy.wcs.utils import pixel_to_skycoord
from mrf.task import MrfTask
from mrf import celestial, display, sbcontrast
import sep
from mrf import download
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib
import artpop
# import nu mpy.ma as np
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb
from matplotlib_scalebar.scalebar import ScaleBar
from photutils import BoundingBox
matplotlib.rcParams.update({'font.size': 20})
from astroquery.sdss import SDSS
from astropy.wcs.utils import pixel_to_skycoord,skycoord_to_pixel
from matplotlib.patches import Ellipse
import shapely.geometry as sg
import shapely.ops as so
from descartes import PolygonPatch
from astropy.table import Table
import astropy.constants as c
import pandas as pd
import os
from os.path import exists


def get_Rad(mass): #Mowla et al. (2019) -- https://ui.adsabs.harvard.edu/abs/2019ApJ...872L..13M/abstract
	r_p = 8.6*u.kpc
	M_p = 10**10.2 * u.M_sun
	a = 0.17
	b = 0.50
	d = 6   
	
	r_80 = r_p * (mass/M_p)**a * ((1/2)*(1 + (mass/M_p)**d))**((b-a)/d)

	r_vir = r_80/0.047
	return r_vir

virial_ = [] # just to get things to run, redefined later
tile_coords = [0, 0] #same as "virial_" patch

### NASA Sloan Atlas -- http://nsatlas.org

NSA_catalog = fits.open(path+'nsa_v0_1_2.fits')
NSA_table = Table(NSA_catalog[1].data)
h=0.7
z_conv = (c.c/(70*(u.km/u.s)/u.Mpc)).decompose().to(u.Mpc).value



### Updated Nearby Galaxies Catalog -- https://ui.adsabs.harvard.edu/abs/2013AJ....145..101K/abstract

LV_ = pd.read_csv(path +'LV_nearbygalaxiescatalog.txt', header = 0, sep = '\s+', comment = '#')
# will assume a mass-to-light of 1 for K-band
LV_coords = SkyCoord(LV_['RA'].values, LV_['Dec'].values, unit = ['hr', 'deg'])



### UW observing status
UWobs = pd.read_csv('https://raw.githubusercontent.com/avapolzin/UWSStatusSearch/main/UWobs.txt', sep = '\t')
coords = SkyCoord(UWobs.RA, UWobs.Dec, unit = ['hr', 'deg'])



def detect_sources(data,snr, band, wcs_cur, mask = None, search_sdss = True, search_radius = 4.,sep_bkg_kwargs = {}, sep_extract_kwargs = {}, plot_sources = True, save_plot = None,photo_z = True):
	"""
	Function used to detect sources in DF images and search for counterparts in the SDSS database
	
	Paramaters
	----------
		data: 2D Array
			Image to be searched, typically a mrf'ed DF tile
		snr: float
			SNR cut, above which to detect sources
		band:
			Observed band 'G' or 'R'
		wcs_cur: Astropy WCS object
			WCS information of data
		
		mask: 2D Array (optional)
			2D array containg pixels to be masked
		
		
		search_sdss: bool (optional)
			Whether to search sdss database for counterparts to detects sources, default True.
			if False then speeds up function call
		photo_z: bool (optional)
			Wether to search SDSS photoz database as backup for galaxies without specz
		search_radius: float (optional)
			Radius to search for counterparts in arcsec, defualt 4
		sep_bkg_kwargs: dict (optional)
			Keywords to pass to sep.Background call
		sep_extract_kwargs: dict (optional)
			Keywords to pass to sep.extract call
		
		plot_sources: bool (optional)
			Whether to make a figure diplaying detected sources, defualt True
		save plot: Str (optional)
			String specifying location of where figure should be saved, default None
	Returns
	-------
		obj_pd: Pandas DataFrame
			Pandas DataFrame containing all of information about the detected sources
"""
	#####
	# Run sep to detect sources
	#####
	datasw = data.byteswap(False).newbyteorder()
	bkg = sep.Background(datasw, mask=mask, **sep_bkg_kwargs)
	obj_tab = sep.extract(data - bkg.back(), snr, err=bkg.rms(), **sep_extract_kwargs )
	obj_pd = pd.DataFrame(obj_tab)

	
	#####
	# Run simple cuts to remove unwanted sources 
	#####
	#remove objects near edge
	obj_pd = obj_pd.query('x < 1002 and x > 5 and y < 1002 and y > 5')
	
	if mask is not None:
		#Check if pixels are masked near source
		mask_phot,_,_ = sep.sum_circle(mask, obj_pd['x'], obj_pd['y'], [2.]*len(obj_pd))
		obj_pd['mask_phot'] = mask_phot
		obj_pd = obj_pd.query('mask_phot < 1')

	obj_pd = obj_pd.reset_index(drop = True)
	coord_all =  pixel_to_skycoord(obj_pd['x']+0.5, obj_pd['y']+0.5, wcs_cur)
	obj_pd['ra'] = coord_all.ra.deg
	obj_pd['dec'] = coord_all.dec.deg
	
	#####
	# Search SDSS database for nearby objects
	#####
	if search_sdss:
		star_near = []
		gal_near = []
		
		star_mag = []
		gal_z = []
		gal_log_ms = []
		gal_photoobjid = []
		gal_specobjid = []
		
		for i, obj in obj_pd.iterrows():
			coord = SkyCoord(ra = obj['ra']*u.deg, dec=obj['dec']*u.deg )
			tab = SDSS.query_region(coord, radius = search_radius*u.arcsec,photoobj_fields = ['ra','dec','objid','mode','type', 'psfMag_g', 'psfMag_r'])
	
			if tab is None:
				star_near.append(False)
				star_mag.append(-99)
				
				gal_near.append(False)
				gal_z.append(-99)
				gal_log_ms.append(-99)
				gal_photoobjid.append(-99)
				gal_specobjid.append(-99)				
				continue

			pd_cur = tab.to_pandas() 
			gal_pd = pd_cur.query('mode == 1 and type == 3').reset_index()
			star_pd = pd_cur.query('mode == 1 and type == 6').reset_index()
			
			if len(star_pd) == 0:
				star_near.append(False)
				star_mag.append(-99)
			else:
				star_near.append(True)
				star_mag.append(np.min(star_pd['psfMag_%s'%band.lower()]))
				
			if len(gal_pd) == 0:
				gal_near.append(False)
				gal_z.append(-99)
				gal_log_ms.append(-99)
				gal_photoobjid.append(-99)
				gal_specobjid.append(-99)
			
			else:
				gal_sep_as = coord.separation( SkyCoord(ra = gal_pd['ra']*u.deg, dec = gal_pd['dec']*u.deg)).to(u.arcsec).value
				gal_near.append(True)
				gal_photoobjid.append(gal_pd['objid'][np.argmin(gal_sep_as)])
				dx = search_radius/60./60.
				SM_tab = SDSS.query_sql('Select ra,dec,z,mstellar_median,specObjID from stellarMassPCAWiscBC03 where ra between %.6f and %.6f and dec between %.6f and %.6f'%(coord.ra.deg - dx,coord.ra.deg + dx, coord.dec.deg - dx, coord.dec.deg + dx ))
				if SM_tab is None:
					gal_z.append(-99)
					gal_log_ms.append(-99)
					gal_specobjid.append(-99)
				else:
					sm_use = SM_tab[np.argmin(coord.separation( SkyCoord(ra = SM_tab['ra']*u.deg, dec = SM_tab['dec']*u.deg)).to(u.arcsec).value )]
					gal_z.append(sm_use['z'])
					gal_log_ms.append(sm_use['mstellar_median'])
					gal_specobjid.append(sm_use['specObjID'])
		
		obj_pd['star_near'] = star_near
		obj_pd['gal_near'] = gal_near
		obj_pd['star_mag'] = star_mag
		obj_pd['gal_spec_z'] = gal_z
		obj_pd['gal_log_ms'] = gal_log_ms
		obj_pd['gal_photoobjid'] =  gal_photoobjid
		obj_pd['gal_specobjid'] = gal_specobjid
	
	
	if photo_z and search_sdss:
		to_query = 'Select objID as gal_photoobjid,z as gal_phot_z, zErr as gal_phot_z_err from Photoz where objID in ('
		to_add = ['%i ,'%id_cur for id_cur in obj_pd.query('gal_near == True').reset_index()['gal_photoobjid']]
		to_query = to_query + ''.join(to_add)[:-1] + ')'
		phz_tab = SDSS.query_sql(to_query)
		obj_pd = obj_pd.join( phz_tab.to_pandas().set_index('gal_photoobjid'), on = 'gal_photoobjid')
		obj_pd['gal_phot_z'] = obj_pd['gal_phot_z'].replace(np.nan,int(-99))
		obj_pd['gal_phot_z_err'] = obj_pd['gal_phot_z_err'].replace(np.nan,int(-99))

	if plot_sources:
		fig, ax = plt.subplots(figsize = (15,15))
		data_sub = data - bkg.back()
		
		if mask is not None: data_sub[np.where(mask == 1)] = 0
			
		m, s = np.mean(data_sub), np.std(data_sub)
		im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
			   vmin=m-s, vmax=m+s, origin='lower')

		# plot an ellipse for each object
		for i,obj in obj_pd.iterrows():
			e = Ellipse(xy=(obj['x'], obj['y']),
				width=6*obj['a'],
				height=6*obj['b'],
				angle=obj['theta'] * 180. / np.pi)
			e.set_facecolor('none')
			e.set_edgecolor('red')
			ax.add_artist(e)
		if save_plot is not None: plt.savefig(save_plot)
	return obj_pd

#modifying to confirm isolated with projected virial radii
#also modifying to plot individual candidates not all in one image
# not going to plot HSC cut out because that's not working for me right now -- will figure out later
# from unagi import hsc,task
# pdr2 = hsc.Hsc(dr='pdr2')
def plot_cutouts(res, obj_pd, decals_fits, virial_radii = virial_, field_center = [tile_coords[0], tile_coords[1]], save_path = None, df_size = 50, hr_size = 20):
	"""
	Function used to plot cutouts of detected sources for exploratory purposes
	
	Paramaters
	----------
		res: mrf results objects
		
		obj_pd: Pandas Dataframe
			DataFrame containing information on sources, generally the output of detect.detect_sources
		
		decals_fits: Astropy fits HDU
			Fits file containing the un-magnifying DECALS images
		
		save_name: str (optional)
			Location to save figure
		
		df_size: float (optional)
			size in arcsec for DF cutout, default 50
		
		hr_size: float (optional)
			size in arcsec for Decals and HSC cutouts, defualt 20
	
	returns:
		Figure
	
"""
	num_obj = len(obj_pd)
	i = 0
	
	for i,obj in obj_pd.iterrows():
		fig, axes = plt.subplots(1,5, figsize = (4*14,12), gridspec_kw = {'width_ratios':[1, 1, 1, 1, 1.5], 'height_ratios':[1]})
		
		if num_obj == 1:
			ax_df = axes[0]
			ax_hrfl = axes[1]
			ax_lrfl = axes[2]
			ax_dec = axes[3]
#			 ax_hsc = axes[4]
			ax_iso = axes[4]
		else:
			ax_df = axes[0]
			ax_hrfl = axes[1]
			ax_lrfl = axes[2]
			ax_dec = axes[3]
#			 ax_hsc = axes[4]
			ax_iso = axes[4]
		
		
		#Calculate properties of object
		coord = SkyCoord(ra = obj['ra']*u.deg, dec = obj['dec']*u.deg)
		
		
		#Get df_cutout
		df_img = celestial.img_cutout(res.lowres_final.image, res.lowres_final.wcs, coord.ra/u.deg *u.deg,coord.dec/u.deg *u.deg, size = [df_size,df_size], pixel_scale = 2.5, save = False)
	
		#Get mrf outputs
		hr_flux_mod = celestial.img_cutout(res.hires_fluxmod,res.hires_img.wcs, coord.ra/u.deg *u.deg,coord.dec/u.deg *u.deg, size = [df_size,df_size], pixel_scale = 0.8333, save = False) 
		lr_flux_mod = celestial.img_cutout(res.lowres_model.image,res.lowres_model.wcs, coord.ra/u.deg *u.deg,coord.dec/u.deg *u.deg, size = [df_size,df_size], pixel_scale = 2.5, save = False) 
		#Get decals cutout
		dec_img = celestial.img_cutout(decals_fits[0].data, WCS(decals_fits[0].header), coord.ra.deg,coord.dec.deg, size = [hr_size,hr_size],pixel_scale = 0.262, save = False)

#		 #Download hsc_cutout
#		 hsc_img = task.hsc_cutout(coord, cutout_size = hr_size/2.*u.arcsec, filters = 'g',archive = pdr2,dr = 'pdr2', use_saved=False, verbose=True, variance=False, mask=False, save_output=False)
#		 hsc_wcs = WCS(hsc_img[1].header)
	
		#Plot images
		display.display_single(df_img[0].data, pixel_scale = 2.5,ax = ax_df,contrast = 0.001, scale_bar_y_offset  = 2)
		
		
		f_hr = np.sum(hr_flux_mod[0].data == 0) / (hr_flux_mod[0].data.shape[0]*hr_flux_mod[0].data.shape[1]) * 100
		if f_hr == 100:
			f_hr -= 1
	   
		display.display_single(hr_flux_mod[0].data, pixel_scale = 0.8333,ax = ax_hrfl,contrast = 0.001, lower_percentile=98, upper_percentile = 100, scale_bar_y_offset  = 2)
		display.display_single(lr_flux_mod[0].data, pixel_scale = 2.5,ax = ax_lrfl,contrast = 0.001, scale_bar_y_offset  = 2)
#		 display.display_single(hsc_img[1].data, ax = ax_hsc,contrast = 0.03, scale_bar_y_offset  = 2, pixel_scale = 0.168)
		display.display_single(dec_img[0].data, pixel_scale = 0.262, ax = ax_dec,contrast = 0.03, scale_bar_y_offset  = 2)

	
		#Plot outline of smaller cutouts on larger cutouts
		lr_box_loc = df_img[0].wcs.world_to_pixel_values(dec_img[0].wcs.calc_footprint()*u.deg )
		lr_box_loc = np.vstack([lr_box_loc,lr_box_loc[0]])

		hr_box_loc = hr_flux_mod[0].wcs.world_to_pixel_values(dec_img[0].wcs.calc_footprint()*u.deg )
		hr_box_loc = np.vstack([hr_box_loc,hr_box_loc[0]])
	
		ax_df.plot(lr_box_loc[:,0], lr_box_loc[:,1], 'w-', lw = 2)
		ax_lrfl.plot(lr_box_loc[:,0], lr_box_loc[:,1], 'w-', lw = 2)
		ax_hrfl.plot(hr_box_loc[:,0], hr_box_loc[:,1], 'w-', lw = 2)
	
		#Define all panels
		ax_df.set_title('obj %i, mrf_final - 50`` x 50`` '%(i) )
		ax_lrfl.set_title('mrf low res flux model - 50`` x 50 ``')
		ax_hrfl.set_title('mrf high res flux model - 50`` x 50 ``')
		ax_dec.set_title('Legacy Survey - 20`` x 20``')
#		 ax_hsc.set_title('HSC - 20`` x 20``')
	
				
		for m in virial_:
			ax_iso.add_patch(PolygonPatch(m, fc = 'gray', ec = 'gray', alpha = 0.6, zorder = 0))
		ax_iso.scatter(coord.ra.deg, coord.dec.deg, c = 'mediumvioletred', s = 200, marker = 'X')
		ax_iso.scatter(RA_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values >= 10)], Dec_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values >= 10)], 
			color = 'red', marker = 'X', label = r'$M_* >= 10^{10} \, M_\odot$', s = 150)
		ax_iso.scatter(RA_[(dist_ <= 20) & (masses_ >= 10**10)], Dec_[(dist_ <= 20) & (masses_ >= 10**10)], color = 'red', marker = 'X', s = 150)

		ax_iso.scatter(RA_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 10) & (loglum_LV.values >= 7)], 
			Dec_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 10) & (loglum_LV.values >= 7)], 
			color = 'k', marker = 'x', label = r'$M_* <= 10^{10} \, M_\odot$ and $M_* >= 10^{7} \, M_\odot$', s = 150)
		ax_iso.scatter(RA_[(dist_ <= 20) & (masses_ <= 10**10) & (masses_ >= 10**7)], Dec_[(dist_ <= 20) & (masses_ <= 10**10) & (masses_ >= 10**7)], color = 'k', marker = 'x', s = 150)

		ax_iso.scatter(RA_LV[~np.isfinite(loglum_LV.values)], Dec_LV[~np.isfinite(loglum_LV.values)], 
			color = 'k', label = r'$M_* <= 10^{7} \, M_\odot$', s = 75)
		ax_iso.scatter(RA_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 7)], Dec_LV[np.isfinite(loglum_LV.values) & (loglum_LV.values <= 7)], 
			color = 'k', s = 75)
		ax_iso.scatter(RA_[(dist_ <= 20) & (masses_ <= 10**7)], Dec_[(dist_ <= 20) & (masses_ <= 10**7)], color = 'k', s = 75)
		ax_iso.set_title('environment')
		ax_iso.set_xlim([field_center[0] - 2, field_center[0] + 2])
		ax_iso.set_ylim([field_center[1] - 2, field_center[1] + 2])
		ax_iso.invert_xaxis()
		ax_iso.axis('off')
	
		#Query and plot source from SDSS Photoobj catalog
		#Blue are galaxies, red a stars
		tab = SDSS.query_region(coord, radius = hr_size/1.5*u.arcsec,photoobj_fields = ['ra','dec','objid','mode','type'])
		if tab is not None:
			for src in tab:
				if src['mode'] != 1: continue
		
#				 x,y = skycoord_to_pixel(SkyCoord(ra = src['ra']*u.deg,dec = src['dec']*u.deg), hsc_wcs)
#				 e = Ellipse(xy=(x, y), width=16, height=16, angle=0,lw = 2)
#				 e.set_facecolor('none')
#				 if src['type'] == 3:
#					 e.set_edgecolor('blue')
#				 else:
#					 e.set_edgecolor('red')
#				 ax_hsc.add_artist(e)
		
				x1,y1 = skycoord_to_pixel(SkyCoord(ra = src['ra']*u.deg,dec = src['dec']*u.deg), dec_img[0].wcs)
				e1 = Ellipse(xy=(x1, y1), width=16*0.168/0.262, height = 16*0.168/0.262, angle=0,lw = 2)
				e1.set_facecolor('none')
				if src['type'] == 3:
					e1.set_edgecolor('blue')
				else:
					e1.set_edgecolor('red')
				ax_dec.add_artist(e1)
		
		#Plot geometry of detected source
		obj_x,obj_y = skycoord_to_pixel(coord, df_img[0].wcs, origin = 0)
		e = Ellipse(xy=(obj_x, obj_x),
				width=2*obj['a'],
				height=2*obj['b'],
				angle=obj['theta'] * 180. / np.pi)
		e.set_facecolor('none')
		e.set_edgecolor('k')
		ax_df.add_artist(e)
	
		obj_x,obj_y = skycoord_to_pixel(coord, dec_img[0].wcs)
		e = Ellipse(xy=(obj_x, obj_y),
			width=2*obj['a']*2.5/0.262,
			height=2*obj['b']*2.5/0.262,
			angle=obj['theta'] * 180. / np.pi)
		e.set_facecolor('none')
		e.set_edgecolor('k')
		ax_dec.add_artist(e)
	
	
#		 obj_x,obj_y = skycoord_to_pixel(coord, hsc_wcs)
#		 e = Ellipse(xy=(obj_x, obj_y),
#			 width=2*obj['a']*2.5/0.168,
#			 height=2*obj['b']*2.5/0.168,
#			 angle=obj['theta'] * 180. / np.pi)
#		 e.set_facecolor('none')
#		 e.set_edgecolor('k')
#		 ax_hsc.add_artist(e)
	
		#fig.subplots_adjust(wspace = -1,hspace = 0 )
		if save_path is not None: 
			plt.savefig(save_path + str(coord.ra) + str(coord.dec) + '.png')#, bbox_inches = 'tight')
	return fig






tiles_out = glob.glob('../DF*')

task = MrfTask('updated_g2.yaml')
for i in tiles_out:
	print(i)
	short_name = i[3:-6]
	if exists(short_name):
		print('Skipping, MRF already run on tile.')
		continue
	tile = fits.open(i)
	for j in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PC1_1', 'PC2_2']:
		tile[0].header.remove(j)
	tile_WCS = WCS(tile[0].header, tile)
	tile_dat = tile[0].data
	
	
	if len(tile_dat[np.isfinite(tile_dat)]) == 0:
		continue
		
	tile_dat[~np.isfinite(tile_dat)] = 0
	

	tile[0].header['CD1_1'] = -0.00069444444444444 
	tile[0].header['CD1_2'] = 0.
	tile[0].header['CD2_1'] = 0.
	tile[0].header['CD2_2'] = 0.00069444444444444 
	tile[0].header['PC1_1'] = 1.0
	tile[0].header['PC2_2'] = 1.0
	tile[0].header['BACKVAL'] = 0.

	
	fits.writeto(i, tile_dat, tile[0].header, overwrite=True)

####### get virial radii for things with d < 20 Mpc for tile #########

	tile_coords = utils.pixel_to_skycoord(tile_dat.shape[0]/2, tile_dat.shape[1]/2, 
	                             wcs = tile_WCS)

	field = sg.box(tile_coords.ra.deg - 0.7, tile_coords.dec.deg - 0.7, tile_coords.ra.deg + 0.7, tile_coords.dec.deg + 0.7)
	virial_ = []

	## Nearby Galaxies Catalog	
	RA_LV = LV_coords.ra.deg[np.logical_and((abs(tile_coords.ra.deg - LV_coords.ra.deg) <= 20), (abs(tile_coords.dec.deg - LV_coords.dec.deg) <= 20))]
	Dec_LV = LV_coords.dec.deg[np.logical_and((abs(tile_coords.ra.deg - LV_coords.ra.deg) <= 20), (abs(tile_coords.dec.deg - LV_coords.dec.deg) <= 20))]
	dist_LV = LV_['dist_Mpc'][np.logical_and((abs(tile_coords.ra.deg - LV_coords.ra.deg) <= 20), (abs(tile_coords.dec.deg - LV_coords.dec.deg) <= 20))]
	loglum_LV = LV_['loglumK'][np.logical_and((abs(tile_coords.ra.deg - LV_coords.ra.deg) <= 20), (abs(tile_coords.dec.deg - LV_coords.dec.deg) <= 20))]
	loglum_LV.reset_index(drop = True)

	if len(loglum_LV) > 0:
		for k in range(len(loglum_LV)):
			mass_LV = 10**loglum_LV.values[k]
			ang_rad = (get_Rad(mass_LV*u.Msun)/(dist_LV.values[k]*u.Mpc)).decompose()*u.rad.to(u.deg)
			virial_.append(sg.Point(RA_LV[k], Dec_LV[k]).buffer(ang_rad))

	## NASA Sloan ATLAS
	RA_ = NSA_table['RA'][np.logical_and((abs(tile_coords.ra.deg - NSA_table['RA']) <= 20), 
	   (abs(tile_coords.dec.deg - NSA_table['DEC']) <= 20))]
	Dec_ = NSA_table['DEC'][np.logical_and((abs(tile_coords.ra.deg - NSA_table['RA']) <= 20), 
	   (abs(tile_coords.dec.deg - NSA_table['DEC']) <= 20))]
	dist_ = z_conv*NSA_table['ZDIST'][np.logical_and((abs(tile_coords.ra.deg - NSA_table['RA']) <= 20), 
	   (abs(tile_coords.dec.deg - NSA_table['DEC']) <= 20))]

	masses_ = h**2 * NSA_table['MASS'][np.logical_and((abs(tile_coords.ra.deg - NSA_table['RA']) <= 20), 
	   (abs(tile_coords.dec.deg - NSA_table['DEC']) <= 20))]

	if len(dist_[dist_ <= 20]) > 0:
		for j in range(len(dist_)):
			if dist_[j] <= 20:
				ang_rad = (get_Rad(masses_[j]*u.Msun)/(dist_[j]*u.Mpc)).decompose()*u.rad.to(u.deg)
				test = sg.Point(RA_[j], Dec_[j]).buffer(ang_rad)
				virial_.append(sg.Point(RA_[j], Dec_[j]).buffer(ang_rad))
	
	
######################################################################



	loc = pixel_to_skycoord(tile_dat.shape[0]/2, tile_dat.shape[1]/2, tile_WCS)

	if not exists('../DECaLS_download/' + short_name + '_g.fits'):
		download.download_decals_large(loc.ra.deg, loc.dec.deg, band = 'g', 
										layer = 'dr9-north', size = 0.9 * u.deg, 
										output_dir = '../DECaLS_download/', output_name = short_name, 
										verbose = False)
	if not exists('../DECaLS_download/' + short_name + '_r.fits'):
		download.download_decals_large(loc.ra.deg, loc.dec.deg, band = 'r', 
										layer = 'dr9-north', size = 0.9 * u.deg, 
										output_dir = '../DECaLS_download/', output_name = short_name, 
										verbose = False)
	img_lowres = i
	img_hires_b = '../DECaLS_download/' + short_name + '_g.fits'
	img_hires_r = '../DECaLS_download/' + short_name + '_g.fits'
	results = task.run(img_lowres, img_hires_b, img_hires_r, 
					   output_name=short_name + '_g', verbose = True, certain_gal_cat = None, 
					   wide_psf = True, skip_SE = False, skip_mast = False, skip_resize = False)

	os.mkdir(short_name)

	plot_cutouts(results, detect_sources(fits.open(short_name+'_g_final.fits')[0].data, 5, 'G', 
			WCS(fits.open(short_name+'_g_final.fits')[0].header)), 
			fits.open('../DECaLS_download/' + short_name + '_g.fits'), virial_radii = virial_, field_center = [tile_coords.ra.deg, tile_coords.dec.deg], 
	save_path = short_name + '/')
