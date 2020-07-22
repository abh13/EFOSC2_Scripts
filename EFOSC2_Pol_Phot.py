#!/usr/bin/python
# -*- coding: utf-8 -*-
# EFOSC2 polarisation photometry script v1.0
# Created by Adam Higgins
# Email: abh13@le.ac.uk

__doc__ = """ Script runs photometry for the polarisation images from EFOSC2
and outputs flux information for ordinary and extraordinary beams.
For usage please go to https://github.com/abh13/EFOSC2_Scripts.

File names for each half-wave plate angle should be:
0ang.fits,
225ang.fits,
45ang.fits,
675ang.fits
"""

from astropy.io import fits
from astropy.table import vstack
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization import ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils import aperture_photometry
from photutils import RectangularAnnulus
from photutils.utils import calc_total_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import argparse


def get_args():
	""" Parse command line arguments """
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("Directory",metavar="DIR",type=str,action="store",
		help="Required directory")
	parser.add_argument("--ap",type=float,default=2.0,dest='aperture',
		help="Source aperture diameter X*FWHM (default = 2.0)")
	parser.add_argument("--fwhm",type=float,default=0,dest='fwhm',
		help="Manually set the FWHM (default = AUTO)")
		
	args = parser.parse_args()
	folder_path = args.__dict__['Directory']
	apermul = args.aperture
	fwhm = args.fwhm
	return folder_path,apermul,fwhm


def efosc2_pol_phot(folder_path,apermul,fwhm):
	""" Script runs photometry for EFOSC2 polarimetry images """

	# Read in four wave plate angle files and set up arrays for later
	file_ang0 = os.path.join(folder_path,'0ang.fits')
	file_ang225 = os.path.join(folder_path,'225ang.fits')
	file_ang45 = os.path.join(folder_path,'45ang.fits')
	file_ang675 = os.path.join(folder_path,'675ang.fits')

	files = [file_ang0,file_ang225,file_ang45,file_ang675]
	angle = ['0','225','45','675']
	ang_dec = ['0','22.5','45','67.5']
	label = ['$0^{\circ}$ image','$22.5^{\circ}$ image',
		'$45^{\circ}$ image','$67.5^{\circ}$ image']
		
	# Set up array to store the number of sources per half-wave plate
	numsource = []
	
	# Loop over files for the four wave plate files
	for k in range(0,len(angle),1):

		# Open fits file, extract pixel flux data and remove saturated pixels
		try:
			hdulist = fits.open(files[k])
			image_data = hdulist[0].data
			
		except FileNotFoundError as e:
			print("Cannot find the fits file(s) you are looking for!")
			print("Please check the input!")
			sys.exit()
		
		# Remove bad pixels and mask edges
		image_data[image_data > 60000] = 0
		image_data[image_data < 0] = 0    
		rows = len(image_data[:,0])
		cols = len(image_data[0,:])
		hdulist.close()

		# Calculate estimate of background using sigma-clipping and calculate
		# number of pixels used in the background region that were not
		# clipped! This is done in a small area near the optical axis.
		go_bmean, go_bmedian, go_bstd = sigma_clipped_stats(image_data
			[510:568,520:580],sigma=3.0,maxiters=5)
		ge_bmean, ge_bmedian, ge_bstd = sigma_clipped_stats(image_data
			[446:504,520:580],sigma=3.0,maxiters=5)
		mask_o = sigma_clip(image_data[510:568,520:580],sigma=3.0,maxiters=5,
			masked=True)
		mask_e = sigma_clip(image_data[446:504,520:580],sigma=3.0,maxiters=5,
			masked=True)
		ann_area_o = np.ma.MaskedArray.count(mask_o)
		ann_area_e = np.ma.MaskedArray.count(mask_e)
		
		# Detect sources using DAO star finder
		daofind_o = DAOStarFinder(fwhm=5,threshold=5*go_bstd,
			exclude_border=True)
		daofind_e = DAOStarFinder(fwhm=5,threshold=5*ge_bstd,
			exclude_border=True)
		sources_o = daofind_o(image_data[522:552,535:565])
		sources_e = daofind_e(image_data[462:492,535:565])
		
		if (len(sources_o) < 1 or len(sources_e) < 1):
			print("No source detected in",ang_dec[k],"degree image")
			sys.exit()
			
		if len(sources_o) != len(sources_e):
			print("Unequal number of sources detected in o and e images!")
			sys.exit()
		
		glob_bgm = [go_bmean,ge_bmean]
		glob_bgerr = [go_bstd,ge_bstd]
		
		# Convert the source centroids back into detector pixels
		sources_o['xcentroid'] = sources_o['xcentroid'] + 535
		sources_o['ycentroid'] = sources_o['ycentroid'] + 522
		sources_e['xcentroid'] = sources_e['xcentroid'] + 535
		sources_e['ycentroid'] = sources_e['ycentroid'] + 462

		# Estimate the FWHM of the source by simulating a 2D Gaussian
		# This is only done on the 0 angle image ensuring aperture sizes
		# are equal for all half-wave plate angles. If a user specified
		# FWHM is given, then the estimation is not used.
		if fwhm == 0.0:
			xpeaks_o = []
			xpeaks_e = []
			ypeaks_o = []
			ypeaks_e = []
			fwhm = []
			
			for i in range(0,len(sources_o),1):			
				data_o = image_data[525:550,535:565]
				xpeaks_o.append(int(sources_o[i]['xcentroid']) - 535)
				ypeaks_o.append(int(sources_o[i]['ycentroid']) - 525)
					
				data_e = image_data[465:490,535:560]
				xpeaks_e.append(int(sources_e[i]['xcentroid']) - 535)
				ypeaks_e.append(int(sources_e[i]['ycentroid']) - 465)
				
				min_count_o = np.min(data_o)
				min_count_e = np.min(data_e)
				max_count_o = data_o[ypeaks_o[i],xpeaks_e[i]]
				max_count_e = data_e[ypeaks_o[i],xpeaks_e[i]]
				half_max_o = (max_count_o + min_count_o)/2
				half_max_e = (max_count_e + min_count_e)/2
				
				# Crude calculation for each source
				nearest_above_x_o = ((np.abs(data_o[ypeaks_o[i],
					xpeaks_o[i]:-1] - half_max_o)).argmin())
				nearest_below_x_o = ((np.abs(data_o[ypeaks_o[i],0:
					xpeaks_o[i]] - half_max_o)).argmin())
				nearest_above_x_e = ((np.abs(data_e[ypeaks_e[i],
					xpeaks_e[i]:-1] - half_max_e)).argmin())
				nearest_below_x_e = ((np.abs(data_e[ypeaks_e[i],0:
					xpeaks_e[i]] - half_max_e)).argmin())
				nearest_above_y_o = ((np.abs(data_o[ypeaks_o[i]:-1,
					xpeaks_o[i]] - half_max_o)).argmin())
				nearest_below_y_o = ((np.abs(data_o[0:ypeaks_o[i],
					xpeaks_o[i]] - half_max_o)).argmin())
				nearest_above_y_e = ((np.abs(data_e[ypeaks_e[i]:-1,
					xpeaks_e[i]] - half_max_e)).argmin())
				nearest_below_y_e = ((np.abs(data_e[0:ypeaks_e[i],
					xpeaks_e[i]] - half_max_e)).argmin())
				fwhm.append((nearest_above_x_o + (xpeaks_o[i] -
					nearest_below_x_o)))
				fwhm.append((nearest_above_y_o + (ypeaks_o[i] -
					nearest_below_y_o)))
				fwhm.append((nearest_above_x_e + (xpeaks_e[i] -
					nearest_below_x_e)))
				fwhm.append((nearest_above_y_e + (ypeaks_e[i] -
					nearest_below_y_e)))
			
			fwhm = np.mean(fwhm)
		
		# Stack both ord and exord sources together
		tot_sources = vstack([sources_o,sources_e])
				
		# Store the ordinary and extraordinary beam source images and
		# create apertures for aperture photometry 
		positions = np.swapaxes(np.array((tot_sources['xcentroid'],
			tot_sources['ycentroid']),dtype='float'),0,1)
		aperture = CircularAperture(positions, r=0.5*apermul*fwhm)
		phot_table = aperture_photometry(image_data,aperture)   
					  
		# Set up arrays of ord and exord source parameters
		s_id = np.zeros([len(np.array(phot_table['id']))])
		xp = np.zeros([len(s_id)])
		yp = np.zeros([len(s_id)])
		fluxbgs = np.zeros([len(s_id)])
		mean_bg = np.zeros([len(s_id)])
		bg_err = np.zeros([len(s_id)])
		s_area = []
		
		for i in range(0,len(np.array(phot_table['id'])),1):
			s_id[i] = np.array(phot_table['id'][i])
			xpos = np.array(phot_table['xcenter'][i])
			ypos = np.array(phot_table['ycenter'][i])
			xp[i] = xpos
			yp[i] = ypos
			s_area.append(np.pi*(0.5*apermul*fwhm)**2)
			j = i%2				
			fluxbgs[i] = (phot_table['aperture_sum'][i] -
				aperture.area*glob_bgm[j])
			mean_bg[i] = glob_bgm[j]
			bg_err[i] = glob_bgerr[j]			
		
		# Create and save the image in z scale and overplot the ordinary and
		# extraordinary apertures and local background annuli if applicable
		fig = plt.figure()
		zscale = ZScaleInterval(image_data)
		norm = ImageNormalize(stretch=SqrtStretch(),interval=zscale)
		image = plt.imshow(image_data,cmap='gray',origin='lower',norm=norm)
		bg_annulus_o = RectangularAnnulus((550,539),w_in=0.1,w_out=60,h_out=58,
			theta=0)
		bg_annulus_e = RectangularAnnulus((550,475),w_in=0.1,w_out=60,h_out=58,
			theta=0)
		bg_annulus_o.plot(color='skyblue',lw=1.5,alpha=0.5)
		bg_annulus_e.plot(color='lightgreen',lw=1.5,alpha=0.5)
		
		for i in range(0,len(np.array(phot_table['id'])),1):
			aperture = CircularAperture((xp[i],yp[i]),r=0.5*apermul*fwhm)
			
			if i < int(len(np.array(phot_table['id']))/2):
				aperture.plot(color='blue',lw=1.5,alpha=0.5)
		
			else:
				aperture.plot(color='green',lw=1.5,alpha=0.5)
			
		plt.xlim(500,600)
		plt.ylim(425,575)
		plt.title(label[k])
		image_fn = folder_path + angle[k] + '_image.png'
		fig.savefig(image_fn)

		# Create dataframes for photometry results
		cols = ['xpix','ypix','fluxbgs','sourcearea','meanbg','bgerr',
			'bgarea']
		df_o = pd.DataFrame(columns=cols)
		df_e = pd.DataFrame(columns=cols)
		
		for i in range(0,len(np.array(phot_table['id'])),1):
			if 0 <= i < int(len(np.array(phot_table['id']))/2):
				df_o = df_o.append({cols[0]:xp[i],cols[1]:yp[i],
					cols[2]:fluxbgs[i],cols[3]:s_area[i],cols[4]:mean_bg[i],
					cols[5]:bg_err[i],cols[6]:ann_area_o},ignore_index=True)
					
			else:
				df_e = df_e.append({cols[0]:xp[i],cols[1]:yp[i],
					cols[2]:fluxbgs[i],cols[3]:s_area[i],cols[4]:mean_bg[i],
					cols[5]:bg_err[i],cols[6]:ann_area_e},ignore_index=True)
		
		# Save dataframes to text files
		df_o.to_string(folder_path+'angle'+angle[k]+'_ord.txt',
			index=False,justify='left')
		
		df_e.to_string(folder_path+'angle'+angle[k]+'_exord.txt',
			index=False,justify='left')
		
		# Save the number of sources in each beam to a list
		numsource.append(int(len(np.array(phot_table['id']))/2))
	
	# Print number of sources per half-wave plate image and FWHM
	print("FWHM =",fwhm,"pixels")
	for i in range(0,len(numsource),1):
		print("No of sources detected at",ang_dec[i],"degrees:",numsource[i])
	
	return 0

	
def main():
	""" Run script from command line """
	folder_path,apermul,fwhm = get_args()
	return efosc2_pol_phot(folder_path,apermul,fwhm)

	
if __name__ == '__main__':
    sys.exit(main())