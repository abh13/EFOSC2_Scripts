#!/usr/bin/python
# -*- coding: utf-8 -*-
# EFOSC2 polarisation calibration script v1.0
# Created by Adam Higgins, Stefano Covino and Klaas Wiersema
# Email: abh13@le.ac.uk, k.wiersema@warwick.ac.uk

__doc__ = """ Calibration script used for reducing optical linear polarimetric
observations made using the EFOSC2 instrument on-board the NTT at La Silla.
For usage please go to https://github.com/abh13/EFOSC2_Scripts.
"""

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import emcee
import corner
import argparse


def get_args():
	""" Parse command line arguments """
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("Directory",metavar="DIR",type=str,action="store",
		help="Required directory")
	parser.add_argument("Filter",metavar="FILTER",type=str,action="store",
		help="Observed filter - choices are V/B/R/U/i")
	parser.add_argument("Par Angle",metavar="PAR",type=float,action="store",
		help="Parallactic Angle (deg)")
	parser.add_argument("--gain",type=float,default=1.1,dest='gain',
		help="Manually choose the gain - electrons per ADU (default = 1.1)")
		
	args = parser.parse_args()
	folder_path = args.__dict__['Directory']
	wave_band = args.__dict__['Filter']
	par_ang = args.__dict__['Par Angle']
	gain = args.gain
	return folder_path,wave_band,par_ang,gain


def efosc2_cal_mm(folder_path,standard_star_file,mirror_props_file,waveband,
	par_ang,q_values,u_values):
	""" Mueller Matrix method for EFOSC2 instrumental calibration """
	
	
	def MirrorMatrixComplex(n,k,theta):
		# Mueller matrix for metallic reflection
		p = n**2 - k**2 - np.sin(theta)**2
		q = 2*n*k
		s = np.sin(theta)*np.tan(theta)
		r_pos = np.sqrt(p + np.sqrt(p**2 + q**2))/np.sqrt(2)
		r_neg = np.sqrt(-p + np.sqrt(p**2 + q**2))/np.sqrt(2)
		rho = np.sqrt((np.sqrt(p**2 + q**2) + s**2 - 2*s*r_pos)/
			(np.sqrt(p**2 + q**2) + s**2 + 2*s*r_pos))
		delta = np.arctan2(2*s*r_neg,np.sqrt(p**2 + q**2) - s**2)
		row1 = [1 + rho**2, 1 - rho**2, 0, 0]
		row2 = [1 - rho**2, 1 + rho**2, 0, 0]
		row3 = [0, 0, -2*rho*np.cos(delta), -2*rho*np.sin(delta)]
		row4 = [0, 0, 2*rho*np.sin(delta), -2*rho*np.cos(delta)]
		return 0.5*np.matrix([row1, row2, row3, row4])
		

	def RotationMatrix(phi):
		# Mueller rotation matrix
		row1 = [1, 0, 0, 0]
		row2 = [0, np.cos(2*phi), np.sin(2*phi), 0]
		row3 = [0, -np.sin(2*phi), np.cos(2*phi), 0]
		row4 = [0, 0, 0, 1]
		return np.matrix([row1, row2, row3, row4])
		

	def EFOSC2_Matrix(par_angle,n,k,offset):
		# Final matrix for EFOSC2 instrument
		m_rot1 = RotationMatrix(np.radians(offset))
		m_rot2 = RotationMatrix(np.radians(-1*(par_angle)))
		m_m3 = MirrorMatrixComplex(n,k,np.radians(45)) #M3 mirror
		m_rot3 = RotationMatrix(np.radians(-1*(par_angle)))
		m_mirror = MirrorMatrixComplex(n,k,np.radians(0))
		return m_rot1*m_rot2*m_m3*m_rot3*m_mirror
		

	def instrument_pol_function(waveband,q,u,par_angle,f,offset):
		# Function multiples the EFOSC2 matrix with the measured stoke
		# parameters
		n = ni(waveband)
		k = ki(waveband)
		stokes = np.matrix([1, q, u, 0]).transpose()
		s = EFOSC2_Matrix(par_angle,f*n,f*k,offset)*stokes
		return s
		

	def lnprior(parms):
		# Setting priors on physical limits
		f, offset = parms
		if 0 < f < 3 and -180 < offset < 180:
			return 0.0
		return -np.inf
		

	def lnprob(parms,waveband,par_angle,q,u,q_err,u_err):
		# Posterior probability
		lp = lnprior(parms)
		
		if not np.isfinite(lp):
			return -np.inf
			
		f, offset = parms
		q_mod = np.zeros(len(q))
		u_mod = np.zeros(len(q))
		
		for i in range(len(q)):
			s = instrument_pol_function(waveband,0,0,par_angle[i],f,offset)
			q_mod[i] = s[1,0]/s[0,0]
			u_mod[i] = s[2,0]/s[0,0]
			
		q_ls = -0.5*(np.sum((((q - q_mod)/q_err)**2) + np.log(q_err**2)) 
			+ len(q)*np.log(2*np.pi))
		u_ls = -0.5*(np.sum((((u - u_mod)/u_err)**2) + np.log(u_err**2))
			+ len(u)*np.log(2*np.pi))
		result = lp + q_ls + u_ls
		return result
		
	
	def gelman_rubin(chain):
		# Gelman-Rubin test for chain convergence
		ssq = np.var(chain, axis=1, ddof=1)
		w = np.mean(ssq, axis=0)
		theta_b = np.mean(chain, axis=1)
		theta_bb = np.mean(theta_b, axis=0)
		m = float(chain.shape[0])
		n = float(chain.shape[1])
		B = n / (m - 1.0) * np.sum((theta_bb - theta_b)**2, axis=0)
		var_theta = (n - 1.0) / n * w + 1.0 / n * B
		statistic = np.sqrt(var_theta / w)
		return statistic
	

	# Read in unpolarised standard star data
	data = np.genfromtxt(standard_star_file,delimiter=',',dtype=float,
		usecols=(1,2,3,4,5))
	un_par_angle = data[:,0]
	un_q = data[:,1]
	un_qerr = data[:,2]
	un_u = data[:,3]
	un_uerr = data[:,4]

	# Refractive index properties of the aluminium mirror
	data = np.genfromtxt(mirror_props_file,delimiter = ',',skip_header=1)
	mask = np.ma.masked_outside(data[:,0],0.2,2.9)
	lambdam = data[:,0][mask.mask == False]
	nm = data[:,1][mask.mask == False]
	km = data[:,2][mask.mask == False]
	ni = interp1d(lambdam, nm, kind='linear', bounds_error=False)
	ki = interp1d(lambdam, km, kind='linear', bounds_error=False)

	# Guess angle offset and multiplication factors 
	nwalkers, ndim, nsteps = 20, 2, 2500
	guess = np.zeros([nwalkers,ndim])
	
	for i in range(nwalkers):
		angle_0 = np.random.uniform(-90,90)
		f_0 = np.random.uniform(0.5,2.0)
		guess[i] = [f_0,angle_0]
		
	sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,
		args=(waveband,un_par_angle,un_q,un_u,un_qerr,un_uerr))
	
	# Run MCMC
	print("Running MCMC...")
	sampler.run_mcmc(guess,nsteps)	
	print("Done!\n")
	
	# Samples, both including burn-in and cleaned with modified reflective
	# boundaries, and corresponding probabilities
	samples = sampler.chain
	samples[samples < -90] = samples[samples < -90] + 180
	samples[samples > 90] = samples[samples > 90] - 180
	samples_c = samples[:,250:,:]
	samples_cf = samples_c.reshape((-1,ndim))
	samp_prob_cf = sampler.lnprobability[:,250:].reshape((-1))

	# Run Gelman-Rubin test for convergence
	mf_gr, do_gr = gelman_rubin(samples_c)
	mf_conv = []
	do_conv = []
	
	if mf_gr < 1.1:
		mf_conv = "- chains converged!"
		
	else:
		mf_conv = "- chains did not converge!"
		
	if do_gr < 1.1:
		do_conv = "- chains converged!"
		
	else:
		do_conv = "- chains did not converge!"
	
	# Find the best fit (highest probability) parameter set
	best_samp = np.argmax(samp_prob_cf)
	params_best = [samples_cf[best_samp,0],samples_cf[best_samp,1]]
	
	# Print results of MCMC and 1 sigma errors
	p1, p2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
		zip(*np.percentile(samples_cf, [16, 50, 84],axis=0)))
	print("Refractive index Multiplication factor: {0} (+{1} -{2})"
		.format(round(p1[0],3),round(p1[1],3),round(p1[2],3)))
	print("Gelman-Rubin Statistic:",mf_gr,mf_conv)
	print("\nDetector offset (degrees): {0} (+{1} -{2})"
		.format(round(p2[0],2),round(p2[1],2),round(p2[2],2)))
	print("Gelman-Rubin Statistic:",do_gr,do_conv,"\n")

	# Print out numerical values for the two mirror matrices
	print("Mirror Matrix (0 degrees):")
	print(MirrorMatrixComplex(params_best[0]*ni(waveband),
		params_best[0]*ki(waveband),np.radians(0)))
	print("\nMirror Matrix (45 degrees):")
	print(MirrorMatrixComplex(params_best[0]*ni(waveband),
		params_best[0]*ki(waveband),np.radians(45)),"\n")

	# Plot the most probable fit and save the model for the instrumental
	# polarisation	
	xx = np.linspace(-180,180,360)
	qmod = []
	umod = []
	
	for i in range(len(xx)):
		un_instmod = instrument_pol_function(waveband,0,0,xx[i],
			params_best[0],params_best[1])
		qmod.append(un_instmod[1,0]/un_instmod[0,0])
		umod.append(un_instmod[2,0]/un_instmod[0,0])

	plt.figure()
	one = plt.errorbar(un_par_angle,un_q,yerr=un_qerr,color='red',
		fmt='.',label='Q')
	two = plt.errorbar(un_par_angle,un_u,yerr=un_uerr,color='blue',
		fmt='.',label='U')
	plt.plot(xx,qmod,color='black')
	plt.plot(xx,umod,color='black')
	plt.xlabel('Parallactic Angle ($^{\circ}$)')
	plt.xlim(-180,180)
	plt.legend(handles=[one,two])
	modelfile = folder_path + 'best_model.png'
	plt.savefig(modelfile)

	# Plot and save corner plot
	fig = corner.corner(samples_cf,labels=["MF","$\phi_{offset}$"])
	cornerfile = folder_path + 'pol_param_corner.png'
	fig.savefig(cornerfile)

	# Plot and save walker paths
	fig, ax = plt.subplots(2,1,sharex=True)
	ax[0].plot(samples[:, :, 0].T, color='grey')
	ax[0].axhline(params_best[0], color='red')
	ax[0].set_ylabel("MF")
	ax[1].plot(samples[:, :, 1].T, color='grey')
	ax[1].axhline(params_best[1], color='red')
	ax[1].set_ylabel("$\phi_{offset}$")
	pathfile = folder_path + 'walker_paths.png'
	fig.savefig(pathfile)

	# Calculate real polarisations from raw measured values using the EFOSC2
	# matrix and estimated offset and Mf parameters calculated above
	real_q = []
	real_u = []
	
	for i in range(len(q_values)):
		stokes = np.matrix([1,q_values[i],u_values[i],0]).transpose()
		Emat = EFOSC2_Matrix(par_ang,params_best[0]*ni(waveband),
			params_best[0]*ki(waveband),params_best[1])
		result = Emat.I*stokes
		real_q.append(result[1,0]/result[0,0])
		real_u.append(result[2,0]/result[0,0])
		
	return real_q,real_u


def efosc2_cal_sa(folder_path,standard_star_file,mirror_props_file,waveband,
	par_ang,q_values,u_values):
	""" Semi-analytical method for EFOSC2 instrumental calibration """
	
	
	def modq(x,p1,p2):
		# Analytical model for Q
		return p1*np.cos(np.deg2rad((2*x+p2)))
		

	def modu(x,p1,p2):
		# Analytical model for U
		return p1*np.cos(np.deg2rad((2*x+p2-90)))

	
	def lnprior(parms):
		# Set up priors using physical limits
		p1, p2 = parms
		if 0 < p1 < 0.1 and 0 < p2 < 180:
			return 0.0
		return -np.inf
		

	def lnprob(parms, x, y1, y2, y1_err, y2_err):
		# Define the posterior
		lp = lnprior(parms)
		
		if not np.isfinite(lp):
			return -np.inf
			
		p1, p2 = parms
		y1_ls = -0.5*(np.sum((((y1-p1*np.cos(np.deg2rad((2*x+p2))))/y1_err)
			**2) + np.log(y1_err**2) + len(y1)*np.log(2*np.pi)))
		y2_ls = -0.5*(np.sum((((y2-p1*np.cos(np.deg2rad((2*x+p2-90))))/y2_err)
			**2) + np.log(y2_err**2)) + len(y2)*np.log(2*np.pi))
		result = y1_ls + y2_ls
		return result

		
	def gelman_rubin(chain):
		# Gelman-Rubin test for chain convergence
		ssq = np.var(chain, axis=1, ddof=1)
		w = np.mean(ssq, axis=0)
		theta_b = np.mean(chain, axis=1)
		theta_bb = np.mean(theta_b, axis=0)
		m = float(chain.shape[0])
		n = float(chain.shape[1])
		B = n / (m - 1.0) * np.sum((theta_bb - theta_b)**2, axis=0)
		var_theta = (n - 1.0) / n * w + 1.0 / n * B
		statistic = np.sqrt(var_theta / w)
		return statistic
		

	# Read in standard star data
	data = np.genfromtxt(standard_star_file,delimiter=',')
	sourcenames = data[:,0]
	x_data = data[:,1]
	y1 = data[:,2]
	y1_err = data[:,3]
	y2 = data[:,4]
	y2_err = data[:,5]
	xx = np.linspace(-180,180,200)

	# Set up initial guesses for parameters
	nwalkers, ndim, nsteps = 20, 2, 2500
	guess = np.zeros([nwalkers,ndim])
	
	for i in range(nwalkers):
		p1_0 = np.random.uniform(0.01,0.09)
		p2_0 = np.random.uniform(10,170)
		guess[i] = [p1_0,p2_0]
	
	sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,
		args=(x_data,y1,y2,y1_err,y2_err))
		
	# Run MCMC
	print("Running MCMC...")
	sampler.run_mcmc(guess,nsteps)
	print("Done!\n")
	
	# Collect samples, both with burn in and cleaned, and corresponding 
	# probabilities
	samples = sampler.chain
	samples_c = samples[:,250:,:]
	samples_cf = samples_c.reshape((-1,ndim))
	samp_prob_cf = sampler.lnprobability[:,250:].reshape((-1))
	
	# Run Gelman-Rubin test for chain convergence
	p1_gr, p2_gr = gelman_rubin(samples_c)
	p1_conv = []
	p1_conv = []
	
	if p1_gr < 1.1:
		p1_conv = "- chains converged!"
		
	else:
		p1_conv = "- chains did not converge!"
		
	if p2_gr < 1.1:
		p2_conv = "- chains converged!"
		
	else:
		p2_conv = "- chains did not converge!"
	
	# Find the best fit (highest probability) parameter set
	best_samp = np.argmax(samp_prob_cf)
	params_best = [samples_cf[best_samp,0],samples_cf[best_samp,1]]

	# Print the parameter medians and uncertainties at 1 sigma
	p1, p2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
		zip(*np.percentile(samples_cf, [16, 50, 84],axis=0)))

	print("EFOSC2 PQ Instrumental Polarisation: ")
	print("""{0}(±{1})*cos(2x + {2}(±{3}))\n""".format(round(p1[0],3),
		round((p1[1]+p1[2])/2,4),round(p2[0],3),round((p2[1]+p2[2])/2,3)))
	print("EFOSC2 PU Instrumental Polarisation: ")
	print("""{0}(±{1})*cos(2x + {2}(±{3}))\n""".format(round(p1[0],3),round(
		(p1[1]+p1[2])/2,4),round((p2[0]-90),3),round((p2[1]+p2[2])/2,3)))
	print("Gelman-Rubin Statistic (Amplitude):",p1_gr,p1_conv)
	print("Gelman-Rubin Statistic (Angle Offset):",p2_gr,p2_conv,"\n")

	# Plot data and most probable models
	y1_mod = modq(xx,*params_best)
	y2_mod = modu(xx,*params_best)

	plt.figure()
	one = plt.errorbar(x_data,y1,yerr=y1_err,color='red',fmt='.',label='Q')
	two = plt.errorbar(x_data,y2,yerr=y2_err,color='blue',fmt='.',label='U')
	plt.plot(xx,y1_mod,color='black')
	plt.plot(xx,y2_mod,color='black')
	plt.xlabel('Parallactic Angle ($^{\circ}$)')
	plt.xlim(-180,180)
	plt.legend(handles=[one,two])
	modelfile = folder_path + 'best_model.png'
	plt.savefig(modelfile)

	# Plot and save corner plot
	fig = corner.corner(samples_cf,labels=["Wave Amp","$\phi$"])
	cornerfile = folder_path + 'pol_param_corner.png'
	fig.savefig(cornerfile)

	# Plot and save walker paths
	fig, ax = plt.subplots(2,1,sharex=True)
	ax[0].plot(samples[:, :, 0].T, color='grey')
	ax[0].axhline(params_best[0], color='red')
	ax[0].set_ylabel("Wave Amp")
	ax[1].plot(samples[:, :, 1].T, color='grey')
	ax[1].axhline(params_best[1], color='red')
	ax[1].set_ylabel("$\phi$")
	pathfile = folder_path + 'walker_paths.png'
	fig.savefig(pathfile)

	# Calculate real q and u
	real_q = []
	real_u = []
	
	for i in range(len(q_values)):
		real_q.append(q_values[i] - modq(par_ang,*params_best))
		real_u.append(u_values[i] - modu(par_ang,*params_best))
		
	return real_q,real_u


def efosc2_pol(folder_path,wave_band,par_ang,gain):
	""" This pipeline accounts for the calibration of instrumental
	polarisation of the NTT EFOSC2 instrument (the functions above).
	"""


	def beam_data(angle_file):
		# Extracts data for all targets per angle of selected beam
		total_data = {}
		
		cols = ['x','y','flux','area','msky','st_dev','n_sky']
		beam_data = pd.read_csv(angle_file,header=0,names=cols,
			delim_whitespace=True)
	
		return beam_data

	
	def flux_error(beam_info,target_list,gain):
		# Calculates the flux uncertainty for each source per angle per beam
		flux_error = []
		k = 1
		nd = 1
		eta = 1
	   
		for i in range(0,len(target_list),1):		
			flux_err1 = beam_info.flux[i]/(gain*eta*nd)
			flux_err2 = (beam_info.area[i]*beam_info.st_dev[i]*
				beam_info.st_dev[i])
			flux_err3 = ((k/beam_info.n_sky[i])*
				(beam_info.area[i]*beam_info.st_dev[i])**2)
			
			flux_error_calc = np.sqrt(flux_err1 + flux_err2 + flux_err3)
			flux_error.append(flux_error_calc)

		return flux_error

	
	def norm_flux(ordin_beam,extra_beam,ordin_fluxerr,extra_fluxerr,
		target_list):
		# Calculates the normalised flux per angle for each beam and the error
		# on the flux
		norm_flux_value = []
		norm_flux_err = []
		
		for i in range(0,len(target_list),1):		
			nf1 = (ordin_beam.flux[i]-extra_beam.flux[i])
			nf2 = (ordin_beam.flux[i]+extra_beam.flux[i])
			norm_flux = nf1/nf2
			norm_flux_value.append(norm_flux)
			a = np.sqrt((ordin_fluxerr[i]**2)+(extra_fluxerr[i]**2))
			b = (ordin_beam.flux[i]-extra_beam.flux[i])
			c = (ordin_beam.flux[i]+extra_beam.flux[i])
			norm_flux_e = norm_flux*np.sqrt(((a/b)**2)+((a/c)**2))
			norm_flux_err.append(norm_flux_e)

		return(norm_flux_value,norm_flux_err)

	
	def pol_param(norm_flux_0,norm_flux_22,norm_flux_45,norm_flux_67,
		target_list):
		# Calculates the measured Q, U, and P values of the objects
		q_values = []
		u_values = []
		p_values = []
		dtr = np.pi/180

		for i in range(0,len(target_list),1):	
			q = ((0.5*norm_flux_0[i]*np.cos(4*0*dtr))+(0.5*norm_flux_22[i]*
				np.cos(4*22.5*dtr))+(0.5*norm_flux_45[i]*np.cos(4*45*dtr))+
				(0.5*norm_flux_67[i]*np.cos(4*67.5*dtr)))
			u = ((0.5*norm_flux_0[i]*np.sin(4*0*dtr))+(0.5*norm_flux_22[i]*
				np.sin(4*22.5*dtr))+(0.5*norm_flux_45[i]*np.sin(4*45*dtr))+
				(0.5*norm_flux_67[i]*np.sin(4*67.5*dtr)))
			p = np.sqrt((q**2)+(u**2))
			q_values.append(q)
			u_values.append(u)
			p_values.append(p)
			
		return(q_values,u_values,p_values)

	
	def calc_theta(real_u,real_q):
		# Calculates theta for all objects
		theta_values = []
		dtr = np.pi/180

		for i in range(0,len(target_list),1):		
			theta = 0.5*np.arctan(real_u[i]/real_q[i])*(1/dtr)
			theta_values.append(theta)
			
		return theta_values


	def position_angle(theta_values,real_q,real_u,target_list):
		# Calculate proper position angles
		corr_theta_values = []
		
		for i in range(0,len(target_list),1):

			if real_q[i] < 0:
				corr_theta_values.append(theta_values[i]+90)

			if real_q[i] > 0 and real_u[i] > 0:
				corr_theta_values.append(theta_values[i]+0)

			if real_q[i] > 0 and real_u[i] < 0:
				corr_theta_values.append(theta_values[i]+180)

		return corr_theta_values
		

	def parameter_errors(norm_flux_err_0,norm_flux_err_22,norm_flux_err_45,
		norm_flux_err_67,real_p,real_q,real_u,ordin_data_0,extra_data_0,
		ordin_data_22,extra_data_22,ordin_data_45,extra_data_45,ordin_data_67,
		extra_data_67,target_list):
		# Calculate errors on Q, U, P, Theta, SD of the average flux per angle
		q_errors = []
		u_errors = []
		sig_p = []
		flux_sig = []
		theta_errors = []
		dtr = np.pi/180

		for i in range(0,len(target_list),1):		
			q_errors.append(np.sqrt(((0.5*norm_flux_err_0[i]*np.cos(4*0*dtr))
				**2)+((0.5*norm_flux_err_22[i]*np.cos(4*22.5*dtr))**2)+
				((0.5*norm_flux_err_45[i]*np.cos(4*45*dtr))**2)+
				((0.5*norm_flux_err_67[i]*np.cos(4*67.5*dtr))**2)))
			u_errors.append(np.sqrt(((0.5*norm_flux_err_0[i]*np.sin(4*0*dtr))
				**2)+((0.5*norm_flux_err_22[i]*np.sin(4*22.5*dtr))**2)+
				((0.5*norm_flux_err_45[i]*np.sin(4*45*dtr))**2)+
				((0.5*norm_flux_err_67[i]*np.sin(4*67.5*dtr))**2)))
			flux_sig.append(1/(np.sqrt((ordin_data_0.flux[i]+
				extra_data_0.flux[i]+ordin_data_22.flux[i]+
				extra_data_22.flux[i]+ordin_data_45.flux[i]+
				extra_data_45.flux[i]+ordin_data_67.flux[i]+
				extra_data_67.flux[i])/4)))

		for j in range(0,len(target_list),1):
			sig_p.append(np.sqrt((real_q[j]**2*q_errors[j]**2+real_u[j]**2
				*u_errors[j]**2)/(real_q[j]**2+real_u[j]**2)))

		for k in range(0,len(target_list),1):
			theta_errors.append(sig_p[k]/(2*real_p[k]*dtr))

		return(q_errors,u_errors,sig_p,flux_sig,theta_errors)  

	
	def estimated_polarisation(ordin_data_0,extra_data_0,ordin_data_22,
		extra_data_22,ordin_data_45,extra_data_45,ordin_data_67,extra_data_67,
		ordin_fluxerr_0,ordin_fluxerr_22,ordin_fluxerr_45,ordin_fluxerr_67,
		extra_fluxerr_0,extra_fluxerr_22,extra_fluxerr_45,extra_fluxerr_67,
		real_p,sig_p,target_list):
		# Calculate etap (rough estimate of flux snr) and then use MAS
		# estimator from Plaszczynski et al. 2015 to correct for bias		
		snr_f0 = []
		snr_f22 = []
		snr_f45 = []
		snr_f67 = []
		snr_fav = []
		eta = []
		p_corr = []
		
		for i in range(0,len(target_list),1):
			snr_f0.append((ordin_data_0.flux[i]+extra_data_0.flux[i])/
				np.sqrt((ordin_fluxerr_0[i]**2)+(extra_fluxerr_0[i]**2)))
			snr_f22.append((ordin_data_22.flux[i]+extra_data_22.flux[i])/
				np.sqrt((ordin_fluxerr_22[i]**2)+(extra_fluxerr_22[i]**2)))
			snr_f45.append((ordin_data_45.flux[i]+extra_data_45.flux[i])/
				np.sqrt((ordin_fluxerr_45[i]**2)+(extra_fluxerr_45[i]**2)))
			snr_f67.append((ordin_data_67.flux[i]+extra_data_67.flux[i])/
				np.sqrt((ordin_fluxerr_67[i]**2)+(extra_fluxerr_67[i]**2)))

		for j in range(0,len(target_list),1):
			snr_fav.append((snr_f0[j]+snr_f22[j]+snr_f45[j]+snr_f67[j])/4)

		for k in range(0,len(target_list),1):
			eta.append(real_p[k]*snr_fav[k])

		for l in range(0,len(target_list),1):
			p_corr.append(real_p[l]-(sig_p[l]**2*(1-np.exp(-(real_p[l]**2/
				sig_p[l]**2)))/(2*real_p[l])))
		
		return(eta,p_corr)
	
	
	# Begin by reading in files
	file_name_ord0 = 'angle0_ord.txt'
	file_name_ord22 = 'angle225_ord.txt'
	file_name_ord45 = 'angle45_ord.txt'
	file_name_ord67 = 'angle675_ord.txt'
	file_name_ext0 = 'angle0_exord.txt'
	file_name_ext22 = 'angle225_exord.txt'
	file_name_ext45 = 'angle45_exord.txt'
	file_name_ext67 = 'angle675_exord.txt'

	ordin_0 = os.path.join(folder_path,file_name_ord0)
	ordin_22 = os.path.join(folder_path,file_name_ord22)
	ordin_45 = os.path.join(folder_path,file_name_ord45)
	ordin_67 = os.path.join(folder_path,file_name_ord67)
	extra_0 = os.path.join(folder_path,file_name_ext0)
	extra_22 = os.path.join(folder_path,file_name_ext22)
	extra_45 = os.path.join(folder_path,file_name_ext45)
	extra_67 = os.path.join(folder_path,file_name_ext67)

	# Defines two lists of ordinary and extra ordinary files to extract data
	ordinary_beam = [ordin_0,ordin_22,ordin_45,ordin_67]
	extra_beam = [extra_0,extra_22,extra_45,extra_67]
	
	# Raise Error if files or folder cannot be found
	try:
		ordin_data_0 = beam_data(ordinary_beam[0])
		ordin_data_22 = beam_data(ordinary_beam[1])
		ordin_data_45 = beam_data(ordinary_beam[2])
		ordin_data_67 = beam_data(ordinary_beam[3])
		extra_data_0 = beam_data(extra_beam[0])
		extra_data_22 = beam_data(extra_beam[1])
		extra_data_45 = beam_data(extra_beam[2])
		extra_data_67 = beam_data(extra_beam[3])
		
	except FileNotFoundError as e:
		print('Cannot find the folder or files you are looking for')
		sys.exit()

	# Creates target list of sources
	target_list = []
	for i in range(0,len(ordin_data_0.x),1):	
		name = 'Source '+ str(i+1)
		target_list.append(name)

	# Ensure all angles in both ordinary and extraordinary beams have the
	# same number of sources
	if (len(ordin_data_0.x) or len(ordin_data_22.x) or len(ordin_data_45.x)
		or len(ordin_data_67.x)) != (len(extra_data_0.x) or
		len(extra_data_22.x) or len(extra_data_45.x) or len(extra_data_67.x)):
		
		print('One or more data files have unequal numbers of sources!')
		sys.exit()

	# Calculate and store flux errors
	ordin_fluxerr_0 = flux_error(ordin_data_0,target_list,gain)
	ordin_fluxerr_22 = flux_error(ordin_data_22,target_list,gain)
	ordin_fluxerr_45 = flux_error(ordin_data_45,target_list,gain)
	ordin_fluxerr_67 = flux_error(ordin_data_67,target_list,gain)

	extra_fluxerr_0 = flux_error(extra_data_0,target_list,gain)
	extra_fluxerr_22 = flux_error(extra_data_22,target_list,gain)
	extra_fluxerr_45 = flux_error(extra_data_45,target_list,gain)
	extra_fluxerr_67 = flux_error(extra_data_67,target_list,gain)
	
	# Calculate and store normalised flux values and errors
	norm_flux_0,norm_flux_err_0 = norm_flux(ordin_data_0,extra_data_0,
		ordin_fluxerr_0,extra_fluxerr_0,target_list)
	norm_flux_22,norm_flux_err_22 = norm_flux(ordin_data_22,extra_data_22,
		ordin_fluxerr_22,extra_fluxerr_22,target_list)
	norm_flux_45,norm_flux_err_45 = norm_flux(ordin_data_45,extra_data_45,
		ordin_fluxerr_45,extra_fluxerr_45,target_list)
	norm_flux_67,norm_flux_err_67 = norm_flux(ordin_data_67,extra_data_67,
		ordin_fluxerr_67,extra_fluxerr_67,target_list)
		
	# Calculate and store Q, U, and P values
	q_values,u_values,p_values = pol_param(norm_flux_0,norm_flux_22,
		norm_flux_45,norm_flux_67,target_list)
  
	# Account for EFOSC2 instrumental polarisation. If the waveband isn't V,
	# B, R, U or i then the program terminates
	mirror_props_file = 'METALS_Aluminium_Rakic.txt'
	print("")	
	
	if wave_band == 'V':
		real_q,real_u = efosc2_cal_mm(folder_path,'pv_standards.txt',
			mirror_props_file,0.547,par_ang,q_values,u_values)
			
	elif wave_band == 'R':
		real_q,real_u = efosc2_cal_mm(folder_path,'pr_standards.txt',
			mirror_props_file,0.643,par_ang,q_values,u_values)
			
	elif wave_band == 'B':
		real_q,real_u = efosc2_cal_mm(folder_path,'pb_standards.txt',
			mirror_props_file,0.440,par_ang,q_values,u_values)
	
	elif wave_band == 'U':
		real_q,real_u = efosc2_cal_mm(folder_path,'pu_standards.txt',
			mirror_props_file,0.354,par_ang,q_values,u_values)
	
	elif wave_band == 'i':
		real_q,real_u = efosc2_cal_mm(folder_path,'pi_standards.txt',
			mirror_props_file,0.793,par_ang,q_values,u_values)
			
	else:
		print("Code does not calibrate for this filter! Please check input!!")
		sys.exit()

	# Calculate real value of p
	real_p = []
	
	for i in range(0,len(target_list),1):
		real_p.append(np.sqrt((real_q[i]**2)+(real_u[i]**2)))
		
	# Calculate theta values
	theta_values = calc_theta(real_u,real_q)
	corr_theta_values = position_angle(theta_values,real_q,real_u,target_list)
	
	# Store Q, U and P, Theta and P error values in following arrays
	data_array = parameter_errors(norm_flux_err_0,norm_flux_err_22,
		norm_flux_err_45,norm_flux_err_67,real_p,real_q,real_u,ordin_data_0,
		extra_data_0,ordin_data_22,extra_data_22,ordin_data_45,extra_data_45,
		ordin_data_67,extra_data_67,target_list)
		
	q_errors,u_errors,sig_p,flux_sig,theta_errors = data_array

	# Eta, estimator name and corrected P values stored in following arrays
	pol_values = estimated_polarisation(ordin_data_0,extra_data_0,
		ordin_data_22,extra_data_22,ordin_data_45,extra_data_45,ordin_data_67,
		extra_data_67,ordin_fluxerr_0,ordin_fluxerr_22,ordin_fluxerr_45,
		ordin_fluxerr_67,extra_fluxerr_0,extra_fluxerr_22,extra_fluxerr_45,
		extra_fluxerr_67,real_p,sig_p,target_list)
	
	eta_values,p_corr_values = pol_values
	
	# Convert pol values into percentages and round to 5 s.f
	q_values = [round(x*100,5) for x in q_values]
	real_q = [round(x*100,5) for x in real_q]
	q_errors = [round(x*100,5) for x in q_errors]
	u_values = [round(x*100,5) for x in u_values]
	real_u = [round(x*100,5) for x in real_u]
	u_errors = [round(x*100,5) for x in u_errors]
	p_values = [round(x*100,5) for x in p_values]
	real_p = [round(x*100,5) for x in real_p]
	sig_p = [round(x*100,5) for x in sig_p]
	p_corr_values = [round(x*100,5) for x in p_corr_values]
	corr_theta_values = [round(x,5) for x in corr_theta_values]
	theta_errors = [round(x,5) for x in theta_errors]
	eta_values = [round(x,5) for x in eta_values]
	snr = [round(x/y,5) for x, y in zip(p_corr_values,sig_p)]

	# Create dataframe and save results to file
	cols = ['Qm(%)','Qr(%)','Q Err(%)','Um(%)','Ur(%)','U Err(%)','Pm(%)',
		'Pr(%)','SNR','Sig_P(%)','Pcorr(%)','Angle','Angle Err']
	df = pd.DataFrame({cols[0]:q_values,cols[1]:real_q,cols[2]:q_errors,
		cols[3]:u_values,cols[4]:real_u,cols[5]:u_errors,cols[6]:p_values,
		cols[7]:real_p,cols[8]:snr,cols[9]:sig_p,cols[10]:p_corr_values,
		cols[11]:corr_theta_values,cols[12]:theta_errors})
	df.to_string(folder_path+'source_results.txt',index=False,justify='left')

	# Close matplotlib windows
	plt.close('all')
	return 0
	
	
def main():
	""" Run the script """
	folder_path,wave_band,par_ang,gain = get_args()
	return efosc2_pol(folder_path,wave_band,par_ang,gain)

	
if __name__ == '__main__':
    sys.exit(main())