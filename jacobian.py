import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.neighbors import KernelDensity
import os
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from util import *

def jacobian_1d(data, z, z_bins = 50, nbins = 50, print_figures = True, savepath = './'): 
	'''
	Calculates the Jacobian of a nonlinear transformation of the input trajectory; assumes the latent space is one-dimensional.
	---------

	INPUT:
	 
	data (N, n): Input trajectory of size N frames by n features. Right now, n must be equal 2

	z (N, m): Trajectory of the latent space variables of size N frames by m features

	z_bins (int): Number of bins for histogramming in the latent space
					   
	nbins (int): Number of bins for histogramming in the input space
	  
	print_figures (bool): Boolean argument specifying whether the projections of the latent space coordinates
						  onto the input surface should be written to file (True) or not (False)

	savepath (str): Path for saving figures. Only used if print_figures = True

	OUTPUT:

	z1_hist (nbins, nbins): Projection of z onto the two-dimensional input space
	z2_hist (nbins, nbins): Projection of z onto the two-dimensional input space

	dzdx1, dzdx2 (nbins, nbins): Gradient of z with respect to the first input
	dzdy1, dzdy2 (nbins, nbins): Gradient of z with respect to the second input
	dzdx1_traj, dzdx2_traj (N,): Trajectory of the gradient with respect to the first input
	dzdy1_traj, dzdy2_traj (N,): Trajectory of the gradient with respect to the second input

	'''

	xbins = np.linspace(data[:,0].min() - 1e-9, data[:,0].max() + 1e-9, nbins + 2)
	ybins = np.linspace(data[:,1].min() - 1e-9, data[:,1].max() + 1e-9, nbins + 2)
	hist = np.histogram2d(data[:,0], data[:,1], bins = [xbins, ybins])

	dx = xbins[1] - xbins[0]
	dy = ybins[1] - ybins[0]

	# histogram

	z1_hist, dummy0, dummy1 = np.histogram2d(data[:,0], data[:,1], weights = z[:,0], bins = [xbins, ybins])
	z1_hist = np.divide(z1_hist, hist[0])

	dzdx1, dzdy1 = np.gradient(z1_hist)

	xx, yy = np.meshgrid((xbins[1:] + xbins[:-1]) / 2, (ybins[1:] + ybins[:-1]) / 2)

	if print_figures:
		os.makedirs(savepath + 'figures', exist_ok = True)
		im = plt.contourf(xx, yy, z1_hist.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, z1_hist.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'z$_1$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/z1_projected.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdx1.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdx1.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_1}{\partial x}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz1dx_proj.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdy1.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdy1.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_1}{\partial y}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz1dy.pdf', dpi = 300)
		plt.show()
		plt.close()


	xhist, xbins = np.histogram(data[:,0], bins = nbins + 1)
	yhist, ybins = np.histogram(data[:,1], bins = nbins + 1)

	xbins = (xbins[1:] + xbins[:-1]) / 2
	ybins = (ybins[1:] + ybins[:-1]) / 2

	xbin_traj = np.digitize(data[:,0], xbins) - 1
	ybin_traj = np.digitize(data[:,1], ybins) - 1

	bin_list = np.column_stack([xbin_traj, ybin_traj])

	dzdx1_traj = np.zeros(len(z))
	dzdy1_traj = np.zeros(len(z))

	for k, xbin, ybin in list_enumerate(bin_list):
		#if k % 10000 == 0: print(k)
		dzdx1_traj[k] = dzdx1[xbin, ybin]
		dzdy1_traj[k] = dzdy1[xbin, ybin]

	# give nans zero weight in the succeding analysis
	dzdx1_traj = np.nan_to_num(dzdx1_traj)
	dzdy1_traj = np.nan_to_num(dzdy1_traj)

			
	return z1_hist, dzdx1, dzdy1, dzdx1_traj, dzdy1_traj


def jacobian_2d(data, z, z_bins = 50, nbins = 50, beta = 1, print_figures = True, savepath = './'): 
	'''
	Calculates the Jacobian of a nonlinear transformation of the input trajectory; assumes the latent space is two-dimensional.
	---------

	INPUT:
	 
	data (N, n): Input trajectory of size N frames by n features. Right now, n must be equal 2

	z (N, m): Trajectory of the latent space variables of size N frames by m features

	z_bins (int): Number of bins for histogramming in the latent space
					   
	nbins (int): Number of bins for histogramming in the input space
					   
	print_figures (bool): Boolean argument specifying whether the projections of the latent space coordinates
						  onto the input surface should be written to file (True) or not (False)

	savepath (str): Path for saving figures. Only used if print_figures = True
					   
	OUTPUT:

	z1_hist (nbins, nbins): Projection of z onto the two-dimensional input space
	z2_hist (nbins, nbins): Projection of z onto the two-dimensional input space

	dzdx1, dzdx2 (nbins, nbins): Gradient of z with respect to the first input
	dzdy1, dzdy2 (nbins, nbins): Gradient of z with respect to the second input
	dzdx1_traj, dzdx2_traj (N,): Trajectory of the gradient with respect to the first input
	dzdy1_traj, dzdy2_traj (N,): Trajectory of the gradient with respect to the second input

	'''

	xbins = np.linspace(data[:,0].min() - 1e-9, data[:,0].max() + 1e-9, nbins + 2)
	ybins = np.linspace(data[:,1].min() - 1e-9, data[:,1].max() + 1e-9, nbins + 2)
	hist = np.histogram2d(data[:,0], data[:,1], bins = [xbins, ybins])

	dx = xbins[1] - xbins[0]
	dy = ybins[1] - ybins[0]

	# histogram

	z1_hist, dummy0, dummy1 = np.histogram2d(data[:,0], data[:,1], weights = z[:,0], bins = [xbins, ybins])
	z1_hist = np.divide(z1_hist, hist[0])
	z2_hist, xbins, ybins = np.histogram2d(data[:,0], data[:,1], weights = z[:,1], bins = [xbins, ybins])
	z2_hist = np.divide(z2_hist, hist[0])

	dzdx1, dzdy1 = np.gradient(z1_hist)
	dzdx2, dzdy2 = np.gradient(z2_hist)

	xx, yy = np.meshgrid((xbins[1:] + xbins[:-1]) / 2, (ybins[1:] + ybins[:-1]) / 2)

	if print_figures:
		os.makedirs(savepath + 'figures', exist_ok = True)
		im = plt.contourf(xx, yy, z1_hist.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, z1_hist.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'z$_1$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/z1_projected.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, z2_hist.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, z2_hist.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'z$_2$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/z2_projected.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdx1.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdx1.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_1}{\partial x}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz1dx_proj.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdy1.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdy1.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_1}{\partial y}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz1dy.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdx2.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdx2.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_2}{\partial x}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz2dx.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdy2.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdy2.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_2}{\partial y}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz2dy.pdf', dpi = 300)
		plt.show()
		plt.close()


	xhist, xbins = np.histogram(data[:,0], bins = nbins + 1)
	yhist, ybins = np.histogram(data[:,1], bins = nbins + 1)

	xbins = (xbins[1:] + xbins[:-1]) / 2
	ybins = (ybins[1:] + ybins[:-1]) / 2

	xbin_traj = np.digitize(data[:,0], xbins) - 1
	ybin_traj = np.digitize(data[:,1], ybins) - 1

	bin_list = np.column_stack([xbin_traj, ybin_traj])

	dzdx1_traj = np.zeros(len(z))
	dzdy1_traj = np.zeros(len(z))
	dzdx2_traj = np.zeros(len(z))
	dzdy2_traj = np.zeros(len(z))

	for k, xbin, ybin in list_enumerate(bin_list):
		#if k % 10000 == 0: print(k)
		dzdx1_traj[k] = dzdx1[xbin, ybin]
		dzdy1_traj[k] = dzdy1[xbin, ybin]
		dzdx2_traj[k] = dzdx2[xbin, ybin]
		dzdy2_traj[k] = dzdy2[xbin, ybin]

	# give nans zero weight in the succeding analysis
	dzdx1_traj = np.nan_to_num(dzdx1_traj)
	dzdy1_traj = np.nan_to_num(dzdy1_traj)
	dzdx2_traj = np.nan_to_num(dzdx2_traj)
	dzdy2_traj = np.nan_to_num(dzdy2_traj)

			
	return z1_hist, z2_hist, dzdx1, dzdy1, dzdx2, dzdy2, dzdx1_traj, dzdy1_traj, dzdx2_traj, dzdy2_traj

def calc_energy_entropy_1d(data, z, dzdx1_traj, dzdy1_traj, Ut, nbins = 50, beta = 1, print_figures = True, NPT = False, KDE = False,
						  bandwidth1 = 1.0, bandwidth2 = 1.0, savepath = './'):
	'''
	Calculates the Jacobian of a nonlinear transformation of the input trajectory; assumes the latent space is two-dimensional.
	---------

	INPUT:
	 
	data (N, n): Input trajectory of size N frames by n features. Right now, n must be equal 2

	z (N, m): Trajectory of the latent space variables of size N frames by m features
						  
	dzdx1_traj (N,): Trajectory of the gradient along the first data coordinate for the latent coordinate.
								 
	dzdy1_traj (N,): Trajectory of the gradient along the second data coordinate for the latent coordinate.
								 
	Ut (N,): Trajectory of the energy as a function of time from the underlying simulation; if the underlying simulation is run
			 in the NPT ensemble, then this array should contain the appropriate enthalpy, using a method such as that described in
			 Kolias et al. Adv. Theory Simul., 2020, 3, 2000092
								 
	nbins (int): Number of bins for histogramming in the input space

	beta (float): Inverse thermal energy of the system, i.e. 1 / kT

	print_figures (bool): Boolean argument specifying whether the projections of the latent space coordinates
						  onto the input surface should be written to file (True) or not (False)
						  
	NPT (bool): Boolean specifying whether the input data originates from a simulation performed in the NPT ensemble or not.
				Only determines how the free-energy plots are labeled (i.e. whether G(z) or A(z))
				
	KDE (bool): Whether or not to use a Gaussian KDE with bandwiths specified by bandwidth1 and bandwidth2 to calculate the histogram
				for dU and dG.

	bandwidth1 (float): Bandwidths for the Gaussian KDE estimator along the latent space coordinate
															
	savepath (str): Path for saving figures. Only used if print_figures = True
						
	OUTPUT:

	z1_bins (nbins,): Bins used to histogram along the latent space coordinate

	dG1 (nbins,): Delta U along the latent space coordinate. Will either be
					   the Helmoltz or Gibbs free energy, depending on the value input for 'NPT'

	dU1 (nbins,): Delta U along the first latent space coordinate, respectively

	dS1 (nbins,): Delta S along the first latent space coordinate, respectively
	'''

	jac1 = np.linalg.norm(np.array([dzdx1_traj, dzdy1_traj]), axis = 0)

	z1_bins = np.linspace(z[:,0].min() - 1e-9, z[:,0].max() + 1e-9, nbins + 1)

	if KDE:    
		path = 'bandwidth1_' + str(bandwidth1) + '_bandwidth2_' + str(bandwidth2) + '/'
		os.makedirs(saveapath + 'bandwidth1_' + str(bandwidth1) + '_bandwidth2_' + str(bandwidth2), exist_ok = True)
		KD1 = KernelDensity(bandwidth = bandwidth1)
		KD1.fit(z[:,0][:,np.newaxis], sample_weight = weights * np.linalg.norm(np.array([dzdx1_traj, dzdy1_traj])))
		grid1 = np.linspace(np.min(z[:,0]),np.max(z[:,0]),50)
		samp1 = KD1.score_samples(grid1[:, np.newaxis])
		samp1 = samp1.reshape(50)
		p1 = np.exp(samp1)

		G1 = -kT * np.log(p1)
		G11 = G1 - G1[G1.argmin()]
		U1_fuzzy, counter1 = fuzzy_histogram(z[:,0], Ut * jac1, grid1, binwidth = binwidth1)

		U1_fuzzy = np.divide(U1_fuzzy, counter1)
		U1 = U1_fuzzy - U1_fuzzy[G1.argmin()]
		
	else:

		G1_hist, G1_bins = np.histogram(z[:,0], bins = z1_bins, weights = jac1)

		# unitless
		G11 = -np.log(np.nan_to_num(G1_hist / (G1_hist.sum())) + 1e-10)

		# add weights from the Jacobian

		z1_bins = np.linspace(z[:,0].min() - 1e-9, z[:,0].max() + 1e-9, nbins + 1)

		U1 = np.zeros(len(z1_bins) - 1)
		counter1 = np.zeros_like(U1)

		z1_bins = np.linspace(z[:,0].min() - 1e-9, z[:,0].max() + 1e-9, nbins + 1)

		U1_hist, U1_bins = np.histogram(z[:,0], bins = z1_bins, weights = Ut * jac1)
		counter1, dummy = np.histogram(z[:,0], bins = z1_bins, weights = jac1)

		# unitless
		U1 = np.divide(U1_hist, counter1)
		
	ref = np.nanargmin(G11)

	if print_figures:
		os.makedirs(savepath + 'figures', exist_ok = True)
		fig, ax0 = plt.subplots()
		if NPT:
			ax0.plot((z1_bins[1:] + z1_bins[:-1]) / 2, G11 - G11[ref], c = 'k', lw = 2, label = r'$\Delta$G(z$_1$)')
			ax0.plot((z1_bins[1:] + z1_bins[:-1]) / 2, U1 - U1[ref], c = 'r', lw = 2, label = r'$\Delta$H(z$_1$)')
			ybox1 = TextArea(r"$\Delta$G(z$_1$)/k$_B$T", textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox2 = TextArea("or ",     textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox3 = TextArea(r"$\Delta$H(z$_1$)/k$_B$T", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
		else:
			ax0.plot((z1_bins[1:] + z1_bins[:-1]) / 2, G11 - G11[ref], c = 'k', lw = 2, label = r'$\Delta$A(z$_1$)')
			ax0.plot((z1_bins[1:] + z1_bins[:-1]) / 2, U1 - U1[ref], c = 'r', lw = 2, label = r'$\Delta$U(z$_1$)')
			ybox1 = TextArea(r"$\Delta$A(z$_1$)/k$_B$T", textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox2 = TextArea("or ",     textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox3 = TextArea(r"$\Delta$U(z$_1$)/k$_B$T", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
		ax1 = ax0.twinx()
		ax1.plot((z1_bins[1:] + z1_bins[:-1]) / 2, (-U1 + U1[ref] + G11 - G11[ref]), c = 'b', lw = 2)
		ax0.tick_params(axis = 'both', size = 12)
		ax1.tick_params(axis = 'y', size = 12, labelcolor = 'b')
		ax0.set_xlabel(r'z$_1$', fontsize = 16)
		#ax0.set_ylabel(r'$\Delta$A(z$_1$) or $\Delta$U(z$_1$), kJ / mol', fontsize = 16)
		ax1.set_ylabel(r'-$\Delta$S(z$_1$)/k$_B$', fontsize = 16, color = 'b')
		ax0.legend(loc = 'upper left')
		#ax1.legend(loc = 'upper right', labelcolor = 'b')
		ax0.set_ylim((-0.75, 4))
		ax1.set_ylim((-0.75, 4))
		ybox = VPacker(children=[ybox1, ybox2, ybox3],align="bottom", pad=0, sep=5)

		anchored_ybox = AnchoredOffsetbox(loc=10, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.17, 0.49), 
										  bbox_transform=ax0.transAxes, borderpad=0.)

		ax0.add_artist(anchored_ybox)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/z1_energy_entropy.pdf', dpi = 300)
		plt.show()
		plt.close()

		
		dU1 = U1 - U1[ref]
		dG1 = G11 - G11[ref]
		dS1 = (U1 - U1[ref] - G11 + G11[ref])

	return z1_bins, dG1, dU1, dS1 

def calc_energy_entropy_2d(data, z, dzdx1_traj, dzdy1_traj, dzdx2_traj, dzdy2_traj, Ut, nbins = 50, beta = 1, print_figures = True, NPT = False, KDE = False,
						  bandwidth1 = 1.0, bandwidth2 = 1.0, savepath = './'):
	'''
	Calculates the Jacobian of a nonlinear transformation of the input trajectory; assumes the latent space is two-dimensional.
	---------

	INPUT:
	 
	data (N, n): Input trajectory of size N frames by n features. Right now, n must be equal 2

	z (N, m): Trajectory of the latent space variables of size N frames by m features
						  
	dzdx1_traj, dzdx2_traj (N,): Trajectory of the gradient along the first data coordinate for the first and second 
								 latent coordinates, respectively.
								 
	dzdy1_traj, dzdy2_traj (N,): Trajectory of the gradient along the second data coordinate for the first and second 
								 latent coordinates, respectively.
								 
	Ut (N,): Trajectory of the energy as a function of time from the underlying simulation; if the underlying simulation is run
			 in the NPT ensemble, then this array should contain the appropriate enthalpy, using a method such as that described in
			 Kolias et al. Adv. Theory Simul., 2020, 3, 2000092
								 
	nbins (int): Number of bins for histogramming in the input space

	beta (float): Inverse thermal energy of the system, i.e. 1 / kT

	print_figures (bool): Boolean argument specifying whether the projections of the latent space coordinates
						  onto the input surface should be written to file (True) or not (False)
						  
	NPT (bool): Boolean specifying whether the input data originates from a simulation performed in the NPT ensemble or not.
				Only determines how the free-energy plots are labeled (i.e. whether G(z) or A(z))
				
	KDE (bool): Whether or not to use a Gaussian KDE with bandwiths specified by bandwidth1 and bandwidth2 to calculate the histogram
				for dU and dG.

	bandwidth1, bandwidth2 (float): Bandwidths for the Gaussian KDE estimator along the first and second latent space coordinates,
									respectively.
															
	savepath (str): Path for saving figures. Only used if print_figures = True
						
	OUTPUT:

	z1_bins, z2_bins (nbins,): Bins used to histogram along the two latent space coordinates

	dG1, dG2 (nbins,): Delta U along the first and second latent space coordinates, respectively. Will either be
					   the Helmoltz or Gibbs free energy, depending on the value input for 'NPT'

	dU1, dU2 (nbins,): Delta U along the first and second latent space coordinates, respectively

	dS1, dS2 (nbins,): Delta S along the first and second latent space coordinates, respectively
	'''

	jac1 = np.linalg.norm(np.array([dzdx1_traj, dzdy1_traj]), axis = 0)
	jac2 = np.linalg.norm(np.array([dzdx2_traj, dzdy2_traj]), axis = 0)

	z1_bins = np.linspace(z[:,0].min() - 1e-9, z[:,0].max() + 1e-9, nbins + 1)
	z2_bins = np.linspace(z[:,1].min() - 1e-9, z[:,1].max() + 1e-9, nbins + 1)

	if KDE:    
		path = 'bandwidth1_' + str(bandwidth1) + '_bandwidth2_' + str(bandwidth2) + '/'
		os.makedirs(savepath + 'bandwidth1_' + str(bandwidth1) + '_bandwidth2_' + str(bandwidth2), exist_ok = True)
		KD1 = KernelDensity(bandwidth = bandwidth1)
		KD1.fit(z[:,0][:,np.newaxis], sample_weight = weights * np.linalg.norm(np.array([dzdx1_traj, dzdy1_traj])))
		grid1 = np.linspace(np.min(z[:,0]),np.max(z[:,0]),50)
		samp1 = KD1.score_samples(grid1[:, np.newaxis])
		samp1 = samp1.reshape(50)
		p1 = np.exp(samp1)

		KD2 = KernelDensity(bandwidth = bandwidth2)
		KD2.fit(z[:,1][:,np.newaxis], sample_weight = weights * np.linalg.norm(np.array([dzdx2_traj, dzdy2_traj])))
		grid2 = np.linspace(np.min(z[:,1]),np.max(z[:,1]),50)
		samp2 = KD2.score_samples(grid2[:, np.newaxis])
		samp2 = samp2.reshape(50)
		p2 = np.exp(samp2)

		G1 = -kT * np.log(p1)
		G11 = G1 - G1[G1.argmin()]
		G2 = -kT * np.log(p2)
		G22 = G2 - G2[G2.argmin()]
		U1_fuzzy, counter1 = fuzzy_histogram(z[:,0], Ut * jac1, grid1, binwidth = binwidth1)
		U2_fuzzy, counter2 = fuzzy_histogram(z[:,1], Ut * jac2, grid2, binwidth = binwidth2)

		U1_fuzzy = np.divide(U1_fuzzy, counter1)
		U1 = U1_fuzzy - U1_fuzzy[G1.argmin()]
		U2_fuzzy = np.divide(U2_fuzzy, counter2)
		U2 = U2_fuzzy - U2_fuzzy[G2.argmin()]
		
	else:

		G1_hist, G1_bins = np.histogram(z[:,0], bins = z1_bins, weights = jac1)
		G2_hist, G2_bins = np.histogram(z[:,1], bins = z2_bins, weights = jac2)

		# unitless
		G11 = -np.log(np.nan_to_num(G1_hist / (G1_hist.sum())) + 1e-10)
		G22 = -np.log(np.nan_to_num(G2_hist / (G2_hist.sum())) + 1e-10)

		# add weights from the Jacobian

		z1_bins = np.linspace(z[:,0].min() - 1e-9, z[:,0].max() + 1e-9, nbins + 1)
		z2_bins = np.linspace(z[:,1].min() - 1e-9, z[:,1].max() + 1e-9, nbins + 1)

		U1 = np.zeros(len(z1_bins) - 1)
		U2 = np.zeros(len(z2_bins) - 1)
		counter1 = np.zeros_like(U1)
		counter2 = np.zeros_like(U2)

		z1_bins = np.linspace(z[:,0].min() - 1e-9, z[:,0].max() + 1e-9, nbins + 1)
		z2_bins = np.linspace(z[:,1].min() - 1e-9, z[:,1].max() + 1e-9, nbins + 1)

		U1_hist, U1_bins = np.histogram(z[:,0], bins = z1_bins, weights = Ut * jac1)
		counter1, dummy = np.histogram(z[:,0], bins = z1_bins, weights = jac1)
		U2_hist, U2_bins = np.histogram(z[:,1], bins = z2_bins, weights = Ut * jac2)
		counter2, dummy = np.histogram(z[:,1], bins = z2_bins, weights = jac2)
        

		# unitless
		U1 = np.divide(U1_hist, counter1)
		U2 = np.divide(U2_hist, counter2)
		
	ref = np.nanargmin(G11)

	if print_figures:
		os.makedirs(savepath + 'figures', exist_ok = True)
		fig, ax0 = plt.subplots()
		if NPT:
			ax0.plot((z1_bins[1:] + z1_bins[:-1]) / 2, G11 - G11[ref], c = 'k', lw = 2, label = r'$\Delta$G(z$_1$)')
			ax0.plot((z1_bins[1:] + z1_bins[:-1]) / 2, U1 - U1[ref], c = 'r', lw = 2, label = r'$\Delta$H(z$_1$)')
			ybox1 = TextArea(r"$\Delta$G(z$_1$)/k$_B$T", textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox2 = TextArea("or ",     textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox3 = TextArea(r"$\Delta$H(z$_1$)/k$_B$T", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
		else:
			ax0.plot((z1_bins[1:] + z1_bins[:-1]) / 2, G11 - G11[ref], c = 'k', lw = 2, label = r'$\Delta$A(z$_1$)')
			ax0.plot((z1_bins[1:] + z1_bins[:-1]) / 2, U1 - U1[ref], c = 'r', lw = 2, label = r'$\Delta$U(z$_1$)')
			ybox1 = TextArea(r"$\Delta$A(z$_1$)/k$_B$T", textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox2 = TextArea("or ",     textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox3 = TextArea(r"$\Delta$U(z$_1$)/k$_B$T", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
		ax1 = ax0.twinx()
		ax1.plot((z1_bins[1:] + z1_bins[:-1]) / 2, (-U1 + U1[ref] + G11 - G11[ref]), c = 'b', lw = 2)
		ax0.tick_params(axis = 'both', size = 12)
		ax1.tick_params(axis = 'y', size = 12, labelcolor = 'b')
		ax0.set_xlabel(r'z$_1$', fontsize = 16)
		#ax0.set_ylabel(r'$\Delta$A(z$_1$) or $\Delta$U(z$_1$), kJ / mol', fontsize = 16)
		ax1.set_ylabel(r'-$\Delta$S(z$_1$)/k$_B$', fontsize = 16, color = 'b')
		ax0.legend(loc = 'upper left')
		#ax1.legend(loc = 'upper right', labelcolor = 'b')
		ax0.set_ylim((-0.75, 4))
		ax1.set_ylim((-0.75, 4))
		ybox = VPacker(children=[ybox1, ybox2, ybox3],align="bottom", pad=0, sep=5)

		anchored_ybox = AnchoredOffsetbox(loc=10, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.17, 0.49), 
										  bbox_transform=ax0.transAxes, borderpad=0.)

		ax0.add_artist(anchored_ybox)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/z1_energy_entropy.pdf', dpi = 300)
		plt.show()
		plt.close()

		U2 = np.nan_to_num(U2, nan = np.nanmax(U2))
		ref = np.nanargmin(G22)
		fig, ax0 = plt.subplots()
		if NPT:
			ax0.plot((z2_bins[1:] + z2_bins[:-1]) / 2, G22 - G22[ref], c = 'k', lw = 2, label = r'$\Delta$G(z$_2$)')
			ax0.plot((z2_bins[1:] + z2_bins[:-1]) / 2, U2 - U2[ref], c = 'r', lw = 2, label = r'$\Delta$H(z$_2$)')
			ybox1 = TextArea(r"$\Delta$G(z$_2$)/k$_B$T", textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox2 = TextArea("or ",     textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox3 = TextArea(r"$\Delta$H(z$_2$)/k$_B$T", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
		else:
			ax0.plot((z2_bins[1:] + z2_bins[:-1]) / 2, G22 - G22[ref], c = 'k', lw = 2, label = r'$\Delta$A(z$_2$)')
			ax0.plot((z2_bins[1:] + z2_bins[:-1]) / 2, U2 - U2[ref], c = 'r', lw = 2, label = r'$\Delta$U(z$_2$)')
			ybox1 = TextArea(r"$\Delta$A(z$_2$)/k$_B$T", textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox2 = TextArea("or ",     textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
			ybox3 = TextArea(r"$\Delta$U(z$_2$)/k$_B$T", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
		ax1 = ax0.twinx()
		ax1.plot((z2_bins[1:] + z2_bins[:-1]) / 2, (-U2 + U2[ref] + G22 - G22[ref]), c = 'b', lw = 2)
		ax0.tick_params(axis = 'both', size = 12)
		ax1.tick_params(axis = 'y', size = 12, labelcolor = 'b')
		ax0.set_xlabel(r'z$_2$', fontsize = 16)
		ax1.set_ylabel(r'-$\Delta$S(z$_2$)/k$_B$', fontsize = 16, color = 'b')
		ax0.legend(loc = 'upper left')
		ybox = VPacker(children=[ybox1, ybox2, ybox3],align="bottom", pad=0, sep=5)

		anchored_ybox = AnchoredOffsetbox(loc=10, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.15, 0.49), 
										  bbox_transform=ax0.transAxes, borderpad=0.)

		ax0.add_artist(anchored_ybox)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/z2_free_energy_entropy.pdf', dpi = 300)
		plt.show()
		plt.close()
		
		dU1 = U1 - U1[ref]
		dU2 = U2 - U2[ref]
		dG1 = G11 - G11[ref]
		dG2 = G22 - G22[ref]
		dS1 = (U1 - U1[ref] - G11 + G11[ref])
		dS2 = (U2 - U2[ref] - G22 + G22[ref])

	return z1_bins, z2_bins, dG1, dG2, dU1, dU2, dS1, dS2