#!/usr/bin/env python
# coding: utf-8

# # Example notebook to extract shear profile from Abell 360

# In[24]:


# general python packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
import pandas as pd
import re


# In[46]:


from lsst.daf.butler import Butler
import lsst.geom as geom
import lsst.afw.geom as afwGeom
import clmm
from clmm import GalaxyCluster, ClusterEnsemble, GCData, Cosmology
from clmm import Cosmology, utils


# In[26]:


repo = '/repo/main'
#other collections can be tested, just an example here 
collection = 'LSSTComCam/runs/DRP/DP1/w_2025_08/DM-49029'
butler = Butler(repo, collections=collection)


# In[27]:


skymap = butler.get('skyMap', skymap='lsst_cells_v1')


# ## Find tracts and patches for Abell 360 and load the catalogs

# In[28]:


# Define BCG coordinates and search radius (in degrees)
ra_bcg, dec_bcg, delta = 37.862, 6.98, 0.5

# Define corner points of the square region around the BCG
radec = [
    geom.SpherePoint(ra_bcg + dra, dec_bcg + ddec, geom.degrees)
    for dra in (-delta, delta)
    for ddec in (-delta, delta)
]

# Query the sky map for overlapping tracts and patches
tracts_and_patches = skymap.findTractPatchList(radec)

#print(tracts_and_patches)

tp_dict = {}
for tract_info, patch_list in tracts_and_patches:
    tract_idx = tract_info.getId()
    patches = [patch.sequential_index for patch in patch_list]
    tp_dict[tract_idx] = patches


# In[29]:


# Get the object catalog of these patches
datasetType = 'objectTable'
init_cat = pd.DataFrame() #initial catalog

# Apply some selections to select only galaxies and good quality objects 
for tract in tp_dict.keys():
    print(f'Loading objects from tract {tract}, patches:{tp_dict[tract]}')
    for patch in tp_dict[tract]:
        dataId = {'tract': tract, 'patch' : patch ,'skymap':'lsst_cells_v1'}
        obj_cat = butler.get(datasetType, dataId=dataId) #objects after selection
        filt = obj_cat['detect_isPrimary']==True
        filt &= obj_cat['r_cModel_flag']== False
        filt &= obj_cat['g_cModel_flag']== False
        filt &= obj_cat['z_cModel_flag']== False
        filt &= obj_cat['i_cModel_flag']== False
        filt &= obj_cat['r_cModelFlux']>0
        filt &= obj_cat['i_cModelFlux']>0
        filt &= obj_cat['g_cModelFlux']>0
        filt &= obj_cat['z_cModelFlux']>0
        filt &= obj_cat['refExtendedness'] == True

        init_cat = pd.concat([init_cat, obj_cat[filt]], ignore_index=True) #catalog after selection


# ## Select a circular field close (<0.1 deg) to the BCG in order to identify Red Sequence

# In[30]:


c1 = SkyCoord(init_cat['coord_ra'].values*u.deg, init_cat['coord_dec'].values*u.deg)
c2 = SkyCoord(ra_bcg*u.deg, dec_bcg*u.deg)
#angular separation between BCG and the object 
sep = c1.separation(c2)

filt = sep.deg < 0.1 # select a region close to the cluster center for RS indentification
init_cat_rs = init_cat[filt] # catalog for RS identification

plt.scatter(init_cat['coord_ra'], init_cat['coord_dec'], marker='.', s=0.1, alpha=1)
plt.scatter(init_cat_rs['coord_ra'], init_cat_rs['coord_dec'], marker='.', s=0.1, alpha=1)
plt.ylabel('dec')
plt.xlabel('ra')


# In[31]:


#number of galaxies selected
len(init_cat), len(init_cat_rs)


# In[32]:


#Convert fluxes to magnitudes
mag_i = -2.50 * np.log10(init_cat_rs['i_cModelFlux']) + 31.4
mag_r = -2.50 * np.log10(init_cat_rs['r_cModelFlux']) + 31.4
mag_g = -2.50 * np.log10(init_cat_rs['g_cModelFlux']) + 31.4
mag_z = -2.50 * np.log10(init_cat_rs['z_cModelFlux']) + 31.4
raT=init_cat_rs['coord_ra']
decT=init_cat_rs['coord_dec']

#Plots all band differences to spot the red sequence 
fig, axis = plt.subplots(3, 2, figsize=(10, 10))
def customize_plot(ax, xdata, ydata, title, ylines=None):
    ax.scatter(xdata, ydata, marker='o', c=mag_i, cmap='plasma', s=0.1, alpha=1)
    ax.set_title(title)
    
    # 1. Add more ticks
    ax.set_xticks(np.linspace(np.min(xdata), np.max(xdata), 10))
    ax.set_yticks(np.linspace(np.min(ydata), np.max(ydata), 10))

    # 2. Add horizontal lines
    if ylines:
        for yval in ylines:
            ax.axhline(y=yval, color='red', linestyle='--', linewidth=0.8)

# Apply to each subplot
customize_plot(axis[0, 0], mag_r, mag_r - mag_i, "r - i vs r", ylines=[0.2, 0.8])
customize_plot(axis[1, 0], mag_r, mag_r - mag_z, "r - z vs r", ylines=[0.25, 1.3])
customize_plot(axis[2, 0], mag_r, mag_i - mag_z, "i - z vs r", ylines=[0, 0.7])
customize_plot(axis[0, 1], mag_r, mag_g - mag_z, "g - z vs r", ylines=[1.5, 2.7])
customize_plot(axis[1, 1], mag_r, mag_g - mag_i, "g - i vs r", ylines=[1.1, 2.3])
customize_plot(axis[2, 1], mag_r, mag_g - mag_r, "g - r vs r", ylines=[1, 1.8])
plt.tight_layout()


# ## Selecting the Red Sequence 

# In[33]:


# Building a mask by watching at regions where the RS is there in all bands differences 
maskp = (mag_r-mag_i>0.2) & (mag_r-mag_i<0.8) & (mag_g-mag_z>1.5) & (mag_g-mag_z<2.7) &  (mag_g-mag_r<1.8) & (mag_g-mag_r>1) &  (mag_r-mag_z<1.3) & (mag_r-mag_z>0.25) &  (mag_i-mag_z<0.7) & (mag_i-mag_z>0) &  (mag_g-mag_i<2.3) & (mag_g-mag_i>1.1) & (mag_r >= 18) & (mag_r <= 24)

#Show all galaxies and red-sequence selected galaxies in green-ish colors 
plt.scatter(mag_r, mag_r-mag_i, marker='o', c = mag_r, cmap='plasma', s=0.3, alpha=1)
plt.scatter(mag_r[maskp], mag_r[maskp]-mag_i[maskp], marker='o', c = mag_r[maskp], cmap='viridis', s=0.3, alpha=1)
plt.xlabel('r')
plt.ylabel('r-i')
plt.colorbar()


# In[34]:


#show the distribution of selected red-sequence galaxies
plt.scatter(raT[maskp], decT[maskp], marker='o', s=0.5, alpha=1)
plt.xlabel('ra')
plt.ylabel('dec')

#Number of galaxies in the RS catalog (~in the A360 cluster)
print(f'RS catalog size: {np.sum(maskp)}')
#Mass-richness relationship (eg here: https://arxiv.org/pdf/2210.09530) would give a mass of >1e15 solar mass for 557 galaxy members. 


# ## Building a catalog where red-sequence galaxies are removed 

# In[35]:


# Get object IDs of RS galaxies
RS_id_list = mag_r.index[maskp]  # Use the index to avoid misalignment

# Spatial filter of 0.5 to build the weak lensing catalog
cat_wl = init_cat[sep.deg < 0.5]

# Remove RS galaxies from the spatially filtered catalog
cat_wl = cat_wl[~cat_wl.index.isin(RS_id_list)]

#plotting original catalgo and the one used for weak lensing studies 
plt.scatter(init_cat['coord_ra'], init_cat['coord_dec'], marker='.', s=0.1, alpha=1)
plt.scatter(cat_wl['coord_ra'], cat_wl['coord_dec'], marker='.', s=0.1, alpha=1)
plt.ylabel('dec')
plt.xlabel('ra')


# In[36]:


len(cat_wl)


# ## Filter to keep only sources with well-measured shapes in i band (to extract ellipticities) 

# In[37]:


#From Celine's notebook 
# Filters to keep sources with good-quality measured shape in i band
source_filt = np.isfinite(cat_wl['i_hsmShapeRegauss_e1'])
source_filt &= np.isfinite(cat_wl['i_hsmShapeRegauss_e2'])
source_filt &= np.sqrt(cat_wl['i_hsmShapeRegauss_e1']**2 + cat_wl['i_hsmShapeRegauss_e2']**2) < 4
source_filt &= cat_wl['i_hsmShapeRegauss_sigma']<= 0.4 #remove quite a lot of sources 
source_filt &= cat_wl['i_hsmShapeRegauss_flag'] == 0
source_filt &= cat_wl['i_blendedness'] < 0.42
source_filt &= cat_wl['i_iPSF_flag']==0

# Resolution factor quality cut - according to Mandelbaum (2018) paper:
# "we use the resolution factor R2 which is defined using the trace of the moment matrix of the PSF TP and 
# of the observed (PSF-convolved) galaxy image TI as: R2 = 1- TP/TI"
# Best guess to translate that in terms of ComCam objectTable catalog output...
# Needs to be double checked
res = 1- (cat_wl['i_ixxPSF']+ cat_wl['i_iyyPSF']) / (cat_wl['i_ixx']+ cat_wl['i_iyy'])

plt.hist(res, bins=100, alpha=0.5, label='resolution')
plt.yscale("log") 

source_filt &= res >= 0.3

# Remove  brightest objects that are likely foreground objects
mag_i = -2.50 * np.log10(cat_wl['i_cModelFlux']) + 31.4
source_filt &= mag_i > 20. 

print(f'Source sample size: {np.sum(source_filt)}')


# In[38]:


# ra,dec distribution of selected galaxies 
plt.scatter(cat_wl['coord_ra'], cat_wl['coord_dec'], marker='o', s=0.1, alpha=1)
plt.scatter(cat_wl['coord_ra'][source_filt], cat_wl['coord_dec'][source_filt], marker='o', s=0.1, alpha=1)
plt.ylabel('dec')
plt.xlabel('ra')


# In[49]:


#Showing orientation and size of ellipticities of galaxies 
ra = cat_wl['coord_ra'][source_filt]
dec = cat_wl['coord_dec'][source_filt]
e1 = cat_wl['i_hsmShapeRegauss_e1'][source_filt]
e2 = cat_wl['i_hsmShapeRegauss_e2'][source_filt]
e_err = cat_wl['i_hsmShapeRegauss_sigma'][source_filt]

e_mag = np.sqrt(e1**2 + e2**2)*0.01 #magnitude of ellipticity
theta = 0.5 * np.arctan2(e2, e1)  # orientation angle

dx = e_mag * np.cos(theta)
dy = e_mag * np.sin(theta)

plt.figure(figsize=(6, 6))
plt.quiver(ra, dec, dx, dy, angles='xy', scale=1, width=0.002)
plt.title("Galaxy Ellipticity Vectors")
plt.xlabel("ra")
plt.ylabel("dec")
plt.axis('equal')
plt.show()


# ## Setting up HSC-Y1 shear calibration

# In[41]:


#Remove unecessary columns in the catalog
# Define pattern (example: drop all columns starting with "temp_")
pattern = r"^g_|^z_"  
# Drop columns matching the pattern
cat_wl = cat_wl.drop(columns=[col for col in cat_wl.columns if re.match(pattern, col)])

astropy_table = Table.from_pandas(cat_wl[source_filt])
astropy_table.write('source_sample.fits', format="fits", overwrite=True)


# Now, we have the source_sample.fits file created. To get the calibrated file, need to
# 
# 1) open a terminal in the interface and clone the following gitlab: https://github.com/PrincetonUniversity/hsc-y1-shear-calib
# 2) modify the utilities.py file to change the name of columns to the ones of LSST DP1 in the following functions : get_snr, get_res, get_psf_ellip (typically: iflux_cmodel -> i_cModelFlux ; iflux_cmodel_err -> i_cModelFluxErr ; ishape_hsm_regauss_resolution -> i_hsmShapeRegauss_sigma ; ishape_sdss_psf_ixx -> i_ixxPSF)
# 3) move the source_sample.fits file in hsc-y1-shear-calib directory
# 4) run the following command: python gen_hsc_calibrations.py source_sample.fits source_sample_calib.fits (or other input_file.fits output_file.fits)
# 5) This will create the file source_sample_calib.fits, move it to the place where you have the notebook

# ## Get the calibration quantities

# In[42]:


with fits.open('source_sample_calib.fits') as hdul:
    # Assuming data is in the first HDU (if not, change the index as needed)
    data = hdul[1].data

    # Convert the FITS data to an Astropy Table
    table = Table(data)

e_rms = table["ishape_hsm_regauss_derived_rms_e"]
m = table["ishape_hsm_regauss_derived_shear_bias_m"]
c1 = table["ishape_hsm_regauss_derived_shear_bias_c1"]
c2 = table["ishape_hsm_regauss_derived_shear_bias_c2"]
weight = table["ishape_hsm_regauss_derived_shape_weight"]

to_use = np.isfinite(weight)*np.isfinite(e_rms)*np.isfinite(m)*np.isfinite(c1)*np.isfinite(c2)

e1_0 = e1[to_use]
e2_0 = e2[to_use]
e_rms = e_rms[to_use]
c1 = c1[to_use]
c2 = c2[to_use]
m = m[to_use]
weight = weight[to_use]

print(f'Number of sources with calibration: {np.sum(to_use)}')


# ## Apply the calibration

# In[43]:


# From Shenming's CLMM demo on using HSC data
def apply_shear_calibration(e1_0, e2_0, e_rms, m, c1, c2, weight):
    R = 1.0 - np.sum(weight * e_rms**2.0) / np.sum(weight)
    m_mean = np.sum(weight * m) / np.sum(weight)
    c1_mean = np.sum(weight * c1) / np.sum(weight)
    c2_mean = np.sum(weight * c2) / np.sum(weight)
    print("R, m_mean, c1_mean, c2_mean: ", R, m_mean, c1_mean, c2_mean)

    g1 = (e1_0 / (2.0 * R) - c1) / (1.0 + m_mean)
    g2 = (e2_0 / (2.0 * R) - c2) / (1.0 + m_mean)

    return g1, g2


# In[44]:


g1, g2 = apply_shear_calibration(e1_0, e2_0, e_rms, m, c1, c2, weight)


# In[45]:


plt.hist(m, bins=100, alpha=0.2,label='m');
plt.legend()


# ## Use CLLM to extract shear profile

# In[47]:


cosmo = clmm.Cosmology(H0=70.0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045, Omega_k0=0.0)


# In[50]:


#prepare a CLMM GCData table using the catalog
galcat = GCData()
galcat['ra'] = ra[to_use]
galcat['dec'] = dec[to_use]
# galcat['e1'] = e1[to_use]
# galcat['e2'] = e2[to_use]
galcat['e1'] = g1
galcat['e2'] = g2
galcat['e_err'] = e_err[to_use]/2.  # factor 2 to account for conversion between e and g

galcat['z'] = np.zeros(len(ra[to_use])) # CLMM needs a redshift column for the source, even if not used


# In[51]:


#Create the corresponding CLMM galaxy cluster object
cluster_id = "Abell 360"
gc_object1 = clmm.GalaxyCluster(cluster_id, ra_bcg, dec_bcg, 0.22, galcat, coordinate_system='euclidean')
gc_object1.compute_tangential_and_cross_components(add=True);


# In[52]:


#Compute the lensing weights using CLMM -  to be checked...
gc_object1.compute_galaxy_weights(
        shape_component1="e1",
        shape_component2="e2",
        use_shape_error=True,
        shape_component1_err="e_err",
        shape_component2_err="e_err",
        use_shape_noise=True,
        weight_name="w_ls",
        cosmo=cosmo,
        add=True,
    ) 


# In[53]:


# Radial binning, either in Mpc or degrees
bins_mpc = clmm.make_bins(0.7,5,nbins=5, method='evenlog10width')
bins_deg = clmm.make_bins(0.1,0.5,nbins=5, method='evenlog10width')


# In[73]:


#Radial profile computation
gc_object1.make_radial_profile(bins=bins_mpc, bin_units='Mpc', add=True, cosmo=cosmo, overwrite=True, use_weights=True);

## Alternatively, angular radial binning (no need for a cosmology then)
#gc_object1.make_radial_profile(bins=bins_deg, bin_units='degrees', add=True, overwrite=True, use_weights=True);


# In[74]:


# Check the profile table
gc_object1.profile


# In[79]:


#Also use CLMM to get a typical model for a cluster at that redshift, assuming the DESC SRD n(z)
moo = clmm.Modeling(massdef="mean", delta_mdef=200, halo_profile_model="nfw")

moo.set_cosmo(cosmo)
moo.set_concentration(4)
moo.set_mass(1.0e15)
#moo.set_mass(4.0e14)

z_cl = gc_object1.z

# source properties
# assume sources redshift following a the DESC Science Roadmap Document distribution. This will need updating.

z_distrib_func = utils.redshift_distributions.desc_srd  

# Compute first beta (e.g. eq(6) of WtGIII paper)
beta_kwargs = {
    "z_cl": z_cl,
    "z_inf": 10.0,
    "cosmo": cosmo,
    "z_distrib_func": z_distrib_func,
}
beta_s_mean = utils.compute_beta_s_mean_from_distribution(**beta_kwargs)
beta_s_square_mean = utils.compute_beta_s_square_mean_from_distribution(**beta_kwargs)

rproj = np.logspace(np.log10(0.3),np.log10(7.), 100)

gt_z = moo.eval_reduced_tangential_shear(
    rproj, z_cl, [beta_s_mean, beta_s_square_mean], z_src_info="beta", approx="order2"
)


# In[81]:


#Also use CLMM to get a typical model for a cluster at that redshift, assuming the DESC SRD n(z)
moo2 = clmm.Modeling(massdef="mean", delta_mdef=200, halo_profile_model="nfw")

moo2.set_cosmo(cosmo)
moo2.set_concentration(4)
moo2.set_mass(4.0e14)

z_cl = gc_object1.z
z_distrib_func = utils.redshift_distributions.desc_srd  

# Compute first beta (e.g. eq(6) of WtGIII paper)
beta_kwargs = {
    "z_cl": z_cl,
    "z_inf": 10.0,
    "cosmo": cosmo,
    "z_distrib_func": z_distrib_func,
}
beta_s_mean = utils.compute_beta_s_mean_from_distribution(**beta_kwargs)
beta_s_square_mean = utils.compute_beta_s_square_mean_from_distribution(**beta_kwargs)

rproj = np.logspace(np.log10(0.3),np.log10(7.), 100)

gt_z2 = moo2.eval_reduced_tangential_shear(
    rproj, z_cl, [beta_s_mean, beta_s_square_mean], z_src_info="beta", approx="order2"
)


# In[82]:


plt.errorbar(gc_object1.profile['radius'], gc_object1.profile['gt'], gc_object1.profile['gt_err'], 
             ls='', marker='.', label='tangential')
plt.errorbar(gc_object1.profile['radius']*1.02, gc_object1.profile['gx'], gc_object1.profile['gx_err'], 
             ls='', marker='.', label='cross')
plt.plot(rproj, gt_z, label='NFW (model, not fit), M200m=1e15 Msun, c=4, n(z)=SRD', ls=':')
plt.plot(rproj, gt_z2, label='NFW (model, not fit), M200m=4e14 Msun, c=4, n(z)=SRD', ls=':')


plt.xscale('log')
plt.axhline(0.0, color='k', ls=':')
plt.ylim([-0.03,0.08])
plt.xlim([0.7,7])
#plt.yscale('log')
plt.xlabel('R [Mpc]')
plt.ylabel('reduced shear')
plt.legend(loc=1)


# In[ ]:




