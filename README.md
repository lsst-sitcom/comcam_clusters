# comcam_clusters
Notebooks for the analysis of galaxy clusters in ComCam data

`ComCam_StarterKit` - Tutorial notebook on loading ComCam data with Butler. Loads the relevant coadds and catalogs for Abell 360, displays a `gri` image along with a red-sequence plot. 

`ACO360_WL_HSCcalib_CLMM` - First attempt at getting a shear profile around the Abell 360 cluster from ComCam data, using color cut source selection, HSC calibration, and CLMM for the shear profile measurement. Main steps are there but a lot remain to be checked/improved.

`ACO360_WL_HSCcalib_MassMap` - Computing a mass_map around Abell 360 from ComCam data. Borrows from calibration/selection of `ACO360_WL_HSCcalib_CLMM`

`ACO360_PSF_properties` - Check the PSF correction in Abell 360 field. Whisker plots, etc.

`ACO360_rgb_and_WLsources` - Make RGB images of all the patches in the field of Abell 360, with sources overlaid.
