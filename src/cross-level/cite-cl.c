/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2020 David Frantz

FORCE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FORCE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FORCE.  If not, see <http://www.gnu.org/licenses/>.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions for citations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "cite-cl.h"


FILE *_cite_fp_ = NULL;

cite_t _cite_me_[_CITE_LENGTH_] = {
  { "FORCE - the overall framework", 
    "Frantz, D. (2019). FORCE—Landsat + Sentinel-2 Analysis Ready Data "
    "and Beyond. Remote Sensing, 11, 1124", 
    false },
  { "Level 2 Processing - general concept, radiometric  correction, "
    "cloud masking, cubing", 
    "Frantz, D., Röder, A., Stellmes, M., & Hill, J. (2016). An "
    "Operational Radiometric Landsat Preprocessing Framework for "
    "Large-Area Time Series Applications. IEEE Transactions on "
    "Geoscience and Remote Sensing, 54, 3928-3943", 
    false },
  { "Atmospheric correction validation and intercomparison", 
    "Doxani, G., Vermote, E., Roger, J.-C., Gascon, F., Adriaensen, S., "
    "Frantz, D., Hagolle, O., Hollstein, A., Kirches, G., Li, F., "
    "Louis, J., Mangin, A., Pahlevan, N., Pflug, B., & Vanhellemont, Q. "
    "(2018). Atmospheric Correction Inter-Comparison Exercise. Remote "
    "Sensing, 10, 352", 
    false },
  { "Fallback to AOD climatology", 
    "Frantz, D., Röder, A., Stellmes, M., & Hill, J. (2015). On the "
    "Derivation of a Spatially Distributed Aerosol Climatology for its "
    "Incorporation in a Radiometric Landsat Preprocessing Framework. "
    "Remote Sensing Letters, 6, 647-656", 
    false },
  { "Cloud detection", 
    "Frantz, D., Röder, A., Udelhoven, T., & Schmidt, M. (2015). "
    "Enhancing the Detectability of Clouds and Their Shadows in "
    "Multitemporal Dryland Landsat Imagery: Extending Fmask. IEEE "
    "Geoscience and Remote Sensing Letters, 12, 1\n"
    "Frantz, D., Haß, E., Uhl, A., Stoffels, J., & Hill, J. (2018). "
    "Improvement of the Fmask algorithm for Sentinel-2 images: "
    "Separating clouds from bright surfaces based on parallax effects. "
    "Remote Sensing of Environment, 215, 471-481\n"
    "Zhu, Z., & Woodcock, C.E. (2012). Object-Based Cloud and Cloud "
    "Shadow Detection in Landsat Imagery. Remote Sensing of "
    "Environment, 118, 83-94\n"
    "Zhu, Z., Wang, S., & Woodcock, C.E. (2015). Improvement and "
    "Expansion of the Fmask Algorithm: Cloud, Cloud Shadow, and Snow "
    "Detection for Landsats 4–7, 8, and Sentinel 2 Images. Remote "
    "Sensing of Environment, 159, 269-277\n"
    "Baetens, L., Desjardins, C., & Hagolle, O. (2019). Validation of "
    "Copernicus Sentinel-2 Cloud Masks Obtained from MAJA, Sen2Cor, and "
    "FMask Processors Using Reference Cloud Masks Generated with a "
    "Supervised Active Learning Procedure. Remote Sensing, 11, 433", 
    false },
  { "Improving the spatial resolution of coarser data using ImproPhe", 
    "Frantz, D., Stellmes, M., Röder, A., Udelhoven, T., Mader, S., & "
    "Hill, J. (2016). Improving the Spatial Resolution of Land Surface "
    "Phenology by Fusing Medium- and Coarse-Resolution Inputs. IEEE "
    "Transactions on Geoscience and Remote Sensing, 54, 4153-4164", 
    false },
  { "Improving the spatial resolution of coarser data using STARFM", 
    "Gao, F., Masek, J., Schwaller, M., & Hall, F. (2006). On the "
    "Blending of the Landsat and MODIS Surface Reflectance: "
    "Predicting Daily Landsat Surface Reflectance. IEEE Transactions "
    "on Geoscience and Remote Sensing, 44, 2207-2218", 
    false },
  { "Improving the spatial resolution of coarser data using regression", 
    "Hill, J., Diemer, C., Stöver, T., & Udelhoven, T. (1999). A Local "
    "Correlation approach for the fusion of image bands with different "
    "spatial resolutions. International Archives of Photogrammetry and "
    "Remote Sensing, 32", 
    false },
  { "Pixel-based compositing using parametric weighting", 
    "Frantz, D., Röder, A., Stellmes, M., & Hill, J. (2017). Phenology-"
    "adaptive pixel-based compositing using optical earth observation "
    "imagery. Remote Sensing of Environment, 190, 331-34\n"
    "Griffiths, P., van der Linden, S., Kuemmerle, T., & Hostert, P. "
    "(2013). A Pixel-Based Landsat Compositing Algorithm for Large Area "
    "Land Cover Mapping. IEEE Journal of Selected Topics in Applied Earth "
    "Observations and Remote Sensing, 6, 2088-2101\n"
    "White, J.C., Wulder, M.A., Hobart, G.W., Luther, J.E., Hermosilla, "
    "T., Griffiths, P., Coops, N.C., Hall, R.J., Hostert, P., Dyk, A., "
    "& Guindon, L. (2014). Pixel-Based Image Compositing for Large-Area "
    "Dense Time Series Applications and Science. Canadian Journal of "
    "Remote Sensing, 40, 192-",    
    false },
  { "Water vapor database for atmospheric correctiion of Landsat", 
    "Frantz, D., Stellmes, M., & Hostert, P. (2019). A Global MODIS "
    "Water Vapor Database for the Operational Atmospheric Correction of "
    "Historic and Recent Landsat Imagery. Remote Sensing, 11, 257\n"
    "Frantz, D., Stellmes, M., Ernst, S. (2021). Water vapor database for "
    "atmospheric correction of Landsat imagery. Zenodo", 
    false },
  { "BRDF correction", 
    "Wanner, W., Li, X., & Strahler, A.H. (1995). On the derivation of "
    "kernels for kernel-driven models of bidirectional reflectance. "
    "Journal of Geophysical Research: Atmospheres, 100, 21077-21089"
    "Wanner, W., Strahler, A.H., Hu, B., Lewis, P., Muller, J.-P., Li, X., "
    "Schaaf, C.L.B., & Barnsley, M.J. (1997). Global retrieval of bidirect"
    "ional reflectance and albedo over land from EOS MODIS and MISR data: "
    "Theory and algorithm. Journal of Geophysical Research: Atmospheres, "
    "102, 17143-17161\n"
    "Roy, D.P., Zhang, H.K., Ju, J., Gomez-Dans, J.L., Lewis, P.E., Schaaf, "
    "C.B., Sun, Q., Li, J., Huang, H., & Kovalskyy, V. (2016). A General "
    "Method to Normalize Landsat Reflectance Data to Nadir BRDF Adjusted "
    "Reflectance. Remote Sensing of Environment, 176, 255-271\n"
    "Roy, D.P., Li, J., Zhang, H.K., Yan, L., Huang, H., & Li, Z. (2017). "
    "Examination of Sentinel-2A multi-spectral instrument (MSI) reflectance "
    "anisotropy and the suitability of a general method to normalize MSI "
    "reflectance to nadir BRDF adjusted reflectance. Remote Sensing of "
    "Environment, 199, 25-38\n"
    "Roy, D., Li, Z., & Zhang, H. (2017). Adjustment of Sentinel-2 Multi-"
    "Spectral Instrument (MSI) Red-Edge Band Reflectance to Nadir BRDF "
    "Adjusted Reflectance (NBAR) and Quantification of Red-Edge Band "
    "BRDF Effects. Remote Sensing, 9, 1325\n"
    "Zhang, H. K., Roy, D. P., & Kovalskyy, V. (2015). Optimal solar geometry "
    "definition for global long-term Landsat time-series bidirectional "
    "reflectance normalization. IEEE Transactions on Geoscience and Remote Sensing, "
    "54(3), 1410-1418\n"
    "Li, Z., Zhang, H. K., & Roy, D. P. (2019). Investigation of Sentinel-2 "
    "bidirectional reflectance hot-spot sensing conditions. IEEE Transactions "
    "on Geoscience and Remote Sensing, 57(6), 3591-3598",
    false },
  { "Co-Registration Sentinel-2 -> Landsat", 
    "Yan, L., Roy, D.P., Zhang, H., Li, J., & Huang, H. (2016). An "
    "Automated Approach for Sub-Pixel Registration of Landsat-8 Operational "
    "Land Imager (OLI) and Sentinel-2 Multi Spectral Instrument (MSI) "
    "Imagery. Remote Sensing, 8, 520\n"
    "Rufin, P., Frantz, D., Yan, L. & Hostert, P. Operational co-regis-"
    "tration of the Sentinel-2A/B image archive using multi-temporal "
    "Landsat spectral averages. In submission",
    false },
  { "Radiative transfer", 
    "Tanré, D., Herman, M., Deschamps, P.Y., & de Leffe, A. (1979). "
    "Atmospheric Modeling for Space Measurements of Ground Reflectances, "
    "Including Bidirectional Properties. Applied Optics, 18, 3587-3594\n"
    "Tanré, D., Deroo, C., Duhaut, P., Herman, M., Morcrette, J.J., "
    "Perbos, J., & Deschamps, P.Y. (1990). Description of a Computer Code "
    "to Simulate the Satellite Signal in the Solar Spectrum: The 5S Code. "
    "International Journal of Remote Sensing, 11, 659-668",
    false },
  { "Adjacency effect correction", 
    "Tanré, D., Herman, M., & Deschamps, P.Y. (1981). Influence of the "
    "Background Contribution upon Space Measurements of Ground "
    "Reflectance. Applied Optics, 20, 3676-3684\n"
    "Bach, H. (1995). Die Bestimmung hydrologischer und landwirtschaft"
    "licher Oberflächenparameter aus hyperspektralen Fernerkundungsdaten. "
    "Munich, Germany: Geobuch-Verlag",
    false },
  { "AOD estimation", 
    "Royer, A., Charbonneau, L., & Teillet, P.M. (1988). Interannual "
    "Landsat-MSS Reflectance Variation in an Urbanized Temperate Zone. "
    "Remote Sensing of Environment, 24, 423-446\n"
    "Kaufman, Y.J., & Sendra, C. (1988). Algorithm for Automatic Atmo"
    "spheric Corrections to Visible and Near-IR Satellite Imagery. Inter"
    "national Journal of Remote Sensing, 9, 1357-1381",
    false },
  { "Radiometric correction", 
    "Hill, J., & Sturm, B. (1991). Radiometric Correction of Multitemporal "
    "Thematic Mapper Data for Use in Agricultural Land-Cover Classification "
    "and Vegetation Monitoring. International Journal of Remote Sensing, "
    "12, 1471-1491\n"
    "Hill, J. (1993). High Precision Land Cover Mapping and Inventory with "
    "Multi-Temporal Earth Observation Satellite Data: the Ardèche "
    "Experiment. In, Faculty of Geography/Geosciences (p. 121). Trier, "
    "Germany: Trier University",
    false },
  { "Topographic correction", 
    "Kobayashi, S., & Sanga-Ngoie, K. (2008). The Integrated Radiometric "
    "Correction of Optical Remote Sensing Imageries. International Journal "
    "of Remote Sensing, 29, 5957-5985",
    false },
  { "Clear Sky Observations", 
    "Ernst, S., Lymburner, L., & Sixsmith, J. (2018). Implications of "
    "Pixel Quality Flags on the Observation Density of a Continental "
    "Landsat Archive. Remote Sensing, 10, 1570",
    false },
  { "Spectral Temporal Metrics", 
    "Müller, H., Rufin, P., Griffiths, P., Barros Siqueira, A.J., & "
    "Hostert, P. (2015). Mining dense Landsat time series for separating "
    "cropland and pasture in a heterogeneous Brazilian savanna landscape. "
    "Remote Sensing of Environment, 156, 490-499",
    false },
  { "RBF interpolation", 
    "Schwieder, M., Leitão, P.J., da Cunha Bustamante, M.M., Ferreira, "
    "L.G., Rabe, A., & Hostert, P. (2016). Mapping Brazilian savanna "
    "vegetation gradients with Landsat time series. International Journal "
    "of Applied Earth Observation and Geoinformation, 52, 361-370",
    false },
  { "SPLITS", 
    "Mader, S. (2012). A Framework for the Phenological Analysis of Hyper"
    "temporal Remote Sensing Data Based on Polynomial Spline Models. In, "
    "Faculty of Geography/Geosciences (p. 101). Trier, Germany: "
    "Trier University",
    false },
  { "CAT transformation", 
    "Hird, J.N., Castilla, G., McDermid, G.J., & Bueno, I.T. (2016). A "
    "Simple Transformation for Visualizing Non-seasonal Landscape Change "
    "From Dense Time Series of Satellite Data. IEEE Journal of Selected "
    "Topics in Applied Earth Observations and Remote Sensing, 9, 3372-3383"
    "",
    false },
  { "NDVI", 
    "Tucker, C.J. (1979). Red and Photographic Infrared Linear "
    "Combinations for Monitoring Vegetation. Remote Sensing of "
    "Environment, 8, 127-150",
    false },
  { "EVI", 
    "Huete, A., Didan, K., Miura, T., Rodriguez, E.P., Gao, X., & "
    "Ferreira, L.G. (2002). Overview of the Radiometric and Biophysical "
    "Performance of the MODIS Vegetation Indices. Remote Sensing of "
    "Environment, 83, 195-213",
    false },
  { "NBR", 
    "Key, C. H., and N. C. Benson. Landscape assessment: ground measure "
    "of severity, the Composite Burn Index; and remote sensing of sever"
    "ity, the Normalized Burn Ratio. FIREMON: Fire effects monitoring and "
    "inventory system 2004 (2005).",
    false },
  { "SAVI, SARVI, ARVI", 
    "Huete, A. (1988). Soil-Adjusted Vegetation Index (SAVI). Remote Sensing "
    "of Environment, 25, 295-309\n"
    "Kaufman, Y., Tanré, D. (1992). Atmospherically Resistant Vegetation Index "
    "(ARVI) for EOS-MODIS. IEEE Transactions On Geoscience And Remote Sensing, "
    "30 (2), 261-270",
    false },
  { "Tasseled Cap", 
    "Crist, E.P. (1985). A TM Tasseled Cap equivalent transformation for "
    "reflectance factor data. Remote Sensing of Environment, 17, 301-306",
    false },
  { "DI", 
    "Healey, S.P., Cohen, W.B., Zhiqiang, Y., & Krankina, O.N. (2005). "
    "Comparison of Tasseled Cap-based Landsat data structures for use in "
    "forest disturbance detection. Remote Sensing of Environment, "
    "97, 301-310",
    false },
  { "NDBI", 
    "Zha, Y., Gao, J., & Ni, S. (2003). Use of normalized difference "
    "built-up index in automatically mapping urban areas from TM imagery. "
    "International Journal of Remote Sensing, 24, 583-594",
    false },
  { "NDWI", 
    "McFeeters, S.K. (1996). The use of the Normalized Difference Water "
    "Index (NDWI) in the delineation of open water features. International "
    "Journal of Remote Sensing, 17, 1425-1432",
    false },
  { "MNDWI", 
    "Xu, H. (2006). Modification of normalised difference water index "
    "(NDWI) to enhance open water features in remotely sensed imagery. "
    "International Journal of Remote Sensing, 27, 3025-3033",
    false },
  { "NDSI", 
    "Hall, D.K., Riggs, G.A., & Salomonson, V.V. (1995). Development of "
    "Methods for Mapping Global Snow Cover using Moderate Resolution "
    "Imaging Spectroradiometer Data. Remote Sensing of Environment, "
    "54, 127-140",
    false },
  { "SMA", 
    "Smith, M.O., Ustin, S.L., Adams, J.B., & Gillespie, A.R. (1990). "
    "Vegetation in deserts: I. A regional measure of abundance from "
    "multispectral images. Remote Sensing of Environment, 31, 1-26",
    false },
  { "EQUI7 grid and projection",
    "Bauer-Marschallinger, B., Sabel, D., & Wagner, W. (2014). Optimi"
    "sation of global grids for high-resolution remote sensing data. "
    "Computers & Geosciences, 72, 84-93",
    false },
  { "Resolution merge S2 bands",
    "Hass et al. TBD",
    false },
  { "Landscape Metrics",
    "McGarigal K.; Cushman, S.; Ene, E. 2012. FRAGSTATS v4: Spatial Pattern "
    "Analysis Program for Categorical and Continuous Maps. Computer software "
    "program produced by the authors at the University of Massachusetts, Amherst. "
    "Available at: http://www.umass.edu/landeco/research/fragstats/fragstats.html",
    false },
  { "NDTI",
    "Van Deventer, A.P., Ward, A.D., Gowda, P.H., Lyon, J.G. (1997). Using "
    "Thematic Mapper data to identify contrasting soil plains and tillage "
    "practices. Photogrammetric engineering and remote sensing, 63, 87-93",
    false },
  { "NDMI",
    "Gao, B. (1996). NDWI — A normalized difference water index for remote "
    "sensing of vegetation liquid water from space. Remote Sensing of Environment, "
    "58, 3, 257-266",
    false },
  { "Polar metrics",
    "Brooks, B., Lee, D., Pomara, L., Hargrove, W. (2020). Monitoring Broadscale "
    "Vegetational Diversity and Change across North American Landscapes Using Land "
    "Surface Phenology. Forests 11(6), 606",
    false },
  { "kNDVI",
    "Camps-Valls, G., Campos-Taberner, M., Moreno-Martínez, Á., Walther, S., "
    "Duveiller, G., Cescatti, A., Mahecha, M.D., Muñoz-Marí, J., García-Haro, F.J., "
    "Guanter, L., Jung, M., Gamon, J.A., Reichstein, M., & Running, S.W. (2021). "
    "A unified vegetation index for quantifying the terrestrial biosphere. "
    "Science Advances, 7, eabc7447.",
    false }
};


/** This function computes the CRC-8 of the cite suggestions
+++ Return: CRC-8
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
unsigned int cite_crc8(){
unsigned int crc = 0xff;
int i, k;


  for (i=0; i<_CITE_LENGTH_; i++){
    crc ^= _cite_me_[i].cited;
    for (k=0; k<8; k++) crc = crc & 1 ? (crc >> 1) ^ 0xb2 : crc >> 1;
  }

  return crc ^ 0xff;
}


/** This function adds a citation to the CITEME file (if it was not added
+++ before). 
--- i:      ID of citation, _CITE_XXX_ enums can be used
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void cite_me(int i){
  
  
  if (i < 0 || i >= _CITE_LENGTH_){
    printf("warning, index %d in cite_me is out of bounds.\n", i);
    return;
  }

  _cite_me_[i].cited = true;

  return;
  
}


/** This function writes the CITEME file
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void cite_push(char *dname){
unsigned int crc;
char fname[NPOW_10];
char *lock = NULL;
int nchar;
int i;


  crc = cite_crc8();

  nchar = snprintf(fname, NPOW_10, "%s/CITEME_%#02x.txt", dname, crc);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return;}
    
  if (fileexist(fname)) return;

  if ((lock = (char*)lock_file(fname, 60)) == NULL) return;

  if ((_cite_fp_ = fopen(fname, "w")) == NULL){
    printf("Unable to open CITEME file!\n"); 
    return;}

  fprintf(_cite_fp_, "FORCE - Framework for Operational Radiometric "
                     "Correction for Environmental monitoring\n");
  fprintf(_cite_fp_, "Version %s\n", _VERSION_);
  fprintf(_cite_fp_, "Copyright (C) 2013-2020 David Frantz, "
                     "david.frantz@geo.hu-berlin.de\n");

  fprintf(_cite_fp_, "\nFORCE is free software under the terms of the "
                     "GNU General Public License as published by the "
                     "Free Software Foundation, see "
                     "<http://www.gnu.org/licenses/>.\n");

  fprintf(_cite_fp_, "\nThank you for using FORCE! This software is "
                     "being developed in the hope that it will be "
                     "helpful for you and your work.\n");

  fprintf(_cite_fp_, "\nHowever, it is requested that you to use the "
                     "software in accordance with academic standards "
                     "and fair usage. Without this, software like FORCE "
                     "will not survive. This includes citation of the "
                     "software and the scientific publications, proper "
                     "acknowledgement in any public presentation, or an "
                     "offer of co-authorship of scientific articles in "
                     "case substantial help in setting up, modifying or "
                     "running the software is provided by the author(s).\n");

  fprintf(_cite_fp_, "\nHere are suggestions for references to be cited. "
                     "This list is based on your specific parameterization:\n");

  for (i=0; i<_CITE_LENGTH_; i++){

    if (_cite_me_[i].cited){
      fprintf(_cite_fp_, "\n%s:\n", _cite_me_[i].description);
      fprintf(_cite_fp_, "%s\n",    _cite_me_[i].reference);
    }

  }
    
  fclose(_cite_fp_);
  unlock_file(lock);
  
  return;
}

