/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2022 David Frantz

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
This file contains functions for handling CRS
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "crs-cl.h"

#include "gdal.h"           // public (C callable) GDAL entry points
#include "cpl_conv.h"
#include "ogr_srs_api.h"


int epsg_to_wkt(int epsg_code, char *wkt_output){


  OGRSpatialReferenceH srs = OSRNewSpatialReference(NULL);

  if (OSRImportFromEPSG(srs, epsg_code) == OGRERR_NONE){
    
    char *wkt = NULL;

    // 3. Export to WKT string
    OSRExportToWkt(srs, &wkt);

    #ifdef FORCE_DEBUG
    printf("EPSG conversion from EPSG:%d to WKT:\n", epsg_code);
    printf("%s\n", wkt);
    #endif
    
    copy_string(wkt_output, NPOW_10, wkt);

    CPLFree(wkt);

  } else {
    OSRDestroySpatialReference(srs);
    RETURN_ERROR("Could not convert EPSG:%d to WKT.", epsg_code);
  }

  OSRDestroySpatialReference(srs);

  return SUCCESS;
}
