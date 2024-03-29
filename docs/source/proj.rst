.. _proj:

.. |rarrow| unicode:: 8594

Projection and Tiling
=====================

The choice of the output projection is of key importance for generating ARD, but choosing an appropriate one is not trivial and often leads to confusion (especially with additional tiling). This chapter is intended to give some guidance and ideas on how to choose reasonable projection and tiling parameters.

First of all, in most cases, it is good practice to reproject the data to one shared coordinate system because the space agencies ship data in UTM projection with different zones. Thus, if the input data cover different UTM zones, the output data cannot be co-registered easily if reprojection is disabled. Note that tiling is still a valid option, but the results should be used with extreme care. Tiling primarily enables pixel-based operations, but these should not be used with a DO_REPROJ = FALSE / DO_TILE = TRUE parameterization (except for areas that are covered by one UTM zone only).

There exist many projections, and custom projections are also allowed (https://spatialreference.org can be used to find an existing one). We use the GDAL library for reprojection purposes. As such, virtually any projection expressed as Well-Known-Text is valid. However, the appropriate projection should be selected with care. In general, the projection should flatten your study area with minimal distortion. The choice depends on the location, size and extent of your study area, as well as on the desired specifications. Frequently, large area production systems use equal area projections with different projection surfaces for different study areas (e.g. Albers Conic |rarrow| CONUS or Lambert Azimuthal |rarrow| Pan-European). Snyder’s handbook on map projections  gives useful recommendations.

After reprojection, the data can be tiled to an arbitrary grid – similar to the MODIS Land products. Pixel-based operations can be easily used if this option is used. In fact, gridding is necessary for all higher-level FORCE operations (> Level 2). The grid can be freely defined – with consideration of the employed projection. We use square tiles (def. tile: grid cell) and the tile size is specified by the TILE_SIZE parameter, which must be given in output projection units (commonly in meters). TILE_SIZE must be a multiple of RESOLUTION.

The tile originates at ORIGIN_LON / ORIGIN_LAT, which must be given in geographic coordinates. These coordinates are transformed to the output projection (or input projection if DO_REPROJ = FALSE). The tile numbers increase from West to East and North to South. A new tile is generated each TILE_SIZE units. Tile X0000_Y0000, pixel 0/0 is located at ORIGIN_LON/ORIGIN_LAT. Negative tile numbers can occur if the tile origin is not North-West of your study area. Although negative tiles are generally not problematic, we recommend to use a tile origin that is sufficiently far away in the North-West of your study area and does not intersect with any input data (but close enough to be represented reasonably in the projected coordinate system).

Note that a geographic location in the North-West is not necessarily North-West in the output coordinate system, too (see Azimuthal example below). This may result in unexpected – yet valid – behavior. Note that coordinates may be undefined if they are too far away from the origin of the coordinate system (see Transverse Mercator example below); in this case the algorithm will fail. It is good practice to use a point of origin that is relatively close to the study area. The allowed distance varies greatly with different output projections.

.. seealso::
    :ref:`force-tabulate-grid <tabulate-grid>` for creating a shapefile with the grid (for visualization purposes) and :ref:`force-tile-finder <tile-finder>` for identifying the tile ID and pixel coordinate of any geographic coordinate.

Finally, the images are intersected with the grid, then defined as chips. The chips are extracted and saved as individual datasets. Empty tiles (e.g. black image boundary) will not be saved. Cloudy tiles can also be suppressed (MAX_CLOUD_COVER_TILE). If FILE_TILE is used, only the indicated tiles will be output.

Below are a few examples of commonly used projections. Fig. 13 depicts an Albers Equal Area projection, which is often used for the Continental Unites States (CONUS), due to the predominant East-West extent. Fig. 14 depicts a Lambert Azimuthal Equal Area projection, which is often used for areas with equal extent in all directions (e.g. Europe). Note that parallels are projected as circles, thus, a point that is very far away from the origin can have the same x-coordinate as a point in the middle of the coordinate system. This is amplified towards high latitude. Therefore, it is good practice to use a tile origin that is not too far away from the origin and preferably also well away from the poles (e.g. a tile origin at 90N/0E would result in non-intuitive behavior as it would be projected at 90N/10E). Fig. 15 depicts a UTM projection. Note that cylindrical projections are only valid for the area covered by the cylinder, as such, any tile origin outside of the colored area would be invalid and would result in a transformation error. In addition, the precision quickly degrades in x-direction, although it is very favorable for areas with high North-South extent.
 
Fig. 13. Albers Equal Area projection example
center: -96W/23N, standard parallels: 29.5N/45.5N, datum: NAD83, ellipsoid: GRS80.
 
Fig. 14. Lambert Azimuthal Equal Area projection example
center: 10W/52N, false easting/northing: 4321000/3210000, datum: ETRS89, ellipsoid: GRS80.
 
Fig. 15. Universal Transverse Mercator, zone 32 projection example
center: 9W/0N, false easting/northing: 500000/0, datum/ellipsoid: WGS84.

