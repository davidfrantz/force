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
Global definition of tables
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef TABLE_H
#define TABLE_H

#ifdef __cplusplus
extern "C" {
#endif

extern const float _E0_[1991];
extern const float _AW_[1991];
extern const float _AO_[1991];
extern const float _WVL_[1991];
extern const int   _WVL_DIM_;
extern const int _RSR_START_LND04_;
extern const int _RSR_START_LND05_;
extern const int _RSR_START_LND07_;
extern const int _RSR_START_LND08_;
extern const int _RSR_START_SEN2A_;
extern const int _RSR_START_SEN2B_;
extern const float _RSR_[56][1991];
extern const int   _RSR_DIM_;
extern const int   _AERO_WATERLIB_DIM_[2];
extern const float _AERO_WATERLIB_[26][491];
extern const int   _AERO_LANDLIB_DIM_[2];
extern const float _AERO_LANDLIB_[34][1991];

float wavelength(int b_rsr);
float E0(int b_rsr);

#ifdef __cplusplus
}
#endif

#endif

