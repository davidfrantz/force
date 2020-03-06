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
This file contains functions for quality assurance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "quality-cl.h"


/** This function sets any quality bit in the QAI layer
+++ Attention: this function implements no safety measures! 
+++ The stack short memory, bitfield, pixel, and index are all vulnerable
+++ to misuse. Take care.
--- qai:    Quality Assurance Information
--- index:  QAI layer
--- p:      pixel
--- val:    set to this value (typically 0 or 1, but can be another 
            integer, too, in which case, a wider bit field is changed
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_qai(stack_t *qai, int index, int p, short val){

  qai->vshort[0][p] |= (short)(val << index);
}


/** This function reads any quality bit in the QAI layer
+++ Attention: this function implements no safety measures! 
+++ The stack short memory, bitfield, pixel, and index are all vulnerable
+++ to misuse. Take care.
--- qai:       Quality Assurance Information
--- index:     QAI layer
--- p:         pixel
--- bitfields: how many bitfields to read? (typically 1 for binary bit, 
               but can be larger to retrieve multi-bit flags
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short get_qai(stack_t *qai, int index, int p, int bitfields){
int i;
short val = 0;

  for (i=0; i<bitfields; i++) val |= (short)(1 << i);

  return (short)(qai->vshort[0][p] >> index) & val;
}


/** read off/on flag **/
bool get_off(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_OFF_, p, 1);
}


/** read cloud flag **/
char get_cloud(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_CLD_, p, 2);
}


/** read cloud shadow flag **/
bool get_shadow(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_SHD_, p, 1);
}


/** read snow flag **/
bool get_snow(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_SNW_, p, 1);
}


/** read water flag **/
bool get_water(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_WTR_, p, 1);
}


/** read aerosol flag **/
char get_aerosol(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_AOD_, p, 2);
}


/** read subzero reflectance flag **/
bool get_subzero(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_SUB_, p, 1);
}


/** read saturated reflectance flag **/
bool get_saturation(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_SAT_, p, 1);
}


/** read low sun angle flag **/
bool get_lowsun(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_SUN_, p, 1);
}


/** read illumination flag **/
char get_illumination(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_ILL_, p, 2);
}


/** read slope flag **/
bool get_slope(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_SLP_, p, 1);
}


/** read water vapor fill flag **/
bool get_vaporfill(stack_t *qai, int p){

  return get_qai(qai, _QAI_BIT_WVP_, p, 1);
}


/** set off/on flag **/
void set_off(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_OFF_, p, val);
}


/** set cloud flag **/
void set_cloud(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_CLD_, p, val);
}


/** set cloud shadow flag **/
void set_shadow(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_SHD_, p, val);
}


/** set snow flag **/
void set_snow(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_SNW_, p, val);
}


/** set water flag **/
void set_water(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_WTR_, p, val);
}


/** set aerosol flag **/
void set_aerosol(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_AOD_, p, val);
}


/** set subzero reflectance flag **/
void set_subzero(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_SUB_, p, val);
}


/** set saturated reflectance flag **/
void set_saturation(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_SAT_, p, val);
}


/** set low sun angle flag **/
void set_lowsun(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_SUN_, p, val);
}


/** set illumination flag **/
void set_illumination(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_ILL_, p, val);
}


/** set slope flag **/
void set_slope(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_SLP_, p, val);
}


/** set water vapor fill flag **/
void set_vaporfill(stack_t *qai, int p, short val){

  set_qai(qai, _QAI_BIT_WVP_, p, val);
}

