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
XML string handling header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/



#ifndef XML_CL_H
#define XML_CL_H

#include <stdio.h>    // core input and output functions
#include <stdlib.h>   // standard general utilities library
#include <string.h>   // string handling functions

#include "../cross-level/string-cl.h"

#ifdef __cplusplus
extern "C" {
#endif

void get_xml_int_value(const char *line, int *value);
void get_xml_float_value(const char *line, float *value);
void get_xml_double_value(const char *line, double *value);
void get_xml_string_value(const char *line, char *value, size_t size);
void get_xml_int_values(const char *line, int **values, int *num_values);
void get_xml_float_values(const char *line, float **values, int *num_values);
void get_xml_double_values(const char *line, double **values, int *num_values);
void get_xml_string_values(const char *line, char ***values, int *num_values);
void split_xml_line(const char *line, string_t *open_tag, string_t *value, string_t *close_tag);
void get_xml_attribute_int_value(const char *tag, const char *attribute_name, int *attribute_value);
void get_xml_attribute_float_value(const char *tag, const char *attribute_name, float *attribute_value);
void get_xml_attribute_double_value(const char *tag, const char *attribute_name, double *attribute_value);
void get_xml_attribute_string_value(const char *tag, const char *attribute_name, char *attribute_value, size_t size);
void get_xml_attribute_int_values(const char *tag, const char *attribute_name, int **attribute_values, int *num_values);
void get_xml_attribute_float_values(const char *tag, const char *attribute_name, float **attribute_values, int *num_values);
void get_xml_attribute_double_values(const char *tag, const char *attribute_name, double **attribute_values, int *num_values);
void get_xml_attribute_string_values(const char *tag, const char *attribute_name, char ***attribute_values, int *num_values);
void split_xml_tag(const char *tag, string_t *tag_name, string_vector_t *attribute_name, string_vector_t *attribute_value);

#ifdef __cplusplus
}
#endif

#endif

