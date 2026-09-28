/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2026 David Frantz

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
This file contains functions for xml string handling
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "xml-cl.h"



// meant to be used for parsing simple xml lines.
// does not cover all cases, but is sufficient for the metadata files used in FORCE
// is designed to fail if the xml line is not in the expected format, to avoid silent errors


/** Extract an integer value from an XML line
+++ This function extracts an integer value from an XML line
--- line:  xml-formatted line containing the value to be extracted
--- value: pointer to the variable where the extracted value will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_xml_int_value(const char *line, int *value){

  string_t xml_open  = {0};
  string_t xml_value = {0};
  string_t xml_close = {0};

  split_xml_line(line, &xml_open, &xml_value, &xml_close);

  char_to_int(xml_value.string, value);

  free_string(&xml_open);
  free_string(&xml_value);
  free_string(&xml_close);

  return;
}


/** Extract a float value from an XML line
+++ This function extracts a float value from an XML line
--- line:  xml-formatted line containing the value to be extracted
--- value: pointer to the variable where the extracted value will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_xml_float_value(const char *line, float *value){

  string_t xml_open  = {0};
  string_t xml_value = {0};
  string_t xml_close = {0};

  split_xml_line(line, &xml_open, &xml_value, &xml_close);

  char_to_float(xml_value.string, value);

  free_string(&xml_open);
  free_string(&xml_value);
  free_string(&xml_close);

  return;
}


/** Extract a double value from an XML line
+++ This function extracts a double value from an XML line
--- line:  xml-formatted line containing the value to be extracted
--- value: pointer to the variable where the extracted value will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_xml_double_value(const char *line, double *value){

  string_t xml_open  = {0};
  string_t xml_value = {0};
  string_t xml_close = {0};

  split_xml_line(line, &xml_open, &xml_value, &xml_close);

  char_to_double(xml_value.string, value);

  free_string(&xml_open);
  free_string(&xml_value);
  free_string(&xml_close);

  return;
}


/** Extract a string value from an XML line
+++ This function extracts a string value from an XML line
--- line:  xml-formatted line containing the value to be extracted
--- value: pointer to the variable where the extracted value will be stored
--- size:  size of the buffer pointed to by value
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_xml_string_value(const char *line, char *value, size_t size){

  string_t xml_open  = {0};
  string_t xml_value = {0};
  string_t xml_close = {0};

  split_xml_line(line, &xml_open, &xml_value, &xml_close);

  copy_string(value, size, xml_value.string);

  free_string(&xml_open);
  free_string(&xml_value);
  free_string(&xml_close);

  return;
}


/** Extract integer values from an XML line
+++ This function extracts integer values from an XML line
--- line:  xml-formatted line containing the values to be extracted
--- values: pointer to the array where the extracted values will be stored
--- num_values: pointer to the variable where the number of extracted values will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_xml_int_values(const char *line, int **values, int *num_values){

  string_t xml_open  = {0};
  string_t xml_value = {0};
  string_t xml_close = {0};

  split_xml_line(line, &xml_open, &xml_value, &xml_close);

  *num_values = count_occurrences(xml_value.string, " ") + 1;
  alloc((void**)values, *num_values, sizeof(int));

  char *saveptr = NULL;
  char *tokenptr = strtok_r(xml_value.string, " ", &saveptr);

  for (int i=0; i<(*num_values); i++){
    if (tokenptr == NULL){
      EXIT_ERROR("Error: Not enough integer values in line %s. Expected %d values.", line, *num_values);
    }
    char_to_int(tokenptr, &((*values)[i]));
    tokenptr = strtok_r(NULL, " ", &saveptr);
  }

  free_string(&xml_open);
  free_string(&xml_value);
  free_string(&xml_close);

  return;
}


/** Extract float values from an XML line
+++ This function extracts float values from an XML line
--- line:  xml-formatted line containing the values to be extracted
--- values: pointer to the array where the extracted values will be stored
--- num_values: pointer to the variable where the number of extracted values will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_float_values(const char *line, float **values, int *num_values){

  string_t xml_open  = {0};
  string_t xml_value = {0};
  string_t xml_close = {0};

  split_xml_line(line, &xml_open, &xml_value, &xml_close);

  *num_values = count_occurrences(xml_value.string, " ") + 1;
  alloc((void**)values, *num_values, sizeof(float));

  char *saveptr = NULL;
  char *tokenptr = strtok_r(xml_value.string, " ", &saveptr);

  for (int i=0; i<(*num_values); i++){
    if (tokenptr == NULL){
      EXIT_ERROR("Error: Not enough float values in line %s. Expected %d values.", line, *num_values);
    }
    char_to_float(tokenptr, &((*values)[i]));
    tokenptr = strtok_r(NULL, " ", &saveptr);
  }

  free_string(&xml_open);
  free_string(&xml_value);
  free_string(&xml_close);

  return;
}


/** Extract double values from an XML line
+++ This function extracts double values from an XML line
--- line:  xml-formatted line containing the values to be extracted
--- values: pointer to the array where the extracted values will be stored
--- num_values: pointer to the variable where the number of extracted values will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_double_values(const char *line, double **values, int *num_values){

  string_t xml_open  = {0};
  string_t xml_value = {0};
  string_t xml_close = {0};

  split_xml_line(line, &xml_open, &xml_value, &xml_close);

  *num_values = count_occurrences(xml_value.string, " ") + 1;
  alloc((void**)values, *num_values, sizeof(double));

  char *saveptr = NULL;
  char *tokenptr = strtok_r(xml_value.string, " ", &saveptr);

  for (int i=0; i<(*num_values); i++){
    if (tokenptr == NULL){
      EXIT_ERROR("Error: Not enough double values in line %s. Expected %d values.", line, *num_values);
    }
    char_to_double(tokenptr, &((*values)[i]));
    tokenptr = strtok_r(NULL, " ", &saveptr);
  }

  free_string(&xml_open);
  free_string(&xml_value);
  free_string(&xml_close);

  return;
}


/** Extract string values from an XML line
+++ This function extracts string values from an XML line
--- line:  xml-formatted line containing the values to be extracted
--- values: pointer to the array where the extracted values will be stored
--- num_values: pointer to the variable where the number of extracted values will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_string_values(const char *line, char ***values, int *num_values){

  string_t xml_open  = {0};
  string_t xml_value = {0};
  string_t xml_close = {0};

  split_xml_line(line, &xml_open, &xml_value, &xml_close);

  *num_values = count_occurrences(xml_value.string, " ") + 1;
  alloc_2D((void***)values, *num_values, NPOW_10, sizeof(char));

  char *saveptr = NULL;
  char *tokenptr = strtok_r(xml_value.string, " ", &saveptr);

  for (int i=0; i<(*num_values); i++){
    if (tokenptr == NULL){
      EXIT_ERROR("Error: Not enough string values in line %s. Expected %d values.", line, *num_values);
    }
    copy_string((*values)[i], NPOW_10, tokenptr);
    tokenptr = strtok_r(NULL, " ", &saveptr);
  }


  free_string(&xml_open);
  free_string(&xml_value);
  free_string(&xml_close);

  return;
}


/** Split an XML line into its components
+++ This function splits an XML line into its opening tag, value, and closing tag components
--- line:  xml-formatted line to be split
--- open_tag: pointer to the string where the opening tag will be stored
--- value: pointer to the string where the value will be stored
--- close_tag: pointer to the string where the closing tag will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void split_xml_line(const char *line, string_t *open_tag, string_t *value, string_t *close_tag){

  char buffer[NPOW_13];
  copy_string(buffer, NPOW_13, line);

  char *open_tag_start = strchr(buffer, '<');
  if (!open_tag_start){
    EXIT_ERROR("Error: Could not find opening tag in line %s.", line);
  }

  char *open_tag_end = strchr(open_tag_start, '>');
  if (!open_tag_end){
    EXIT_ERROR("Error: Could not find end of opening tag in line %s.", line);
  }

  char *close_tag_start = strchr(open_tag_end, '<');
  if (!close_tag_start){
    EXIT_ERROR("Error: Could not find closing tag in line %s.", line);
  }

  char *close_tag_end = strchr(close_tag_start, '>');
  if (!close_tag_end){
    EXIT_ERROR("Error: Could not find end of closing tag in line %s.", line);
  }

  *open_tag_end = '\0';
  fill_string(open_tag, open_tag_start + 1);

  *close_tag_start = '\0';
  fill_string(value, open_tag_end + 1);

  *close_tag_end = '\0';
  fill_string(close_tag, close_tag_start + 1);

  return;
}


/** Extract an integer attribute from an XML tag
+++ This function extracts an integer attribute from an XML tag
--- tag:  xml-formatted tag containing the attribute
--- attribute_name: name of the attribute to be extracted
--- attribute_value: pointer to the variable where the extracted value will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_attribute_int_value(const char *tag, const char *attribute_name, int *attribute_value){

  string_t xml_tag_name = {0};
  string_vector_t xml_attribute_names = {0};
  string_vector_t xml_attribute_values = {0};

  split_xml_tag(tag, &xml_tag_name, &xml_attribute_names, &xml_attribute_values);

  int pos = vector_contains_pos((const char**)xml_attribute_names.string, xml_attribute_names.number, attribute_name);
  if (pos < 0){
    EXIT_ERROR("Error: Could not find attribute %s in tag %s.", attribute_name, tag);
  }

  char_to_int(xml_attribute_values.string[pos], attribute_value);

  free_string(&xml_tag_name);
  free_string_vector(&xml_attribute_names);
  free_string_vector(&xml_attribute_values);

  return;
}


/** Extract a float attribute from an XML tag
++++ This function extracts a float attribute from an XML tag
+--- tag:  xml-formatted tag containing the attribute
+--- attribute_name: name of the attribute to be extracted
+--- attribute_value: pointer to the variable where the extracted value will be stored
++++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_attribute_float_value(const char *tag, const char *attribute_name, float *attribute_value){

  string_t xml_tag_name = {0};
  string_vector_t xml_attribute_names = {0};
  string_vector_t xml_attribute_values = {0};

  split_xml_tag(tag, &xml_tag_name, &xml_attribute_names, &xml_attribute_values);

  int pos = vector_contains_pos((const char**)xml_attribute_names.string, xml_attribute_names.number, attribute_name);
  if (pos < 0){
    EXIT_ERROR("Error: Could not find attribute %s in tag %s.", attribute_name, tag);
  }

  char_to_float(xml_attribute_values.string[pos], attribute_value);

  free_string(&xml_tag_name);
  free_string_vector(&xml_attribute_names);
  free_string_vector(&xml_attribute_values);

  return;
}


/** Extract a double attribute from an XML tag
+++ This function extracts a double attribute from an XML tag
--- tag:  xml-formatted tag containing the attribute
--- attribute_name: name of the attribute to be extracted
--- attribute_value: pointer to the variable where the extracted value will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_attribute_double_value(const char *tag, const char *attribute_name, double *attribute_value){

  string_t xml_tag_name = {0};
  string_vector_t xml_attribute_names = {0};
  string_vector_t xml_attribute_values = {0};

  split_xml_tag(tag, &xml_tag_name, &xml_attribute_names, &xml_attribute_values);

  int pos = vector_contains_pos((const char**)xml_attribute_names.string, xml_attribute_names.number, attribute_name);
  if (pos < 0){
    EXIT_ERROR("Error: Could not find attribute %s in tag %s.", attribute_name, tag);
  }

  char_to_double(xml_attribute_values.string[pos], attribute_value);

  free_string(&xml_tag_name);
  free_string_vector(&xml_attribute_names);
  free_string_vector(&xml_attribute_values);

  return;
}


/** Extract a string attribute from an XML tag
+++ This function extracts a string attribute from an XML tag
--- tag:  xml-formatted tag containing the attribute
--- attribute_name: name of the attribute to be extracted
--- attribute_value: pointer to the variable where the extracted value will be stored
--- size: size of the attribute_value buffer
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_attribute_string_value(const char *tag, const char *attribute_name, char *attribute_value, size_t size){

  string_t xml_tag_name = {0};
  string_vector_t xml_attribute_names = {0};
  string_vector_t xml_attribute_values = {0};

  split_xml_tag(tag, &xml_tag_name, &xml_attribute_names, &xml_attribute_values);

  int pos = vector_contains_pos((const char**)xml_attribute_names.string, xml_attribute_names.number, attribute_name);
  if (pos < 0){
    EXIT_ERROR("Error: Could not find attribute %s in tag %s.", attribute_name, tag);
  }

  copy_string(attribute_value, size, xml_attribute_values.string[pos]);

  free_string(&xml_tag_name);
  free_string_vector(&xml_attribute_names);
  free_string_vector(&xml_attribute_values);

  return;
}


/** Extract integer values from an XML tag
+++ This function extracts integer values from an XML tag
--- tag:  xml-formatted tag containing the attribute
--- attribute_name: name of the attribute to be extracted
--- attribute_values: pointer to the array where the extracted values will be stored
--- num_values: pointer to the variable where the number of extracted values will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_attribute_int_values(const char *tag, const char *attribute_name, int **attribute_values, int *num_values){

  string_t xml_tag_name = {0};
  string_vector_t xml_attribute_names = {0};
  string_vector_t xml_attribute_values = {0};

  split_xml_tag(tag, &xml_tag_name, &xml_attribute_names, &xml_attribute_values);

  int pos = vector_contains_pos((const char**)xml_attribute_names.string, xml_attribute_names.number, attribute_name);
  if (pos < 0){
    EXIT_ERROR("Error: Could not find attribute %s in tag %s.", attribute_name, tag);
  }

  *num_values = count_occurrences(xml_attribute_values.string[pos], " ") + 1;
  alloc((void**)attribute_values, *num_values, sizeof(int));

  char *saveptr = NULL;
  char *tokenptr = strtok_r(xml_attribute_values.string[pos], " ", &saveptr);

  for (int i=0; i<(*num_values); i++){
    if (tokenptr == NULL){
      EXIT_ERROR("Error: Not enough integer values in tag %s. Expected %d values.", tag, *num_values);
    }
    char_to_int(tokenptr, &((*attribute_values)[i]));
    tokenptr = strtok_r(NULL, " ", &saveptr);
  }

  free_string(&xml_tag_name);
  free_string_vector(&xml_attribute_names);
  free_string_vector(&xml_attribute_values);

  return;
}


/** Extract float values from an XML tag
+++ This function extracts float values from an XML tag
--- tag:  xml-formatted tag containing the attribute
--- attribute_name: name of the attribute to be extracted
--- attribute_values: pointer to the array where the extracted values will be stored
--- num_values: pointer to the variable where the number of extracted values will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_attribute_float_values(const char *tag, const char *attribute_name, float **attribute_values, int *num_values){

  string_t xml_tag_name = {0};
  string_vector_t xml_attribute_names = {0};
  string_vector_t xml_attribute_values = {0};

  split_xml_tag(tag, &xml_tag_name, &xml_attribute_names, &xml_attribute_values);

  int pos = vector_contains_pos((const char**)xml_attribute_names.string, xml_attribute_names.number, attribute_name);
  if (pos < 0){
    EXIT_ERROR("Error: Could not find attribute %s in tag %s.", attribute_name, tag);
  }

  *num_values = count_occurrences(xml_attribute_values.string[pos], " ") + 1;
  alloc((void**)attribute_values, *num_values, sizeof(float));

  char *saveptr = NULL;
  char *tokenptr = strtok_r(xml_attribute_values.string[pos], " ", &saveptr);

  for (int i=0; i<(*num_values); i++){
    if (tokenptr == NULL){
      EXIT_ERROR("Error: Not enough float values in tag %s. Expected %d values.", tag, *num_values);
    }
    char_to_float(tokenptr, &((*attribute_values)[i]));
    tokenptr = strtok_r(NULL, " ", &saveptr);
  }

  free_string(&xml_tag_name);
  free_string_vector(&xml_attribute_names);
  free_string_vector(&xml_attribute_values);

  return;
}


/** Extract double values from an XML tag
+++ This function extracts double values from an XML tag
--- tag:  xml-formatted tag containing the attribute
--- attribute_name: name of the attribute to be extracted
--- attribute_values: pointer to the array where the extracted values will be stored
--- num_values: pointer to the variable where the number of extracted values will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_attribute_double_values(const char *tag, const char *attribute_name, double **attribute_values, int *num_values){

  string_t xml_tag_name = {0};
  string_vector_t xml_attribute_names = {0};
  string_vector_t xml_attribute_values = {0};

  split_xml_tag(tag, &xml_tag_name, &xml_attribute_names, &xml_attribute_values);

  int pos = vector_contains_pos((const char**)xml_attribute_names.string, xml_attribute_names.number, attribute_name);
  if (pos < 0){
    EXIT_ERROR("Error: Could not find attribute %s in tag %s.", attribute_name, tag);
  }

  *num_values = count_occurrences(xml_attribute_values.string[pos], " ") + 1;
  alloc((void**)attribute_values, *num_values, sizeof(double));

  char *saveptr = NULL;
  char *tokenptr = strtok_r(xml_attribute_values.string[pos], " ", &saveptr);

  for (int i=0; i<(*num_values); i++){
    if (tokenptr == NULL){
      EXIT_ERROR("Error: Not enough double values in tag %s. Expected %d values.", tag, *num_values);
    }
    char_to_double(tokenptr, &((*attribute_values)[i]));
    tokenptr = strtok_r(NULL, " ", &saveptr);
  }

  free_string(&xml_tag_name);
  free_string_vector(&xml_attribute_names);
  free_string_vector(&xml_attribute_values);

  return;
}


/** Extract string values from an XML tag
+++ This function extracts string values from an XML tag
--- tag:  xml-formatted tag containing the attribute
--- attribute_name: name of the attribute to be extracted
--- attribute_values: pointer to the array where the extracted values will be stored
--- num_values: pointer to the variable where the number of extracted values will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void get_xml_attribute_string_values(const char *tag, const char *attribute_name, char ***attribute_values, int *num_values){

  string_t xml_tag_name = {0};
  string_vector_t xml_attribute_names = {0};
  string_vector_t xml_attribute_values = {0};

  split_xml_tag(tag, &xml_tag_name, &xml_attribute_names, &xml_attribute_values);

  int pos = vector_contains_pos((const char**)xml_attribute_names.string, xml_attribute_names.number, attribute_name);
  if (pos < 0){
    EXIT_ERROR("Error: Could not find attribute %s in tag %s.", attribute_name, tag);
  }

  *num_values = count_occurrences(xml_attribute_values.string[pos], " ") + 1;
  alloc_2D((void***)attribute_values, *num_values, NPOW_10, sizeof(char));

  char *saveptr = NULL;
  char *tokenptr = strtok_r(xml_attribute_values.string[pos], " ", &saveptr);

  for (int i=0; i<(*num_values); i++){
    if (tokenptr == NULL){
      EXIT_ERROR("Error: Not enough string values in tag %s. Expected %d values.", tag, *num_values);
    }
    copy_string((*attribute_values)[i], NPOW_10, tokenptr);
    tokenptr = strtok_r(NULL, " ", &saveptr);
  }


  free_string(&xml_tag_name);
  free_string_vector(&xml_attribute_names);
  free_string_vector(&xml_attribute_values);

  return;
}


/** Split an XML tag into its components
+++ This function splits an XML tag into its name and attributes
--- tag:  xml-formatted tag to be split
--- tag_name: pointer to the string where the tag name will be stored
--- attribute_name: pointer to the vector where the attribute names will be stored
--- attribute_value: pointer to the vector where the attribute values will be stored
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
void split_xml_tag(const char *tag, string_t *tag_name, string_vector_t *attribute_name, string_vector_t *attribute_value){


  if (strstr(tag, " ") == NULL || strstr(tag, "=") == NULL || strstr(tag, "\"") == NULL){
      fill_string(tag_name, tag);
      attribute_name->number = 0;
      attribute_value->number = 0;
      return;
  }

  char buffer[NPOW_13];
  copy_string(buffer, NPOW_13, tag);

  replace_string(buffer, "<", "", NPOW_13);
  replace_string(buffer, ">", "", NPOW_13);

  char *saveptr = NULL;
  char *tokenptr = strtok_r(buffer, " ", &saveptr);
  if (tokenptr == NULL){
    EXIT_ERROR("Error: Could not parse tag %s.", tag);
  }

  fill_string(tag_name, tokenptr);
printf("tag_name: %s\n", tag_name->string);

  int i = 0;
  while (tokenptr != NULL){

    // if not an attribute, skip to next token
    if (strstr(tokenptr, "=") == NULL){
      tokenptr = strtok_r(NULL, " ", &saveptr);
      continue;
    }

    char attribute[NPOW_10];
    copy_string(attribute, NPOW_10, tokenptr);
    
    char *saveptr_attr = NULL;
    char *tokenptr_attr = strtok_r(attribute, "=\"", &saveptr_attr);
    if (tokenptr_attr == NULL){
      EXIT_ERROR("Error: Could not find attribute in tag %s.", tag);
    }
    fill_string_vector(attribute_name, i, tokenptr_attr);
    
    tokenptr_attr = strtok_r(NULL, "\"", &saveptr_attr);
    if (tokenptr_attr == NULL){
      EXIT_ERROR("Error: Could not find value for attribute %s in tag %s.", attribute_name->string[i], tag);
    }
    fill_string_vector(attribute_value, i, tokenptr_attr);

    tokenptr = strtok_r(NULL, " ", &saveptr);
    i++;

  }

  return;
}
