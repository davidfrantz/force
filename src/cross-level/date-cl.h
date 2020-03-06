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
Date header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef DATE_CL_H
#define DATE_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <math.h>    // common mathematical functions
#include <string.h>  // string handling functions

#include "../cross-level/const-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int ce;      // days since current era
  int day;     // day
  int doy;     // day-of-year
  int week;    // week
  int month;   // month
  int quarter; // quarter
  int year;    // year
  int hh;      // hour
  int mm;      // minute
  int ss;      // second
  int tz;      // timezone
} date_t;

void doy2md(int doy, int *m, int *d);
int doy2m(int doy);
int doy2d(int doy);
int md2doy(int m, int d);
int doy2week(int doy);
int week2doy(int week);
int doy2quarter(int doy);
int m2quarter(int month);
int date2ce(int y, int m, int d);
int doy2ce(int doy, int y);
void ce2date(int ce, int *y, int *m, int *d);
void ce2doy(int ce, int* doy, int* y);
int  ce2year(int ce);
void compact_date(int y, int m, int d, char formatted[], size_t size);
void long_date(int y, int m, int d, int hh, int mm, int ss, int tz, char formatted[], size_t size);
void init_date(date_t *date);
void set_date(date_t *date, int year, int month, int day);
void set_date_year(date_t *date, int year);
void set_date_month_day(date_t *date, int month, int day);
void set_date_ce(date_t *date, int ce);
void set_date_doy(date_t *date, int doy);
void set_date_week(date_t *date, int week);
void set_date_month(date_t *date, int month);
void set_date_quarter(date_t *date, int quarter);
void set_time(date_t *date, int hh, int mm, int ss);
void set_secs(date_t *date, int ss);
void copy_date(date_t *from, date_t *to);
void print_date(date_t *date);

#ifdef __cplusplus
}
#endif

#endif

