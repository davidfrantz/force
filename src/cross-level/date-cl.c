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

/** The date implementation included in this file, is based on code from 
+++ SpliTS, a framework for spline analysis of  time series. SpliTS is 
+++ licensed under the terms of the GNU General Public License (>= 3).
+++ http://sebastian-mader.net/splits/
+++ SpliTS Copyright (C) 2010-2016 Sebastian Mader
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions for converting dates
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "date-cl.h"


/** This function converts a day-of-year value to month and day. Note that
+++ leap years are ignored, a year has 365 days.
--- doy:    day-of-year
--- m:      month (returned)
--- d:      day   (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void doy2md(int doy, int *m, int *d){

  if (doy>334){
    *m = 12; *d = doy-334; return;
  } else if (doy>304){
    *m = 11; *d = doy-304; return;
  } else if (doy>273){
    *m = 10; *d = doy-273; return;
  } else if (doy>243){
    *m = 9;  *d = doy-243; return;
  } else if (doy>212){
    *m = 8;  *d = doy-212; return;
  } else if (doy>181){
    *m = 7;  *d = doy-181; return;
  } else if (doy>151){
    *m = 6;  *d = doy-151; return;
  } else if (doy>120){
    *m = 5;  *d = doy-120; return;
  } else if (doy>90){
    *m = 4;  *d = doy-90;  return;
  } else if (doy>59){
    *m = 3;  *d = doy-59;  return;
  } else if (doy>31){
    *m = 2;  *d = doy-31;  return;
  } else {
    *m = 1;  *d = doy;     return;
  }

  return;
}


/** This function converts a day-of-year value to month. Note that
+++ leap years are ignored, a year has 365 days.
--- doy:    day-of-year
--- m:      month (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int doy2m(int doy){

  if (doy > 334){
    return 12;
  } else if (doy>304){
    return 11;
  } else if (doy>273){
    return 10;
  } else if (doy>243){
    return 9;
  } else if (doy>212){
    return 8;
  } else if (doy>181){
    return 7;
  } else if (doy>151){
    return 6;
  } else if (doy>120){
    return 5;
  } else if (doy>90){
    return 4;
  } else if (doy>59){
    return 3;
  } else if (doy>31){
    return 2;
  } else {
    return 1;
  }

}


/** This function converts a day-of-year value to day. Note that
+++ leap years are ignored, a year has 365 days.
--- doy:    day-of-year
--- d:      day   (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int doy2d(int doy){

  if (doy>334){
    return doy-334;
  } else if (doy>304){
    return doy-304;
  } else if (doy>273){
    return doy-273;
  } else if (doy>243){
    return doy-243;
  } else if (doy>212){
    return doy-212;
  } else if (doy>181){
    return doy-181;
  } else if (doy>151){
    return doy-151;
  } else if (doy>120){
    return doy-120;
  } else if (doy>90){
    return doy-90;
  } else if (doy>59){
    return doy-59;
  } else if (doy>31){
    return doy-31;
  } else {
    return doy;
  }

}


/** This function converts month and day values to day-of-year. Note that 
+++ leap years are ignored, a year has 365 days.
--- m:      month
--- d:      day
+++ Return: day-of-year
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int md2doy(int m, int d){
int doy;

  switch(m){
    case 1:  doy = d;     break;
    case 2:  doy = 31+d;  break;
    case 3:  doy = 59+d;  break;
    case 4:  doy = 90+d;  break;
    case 5:  doy = 120+d; break;
    case 6:  doy = 151+d; break;
    case 7:  doy = 181+d; break;
    case 8:  doy = 212+d; break;
    case 9:  doy = 243+d; break;
    case 10: doy = 273+d; break;
    case 11: doy = 304+d; break;
    case 12: doy = 334+d; break;
    default: doy = 0;
  }

  return doy;
}


/** This function converts a day-of-year value to week-of-year. Note that
+++ the 1st week starts at DOY 1, a year has 52 weeks (The last week con-
+++ a couple more days).
--- doy:    day-of-year
+++ Return: week-of-year
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int doy2week(int doy){
float w;

  if ((w = ceil(doy/7.0)) > 52) return 52; else return (int)w;
}


/** This function converts a week-of-year value to day-of-year. Note that
+++ the 1st week starts at DOY 1, a year has 52 weeks (The last week con-
+++ a couple more days).
--- week: week-of-year
--- Return: day-of-year (1st day in week)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int week2doy(int week){
  
  return (week-1)*7+1;
}


/** This function converts a day-of-year value to quarter-of-year.  
--- doy:    day-of-year
+++ Return: quarter-of-year
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int doy2quarter(int doy){
int m, d, q;

  doy2md(doy, &m, &d);
  q = m2quarter(m);

  return q;
}


/** This function converts a month-of-year value to quarter-of-year.  
--- month:  month-of-year
+++ Return: quarter-of-year
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int m2quarter(int month){
int q;

  q = ceil(month/3.0);
  return q;
}


/** This function converts a quarter-of-year value to day-of-year.  
+++ quarter: quarter-of-year
--- doy:     day-of-year (1st day in quarter)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int quarter2doy(int quarter){
  
  
  return md2doy((quarter-1)*3+1, 1);
}


/** This function converts a date to days since CE.  Note that leap years 
+++ are ignored, a year has 365 days.
--- m:      month
--- d:      day
--- y:      year
+++ Return: days since current era
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int date2ce(int y, int m, int d){

  return y*365+md2doy(m,d);
}


/** This function converts day-of-year and year values to days since CE. 
+++ Note that leap years are ignored, a year has 365 days.
--- doy:    day-of-year
--- y:      year
+++ Return: days since current era
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int doy2ce(int doy, int y){

  return y*365+doy;
}


/** This function converts a days since CE value to a a date. Note that 
+++ leap years are ignored, a year has 365 days.
--- ce:     days since current era
--- m:      month (returned)
--- d:      day   (returned)
--- y:      year  (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void ce2date(int ce, int *y, int *m, int *d){
int year, tmp;

  year = (int)floor(ce/365.0);
  tmp = ce-year*365;

  if (tmp==0){
    *y = year-1;
    *m = 12;
    *d = 31;
  } else {
    *y = year;
    doy2md(tmp, m, d);
  }

  return;
}


/** This function converts a days since CE value to day-of-year and year. 
+++ Note that leap years are ignored, a year has 365 days.
--- ce:     days since current era
--- doy:    day-of-year (returned)
--- y:      year        (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void ce2doy(int ce, int* doy, int* y){
int year, tmp;

  year = (int)floor(ce/365.0);
  tmp = ce-year*365;

  if (tmp==0){
    *y = year-1;
    *doy = 365;
  } else {
    *y = year;
    *doy = tmp;
  }

  return;
}

/** This function converts a days since CE value to year. 
+++ Note that leap years are ignored, a year has 365 days.
--- ce:     days since current era
+++ Return: year
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int  ce2year(int ce){
int year, tmp;

  year = (int)floor(ce/365.0);
  tmp = ce-year*365;

  if (tmp==0) return(year-1);

  return(year);
}


/** This function formats a date using compact date convention YYYYMMDD
--- y:         year
--- m:         month
--- d:         day
--- formatted: buffer that will hold the formatted string
--- size:      length of buffer
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compact_date(int y, int m, int d, char formatted[], size_t size){
int nchar;


  nchar = snprintf(formatted, size, "%04d%02d%02d", y, m, d);
  if (nchar < 0 || nchar >= size){ 
    printf("Buffer Overflow in assembling compact date %04d%02d%02d\n", 
      y, m, d); exit(1);}

  return;
}


/** This function formats a date using long date convention:
+++ YYYY-MM-DDTHH:MM:SS.MSZ or YYYY-MM-DDTHH:MM:SS if tz == 99
--- y:         year
--- m:         month
--- d:         day
--- hh:        hour
--- mm:        minutes
--- ss:        seconds
--- tz:        time zone
--- formatted: buffer that will hold the formatted string
--- size:      length of buffer
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void long_date(int y, int m, int d, int hh, int mm, int ss, int tz, char formatted[], size_t size){
int nchar;



  if (tz != 99){
    nchar = snprintf(formatted, size, "%04d-%02d-%02dT%02d:%02d:%02d.%dZ", 
      y, m, d, hh, mm, ss, tz);
    if (nchar < 0 || nchar >= size){ 
      printf("Buffer Overflow in assembling long date %04d-%02d-%02dT%02d:%02d:%02d.%dZ\n", 
        y, m, d, hh, mm, ss, tz); exit(1);}
  } else {
    nchar = snprintf(formatted, size, "%04d-%02d-%02dT%02d:%02d:%02d", 
      y, m, d, hh, mm, ss);
    if (nchar < 0 || nchar >= size){ 
      printf("Buffer Overflow in assembling long date %04d-%02d-%02dT%02d:%02d:%02d\n", 
        y, m, d, hh, mm, ss); exit(1);}
  }

  return;
}


/** This function initializes a date struct to 0
--- date:   date struct (modified)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_date(date_t *date){

  date->ce      = 0;
  date->day     = 0;
  date->doy     = 0;
  date->week    = 0;
  date->month   = 0;
  date->quarter = 0;
  date->year    = 0;
  date->hh      = 0;
  date->mm      = 0;
  date->ss      = 0;
  date->tz      = 0;

  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- year:   year
--- month:  month
--- day:    day
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_date(date_t *date, int year, int month, int day){

  set_date_year(date, year);
  set_date_month_day(date, month, day);

  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- year:   year
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_date_year(date_t *date, int year){

  date->year = year;
  date->ce   = doy2ce(date->doy, date->year);
  
  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- month:  month
--- day:    day
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_date_month_day(date_t *date, int month, int day){

  date->month   = month;
  date->day     = day;
  date->doy     = md2doy(date->month, date->day);
  date->week    = doy2week(date->doy);
  date->quarter = doy2quarter(date->doy);
  date->ce      = doy2ce(date->doy, date->year);

  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- ce:     days since current era
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_date_ce(date_t *date, int ce){
int m, d, y;

  ce2date(ce, &y, &m, &d);
  set_date(date, y, m, d);
  
  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- doy:    day of year
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_date_doy(date_t *date, int doy){
int m, d;

  doy2md(doy, &m, &d);
  set_date_month_day(date, m, d);
  
  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- doy:    week of year
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_date_week(date_t *date, int week){
int doy;
  
  doy = week2doy(week);
  set_date_doy(date, doy);
  
  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- doy:    month of year
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_date_month(date_t *date, int month){
int doy;
  
  doy = md2doy(month, 1);
  set_date_doy(date, doy);
  
  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- doy:    quarter of year
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_date_quarter(date_t *date, int quarter){
int doy;
  
  doy = quarter2doy(quarter);
  set_date_doy(date, doy);
  
  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- hh:     hour
--- mm:     minutes
--- ss:     seconds
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_time(date_t *date, int hh, int mm, int ss){

  date->hh = hh;
  date->mm = mm;
  date->ss = ss;

  return;
}


/** This function sets values of a date struct
--- date:   date struct (modified)
--- ss:     seconds
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_secs(date_t *date, int ss){
int y, m, d, hh, mm;

  if (ss >= 946080000){
    y = floor(ss/946080000.0); ss -= y*946080000;
  } else y = 0;
  if (ss >= 2592000){
    m = floor(ss/2592000.0); ss -= m*2592000;
  } else m = 0;
  if (ss >= 86400){
    d = floor(ss/86400.0); ss -= d*86400;
  } else d = 0;
  if (ss >= 3600){
    hh = floor(ss/3600.0); ss -= hh*3600;
  } else hh = 0;
  if (ss >= 60){
    mm = floor(ss/60.0); ss -= mm*60;
  } else mm = 0;

  set_date(date, y, m, d);
  set_time(date, hh, mm, ss);

  return;
}


/** This function copies a date struct
--- from:   source date struct
--- to:     target date struct (modified)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void copy_date(date_t *from, date_t *to){

  to->ce      = from->ce;
  to->day     = from->day;
  to->doy     = from->doy;
  to->week    = from->week;
  to->month   = from->month;
  to->quarter = from->quarter;
  to->year    = from->year;
  to->hh      = from->hh;
  to->mm      = from->mm;
  to->ss      = from->ss;
  to->tz      = from->tz;

  return;
}


/** This function prints a date struct
--- date:   date struct
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_date(date_t *date){
  
  printf("Date: %04d-%02d-%02d (%03d/%02d/%d) %02d:%02d:%02d-%02dZ\n", 
    date->year, date->month, date->day, date->doy, date->week, date->ce, 
    date->hh, date->mm, date->ss, date->tz);

  return;
}

