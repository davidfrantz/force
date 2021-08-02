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
This file contains functions for system date support
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "datesys-cl.h"


struct tm *date_set(date_t d);


/** Get current time
+++ This function gets the current time and date and returns a date struct
+++ containing year, month, day, day-of-year, week-of-year and days since
+++ current era. 
--- d:      date struct (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void current_date(date_t *d){
time_t rawtime;
struct tm *timeinfo;


  time(&rawtime);
  timeinfo = localtime(&rawtime);
  d->year  = timeinfo->tm_year+1900;
  d->month = timeinfo->tm_mon+1;
  d->day   = timeinfo->tm_mday;
  d->doy   = md2doy(d->month,d->day);
  d->week  = doy2week(d->doy);
  d->ce    = doy2ce(d->doy,d->year);

  return;
}


/** Set time
+++ This function sets the time to the given date.
+++ This function needs a global variable timeinfo (see force.h/global.c).
+++ date_set needs to be called before using this function.
--- d:      date struct
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
struct tm *date_set(date_t d){
time_t rawtime;
struct tm *timeinfo;


  time(&rawtime);
  timeinfo = localtime(&rawtime);
  timeinfo->tm_year = d.year-1900;
  timeinfo->tm_mon = d.month-1;
  timeinfo->tm_mday = d.day;
  mktime(timeinfo);

  return timeinfo;
}


/** Increment time
+++ This function increments the time with one day, returns a date struct
+++ containing year, month, day, day-of-year, week-of-year and days since
+++ current era. 
--- d:      date struct (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void date_plus(date_t *d){
struct tm *timeinfo;


  timeinfo = date_set(*d);

  timeinfo->tm_mday++;
  mktime(timeinfo);

  d->year  = timeinfo->tm_year+1900;
  d->month = timeinfo->tm_mon+1;
  d->day   = timeinfo->tm_mday;
  d->doy   = md2doy(d->month,d->day);
  d->week  = doy2week(d->doy);
  d->ce    = doy2ce(d->doy,d->year);

  return;
}


/** Decrement time
+++ This function decrements the time with one day, returns a date struct
+++ containing year, month, day, day-of-year, week-of-year and days since
+++ current era. 
--- d:      date struct (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void date_minus(date_t *d){
struct tm *timeinfo;

  timeinfo = date_set(*d);

  timeinfo->tm_mday--;
  mktime(timeinfo);

  d->year  = timeinfo->tm_year+1900;
  d->month = timeinfo->tm_mon+1;
  d->day   = timeinfo->tm_mday;
  d->doy   = md2doy(d->month,d->day);
  d->week  = doy2week(d->doy);
  d->ce    = doy2ce(d->doy,d->year);

  return;
}

int date_overshoot(date_t *d){
	date_t today;
	current_date(&today);

	if (d->doy > today.doy && d->year >= today.year) {
		return 1;
	} else {
		return 0;
	}
}
