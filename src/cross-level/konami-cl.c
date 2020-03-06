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
This file contains konami code easter egg
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "konami-cl.h"


void clear1(float sec);
void clear2(float sec);
void aspect1(int col, float sec);
void aspect2(int col, float sec);
void aspect3(int col, float sec);
void aspect4(int col, float sec);
int ansi(int *c, int *p, int *step, int *col);


/** This function is a hidden easter egg. The function compares the 1st
+++ given argument to the konami code sequence. If so, some 80ish FORCE
+++ animation is shown in an endless loop
--- arg:    commandline arguments
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void check_arg(char *arg){
char achievement[1024] = "CONGRATULATIONS!!! YOU HAVE EARNED A NEW ACHIEVEMENT: <<<LIGHTSABER>>>\n\n";
struct timespec ts;
int i;
int col[3] = { 0, 0, 5 }, p = 0, step = 1, c = 0, col_ansi = 21;
float sec = 0.1;

  if (strcmp(arg, "wwssadadBA") != 0) return;
  
  for (i=0; i<2; i++){ clear1(sec);  clear2(sec);}
    
  
  ts.tv_sec = 0;
  ts.tv_nsec = sec*1e9;
 
  for (i=0; i<strlen(achievement); i++){
    printf("\033[1;38;5;%dm%c\033[0;00m", col_ansi, achievement[i]); fflush(stdout);
    col_ansi = ansi(&c, &p, &step, col);
    nanosleep(&ts, &ts);
  }

  
  col[0] = 0; col[1] = 0; col[2] = 5;
  p = 0; step = 1; c = 0; col_ansi = 21;

  while (1 > 0){
    for (i=0; i<30; i++){
      aspect1(col_ansi, sec);
      col_ansi = ansi(&c, &p, &step, col);
    }
    for (i=0; i<30; i++){
      aspect2(col_ansi, sec);
      col_ansi = ansi(&c, &p, &step, col);
    }
    for (i=0; i<30; i++){
      aspect3(col_ansi, sec);
      col_ansi = ansi(&c, &p, &step, col);
    }
    for (i=0; i<30; i++){
      aspect4(col_ansi, sec);
      col_ansi = ansi(&c, &p, &step, col);
    }
  }
  
  exit(1);
  return;
}


/** This function clears the console, and inserts a +++ field
--- sec:    update frequency
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void clear1(float sec){
struct timespec ts;
int line;

  ts.tv_sec = 0;
  ts.tv_nsec = sec*1e9;

  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);
  printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"); nanosleep(&ts, &ts);

  for (line=0; line<=12; line++) printf("\033[A\r");
  printf("\n");

  return;
}


/** This function clears the console, and inserts a blank field
--- sec:    update frequency
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void clear2(float sec){
struct timespec ts;
int line;

  ts.tv_sec = 0;
  ts.tv_nsec = sec*1e9;

  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);
  printf("                                                                      \n"); nanosleep(&ts, &ts);

  for (line=0; line<=12; line++) printf("\033[A\r");
  printf("\n");

  nanosleep(&ts, &ts);

  return;
}


/** This function prints FORCE with aspect 1
--- col:    text color
--- sec:    update frequency
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void aspect1(int col, float sec){
struct timespec ts;
int line;

  printf("\033[1;38;5;%dm      ___           ___           ___           ___           ___     \033[0;00m\n", col);
  printf("\033[1;38;5;%dm     /\\  \\         /\\  \\         /\\  \\         /\\  \\         /\\  \\    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm    /::\\  \\       /::\\  \\       /::\\  \\       /::\\  \\       /::\\  \\   \033[0;00m\n", col);
  printf("\033[1;38;5;%dm   /:/\\:\\  \\     /:/\\:\\  \\     /:/\\:\\  \\     /:/\\:\\  \\     /:/\\:\\  \\  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm  /::\\~\\:\\  \\   /:/  \\:\\  \\   /::\\~\\:\\  \\   /:/  \\:\\  \\   /::\\~\\:\\  \\ \033[0;00m\n", col);
  printf("\033[1;38;5;%dm /:/\\:\\ \\:\\__\\ /:/__/ \\:\\__\\ /:/\\:\\ \\:\\__\\ /:/__/ \\:\\__\\ /:/\\:\\ \\:\\__\\\033[0;00m\n", col);
  printf("\033[1;38;5;%dm \\/__\\:\\ \\/__/ \\:\\  \\ /:/  / \\/_|::\\/:/  / \\:\\  \\  \\/__/ \\:\\~\\:\\ \\/__/\033[0;00m\n", col);
  printf("\033[1;38;5;%dm      \\:\\__\\    \\:\\  /:/  /     |:|::/  /   \\:\\  \\        \\:\\ \\:\\__\\  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm       \\/__/     \\:\\/:/  /      |:|\\/__/     \\:\\  \\        \\:\\ \\/__/  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm                  \\::/  /       |:|  |        \\:\\__\\        \\:\\__\\    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm                   \\/__/         \\|__|         \\/__/         \\/__/    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm                                                                      \033[0;00m\n", col);
  
  for (line=0; line<=12; line++) printf("\033[A\r");
  printf("\n");
  
  ts.tv_sec = 0;
  ts.tv_nsec = sec*1e9;
  
  nanosleep(&ts, &ts);

  return;
}


/** This function prints FORCE with aspect 2
--- col:    text color
--- sec:    update frequency
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void aspect2(int col, float sec){
struct timespec ts;
int line;

  printf("\033[1;38;5;%dm      ___           ___           ___           ___           ___     \033[0;00m\n", col);
  printf("\033[1;38;5;%dm     /\\__\\         /\\  \\         /\\  \\         /\\__\\         /\\__\\    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm    /:/ _/_       /::\\  \\       /::\\  \\       /:/  /        /:/ _/_   \033[0;00m\n", col);
  printf("\033[1;38;5;%dm   /:/ /\\__\\     /:/\\:\\  \\     /:/\\:\\__\\     /:/  /        /:/ /\\__\\  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm  /:/ /:/  /    /:/  \\:\\  \\   /:/ /:/  /    /:/  /  ___   /:/ /:/ _/_ \033[0;00m\n", col);
  printf("\033[1;38;5;%dm /:/_/:/  /    /:/__/ \\:\\__\\ /:/_/:/__/___ /:/__/  /\\__\\ /:/_/:/ /\\__\\\033[0;00m\n", col);
  printf("\033[1;38;5;%dm \\:\\/:/  /     \\:\\  \\ /:/  / \\:\\/:::::/  / \\:\\  \\ /:/  / \\:\\/:/ /:/  /\033[0;00m\n", col);
  printf("\033[1;38;5;%dm  \\::/__/       \\:\\  /:/  /   \\::/~~/~~~~   \\:\\  /:/  /   \\::/_/:/  / \033[0;00m\n", col);
  printf("\033[1;38;5;%dm   \\:\\  \\        \\:\\/:/  /     \\:\\~~\\        \\:\\/:/  /     \\:\\/:/  /  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm    \\:\\__\\        \\::/  /       \\:\\__\\        \\::/  /       \\::/  /   \033[0;00m\n", col);
  printf("\033[1;38;5;%dm     \\/__/         \\/__/         \\/__/         \\/__/         \\/__/    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm                                                                      \033[0;00m\n", col);
  
  for (line=0; line<=12; line++) printf("\033[A\r");
  printf("\n");
  
  ts.tv_sec = 0;
  ts.tv_nsec = sec*1e9;
  
  nanosleep(&ts, &ts);

  return;
}


/** This function prints FORCE with aspect 3
--- col:    text color
--- sec:    update frequency
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void aspect3(int col, float sec){
struct timespec ts;
int line;

  printf("\033[1;38;5;%dm        ___         ___           ___           ___           ___     \033[0;00m\n", col);
  printf("\033[1;38;5;%dm       /  /\\       /  /\\         /  /\\         /  /\\         /  /\\    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm      /  /:/_     /  /::\\       /  /::\\       /  /:/        /  /:/_   \033[0;00m\n", col);
  printf("\033[1;38;5;%dm     /  /:/ /\\   /  /:/\\:\\     /  /:/\\:\\     /  /:/        /  /:/ /\\  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm    /  /:/ /:/  /  /:/  \\:\\   /  /:/~/:/    /  /:/  ___   /  /:/ /:/_ \033[0;00m\n", col);
  printf("\033[1;38;5;%dm   /__/:/ /:/  /__/:/ \\__\\:\\ /__/:/ /:/___ /__/:/  /  /\\ /__/:/ /:/ /\\\033[0;00m\n", col);
  printf("\033[1;38;5;%dm   \\  \\:\\/:/   \\  \\:\\ /  /:/ \\  \\:\\/:::::/ \\  \\:\\ /  /:/ \\  \\:\\/:/ /:/\033[0;00m\n", col);
  printf("\033[1;38;5;%dm    \\  \\::/     \\  \\:\\  /:/   \\  \\::/~~~~   \\  \\:\\  /:/   \\  \\::/ /:/ \033[0;00m\n", col);
  printf("\033[1;38;5;%dm     \\  \\:\\      \\  \\:\\/:/     \\  \\:\\        \\  \\:\\/:/     \\  \\:\\/:/  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm      \\  \\:\\      \\  \\::/       \\  \\:\\        \\  \\::/       \\  \\::/   \033[0;00m\n", col);
  printf("\033[1;38;5;%dm       \\__\\/       \\__\\/         \\__\\/         \\__\\/         \\__\\/    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm                                                                      \033[0;00m\n", col);  
  
  for (line=0; line<=12; line++) printf("\033[A\r");
  printf("\n");
  
  ts.tv_sec = 0;
  ts.tv_nsec = sec*1e9;
  
  nanosleep(&ts, &ts);

  return;
}


/** This function prints FORCE with aspect 4
--- col:    text color
--- sec:    update frequency
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void aspect4(int col, float sec){
struct timespec ts;
int line;

  printf("\033[1;38;5;%dm                    ___           ___           ___           ___     \033[0;00m\n", col);
  printf("\033[1;38;5;%dm      ___          /  /\\         /  /\\         /  /\\         /  /\\    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm     /  /\\        /  /::\\       /  /::\\       /  /::\\       /  /::\\   \033[0;00m\n", col);
  printf("\033[1;38;5;%dm    /  /::\\      /  /:/\\:\\     /  /:/\\:\\     /  /:/\\:\\     /  /:/\\:\\  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm   /  /:/\\:\\    /  /:/  \\:\\   /  /::\\ \\:\\   /  /:/  \\:\\   /  /::\\ \\:\\ \033[0;00m\n", col);
  printf("\033[1;38;5;%dm  /  /::\\ \\:\\  /__/:/ \\__\\:\\ /__/:/\\:\\_\\:\\ /__/:/ \\  \\:\\ /__/:/\\:\\ \\:\\\033[0;00m\n", col);
  printf("\033[1;38;5;%dm /__/:/\\:\\ \\:\\ \\  \\:\\ /  /:/ \\__\\/~|::\\/:/ \\  \\:\\  \\__\\/ \\  \\:\\ \\:\\_\\/\033[0;00m\n", col);
  printf("\033[1;38;5;%dm \\__\\/  \\:\\_\\/  \\  \\:\\  /:/     |  |:|::/   \\  \\:\\        \\  \\:\\ \\:\\  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm      \\  \\:\\     \\  \\:\\/:/      |  |:|\\/     \\  \\:\\        \\  \\:\\_\\/  \033[0;00m\n", col);
  printf("\033[1;38;5;%dm       \\__\\/      \\  \\::/       |__|:|~       \\  \\:\\        \\  \\:\\    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm                   \\__\\/         \\__\\|         \\__\\/         \\__\\/    \033[0;00m\n", col);
  printf("\033[1;38;5;%dm                                                                      \033[0;00m\n", col);

  for (line=0; line<=12; line++) printf("\033[A\r");
  printf("\n");

  ts.tv_sec = 0;
  ts.tv_nsec = sec*1e9;
  
  nanosleep(&ts, &ts);

  return;
}


/** This function generates continuous ANSII colors
--- c:      R, G, or B
--- p:      position of color
--- step:   step up or down (up to 5)
--- col:    text color
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int ansi(int *c, int *p, int *step, int *col){

  col[*c] += *step;

  if (++*p == 5){
    *step *= -1;
    *p = 0;
    if (--*c < 0) *c = 2;
  }

  return 16+36*col[0]+6*col[1]+col[2];
}
 
