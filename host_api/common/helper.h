/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef HELPER_H
#define HELPER_H

double getTimeinMilliSec();

unsigned int coord( unsigned int i, unsigned int j, unsigned int k, int N[3]);

void print_output(double fftw_time, double fpga_time, int iter, int N[3], char* fname);

#endif // HELPER_H
