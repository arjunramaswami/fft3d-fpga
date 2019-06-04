#include "fft_config.h"

void fourier_transform_3d( int lognr_points, cmplex *h_verify, bool inverse);
void check_correctness(int N, cmplex *h_verify, cmplex *h_outData);
int coord(int iteration, int i, int j, uint32_t *N);

