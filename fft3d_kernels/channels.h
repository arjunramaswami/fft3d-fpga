/******************************************************************************
 *  Authors: Arjun Ramaswami
 *****************************************************************************/
#include "fft3d_config.h"

#ifndef FFT_CHANNELS_H
#define FFT_CHANNELS_H

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel cmplx chaninfft[8] __attribute__((depth(8)));
channel cmplx chanoutfft[8] __attribute__((depth(8)));
channel cmplx chaninfft2[8] __attribute__((depth(8)));
channel cmplx chanoutfft2[8] __attribute__((depth(8)));
channel cmplx chaninfetch[8] __attribute__((depth(8)));

#endif
