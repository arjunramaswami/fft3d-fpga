/******************************************************************************
 *  Authors: Arjun Ramaswami
 *****************************************************************************/

// Declare channels for kernel to kernel communication

#ifndef FFT_CHANNELS_H
#define FFT_CHANNELS_H

#pragma OPENCL EXTENSION cl_intel_channels : enable

channel cmplex chaninfft[8] __attribute__((depth(8)));
channel cmplex chanoutfft[8] __attribute__((depth(8)));

channel cmplex chaninfft2[8] __attribute__((depth(8)));
channel cmplex chanoutfft2[8] __attribute__((depth(8)));

channel cmplex chaninfetch[8] __attribute__((depth(8)));
#endif
