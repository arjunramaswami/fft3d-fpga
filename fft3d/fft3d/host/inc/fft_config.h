/******************************************************************************
 *  Authors: Arjun Ramaswami
 ***************************************************************************/ 

#ifndef FFT_CONFIG_H__
#define FFT_CONFIG_H__

// Determines FFT size 
#ifndef LOGN
#  define LOGN 6
#endif

#ifndef TYPE_FLOAT
#  define TYPE_FLOAT 0
#endif 


#ifndef CMPLEX
#  define CMPLEX

#if TYPE_FLOAT == 1
    typedef float2 cmplex;
#else
    typedef double2 cmplex;
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#endif // CMPLEX

#ifndef PREC
#  define PREC

#if TYPE_FLOAT == 1
    typedef float prec;
#else 
    typedef double prec;
#endif

#endif //prec

#endif
