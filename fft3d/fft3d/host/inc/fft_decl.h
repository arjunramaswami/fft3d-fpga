// FFT operates with complex numbers - store them in a struct
#ifndef PRECISION
#  define PRECISION

typedef struct {
  double x;
  double y;
} double2;

typedef struct {
  float x;
  float y;
} float2;

#endif // PRECISION

#ifndef TYPE_FLOAT
#  define TYPE_FLOAT 0
#endif 

#ifndef DECIMAL
#  define DECIMAL

#if TYPE_FLOAT == 1
typedef float2 cmplex;
typedef float decimal;
#else
typedef double2 cmplex;
typedef double decimal;
#endif

#endif 
