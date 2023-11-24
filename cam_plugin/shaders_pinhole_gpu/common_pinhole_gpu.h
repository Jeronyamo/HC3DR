/////////////////////////////////////////////////////////////////////
/////////////  Required  Shader Features ////////////////////////////
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
#ifndef uint32_t
#define uint32_t uint
#endif
#define FLT_MAX 1e37f
#define FLT_MIN -1e37f
#define FLT_EPSILON 1e-6f
#define DEG_TO_RAD  0.017453293f
#define unmasked
#define half  float16_t
#define half2 f16vec2
#define half3 f16vec3
#define half4 f16vec4
bool  isfinite(float x)            { return !isinf(x); }
float copysign(float mag, float s) { return abs(mag)*sign(s); }
struct CamParameters    ///<! add any parameter you like to this structure
{
  float fov;
  float aspect;
  float nearPlane;
  float farPlane;
  int   spectralMode;
};
const float CAM_LAMBDA_MIN = 360.0f;
const float CAM_LAMBDA_MAX = 830.0f;
struct RayPart1 
{
  float    origin[3]; ///<! ray origin, x,y,z
  uint waves01;   ///<! Packed 2 first waves in 16 bit xy, fixed point; 0x0000 => CAM_LAMBDA_MIN; 0xFFFF => CAM_LAMBDA_MAX; wave[0] stored in less significant bits.
};
struct RayPart2 
{
  float    direction[3]; ///<! normalized ray direction, x,y,z
  uint waves23;      ///<! Packed 2 last waves in 16 bit xy, fixed point; 0x0000 => CAM_LAMBDA_MIN; 0xFFFF => CAM_LAMBDA_MAX; wave[1] stored in less significant bits.
};
const uint RAY_FLAG_IS_DEAD = 0x80000000;
const uint RAY_FLAG_OUT_OF_SCENE = 0x40000000;
const uint RAY_FLAG_HIT_LIGHT = 0x20000000;
const uint RAY_FLAG_HAS_NON_SPEC = 0x10000000;
const uint RAY_FLAG_HAS_INV_NORMAL = 0x08000000;
const float LAMBDA_MIN = 360.0f;
const float LAMBDA_MAX = 830.0f;
const uint SPECTRUM_SAMPLE_SZ = 4;
struct Lite_HitT
{
  float t;
  int   primId; 
  int   instId;
  int   geomId;
};
#define Lite_Hit Lite_HitT
struct SurfaceHitT
{
  vec3 pos;
  vec3 norm;
  vec2 uv;
};
#define SurfaceHit SurfaceHitT
const float GEPSILON = 1e-5f;
const float DEPSILON = 1e-20f;
struct MisData
{
  float matSamplePdf; ///< previous angle pdf (pdfW) that were used for sampling material. if < 0, then material sample was pure specular 
  float cosTheta;     ///< previous dot(matSam.dir, hit.norm)
  float ior;          ///< previous ior
  float dummy;        ///< dummy for 4 float
};
struct RandomGenT
{
  uvec2 state;

};
#define RandomGen RandomGenT

#ifndef SKIP_UBO_INCLUDE
#include "include/CamPinHole_pinhole_gpu_ubo.h"
#endif

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

mat4 translate4x4(vec3 delta)
{
  return mat4(vec4(1.0, 0.0, 0.0, 0.0),
              vec4(0.0, 1.0, 0.0, 0.0),
              vec4(0.0, 0.0, 1.0, 0.0),
              vec4(delta, 1.0));
}

mat4 rotate4x4X(float phi)
{
  return mat4(vec4(1.0f, 0.0f,  0.0f,           0.0f),
              vec4(0.0f, +cos(phi),  +sin(phi), 0.0f),
              vec4(0.0f, -sin(phi),  +cos(phi), 0.0f),
              vec4(0.0f, 0.0f,       0.0f,      1.0f));
}

mat4 rotate4x4Y(float phi)
{
  return mat4(vec4(+cos(phi), 0.0f, -sin(phi), 0.0f),
              vec4(0.0f,      1.0f, 0.0f,      0.0f),
              vec4(+sin(phi), 0.0f, +cos(phi), 0.0f),
              vec4(0.0f,      0.0f, 0.0f,      1.0f));
}

mat4 rotate4x4Z(float phi)
{
  return mat4(vec4(+cos(phi), sin(phi), 0.0f, 0.0f),
              vec4(-sin(phi), cos(phi), 0.0f, 0.0f),
              vec4(0.0f,      0.0f,     1.0f, 0.0f),
              vec4(0.0f,      0.0f,     0.0f, 1.0f));
}

mat4 inverse4x4(mat4 m) { return inverse(m); }
vec3 mul4x3(mat4 m, vec3 v) { return (m*vec4(v, 1.0f)).xyz; }
vec3 mul3x3(mat4 m, vec3 v) { return (m*vec4(v, 0.0f)).xyz; }

float SpectrumAverage(vec4 spec) {
  float sum = spec[0];
  for (uint i = 1; i < SPECTRUM_SAMPLE_SZ; ++i)
    sum += spec[int(i)];
  return sum / float(SPECTRUM_SAMPLE_SZ);
}

uint NextState(inout RandomGen gen) {
  const uint x = (gen.state).x * 17 + (gen.state).y * 13123;
  (gen.state).x = (x << 13) ^ x;
  (gen.state).y ^= (x << 7);
  return x;
}

vec3 XYZToRGB(vec3 xyz) {
  vec3 rgb;
  rgb[0] = +3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
  rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
  rgb[2] = +0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];

  return rgb;
}

vec3 EyeRayDirNormalized(float x, float y, mat4 a_mViewProjInv) {
  vec4 pos = vec4(2.0f*x - 1.0f,2.0f*y - 1.0f,0.0f,1.0f);
  pos = a_mViewProjInv * pos;
  pos /= pos.w;
  return normalize(pos.xyz);
}

float rndFloat1_Pseudo(inout RandomGen gen) {
  const uint x = NextState(gen);
  const uint tmp = (x * (x * x * 15731 + 74323) + 871483);
  const float scale      = (1.0f / 4294967296.0f);
  return (float((tmp)))*scale;
}

uvec2 unpackXY1616(uint packedIndex) {
  uvec2 res; 
  res.x = (packedIndex & 0x0000FFFF);         
  res.y = (packedIndex & 0xFFFF0000) >> 16;   
  return res;
}

vec4 SampleWavelengths(float u, float a, float b) {
  // pdf is 1.0f / (b - a)
  vec4 res;

  res[0] = mix(a, b, u);

  float delta = (b - a) / float(SPECTRUM_SAMPLE_SZ);
  for (uint i = 1; i < SPECTRUM_SAMPLE_SZ; ++i) 
  {
      res[int(i)] = res[i - 1] + delta;
      if (res[int(i)] > b)
        res[int(i)] = a + (res[int(i)] - b);
  }

  return res;
}

uint packXY1616(uint x, uint y) { return (y << 16u) | (x & 0x0000FFFF); }

#define KGEN_FLAG_RETURN            1
#define KGEN_FLAG_BREAK             2
#define KGEN_FLAG_DONT_SET_EXIT     4
#define KGEN_FLAG_SET_EXIT_NEGATIVE 8
#define KGEN_REDUCTION_LAST_STEP    16
#define BASIC_PROJ_LOGIC_H 
#define MAXFLOAT FLT_MAX
#define RTC_RANDOM 
#define CFLOAT_GUARDIAN 
#define SPECTRUM_H 

