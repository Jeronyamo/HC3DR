#pragma once

#include "cglobals.h"

static constexpr uint LIGHT_GEOM_RECT   = 1; 
static constexpr uint LIGHT_GEOM_DISC   = 2;
static constexpr uint LIGHT_GEOM_SPHERE = 3;
static constexpr uint LIGHT_GEOM_DIRECT = 4;
static constexpr uint LIGHT_GEOM_POINT  = 5;

static constexpr uint LIGHT_DIST_LAMBERT = 0;
static constexpr uint LIGHT_DIST_OMNI    = 1;

struct LightSource
{
  float4x4 matrix;    ///<! translation in matrix is always (0,0,0,1)
  float4   pos;       ///<! translation aclually stored here
  float4   intensity; ///<! brightress, i.e. screen value if light is visable directly
  float4   norm;
  
  float2   size;
  float    pdfA;
  uint     geomType;  ///<! LIGHT_GEOM_RECT, LIGHT_GEOM_DISC, LIGHT_GEOM_SPHERE, ...
  
  uint     distType;  ///<! LIGHT_DIST_LAMBERT, LIGHT_DIST_OMNI, ...
  uint     dummy1;
  uint     dummy2;
  uint     dummy3;

  uint     specId;
  uint     texId;
  uint     iesId;
  float    mult;
};

struct DLightSource
{
  float dI_dCx = 0.f, dI_dCy = 0.f, dI_dCz = 0.f, // center(pos)
        dI_dSx = 0.f, dI_dSy = 0.f; // size
  // uint n_samples = 0u;

  DLightSource operator*(float _v) const {
    return { dI_dCx * _v, dI_dCy * _v, dI_dCz * _v, dI_dSx * _v, dI_dSy * _v };
  }

  DLightSource operator+=(const DLightSource &_dls) {
    dI_dCx += _dls.dI_dCx;
    dI_dCy += _dls.dI_dCy;
    dI_dCz += _dls.dI_dCz;
    dI_dSx += _dls.dI_dSx;
    dI_dSy += _dls.dI_dSy;
    return *this;
  }
};

struct LightSample
{
  float3 pos;
  float3 norm;
  bool   isOmni;
};

static inline LightSample areaLightSampleRev(const LightSource* a_pLight, float2 rands)
{
  float2 sampleOff = 2.0f * (float2(-0.5f,-0.5f) + rands) * a_pLight[0].size;  // PLEASE! use 'a_pLight[0].' for a while ... , not a_pLight-> and not *(a_pLight[0])
  if(a_pLight[0].geomType == LIGHT_GEOM_DISC)
  {
    const float offsetX = rands.x * 2.0f - 1.0f;
    const float offsetY = rands.y * 2.0f - 1.0f;
    sampleOff = MapSamplesToDisc(float2(offsetX, offsetY))*a_pLight[0].size.x; 
  }
  const float3 samplePos = mul3x3(a_pLight[0].matrix, float3(sampleOff.x, 0.0f, sampleOff.y)) + to_float3(a_pLight[0].pos) + epsilonOfPos(to_float3(a_pLight[0].pos)) * to_float3(a_pLight[0].norm);
  LightSample res;
  res.pos    = samplePos;
  res.norm   = to_float3(a_pLight[0].norm);
  res.isOmni = false;
  return res;
}

static inline LightSample sphereLightSampleRev(const LightSource* a_pLight, float2 rands)
{
  const float theta = 2.0f * M_PI * rands.x;
  const float phi   = std::acos(1.0f - 2.0f * rands.y);
  const float x     = std::sin(phi) * std::cos(theta);
  const float y     = std::sin(phi) * std::sin(theta);
  const float z     = std::cos(phi);
  const float3 lcenter   = to_float3(a_pLight[0].pos);
  const float  lradius   = a_pLight[0].size.x;
  const float3 samplePos = lcenter + (lradius*1.000001f)*make_float3(x, y, z);
  LightSample res;
  res.pos  = samplePos;
  res.norm = normalize(samplePos - lcenter);
  return res;
}

static inline LightSample directLightSampleRev(const LightSource* a_pLight, float2 rands, float3 illuminationPoint)
{
  const float3 norm = to_float3(a_pLight[0].norm);
  LightSample res;
  res.pos    = illuminationPoint - norm*100000.0f;
  res.norm   = norm;
  res.isOmni = false;
  return res;
}

static inline LightSample pointLightSampleRev(const LightSource* a_pLight)
{
  LightSample res;
  res.pos    = to_float3(a_pLight[0].pos);
  res.norm   = to_float3(a_pLight[0].norm);
  res.isOmni = (a_pLight[0].distType == LIGHT_DIST_OMNI);
  return res;
}
