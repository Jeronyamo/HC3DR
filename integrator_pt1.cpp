#include "integrator_pt.h"
#include "include/crandom.h"
#include "diff_render/adam.h"

#include <chrono>
#include <string>
#include <omp.h>

#include "Image2d.h"
using LiteImage::Image2D;
using LiteImage::Sampler;
using LiteImage::ICombinedImageSampler;
using namespace LiteMath;

void Integrator::InitRandomGens(int a_maxThreads)
{
  m_randomGens.resize(a_maxThreads);
  #pragma omp parallel for default(shared)
  for(int i=0;i<a_maxThreads;i++)
    m_randomGens[i] = RandomGenInit(i);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void Integrator::kernel_InitEyeRay2(uint tid, const uint* packedXY, 
                                   float4* rayPosAndNear, float4* rayDirAndFar, float4* wavelengths, 
                                   float4* accumColor,    float4* accumuThoroughput,
                                   RandomGen* gen, uint* rayFlags, MisData* misData) // 
{
  if(tid >= m_maxThreadId)
    return;

  *accumColor        = make_float4(0,0,0,0);
  *accumuThoroughput = make_float4(1,1,1,1);
  RandomGen genLocal = m_randomGens[tid];
  *rayFlags          = 0;
  *misData           = makeInitialMisData();

  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;
  const float2 pixelOffsets = rndFloat2_Pseudo(&genLocal);
  
  float3 rayDir = EyeRayDirNormalized((float(x) + pixelOffsets.x)/float(m_winWidth), 
                                      (float(y) + pixelOffsets.y)/float(m_winHeight), m_projInv);
  float3 rayPos = float3(0,0,0);

  transform_ray3f(m_worldViewInv, &rayPos, &rayDir);
  
  float tmp = 0.0f;
  if(KSPEC_SPECTRAL_RENDERING !=0 && m_spectral_mode != 0)
  {
    float u = rndFloat1_Pseudo(&genLocal);
    *wavelengths = SampleWavelengths(u, LAMBDA_MIN, LAMBDA_MAX);
    tmp = u;
  }
  else
  {
    const uint32_t sample_sz = sizeof((*wavelengths).M) / sizeof((*wavelengths).M[0]);
    for (uint32_t i = 0; i < sample_sz; ++i) 
      (*wavelengths)[i] = 0.0f;
  }

  RecordPixelRndIfNeeded(pixelOffsets, tmp);
 
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, FLT_MAX);
  *gen           = genLocal;
}

void Integrator::kernel_InitEyeRayFromInput(uint tid, const RayPosAndW* in_rayPosAndNear, const RayDirAndT* in_rayDirAndFar,
                                            float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumuThoroughput, 
                                            RandomGen* gen, uint* rayFlags, MisData* misData, float4* wavelengths)
{
  if(tid >= m_maxThreadId)
    return;

  *accumColor        = make_float4(0,0,0,0);
  *accumuThoroughput = make_float4(1,1,1,1);
  *rayFlags          = 0;
  *misData           = makeInitialMisData();  

  //const int x = int(tid) % m_winWidth;
  //const int y = int(tid) / m_winHeight;

  const RayPosAndW rayPosData = in_rayPosAndNear[tid];
  const RayDirAndT rayDirData = in_rayDirAndFar[tid];

  float3 rayPos = float3(rayPosData.origin[0], rayPosData.origin[1], rayPosData.origin[2]);
  float3 rayDir = float3(rayDirData.direction[0], rayDirData.direction[1], rayDirData.direction[2]);
  transform_ray3f(m_worldViewInv, &rayPos, &rayDir);

  if(KSPEC_SPECTRAL_RENDERING !=0 && m_spectral_mode != 0)
  {
    *wavelengths = float4(rayPosData.wave);
    //const uint2 wavesXY = unpackXY1616(rayPosData.waves01);
    //const uint2 wavesZW = unpackXY1616(rayDirData.waves23);
    //const float scale = (1.0f/65535.0f)*(LAMBDA_MAX - LAMBDA_MIN);
    //*wavelengths = float4(float(wavesXY[0])*scale + LAMBDA_MIN,
    //                      float(wavesXY[1])*scale + LAMBDA_MIN,
    //                      float(wavesZW[0])*scale + LAMBDA_MIN,
    //                      float(wavesZW[1])*scale + LAMBDA_MIN);
  }
  else
    *wavelengths = float4(0,0,0,0);

  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, FLT_MAX);
  *gen           = m_randomGens[tid];
}


void Integrator::kernel_RayTrace2(uint tid, uint bounce, const float4* rayPosAndNear, const float4* rayDirAndFar,
                                 float4* out_hit1, float4* out_hit2, float4* out_hit3, uint* out_instId, uint* rayFlags)
{
  if(tid >= m_maxThreadId)
    return;
  uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;

  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  const CRT_Hit hit   = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);
  RecordRayHitIfNeeded(bounce, hit);

  if(hit.geomId != uint32_t(-1))
  {
    const float2 uv     = float2(hit.coords[0], hit.coords[1]);
    const float3 hitPos = to_float3(rayPos) + (hit.t*0.999999f)*to_float3(rayDir); // set hit slightlyt closer to old ray origin to prevent self-interseaction and e.t.c bugs
    
    // alternative, you may consider Johannes Hanika solution from  Ray Tracing Gems2  
    /////////////////////////////////////////////////////////////////////////////////
    // // get distance vectors from triangle vertices
    // vec3 tmpu = P - A, tmpv = P - B, tmpw = P - C
    // // project these onto the tangent planes
    // // defined by the shading normals
    // float dotu = min (0.0, dot(tmpu , nA))
    // float dotv = min (0.0, dot(tmpv , nB))
    // float dotw = min (0.0, dot(tmpw , nC))
    // tmpu -= dotu*nA
    // tmpv -= dotv*nB
    // tmpw -= dotw*nC
    // // finally P' is the barycentric mean of these three
    // vec3 Pp = P + u*tmpu + v*tmpv + w*tmpw
    /////////////////////////////////////////////////////////////////////////////////

    const uint triOffset  = m_matIdOffsets[hit.geomId];
    const uint vertOffset = m_vertOffset  [hit.geomId];
  
    const uint A = m_triIndices[(triOffset + hit.primId)*3 + 0];
    const uint B = m_triIndices[(triOffset + hit.primId)*3 + 1];
    const uint C = m_triIndices[(triOffset + hit.primId)*3 + 2];

    const float4 data1 = (1.0f - uv.x - uv.y)*m_vNorm4f[A + vertOffset] + uv.y*m_vNorm4f[B + vertOffset] + uv.x*m_vNorm4f[C + vertOffset];
    const float4 data2 = (1.0f - uv.x - uv.y)*m_vTang4f[A + vertOffset] + uv.y*m_vTang4f[B + vertOffset] + uv.x*m_vTang4f[C + vertOffset];

    float3 hitNorm     = to_float3(data1);
    float3 hitTang     = to_float3(data2);
    float2 hitTexCoord = float2(data1.w, data2.w);

    // transform surface point with matrix and flip normal if needed
    //
    hitNorm                = normalize(mul3x3(m_normMatrices[hit.instId], hitNorm));
    hitTang                = normalize(mul3x3(m_normMatrices[hit.instId], hitTang));
    const float flipNorm   = dot(to_float3(rayDir), hitNorm) > 0.001f ? -1.0f : 1.0f; // beware of transparent materials which use normal sign to identity "inside/outside" glass for example
    hitNorm                = flipNorm * hitNorm;
    hitTang                = flipNorm * hitTang; // do we need this ??
    
    if (flipNorm < 0.0f) currRayFlags |=  RAY_FLAG_HAS_INV_NORMAL;
    else                 currRayFlags &= ~RAY_FLAG_HAS_INV_NORMAL;

    const uint midOriginal = m_matIdByPrimId[m_matIdOffsets[hit.geomId] + hit.primId];
    const uint midRemaped  = RemapMaterialId(midOriginal, hit.instId);

    *rayFlags              = packMatId(currRayFlags, midRemaped);
    *out_hit1              = to_float4(hitPos,  hitTexCoord.x); 
    *out_hit2              = to_float4(hitNorm, hitTexCoord.y);
    *out_hit3              = to_float4(hitTang, 0.0f);
    *out_instId            = hit.instId;
  }
  else
    *rayFlags              = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_OUT_OF_SCENE);
}

float4 Integrator::GetLightSourceIntensity(uint a_lightId, const float4* a_wavelengths, float3 a_rayDir)
{
  float4 lightColor = m_lights[a_lightId].intensity;  
  if(KSPEC_SPECTRAL_RENDERING !=0 && m_spectral_mode != 0)
  {
    const uint specId = m_lights[a_lightId].specId;
  
    if(specId < 0xFFFFFFFF)
    {
      // lightColor = SampleSpectrum(m_spectra.data() + specId, *a_wavelengths);
      const uint2 data  = m_spec_offset_sz[specId];
      const uint offset = data.x;
      const uint size   = data.y;
      lightColor = SampleSpectrum(m_wavelengths.data() + offset, m_spec_values.data() + offset, *a_wavelengths, size);
    }
  }
  lightColor *= m_lights[a_lightId].mult;
  
  uint iesId = m_lights[a_lightId].iesId;
  if(iesId != uint(-1))
  {
    float sintheta        = 0.0f;
    const float2 texCoord = sphereMapTo2DTexCoord((-1.0f)*a_rayDir, &sintheta);
    const float4 texColor = m_textures[iesId]->sample(texCoord);
    lightColor *= texColor;
  }

  return lightColor;
}


float3 Integrator::SampleLightSourceByID(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar, 
                                         const float4* wavelengths, const float4* in_hitPart1, const float4* in_hitPart2, const float4* in_hitPart3,
                                         const uint* rayFlags, uint bounce, int lightId, const LightSample &lSam) {
  if(lightId < 0) {
    return {0.f, 0.f, 0.f};
  }

  const uint32_t matId = extractMatId(*rayFlags);
  const float3 ray_dir = to_float3(*rayDirAndFar);
  
  const float4 data1  = *in_hitPart1;
  const float4 data2  = *in_hitPart2;
  const float4 lambda = *wavelengths;

  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.tang = to_float3(*in_hitPart3);
  hit.uv   = float2(data1.w, data2.w);


  const float3 shadowRayDir = normalize(lSam.pos - hit.pos); // explicitSam.direction;
  const float3 shadowRayPos = hit.pos + hit.norm*std::max(maxcomp(hit.pos), 1.0f)*5e-6f;
  const float  hitDist = length(lSam.pos - shadowRayPos);
  const bool   inShadow  = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  const bool   inIllumArea = (dot(shadowRayDir, lSam.norm) < 0.0f);

  if(!inShadow && inIllumArea) 
  {
    const BsdfEval bsdfV    = MaterialEval(matId, lambda, shadowRayDir, (-1.0f)*ray_dir, hit.norm, hit.tang, hit.uv);
    float cosThetaOut       = std::max(dot(shadowRayDir, hit.norm), 0.0f);

    float _lightEvalPdfResOld = LightEvalPDF(lightId, shadowRayPos, shadowRayDir, lSam.pos, lSam.norm);
    // float hitDist2 = dot(lSam.pos - shadowRayPos, lSam.pos - shadowRayPos);
    float _lightEvalPdfRes = (m_lights[lightId].pdfA * hitDist * hitDist) / std::max(-dot(shadowRayDir, lSam.norm), 1e-30f);
    float  lgtPdfW         =   _lightEvalPdfRes / m_lights.size();

    float4 lightColor = m_lights[lightId].intensity * m_lights[lightId].mult;
    // const float4 lightColor = GetLightSourceIntensity(lightId, wavelengths, shadowRayDir);

    // return to_float3(lightColor) * std::max(-dot(shadowRayDir, lSam.norm), 1e-30f) / (hitDist * hitDist);
    return to_float3(lightColor * bsdfV.val) * (cosThetaOut / lgtPdfW);
    
    to_float3(lightColor * bsdfV.val) * ((std::max(dot(shadowRayDir, hit.norm), 0.0f)) *
        m_lights.size() * std::max(-dot(shadowRayDir, lSam.norm), 1e-30f) /
        (m_lights[lightId].pdfA * hitDist * hitDist));
  }
  return {};
}


void Integrator::kernel_SampleLightSource(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar, 
                                          const float4* wavelengths, const float4* in_hitPart1, const float4* in_hitPart2, const float4* in_hitPart3,
                                          const uint* rayFlags, uint bounce,
                                          RandomGen* a_gen, float4* out_shadeColor)
{
  if(tid >= m_maxThreadId)
    return;
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;
    
  const uint32_t matId = extractMatId(currRayFlags);
  const float3 ray_dir = to_float3(*rayDirAndFar);
  
  const float4 data1  = *in_hitPart1;
  const float4 data2  = *in_hitPart2;
  const float4 lambda = *wavelengths;

  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.tang = to_float3(*in_hitPart3);
  hit.uv   = float2(data1.w, data2.w);

  const float2 rands = rndFloat2_Pseudo(a_gen); // don't use single rndFloat4 (!!!)
  const float rndId  = rndFloat1_Pseudo(a_gen); // don't use single rndFloat4 (!!!)
  const int lightId  = std::min(int(std::floor(rndId * float(m_lights.size()))), int(m_lights.size() - 1u));
  RecordLightRndIfNeeded(bounce, lightId, rands);

  if(lightId < 0) // no lights or invalid light id
  {
    *out_shadeColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }
  
  const LightSample lSam = LightSampleRev(lightId, rands, hit.pos);
  const float  hitDist   = std::sqrt(dot(hit.pos - lSam.pos, hit.pos - lSam.pos));

  const float3 shadowRayDir = normalize(lSam.pos - hit.pos); // explicitSam.direction;
  const float3 shadowRayPos = hit.pos + hit.norm*std::max(maxcomp(hit.pos), 1.0f)*5e-6f; // TODO: see Ray Tracing Gems, also use flatNormal for offset
  const bool   inShadow     = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  const bool   inIllumArea  = (dot(shadowRayDir, lSam.norm) < 0.0f) || lSam.isOmni;

  RecordShadowHitIfNeeded(bounce, inShadow);

  if(!inShadow && inIllumArea) 
  {
    const BsdfEval bsdfV    = MaterialEval(matId, lambda, shadowRayDir, (-1.0f)*ray_dir, hit.norm, hit.tang, hit.uv);
    float cosThetaOut       = std::max(dot(shadowRayDir, hit.norm), 0.0f);
    
    float      lgtPdfW      = LightPdfSelectRev(lightId) * LightEvalPDF(lightId, shadowRayPos, shadowRayDir, lSam.pos, lSam.norm);
    float      misWeight    = (m_intergatorType == INTEGRATOR_MIS_PT) ? misWeightHeuristic(lgtPdfW, bsdfV.pdf) : 1.0f;
    const bool isDirect     = (m_lights[lightId].geomType == LIGHT_GEOM_DIRECT); 
    const bool isPoint      = (m_lights[lightId].geomType == LIGHT_GEOM_POINT); 
    
    if(isDirect)
    {
      misWeight = 1.0f;
      lgtPdfW   = 1.0f;
    }
    else if(isPoint)
      misWeight = 1.0f;

    if(m_skipBounce >= 1 && int(bounce) < int(m_skipBounce)-1) // skip some number of bounces if this is set
      misWeight = 0.0f;
    
    const float4 lightColor = GetLightSourceIntensity(lightId, wavelengths, shadowRayDir);
    *out_shadeColor = (lightColor * bsdfV.val / lgtPdfW) * cosThetaOut * misWeight;
  }
  else
    *out_shadeColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
}
// Func 2
void Integrator::kernel_NextBounce(uint tid, uint bounce, const float4* in_hitPart1, const float4* in_hitPart2, const float4* in_hitPart3, const uint* in_instId,
                                   const float4* in_shadeColor, float4* rayPosAndNear, float4* rayDirAndFar, const float4* wavelengths,
                                   float4* accumColor, float4* accumThoroughput, RandomGen* a_gen, MisData* misPrev, uint* rayFlags)
{
  if(tid >= m_maxThreadId)
    return;
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;
    
  const uint32_t matId = extractMatId(currRayFlags);

  // process surface hit case
  //
  const float3 ray_dir = to_float3(*rayDirAndFar);
  const float3 ray_pos = to_float3(*rayPosAndNear);
  const float4 lambda  = *wavelengths;
  
  const float4 data1 = *in_hitPart1;
  const float4 data2 = *in_hitPart2;
  
  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.tang = to_float3(*in_hitPart3);
  hit.uv   = float2(data1.w, data2.w);
  
  const MisData prevBounce = *misPrev;
  const float   prevPdfW   = prevBounce.matSamplePdf;

  // process light hit case
  //
  if(m_materials[matId].mtype == MAT_TYPE_LIGHT_SOURCE)
  {
    const uint   texId     = m_materials[matId].texid[0];
    const float2 texCoordT = mulRows2x4(m_materials[matId].row0[0], m_materials[matId].row1[0], hit.uv);
    const float4 texColor  = m_textures[texId]->sample(texCoordT);
    const uint   lightId   = m_instIdToLightInstId[*in_instId]; 
    
    const float4 emissColor = m_materials[matId].colors[EMISSION_COLOR];
    float4 lightIntensity   = emissColor * texColor;

    if(lightId != 0xFFFFFFFF)
    {
      const float lightCos = dot(to_float3(*rayDirAndFar), to_float3(m_lights[lightId].norm));
      const float lightDirectionAtten = (lightCos < 0.0f || m_lights[lightId].geomType == LIGHT_GEOM_SPHERE) ? 1.0f : 0.0f;
      lightIntensity = GetLightSourceIntensity(lightId, wavelengths, to_float3(*rayDirAndFar))*lightDirectionAtten;
    }

    float misWeight = 1.0f;
    if(m_intergatorType == INTEGRATOR_MIS_PT) 
    {
      if(bounce > 0)
      {
        if(lightId != 0xFFFFFFFF)
        {
          const float lgtPdf  = LightPdfSelectRev(lightId) * LightEvalPDF(lightId, ray_pos, ray_dir, hit.pos, hit.norm);
          misWeight           = misWeightHeuristic(prevPdfW, lgtPdf);
          if (prevPdfW <= 0.0f) // specular bounce
            misWeight = 1.0f;
        }
      }
    }
    else if(m_intergatorType == INTEGRATOR_SHADOW_PT && hasNonSpecular(currRayFlags))
      misWeight = 0.0f;
    
    if(m_skipBounce >= 1 && bounce < m_skipBounce) // skip some number of bounces if this is set
      misWeight = 0.0f;

    float4 currAccumColor      = *accumColor;
    float4 currAccumThroughput = *accumThoroughput;
    
    currAccumColor += currAccumThroughput * lightIntensity * misWeight;
   
    *accumColor = currAccumColor;
    *rayFlags   = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_HIT_LIGHT);
    return;
  }
  
  const uint bounceTmp    = bounce;
  const BsdfSample matSam = MaterialSampleAndEval(matId, bounceTmp, lambda, a_gen, (-1.0f)*ray_dir, hit.norm, hit.tang, hit.uv, misPrev, currRayFlags);
  const float4 bxdfVal    = matSam.val * (1.0f / std::max(matSam.pdf, 1e-20f));
  const float  cosTheta   = std::abs(dot(matSam.dir, hit.norm)); 

  MisData nextBounceData      = *misPrev;        // remember current pdfW for next bounce
  nextBounceData.matSamplePdf = (matSam.flags & RAY_EVENT_S) != 0 ? -1.0f : matSam.pdf; 
  nextBounceData.cosTheta     = cosTheta;   
  *misPrev                    = nextBounceData;

  if(m_intergatorType == INTEGRATOR_STUPID_PT)
  {
    *accumThoroughput *= cosTheta * bxdfVal; 
  }
  else if(m_intergatorType == INTEGRATOR_SHADOW_PT || m_intergatorType == INTEGRATOR_MIS_PT)
  {
    const float4 currThoroughput = *accumThoroughput;
    const float4 shadeColor      = *in_shadeColor;
    float4 currAccumColor        = *accumColor;

    currAccumColor += currThoroughput * shadeColor;
    // currAccumColor.x += currThoroughput.x * shadeColor.x;
    // currAccumColor.y += currThoroughput.y * shadeColor.y;
    // currAccumColor.z += currThoroughput.z * shadeColor.z;
    // if(bounce > 0)
    //   currAccumColor.w *= prevPdfA;

    *accumColor       = currAccumColor;
    *accumThoroughput = currThoroughput*cosTheta*bxdfVal; 
  }

  *rayPosAndNear = to_float4(OffsRayPos(hit.pos, hit.norm, matSam.dir), 0.0f); // todo: use flatNormal for offset
  *rayDirAndFar  = to_float4(matSam.dir, FLT_MAX);
  *rayFlags      = currRayFlags | matSam.flags;
}

void Integrator::kernel_HitEnvironment(uint tid, const uint* rayFlags, const float4* rayDirAndFar, const MisData* a_prevMisData, const float4* accumThoroughput,
                                       float4* accumColor)
{
  if(tid >= m_maxThreadId)
    return;
  const uint currRayFlags = *rayFlags;
  if(!isOutOfScene(currRayFlags))
    return;
  
  // TODO: HDRI maps
  const float4 envData  = GetEnvironmentColorAndPdf(to_float3(*rayDirAndFar));
  // const float3 envColor = to_float3(envData)/envData.w;    // explicitly account for pdf; when MIS will be enabled, need to deal with MIS weight also!

  const float4 envColor = envData;
  if(m_intergatorType == INTEGRATOR_STUPID_PT)     // todo: when explicit sampling will be added, disable contribution here for 'INTEGRATOR_SHADOW_PT'
    *accumColor = (*accumThoroughput) * envColor;
  else
    *accumColor += (*accumThoroughput) * envColor;
}


void Integrator::kernel_ContributeToImage(uint tid, uint channels, const float4* a_accumColor, const RandomGen* gen, const uint* in_pakedXY,
                                          const float4* wavelengths, float* out_color)
{
  
  if(tid >= m_maxThreadId) // don't contrubute to image in any "record" mode
    return;
  
  m_randomGens[tid] = *gen;
  if(m_disableImageContrib !=0)
    return;

  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;
  
  float4 specSamples = *a_accumColor; 
  float4 tmpVal      = specSamples*m_camRespoceRGB;
  float3 rgb         = to_float3(tmpVal);
  if(KSPEC_SPECTRAL_RENDERING!=0 && m_spectral_mode != 0) 
  {
    float4 waves = *wavelengths;
    
    if(m_camResponseSpectrumId[0] < 0)
    {
      const float3 xyz = SpectrumToXYZ(specSamples, waves, LAMBDA_MIN, LAMBDA_MAX, m_cie_x.data(), m_cie_y.data(), m_cie_z.data());
      rgb = XYZToRGB(xyz);
    }
    else
    {
      float4 responceX, responceY, responceZ;
      {
        int specId = m_camResponseSpectrumId[0];
        if(specId >= 0)
        {
          const uint2 data  = m_spec_offset_sz[specId];
          const uint offset = data.x;
          const uint size   = data.y;
          responceX = SampleSpectrum(m_wavelengths.data() + offset, m_spec_values.data() + offset, waves, size);
        }
        else
          responceX = float4(1,1,1,1);

        specId = m_camResponseSpectrumId[1];
        if(specId >= 0)
        {
          const uint2 data  = m_spec_offset_sz[specId];
          const uint offset = data.x;
          const uint size   = data.y;
          responceY = SampleSpectrum(m_wavelengths.data() + offset, m_spec_values.data() + offset, waves, size);
        }
        else
          responceY = responceX;

        specId = m_camResponseSpectrumId[2];
        if(specId >= 0)
        {
          const uint2 data  = m_spec_offset_sz[specId];
          const uint offset = data.x;
          const uint size   = data.y;
          responceZ = SampleSpectrum(m_wavelengths.data() + offset, m_spec_values.data() + offset, waves, size);
        }
        else
          responceZ = responceY;
      }

      float3 xyz = float3(0,0,0);
      for (uint32_t i = 0; i < SPECTRUM_SAMPLE_SZ; ++i) {
        xyz.x += specSamples[i]*responceX[i];
        xyz.y += specSamples[i]*responceY[i];
        xyz.z += specSamples[i]*responceZ[i]; 
      } 

      if(m_camResponseType == CAM_RESPONCE_XYZ)
        rgb = XYZToRGB(xyz);
      else
        rgb = xyz;
    }
  }

  float4 colorRes = m_exposureMult * to_float4(rgb, 1.0f);
  //if(x == 415 && (y == 256-130-1))
  //{
  //  int a = 2;
  //  //colorRes = float4(1,0,0,0);
  //}
  
  if(channels == 1)
  {
    const float mono = 0.2126f*colorRes.x + 0.7152f*colorRes.y + 0.0722f*colorRes.z;
    out_color[y*m_winWidth+x] += mono;
  }
  else if(channels <= 4)
  {
    out_color[(y*m_winWidth+x)*channels + 0] += colorRes.x;
    out_color[(y*m_winWidth+x)*channels + 1] += colorRes.y;
    out_color[(y*m_winWidth+x)*channels + 2] += colorRes.z;
  }
  else
  {
    auto waves = (*wavelengths);
    auto color = (*a_accumColor)*m_exposureMult;
    for(int i=0;i<4;i++) {
      const float t         = (waves[i] - LAMBDA_MIN)/(LAMBDA_MAX-LAMBDA_MIN);
      const int channelId   = std::min(int(float(channels)*t), int(channels)-1);
      const int offsetPixel = int(y)*m_winWidth + int(x);
      const int offsetLayer = channelId*m_winWidth*m_winHeight;
      out_color[offsetLayer + offsetPixel] += color[i];
    }
  }

}

void Integrator::kernel_CopyColorToOutput(uint tid, uint channels, const float4* a_accumColor, const RandomGen* gen, float* out_color)
{
  if(tid >= m_maxThreadId)
    return;
  
  const float4 color = *a_accumColor;

  if(channels == 4)
  {
    out_color[tid*4+0] += color[0];
    out_color[tid*4+1] += color[1];
    out_color[tid*4+2] += color[2];
    out_color[tid*4+3] += color[3];
  }
  else if(channels == 1)
    out_color[tid] += color[0];

  m_randomGens[tid] = *gen;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Integrator::NaivePathTrace(uint tid, uint channels, float* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  float4 wavelengths;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags;
  kernel_InitEyeRay2(tid, m_packedXY.data(), &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &rayFlags, &mis);

  for(uint depth = 0; depth < m_traceDepth + 1; ++depth) // + 1 due to NaivePT uses additional bounce to hit light 
  {
    float4 shadeColor, hitPart1, hitPart2, hitPart3;
    uint instId = 0;
    kernel_RayTrace2(tid, depth, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &hitPart3, &instId, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &hitPart3, &instId, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &mis, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
  }

  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
                       &accumColor);

  kernel_ContributeToImage(tid, channels, &accumColor, &gen, m_packedXY.data(), &wavelengths, 
                           out_color);
}

// This one
void Integrator::PathTrace(uint tid, uint channels, float* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  float4 wavelengths;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags;
  kernel_InitEyeRay2(tid, m_packedXY.data(), &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &rayFlags, &mis);

  for(uint depth = 0; depth < 1; depth++) 
  {
    float4   shadeColor, hitPart1, hitPart2, hitPart3;
    uint instId;
    kernel_RayTrace2(tid, depth, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &hitPart3, &instId, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_SampleLightSource(tid, &rayPosAndNear, &rayDirAndFar, &wavelengths, &hitPart1, &hitPart2, &hitPart3, &rayFlags, depth,
                             &gen, &shadeColor);

    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &hitPart3, &instId, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &mis, &rayFlags);

    if(isDeadRay(rayFlags))
      break;
  }

  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
                        &accumColor);

  kernel_ContributeToImage(tid, channels, &accumColor, &gen, m_packedXY.data(), &wavelengths, out_color);
}

// some simple derivatives

struct DCross {
  static float3 cross(const float3 &v0, const float3 &v1) {
    return { v0.y * v1.z - v0.z * v1.y,
             v0.z * v1.x - v0.x * v1.z,
             v0.x * v1.y - v0.y * v1.x };
  }

// v0 derivatives
  static float3 crossD0X(const float3 &v0, const float3 &v1) { return { 0.f, -v1.z, v1.y }; }
  static float3 crossD0Y(const float3 &v0, const float3 &v1) { return { v1.z, 0.f, -v1.x }; }
  static float3 crossD0Z(const float3 &v0, const float3 &v1) { return { -v1.y, v1.x, 0.f }; }

// v1 derivatives
  static float3 crossD1X(const float3 &v0, const float3 &v1) { return { 0.f, v0.z, -v0.y }; }
  static float3 crossD1Y(const float3 &v0, const float3 &v1) { return { -v0.z, 0.f, v0.x }; }
  static float3 crossD1Z(const float3 &v0, const float3 &v1) { return { v0.y, -v0.x, 0.f }; }
};


struct DDot {
    static float dot(const float3 &v0, const float3 &v1) {
      return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
    }

// v0 derivatives
  static float dotD0X(const float3 &v0, const float3 &v1) { return v1.x; }
  static float dotD0Y(const float3 &v0, const float3 &v1) { return v1.y; }
  static float dotD0Z(const float3 &v0, const float3 &v1) { return v1.z; }

// v1 derivatives
  static float dotD1X(const float3 &v0, const float3 &v1) { return v0.x; }
  static float dotD1Y(const float3 &v0, const float3 &v1) { return v0.y; }
  static float dotD1Z(const float3 &v0, const float3 &v1) { return v0.z; }
};

struct DLightRectParams {
  static float3 pt(const float3 &_center, const float2 &_size, int _edge) {
    switch (_edge) {
      case (4):
      case (0): {
        //--
        return _center + make_float3(-_size.y, 0.f, -_size.x);
      }
      case (1): {
        //-+
        return _center + make_float3(-_size.y, 0.f,  _size.x);
      }
      case (2): {
        //++
        return _center + make_float3( _size.y, 0.f,  _size.x);
      }
      case (3): {
        //+-
        return _center + make_float3( _size.y, 0.f, -_size.x);
      }
    }
    throw std::runtime_error("Error: invalid parameter (how?)\n");
  }

  // don't forget to multiply by (1-t)/t:  v0*(1-_t) + v1*_t
  // dcenter.x, dcenter.y, dcenter.z - it's an identical matrix, so multiply independently
  static float3 dcenterx() {
    return {1.f, 0.f, 0.f};
  }
  static float3 dcentery() {
    return {0.f, 1.f, 0.f};
  }
  static float3 dcenterz() {
    return {0.f, 0.f, 1.f};
  }

  // don't forget to multiply by (1-t)/t:  v0*(1-_t) + v1*_t
  static float3 dsizey(int _edge) {
    switch (_edge) {
      case (4):
      case (0): {
        return {-1.f, 0.f, 0.f};
      }
      case (1): {
        return {-1.f, 0.f, 0.f};
      }
      case (2): {
        return { 1.f, 0.f, 0.f};
      }
      case (3): {
        return { 1.f, 0.f, 0.f};
      }
    }
    throw std::runtime_error("Error: invalid parameter (how?)\n");
  }
  // don't forget to multiply by (1-t)/t:  v0*(1-_t) + v1*_t
  static float3 dsizex(int _edge) {
    switch (_edge) {
      case (4):
      case (0): {
        return { 0.f, 0.f, -1.f};
      }
      case (1): {
        return { 0.f, 0.f,  1.f};
      }
      case (2): {
        return { 0.f, 0.f,  1.f};
      }
      case (3): {
        return { 0.f, 0.f, -1.f};
      }
    }
    throw std::runtime_error("Error: invalid parameter (how?)\n");
  }
};


// Surface equation for secondary edges from the article, returns float
struct DEdgeEquationSec3D {
  static float a(const float3 &_p, const float3 &_m, const float3 &v0, const float3 &v1) {
    return dot(_m - _p, cross(v0 - _p, v1 - _p));
  }
};

struct DNorm {
    static float3 normalize(const float3 &_v) {
      float __len = 1.f / sqrt(_v.x*_v.x + _v.y*_v.y + _v.z*_v.z);
      return { _v.x * __len, _v.y * __len, _v.z * __len };
    }

    static float3 dx(const float3 &_v) {
      float __len = 1.f / sqrt(_v.x*_v.x + _v.y*_v.y + _v.z*_v.z);
      __len = __len * __len * __len;
      return { (_v.y*_v.y + _v.z*_v.z) * __len,
              -(_v.x*_v.y) * __len,
              -(_v.x*_v.z) * __len};
    }

    static float3 dy(const float3 &_v) {
      float __len = 1.f / sqrt(_v.x*_v.x + _v.y*_v.y + _v.z*_v.z);
      __len = __len * __len * __len;
      return {-(_v.y*_v.x) * __len,
               (_v.x*_v.x + _v.z*_v.z) * __len,
              -(_v.y*_v.z) * __len};
    }

    static float3 dz(const float3 &_v) {
      float __len = 1.f / sqrt(_v.x*_v.x + _v.y*_v.y + _v.z*_v.z);
      __len = __len * __len * __len;
      return {-(_v.z*_v.x) * __len,
              -(_v.z*_v.y) * __len,
               (_v.x*_v.x + _v.y*_v.y) * __len};
    }
};



struct ImgMSE {
  std::vector<float> grad;
  uint w, h, params;

  ImgMSE() {}
  ImgMSE(uint _width, uint _height, uint _parameters) :
      w{_width}, h{_height}, params{_parameters} {
    grad.resize(w * h * params);
  }


  void nullify() {
    std::fill(grad.begin(), grad.end(), 0.0f);
  }

  void add(uint _x, uint _y, const DLightSource &_params, const float3 &_f_diff) {
    uint _offset = (_y * w + _x) * params;
    float3 _tmp{_params.dI_dCx * _f_diff};

    grad[_offset    ] += _tmp.x;
    grad[_offset + 1] += _tmp.y;
    grad[_offset + 2] += _tmp.z;

    _tmp = _params.dI_dCy * _f_diff;
    grad[_offset + 3] += _tmp.x;
    grad[_offset + 4] += _tmp.y;
    grad[_offset + 5] += _tmp.z;

    _tmp = _params.dI_dCz * _f_diff;
    grad[_offset + 6] += _tmp.x;
    grad[_offset + 7] += _tmp.y;
    grad[_offset + 8] += _tmp.z;

    _tmp = _params.dI_dSx * _f_diff;
    grad[_offset +  9] += _tmp.x;
    grad[_offset + 10] += _tmp.y;
    grad[_offset + 11] += _tmp.z;

    _tmp = _params.dI_dSy * _f_diff;
    grad[_offset + 12] += _tmp.x;
    grad[_offset + 13] += _tmp.y;
    grad[_offset + 14] += _tmp.z;
  }

  float dmse(const float *_img, const float *_ref, DLightSource &_par) {
    DLightSource _dloss{0.f, 0.f, 0.f, 0.f, 0.f};
    float _loss = 0.f;

    for (uint y = 1u; y < h - 1; ++y) {
      for (uint x = 1u; x < w - 1; ++x) {
        float3 _colDiff{0.f};
        DLightSource _tmp{0.f, 0.f, 0.f, 0.f, 0.f};
        
        // // no convolution
        // _colDiff = colDiff(x    , y    , _img, _ref);

        // 3x3 Gauss
        _colDiff  = colDiff(x - 1, y - 1, _img, _ref) * 0.0625f;
        _colDiff += colDiff(x    , y - 1, _img, _ref) * 0.125f;
        _colDiff += colDiff(x + 1, y - 1, _img, _ref) * 0.0625f;
        _colDiff += colDiff(x - 1, y    , _img, _ref) * 0.125f;
        _colDiff += colDiff(x    , y    , _img, _ref) * 0.25f;
        _colDiff += colDiff(x + 1, y    , _img, _ref) * 0.125f;
        _colDiff += colDiff(x - 1, y + 1, _img, _ref) * 0.0625f;
        _colDiff += colDiff(x    , y + 1, _img, _ref) * 0.125f;
        _colDiff += colDiff(x + 1, y + 1, _img, _ref) * 0.0625f;
        _loss += dot(_colDiff, _colDiff);
        _tmp = pixelGrad(x, y, _colDiff);
        _dloss += _tmp;
      }
    }
    _par = _dloss;
    return _loss;
  }

  float3 colDiff(uint _x, uint _y, const float *_img, const float *_ref) {
    uint ind = _y * w + _x;

    return float3{ _img[ind * 4], _img[ind * 4 + 1], _img[ind * 4 + 2] } -
           float3{ _ref[ind * 4], _ref[ind * 4 + 1], _ref[ind * 4 + 2] };
  }

  DLightSource pixelGrad(uint _x, uint _y, const float3 &_col_diff) {
    uint ind = _y * w + _x;
    DLightSource _tmp{0.f, 0.f, 0.f, 0.f, 0.f};

    _tmp.dI_dCx = dot(float3{ grad[ind * params    ],
                              grad[ind * params + 1],
                              grad[ind * params + 2] }, _col_diff);
    _tmp.dI_dCy = dot(float3{ grad[ind * params + 3],
                              grad[ind * params + 4],
                              grad[ind * params + 5] }, _col_diff);
    _tmp.dI_dCz = dot(float3{ grad[ind * params + 6],
                              grad[ind * params + 7],
                              grad[ind * params + 8] }, _col_diff);
    _tmp.dI_dSx = dot(float3{ grad[ind * params + 9],
                              grad[ind * params + 10],
                              grad[ind * params + 11] }, _col_diff);
    _tmp.dI_dSy = dot(float3{ grad[ind * params + 12],
                              grad[ind * params + 13],
                              grad[ind * params + 14] }, _col_diff);
    return _tmp;
  }
};

uint2 Integrator::getImageIndicesSomehow(const float3 &_pos, const float3 &_dir) {
  uint2 _res{0u, 0u};
  float3 pos2 = _pos + 100.f * _dir;

  float4 pos = m_proj * to_float4(normalize(mul4x3(m_worldView, pos2)), 1.f);
  pos /= pos.w;
  _res.x = floorf((pos.x + 1) * (0.5 * m_winWidth));
  _res.y = floorf((pos.y + 1) * (0.5 * m_winHeight));
  return _res;
}

float2 Integrator::getImageSScoords(const float3 &_pos, const float3 &_dir) {
  float2 _res{0u, 0u};
  float3 pos2 = _pos + 100.f * _dir;

  float4 pos = m_proj * to_float4(normalize(mul4x3(m_worldView, pos2)), 1.f);
  pos /= pos.w;
  _res.x = (pos.x + 1) * (0.5 * m_winWidth);
  _res.y = (pos.y + 1) * (0.5 * m_winHeight);
  return _res;
}

float3 Integrator::dirFromSScoords(float _x, float _y) {
  const float3 rayPos2 = mul4x3(m_worldViewInv, float3(0,0,0));

  float4 __pos = float4(2.f * _x / m_winWidth  - 1.f,
                        2.f * _y / m_winHeight - 1.f, 0.f, 1.f );
  __pos = m_projInv * __pos;
  float3 rayDir2 = normalize(float3{ __pos.x / __pos.w, __pos.y / __pos.w, __pos.z / __pos.w });
  float3 _pos2 = mul4x3(m_worldViewInv, 100.f*rayDir2);
  return normalize(_pos2 - rayPos2);
}



// change this for WAS; from IntegratorDR::kernel_CalcRayColor
void Integrator::getColorAfterIntersection(uint tid, const Lite_Hit* in_hit,
                                     const float2* bars, float4* finalColor) { 
  if (tid >= m_maxThreadId) return;

  const Lite_Hit hit = *in_hit;
  if (hit.geomId == -1) return;

  const uint32_t matId  = m_matIdByPrimId[m_matIdOffsets[hit.geomId] + hit.primId];
  const float4 mdata    = m_materials[matId].colors[GLTF_COLOR_BASE];
  const float2 uv       = *bars;

  const uint triOffset  = m_matIdOffsets[hit.geomId];
  const uint vertOffset = m_vertOffset  [hit.geomId];

  const uint A = m_triIndices[(triOffset + hit.primId)*3 + 0];
  const uint B = m_triIndices[(triOffset + hit.primId)*3 + 1];
  const uint C = m_triIndices[(triOffset + hit.primId)*3 + 2];
  const float4 data1 = (1.0f - uv.x - uv.y)*m_vNorm4f[A + vertOffset] + uv.y*m_vNorm4f[B + vertOffset] + uv.x*m_vNorm4f[C + vertOffset];
  const float4 data2 = (1.0f - uv.x - uv.y)*m_vTang4f[A + vertOffset] + uv.y*m_vTang4f[B + vertOffset] + uv.x*m_vTang4f[C + vertOffset];
  float3 hitNorm     = to_float3(data1);
  float3 hitTang     = to_float3(data2);
  float2 hitTexCoord = float2(data1.w, data2.w);

  const uint   texId     = m_materials[matId].texid[0];
  const float2 texCoordT = mulRows2x4(m_materials[matId].row0[0], m_materials[matId].row1[0], hitTexCoord);
  const float4 texColor  = m_textures[texId]->sample(texCoordT);
  float3 color = (mdata.w > 0.0f) ? clamp(float3(mdata.w,mdata.w,mdata.w), 0.0f, 1.0f) : to_float3(mdata*texColor);
  (*finalColor) = to_float4(color, 0);
}

// from IntegratorDR::CastRayDR
float3 Integrator::getColor1(const float3 &_pos, const float3 &_dir) {
  float4 rayPosAndNear = to_float4(_pos,    0.0f),
         rayDirAndFar  = to_float4(_dir, FLT_MAX);

  Lite_Hit hit;
  float2   baricentrics; 
  if(!kernel_RayTrace(0, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
    return float3(0,0,0);

  float4 finalColor;
  getColorAfterIntersection(0, &hit, &baricentrics, &finalColor);
  return to_float3(finalColor);
}

float3 Integrator::linearToSRGB(float3 _col) {
  for (int i = 0; i < 3; ++i) {
    _col[i] = clamp(_col[i], 0.0f, 1.0f);

    // copy of linearToSRGB - I need final image for correct loss function calculation
    if(_col[i] <= 0.00313066844250063f)
      _col[i] *= 12.92f;
    else
      _col[i] = 1.055f * std::pow(_col[i], 1.0f/2.4f) - 0.055f;
  }
  return _col;
}


// from Integrator::PathTraceFromInputRays (at the bottom)
float3 Integrator::getColor2(const float3 &_pos, const float3 &_dir) {
  float4 accumColor { 0.f, 0.f, 0.f, 0.f }, accumThroughput{ 1.f, 1.f, 1.f, 1.f };
  float4 wavelengths{ 0.f, 0.f, 0.f, 0.f };
  MisData mis = makeInitialMisData();
  uint rayFlags = 0;

  float4 rayPosAndNear = to_float4(_pos,    0.0f),
         rayDirAndFar  = to_float4(_dir, FLT_MAX);

  uint2 __ind = getImageIndicesSomehow(to_float3(rayPosAndNear), to_float3(rayDirAndFar));
  RandomGen gen = m_randomGens[__ind.y * m_winWidth + __ind.x];
  float4   shadeColor, hitPart1, hitPart2, hitPart3;
  uint instId;

  // doesn't depend on tid
  kernel_RayTrace2(0, 0, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &hitPart3, &instId, &rayFlags);
  if(isDeadRay(rayFlags)) return linearToSRGB(to_float3(accumColor));

  kernel_SampleLightSource(0, &rayPosAndNear, &rayDirAndFar, &wavelengths, &hitPart1, &hitPart2, &hitPart3, &rayFlags, 0,
                            &gen, &shadeColor);
  kernel_NextBounce(0, 0, &hitPart1, &hitPart2, &hitPart3, &instId, &shadeColor,
                    &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &mis, &rayFlags);
  if(isDeadRay(rayFlags)) return linearToSRGB(to_float3(accumColor));

  kernel_HitEnvironment(0, &rayFlags, &rayDirAndFar, &mis, &accumThroughput, &accumColor);
  return linearToSRGB(to_float3(accumColor));
}


// _v - point in 3D, _dv_ss - derivative of its screen space projection
float3 Integrator::projectSSderivatives(const float3 & _v, const float2 &_dv_ss) {
  const float3 _pos = mul4x3(m_worldViewInv, float3(0,0,0));
// 1
  float3 _dir = normalize(_v - _pos);
// 2
  float _coef = 100.f;
  float3 pos2 = _pos + _coef * _dir;
// 3
  float3 rayDirNonnorm = mul4x3(m_worldView, pos2);
  float3 rayDir = normalize(rayDirNonnorm);
// 4
  float4 pos = m_proj * to_float4(8 * rayDir, 1.f);
// 5
  pos.x /= pos.w;
  pos.y /= pos.w;
// 6
  float2 _res{0.f};
  _res.x = (pos.x + 1.f) * 0.5f * m_winWidth ;
  _res.y = (pos.y + 1.f) * 0.5f * m_winHeight;

//---------------------------------------------------
// -6
  float4 dpos{0.f};
  dpos.x = _dv_ss.x * 0.5f * m_winWidth ;
  dpos.y = _dv_ss.y * 0.5f * m_winHeight;
// -5
  dpos = float4{dpos.x / pos.w, dpos.y / pos.w, 0.f, -(dpos.x * pos.x + dpos.y * pos.y) / pos.w}; // should denom be squared for w?
// -4
  float3 DrayDir{ dot(m_proj.col(0), dpos), dot(m_proj.col(1), dpos), dot(m_proj.col(2), dpos) };
  DrayDir *= 8;
// -3
  float4 DrayDirNonnorm{ dot(DrayDir, DNorm::dx(rayDirNonnorm)),
                         dot(DrayDir, DNorm::dy(rayDirNonnorm)),
                         dot(DrayDir, DNorm::dz(rayDirNonnorm)),
                         0.f }; // derivative is 0
  float3 dpos2{ dot(m_worldView.col(0), DrayDirNonnorm),
                dot(m_worldView.col(1), DrayDirNonnorm),
                dot(m_worldView.col(2), DrayDirNonnorm) };
// -2
  float3 ddir = _coef * dpos2;
// -1
  float3 dv{ dot(ddir, DNorm::dx(_v - _pos)),
             dot(ddir, DNorm::dy(_v - _pos)),
             dot(ddir, DNorm::dz(_v - _pos)) };
  return dv;
}

float Integrator::sampleSSfrom2Dpoints(const float2 *_v_ss, uint _v_size, int &_edge_int) {
  float _cumulative_len[_v_size + 1];
  _cumulative_len[0] = 0.f;

  for (int i = 1; i < _v_size + 1; ++i) {
    _cumulative_len[i] = length(_v_ss[i % _v_size] - _v_ss[i - 1]) + _cumulative_len[i - 1];
  }

  double _t = (double(std::rand()) / RAND_MAX) * _cumulative_len[_v_size];

  for (int i = 0; i < _v_size; ++i)
    if (_t >= _cumulative_len[i] && _t < _cumulative_len[i + 1]) {
      _edge_int = i;
      return (_t - _cumulative_len[i]) / (_cumulative_len[i + 1] - _cumulative_len[i]);
    }
  printf("Sampling error!\n");
  return 0.f;
}

float Integrator::sampleEdgePtfrom3Dpoints(const float3 *_v_ss, uint _v_size, int &_edge_int) {
  float _cumulative_len[_v_size + 1];
  _cumulative_len[0] = 0.f;

  for (int i = 1; i < _v_size + 1; ++i) {
    _cumulative_len[i] = length(_v_ss[i % _v_size] - _v_ss[i - 1]) + _cumulative_len[i - 1];
  }

  double _t = (double(std::rand()) / RAND_MAX) * _cumulative_len[_v_size];

  for (int i = 0; i < _v_size; ++i)
    if (_t >= _cumulative_len[i] && _t < _cumulative_len[i + 1]) {
      _edge_int = i;
      return (_t - _cumulative_len[i]) / (_cumulative_len[i + 1] - _cumulative_len[i]);
    }
  printf("Sampling error (3D)!\n");
  return 0.f;
}

float Integrator::LightEdgeSamplingStep(float* out_color, const float* a_refImg, float* a_DerivPosImg,
                                        float* a_DerivNegImg, uint _iter, const std::vector<DLightSource> &_shadowDerivative) {
  const uint num_samples = 512u;
  const float norm_coef = 1.f / (num_samples);
  float _loss = 0.f;
  ImgMSE _MSE{uint(m_winWidth), uint(m_winHeight), 16u};

  for (uint i = 0u; i < m_lightInst.size(); ++i) {
    DLightSourceUpdater _lsu = m_lightInst[i];
    LightSource _ls = m_lights[_lsu.lightID];
    float3 _p = mul4x3(m_worldViewInv, float3(0,0,0));
    float3 _center = to_float3(_ls.pos);

    _MSE.nullify();

    if (_ls.geomType == LIGHT_GEOM_RECT) {
      // Rectangle. Edges 01-12-23-30

      // v0 -> v1 -> v2 -> v3 -> v0
      float3 _v[4] = {_center + float3(-_ls.size.y, 0.f, -_ls.size.x),
                      _center + float3(-_ls.size.y, 0.f,  _ls.size.x),
                      _center + float3( _ls.size.y, 0.f,  _ls.size.x),
                      _center + float3( _ls.size.y, 0.f, -_ls.size.x)};
      float2 _v_ss[4] = {getImageSScoords(_p, normalize(_v[0] - _p)),
                         getImageSScoords(_p, normalize(_v[1] - _p)),
                         getImageSScoords(_p, normalize(_v[2] - _p)),
                         getImageSScoords(_p, normalize(_v[3] - _p))};

      for (int n = 0; n < num_samples; ++n) {
        int _edge_ind = 0;
        float _t = sampleSSfrom2Dpoints(_v_ss, 4, _edge_ind);
        // double _t = double(std::rand()) / RAND_MAX;
        float3 v0 = _v[_edge_ind];
        float3 v1 = _v[(_edge_ind + 1) & 3u];

        // float3 _m = v0 + (v1 - v0) * _t;
        float3 _edge_center = 0.5f * (v1 + v0);

        float2 v0ss = _v_ss[_edge_ind],
               v1ss = _v_ss[(_edge_ind + 1) & 3u],
               _center_ss = getImageSScoords(_p, normalize(_center - _p)),
               _edgec_ss  = getImageSScoords(_p, normalize(_edge_center - _p));
        float2 _m_ss = v0ss + (v1ss - v0ss) * _t;

        // check that sample is in frame
        int2 _sscoords{(int)floorf(_m_ss.x), (int)floorf(_m_ss.y)};
        if (_sscoords.x - 1 < 0 || _sscoords.x + 1 > m_winWidth ||
            _sscoords.y - 1 < 0 || _sscoords.y + 1 > m_winHeight)
          continue;

        // normal in SS coords
        float2 _n = normalize(float2{v1ss.y - v0ss.y, v0ss.x - v1ss.x});
        if (dot(_n, normalize(_edgec_ss - _center_ss)) < 0)
          _n *= -1;

        float3 f_in = getColor2(_p, dirFromSScoords((_m_ss - _n * 1.f).x, (_m_ss - _n * 1.f).y)),
              f_out = getColor2(_p, dirFromSScoords((_m_ss + _n * 1.f).x, (_m_ss + _n * 1.f).y));
        float3 f_diff{f_in - f_out};

        // both give the correct result
        // float2 _dv0_ss{v1ss.y - _m_ss.y, _m_ss.x - v1ss.x}, _dv1_ss{_m_ss.y - v0ss.y, v0ss.x - _m_ss.x};
        float2 _dv0_ss{_n.x * _t, _n.y * _t}, _dv1_ss{_n.x * (1.f - _t), _n.y * (1.f - _t)}; // screen space derivatives

        // backprop from dv0ss, dv1ss to dv0, dv1
        float3 _dv0 = projectSSderivatives(v0, _dv0_ss), _dv1 = projectSSderivatives(v1, _dv1_ss);


        // also works in pure 3D! No need to project stuff...
        float3  m = (1.f - _t) * v0 + _t * v1;
        float3 _dm_a = -1.f * cross(v0 - _p, v1 - _p); // with -1 it should always be correct (without dot() check)
        float3 _n_h  = normalize(_dm_a);
        if (dot(normalize(_edge_center - _center), _n_h) < 0.f) {
          _dm_a *= -1.f;
          _n_h  *= -1.f;
        }
        _dv1 = _n_h * (1.f - _t);
        _dv0 = _n_h * _t;
        // new stuff end


        // image derivative and loss:
        uint __ind = (_sscoords.y * m_winWidth + _sscoords.x) << 2;
        // 4-channel image, but only need 3 components
        float3 _colRef = {a_refImg[__ind], a_refImg[__ind+1], a_refImg[__ind+2]},
               _col = {out_color[__ind], out_color[__ind+1], out_color[__ind+2]};
        float3 _colDiff{_col - _colRef};
        // show samples on the images
        // out_color[__ind] = 1.f;
        // out_color[__ind+1] = 0.f;
        // out_color[__ind+2] = 0.f;

        _loss += dot(_colDiff, _colDiff);

// compute derivatives - new
        float3 _dv0dC = float3{ dot(_dv0, DLightRectParams::dcenterx()),
                                dot(_dv0, DLightRectParams::dcentery()),
                                dot(_dv0, DLightRectParams::dcenterz())};
        float2 _dv0dS = float2{ dot(_dv0, DLightRectParams::dsizex(_edge_ind)),
                                dot(_dv0, DLightRectParams::dsizey(_edge_ind))};
        float3 _dv1dC = float3{ dot(_dv1, DLightRectParams::dcenterx()),
                                dot(_dv1, DLightRectParams::dcentery()),
                                dot(_dv1, DLightRectParams::dcenterz())};
        float2 _dv1dS = float2{ dot(_dv1, DLightRectParams::dsizex(_edge_ind+1)),
                                dot(_dv1, DLightRectParams::dsizey(_edge_ind+1))};

        DLightSource __tmpParams{(_dv0dC.x * (1.f - _t) + _dv1dC.x * _t),
                                 (_dv0dC.y * (1.f - _t) + _dv1dC.y * _t),
                                 (_dv0dC.z * (1.f - _t) + _dv1dC.z * _t),
                                 (_dv0dS.x * (1.f - _t) + _dv1dS.x * _t),
                                 (_dv0dS.y * (1.f - _t) + _dv1dS.y * _t)};
        _MSE.add(_sscoords.x, _sscoords.y, __tmpParams, f_diff);
      }
    }
    // Circle. TODO
    // Sphere. TODO

    DLightSource _deriv{0.f, 0.f, 0.f, 0.f, 0.f};
    _MSE.dmse(out_color, a_refImg, _deriv);
    float _mse_coef = 2.f / (m_winHeight * m_winWidth);
    _deriv = _deriv * (norm_coef * _mse_coef);
    // CHECK IF SHADOW CAN OPTIMIZE LIGHT SOURCE, DON'T FORGET "+="
    _deriv = _shadowDerivative[i] * (_mse_coef) * float(i == 0);

    _lsu.update(*m_adams[i], m_pAccelStruct, m_lights, _deriv, _iter);
    m_lightInst[i] = _lsu;
  }
  m_pAccelStruct->CommitScene();
  return _loss *= norm_coef;
}



float3 Integrator::sampleImageConv(const uint2 &_coords, const float *_image) {
  if (_coords.x == 0 || _coords.x + 1 >= m_winWidth || _coords.y == 0 || _coords.y + 1 >= m_winHeight)
    return { _image[( _coords.y * m_winWidth + _coords.x) << 2],
             _image[((_coords.y * m_winWidth + _coords.x) << 2) + 1],
             _image[((_coords.y * m_winWidth + _coords.x) << 2) + 2] };

  uint _00 = ((_coords.y - 1) * m_winWidth + _coords.x - 1) << 2,
       _01 = ((_coords.y - 1) * m_winWidth + _coords.x)     << 2,
       _02 = ((_coords.y - 1) * m_winWidth + _coords.x + 1) << 2,
       _10 = ((_coords.y    ) * m_winWidth + _coords.x - 1) << 2,
       _11 = ((_coords.y    ) * m_winWidth + _coords.x)     << 2,
       _12 = ((_coords.y    ) * m_winWidth + _coords.x + 1) << 2,
       _20 = ((_coords.y + 1) * m_winWidth + _coords.x - 1) << 2,
       _21 = ((_coords.y + 1) * m_winWidth + _coords.x)     << 2,
       _22 = ((_coords.y + 1) * m_winWidth + _coords.x + 1) << 2;
  
  return (1.f / 9) * (float3{_image[_00], _image[_00 + 1], _image[_00 + 2]} +
                      float3{_image[_01], _image[_01 + 1], _image[_01 + 2]} +
                      float3{_image[_02], _image[_02 + 1], _image[_02 + 2]} +
                      float3{_image[_10], _image[_10 + 1], _image[_10 + 2]} +
                      float3{_image[_11], _image[_11 + 1], _image[_11 + 2]} +
                      float3{_image[_12], _image[_12 + 1], _image[_12 + 2]} +
                      float3{_image[_20], _image[_20 + 1], _image[_20 + 2]} +
                      float3{_image[_21], _image[_21 + 1], _image[_21 + 2]} +
                      float3{_image[_22], _image[_22 + 1], _image[_22 + 2]});
}



void Integrator::shadowEdgeSamplingStep(float* out_color, const float* a_refImg, std::vector<DLightSource> &_shadowDerivative) {
  ImgMSE _MSE{uint(m_winWidth), uint(m_winHeight), 5};
  const uint num_samples = 8192u * 2;
  const uint samples_per_ray = 32u * 2;

  std::vector<uint2> samples;

  for (uint l = 0u; l < m_lightInst.size(); ++l) {
    _MSE.nullify();
    DLightSourceUpdater _li = m_lightInst[l];
    LightSource _ls =  m_lights[_li.lightID];
    float3 _center = to_float3(_ls.pos);
    float loss = 0.f;

    for (uint i = 0u; i < num_samples / samples_per_ray; ++i) {
      float2 _sscoords{(float(std::rand()) / RAND_MAX) * m_winWidth,
                       (float(std::rand()) / RAND_MAX) * m_winHeight};
      // _sscoords = {436, 80};
      uint2 __ind2d{ uint(floorf(_sscoords.x)), uint(floorf(_sscoords.y)) };
      uint  __ind = (__ind2d.y * m_winWidth + __ind2d.x) << 2;

      float3 _pos = mul4x3(m_worldViewInv, float3(0,0,0));
      float3 _dir = dirFromSScoords(_sscoords.x, _sscoords.y);

      // {
      //   float3 _v[4] = {_center + float3(-_ls.size.y, 0.f, -_ls.size.x),
      //                   _center + float3(-_ls.size.y, 0.f,  _ls.size.x),
      //                   _center + float3( _ls.size.y, 0.f,  _ls.size.x),
      //                   _center + float3( _ls.size.y, 0.f, -_ls.size.x)};
      //   float3 _centerDir{normalize(_center - _pos)};
      //   float3 _proj[] = { _v[0] - dot(_pos - _v[0], _centerDir) * _centerDir,
      //                      _v[1] - dot(_pos - _v[1], _centerDir) * _centerDir,
      //                      _v[2] - dot(_pos - _v[2], _centerDir) * _centerDir,
      //                      _v[3] - dot(_pos - _v[3], _centerDir) * _centerDir };
      //   float4 _lengths{ length(_proj[1] - _proj[0]),
      //                    length(_proj[2] - _proj[1]),
      //                    length(_proj[3] - _proj[2]),
      //                    length(_proj[0] - _proj[3]) };
      //   _lengths /= _lengths[0] + _lengths[1] + _lengths[2] + _lengths[3];

      //   float2 _v_ss[4] = { getImageSScoords(_pos, normalize(_v[0] - _pos)),
      //                       getImageSScoords(_pos, normalize(_v[1] - _pos)),
      //                       getImageSScoords(_pos, normalize(_v[2] - _pos)),
      //                       getImageSScoords(_pos, normalize(_v[3] - _pos)) };
      //   float4 _len_p{ length(_v_ss[1] - _v_ss[0]),
      //                  length(_v_ss[2] - _v_ss[1]),
      //                  length(_v_ss[3] - _v_ss[2]),
      //                  length(_v_ss[0] - _v_ss[3]) };
      //   _len_p /= _len_p[0] + _len_p[1] + _len_p[2] + _len_p[3];
      //   printf("Ortho: [%1.2f, %1.2f, %1.2f, %1.2f], persp: [%1.2f, %1.2f, %1.2f, %1.2f]\n",
      //         _lengths.x, _lengths.y, _lengths.z, _lengths.w, _len_p.x, _len_p.y, _len_p.z, _len_p.w);
      // }

      uint instId = -1, rayFlags = 0;
      float4 rayPosAndNear{to_float4(_pos, 0.f)}, rayDirAndFar{to_float4(_dir, FLT_MAX)}, hitPart1, hitPart2, hitPart3;

      kernel_RayTrace2(0, 0, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &hitPart3, &instId, &rayFlags);

      if(isDeadRay(rayFlags)) continue;
      // don't sample light sources directly
      {
        bool _eq = false;
        for (auto _lInst : m_lightInst) {
          if (instId == _lInst.instID) {
            _eq = true;
            break;
          }
        }
        if (_eq) continue;
      }

      float3 shadowRayPos = to_float3(hitPart1) + to_float3(hitPart2) * std::max(maxcomp(to_float3(hitPart1)), 1.0f) * 5e-6f;

      // pixel color from the image and reference
      float3 _colRef = sampleImageConv(__ind2d,  a_refImg),
             _col    = sampleImageConv(__ind2d, out_color);
      // float3 _colRef = {a_refImg[__ind], a_refImg[__ind+1], a_refImg[__ind+2]},
      //        _col = {out_color[__ind], out_color[__ind+1], out_color[__ind+2]};
      float3 _colDiff{_col - _colRef};

      loss += dot(_colDiff, _colDiff);
      bool _success = false;
      for (uint s = 0u; s < samples_per_ray; ++s) {
        if (_ls.geomType == LIGHT_GEOM_RECT) {
          float3 _v[4] = {_center + float3(-_ls.size.y, 0.f, -_ls.size.x),
                          _center + float3(-_ls.size.y, 0.f,  _ls.size.x),
                          _center + float3( _ls.size.y, 0.f,  _ls.size.x),
                          _center + float3( _ls.size.y, 0.f, -_ls.size.x)};
          float3 _centerDir{normalize(_center - shadowRayPos)};
          float3 _proj[] = { _v[0] - dot(shadowRayPos - _v[0], _centerDir) * _centerDir,
                             _v[1] - dot(shadowRayPos - _v[1], _centerDir) * _centerDir,
                             _v[2] - dot(shadowRayPos - _v[2], _centerDir) * _centerDir,
                             _v[3] - dot(shadowRayPos - _v[3], _centerDir) * _centerDir };
          float4 _lengths{ length(_proj[1] - _proj[0]), length(_proj[2] - _proj[1]),
                           length(_proj[3] - _proj[2]), length(_proj[0] - _proj[3]) };
          _lengths /= _lengths[0] + _lengths[1] + _lengths[2] + _lengths[3];

          int _edge_ind = 0;
          float _t = sampleEdgePtfrom3Dpoints(_v, 4, _edge_ind);
          // _t = 0.6f;
          // _edge_ind = 0;

          float3  v0 = _v[_edge_ind], v1 = _v[(_edge_ind + 1) & 3u];
          float3 _edge_center = 0.5f * (v0 + v1);
          float3  m = (1.f - _t) * v0 + _t * v1;
          float _hitDist = length(m - shadowRayPos);
          float3 shadowRayDir = normalize(m - shadowRayPos);

          float3 _n_m  = to_float3(_ls.norm);
          if (dot(_n_m, shadowRayDir) > -0.01f) continue;
          float3 _dm_a = -1.f * cross(v0 - shadowRayPos, v1 - shadowRayPos);
          float3 _n_h  = normalize(_dm_a);
          if (dot(normalize(_edge_center - _center), _n_h) < 0.f) {
            _dm_a *= -1.f;
            _n_h  *= -1.f;
          }


          uint _tmpId = 0, _tmpFlags = 0;
          float4 wavelengths, _tmpHit1, _tmpHit2, _tmpHit3;
          float4 _sPos    = to_float4(shadowRayPos, 0.f),
                 _sDirIn  = to_float4(normalize((m - 1.f * _n_h) - shadowRayPos), FLT_MAX),
                 _sDirOut = to_float4(normalize((m + 1.f * _n_h) - shadowRayPos), FLT_MAX);

          // in
          kernel_RayTrace2(0, 0, &_sPos, &_sDirIn,  &_tmpHit1, &_tmpHit2, &_tmpHit3, &_tmpId, &_tmpFlags);
          // if (m_instIdToLightInstId[_tmpId] != _li.lightID) continue;
          uint2 _inds{0u};
          _inds.x = _tmpId;

          float3 h_in = SampleLightSourceByID(0, &rayPosAndNear, &rayDirAndFar, &wavelengths, &hitPart1, &hitPart2, &hitPart3,
                        &rayFlags, 0, m_instIdToLightInstId[_tmpId], { .pos = to_float3(_tmpHit1), .norm = to_float3(_ls.norm), .isOmni = false });

          // out
          _tmpId = 0;
          kernel_RayTrace2(0, 0, &_sPos, &_sDirOut, &_tmpHit1, &_tmpHit2, &_tmpHit3, &_tmpId, &_tmpFlags);
          float3 h_out{0.f};
          if (int(_tmpId) >= 0) {
            h_out = SampleLightSourceByID(0, &rayPosAndNear, &rayDirAndFar, &wavelengths, &hitPart1, &hitPart2, &hitPart3,
                                          &rayFlags, 0, m_instIdToLightInstId[_tmpId], { .pos = to_float3(_tmpHit1), .norm = to_float3(_ls.norm), .isOmni = false });
          }
          float3 h_diff = h_in - h_out;
          // printf("Diff: [%1.2f, %1.2f, %1.2f], dist: %1.3f\n", h_diff.x, h_diff.y, h_diff.z, _hitDist);
          _inds.y = _tmpId;
          // if (length(h_diff) < 0.01f) {
          //   h_diff *= -1.f;
          //   printf("h_in: [%1.2f, %1.2f, %1.2f], h_out: [%1.2f, %1.2f, %1.2f], insts: [%d, %d], curr light: %d\n", h_in.x, h_in.y, h_in.z, h_out.x, h_out.y, h_out.z, _inds.x, _inds.y, _li.instID);
          //   printf("Nh: [%1.2f, %1.2f, %1.2f], Nc: [%1.2f, %1.2f, %1.2f], Nh norm: %1.2f, dot: %1.2f\n", _n_h.x, _n_h.y, _n_h.z,
          //                                                            normalize(_edge_center - _center).x,
          //                                                            normalize(_edge_center - _center).y,
          //                                                            normalize(_edge_center - _center).z,
          //                                                            length(_n_h),
          //                                                            dot(normalize(_edge_center - _center), _n_h));
          //   printf("V0: [%1.2f, %1.2f, %1.2f], V1: [%1.2f, %1.2f, %1.2f], m: [%1.2f, %1.2f, %1.2f], p: [%1.2f, %1.2f, %1.2f]\n\n", v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, m.x, m.y, m.z, shadowRayPos.x, shadowRayPos.y, shadowRayPos.z);
          // }
          // printf("Moving on...\n");
          // printf("V0: [%1.2f, %1.2f, %1.2f], V1: [%1.2f, %1.2f, %1.2f], m: [%1.2f, %1.2f, %1.2f],\np: [%1.2f, %1.2f, %1.2f], Nh: [%1.2f, %1.2f, %1.2f]\n\n",
          //         v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, m.x, m.y, m.z, shadowRayPos.x, shadowRayPos.y, shadowRayPos.z, _n_h.x, _n_h.y, _n_h.z);


          // formula 2
          float _corr_term = 1.f / length(cross(_n_m, _n_h));
          float Jm_norm = 0.f;
          {
            float3 w_t = v0 + (v1 - v0) * _t - shadowRayPos; // same as (m - shadowRayPos)
            float tau_t = dot(w_t, _n_m);
            if (tau_t < -1e-5f || tau_t > 1e-5f) {
              tau_t = dot(m - shadowRayPos, _n_m) / tau_t;
              float3 Jm = tau_t * ((v1 - v0) - w_t * (dot(v1 - v0, _n_m) / dot(w_t, _n_m)));
              Jm_norm = length(v1 - v0); // not len(Jm), because don't need to project anything
            }
          }
          float _coef = 1.f / length(_dm_a) * _corr_term * Jm_norm;
          float3 _dv0 = cross(v1, m), _dv1 = cross(m, v0);
          // formula 2 end -> _coef, _dv0, _dv1
          _coef = 100.f / (_hitDist * _lengths[_edge_ind]);
          _dv0 = _n_h * (1.f - _t);
          _dv1 = _n_h * _t;

          float3 _dv0dC = float3{ dot(_dv0, DLightRectParams::dcenterx()),
                                  dot(_dv0, DLightRectParams::dcentery()),
                                  dot(_dv0, DLightRectParams::dcenterz())};
          float2 _dv0dS = float2{ dot(_dv0, DLightRectParams::dsizex(_edge_ind)),
                                  dot(_dv0, DLightRectParams::dsizey(_edge_ind))};
          float3 _dv1dC = float3{ dot(_dv1, DLightRectParams::dcenterx()),
                                  dot(_dv1, DLightRectParams::dcentery()),
                                  dot(_dv1, DLightRectParams::dcenterz())};
          float2 _dv1dS = float2{ dot(_dv1, DLightRectParams::dsizex(_edge_ind+1)),
                                  dot(_dv1, DLightRectParams::dsizey(_edge_ind+1))};

          DLightSource __tmpParams{(_dv0dC.x * (1.f - _t) + _dv1dC.x * _t),
                                   (_dv0dC.y * (1.f - _t) + _dv1dC.y * _t),
                                   (_dv0dC.z * (1.f - _t) + _dv1dC.z * _t),
                                   (_dv0dS.x * (1.f - _t) + _dv1dS.x * _t),
                                   (_dv0dS.y * (1.f - _t) + _dv1dS.y * _t)};

          {
            float3 __dir{ normalize(_center - shadowRayPos) };
            if (__dir.y > 1e-5f) {
              float3 __norm{ normalize(float3{0.f, 1.f / __dir.y, 0.f} - __dir) };
              // printf("Norm: [%1.2f, %1.2f, %1.2f]\n", __norm.x, __norm.y, __norm.z);
              float3 _dC{ __norm };
              // printf("dC: [%1.2f, %1.2f, %1.2f]\n", _dC.x, _dC.y, _dC.z);

              // float3 __norms[4];
              // for (int i = 0; i < 4; ++i) {
              //   __norms[i] = normalize(-1.f * cross(_v[i] - shadowRayPos,
              //                                       _v[(i + 1) & 3u] - shadowRayPos));
              //   if (dot(normalize(_edge_center - _center), _n_h) < 0.f)
              //     __norms[i] *= -1.f;
              // }
              // float3 _mean{_lengths[0] * __norms[0] +
              //              _lengths[1] * __norms[1] +
              //              _lengths[2] * __norms[2] +
              //              _lengths[3] * __norms[3] };
              // printf("n1: [%1.2f, %1.2f, %1.2f], n2: [%1.2f, %1.2f, %1.2f], n3: [%1.2f, %1.2f, %1.2f], n4: [%1.2f, %1.2f, %1.2f]\n",
              //         __norms[0].x, __norms[0].y, __norms[0].z,
              //         __norms[1].x, __norms[1].y, __norms[1].z,
              //         __norms[2].x, __norms[2].y, __norms[2].z,
              //         __norms[3].x, __norms[3].y, __norms[3].z);
              // printf("Mean: [%1.2f, %1.2f, %1.2f], norm: [%1.2f, %1.2f, %1.2f]\n", _mean.x, _mean.y, _mean.z, __norm.x, __norm.y, __norm.z);

              __tmpParams.dI_dCx = _dC.x;
              __tmpParams.dI_dCy = _dC.y;
              __tmpParams.dI_dCz = _dC.z;
            }
          }
          if (_hitDist < 1.f) {
            printf("Info: instId = %d, norm: [%1.2f, %1.2f, %1.2f]\n", instId, __tmpParams.dI_dCx, __tmpParams.dI_dCy, __tmpParams.dI_dCz);
          }
          else {
            _MSE.add(__ind2d.x, __ind2d.y, __tmpParams, _coef * h_diff);
            _success = true;
          }
        }
      } // for { if {} }
      if (_success) samples.push_back(__ind2d);
    }
    _MSE.dmse(out_color, a_refImg, _shadowDerivative[l]);
    for (auto _pix : samples) {
      uint  __ind = (_pix.y * m_winWidth + _pix.x) << 2;
      out_color[__ind    ] = 0.f;
      out_color[__ind + 1] = 0.f;
      out_color[__ind + 2] = 1.f;
    }
  }
}



float3 Integrator::dSampleLightSource(const float3 &shadowRayPos, const float3 &shadowRayDir, const float3 &_m, const float3 &ray_dir, const float4* wavelengths,
                                      const float4* in_hitPart1, const float4* in_hitPart2, const float4* in_hitPart3,
                                      const uint* rayFlags, int lightId, float hitDist, RandomGen* a_gen) {
  const uint32_t matId = extractMatId(*rayFlags);

  const float4 data1  = *in_hitPart1;
  const float4 data2  = *in_hitPart2;
  const float4 lambda = *wavelengths;

  SurfaceHit hit;
  // hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.tang = to_float3(*in_hitPart3);
  hit.uv   = float2(data1.w, data2.w);

  LightSource _ls = m_lights[lightId];

  const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  const bool inIllumArea = dot(shadowRayDir, to_float3(_ls.norm)) < 0.0f;

  if(!inShadow && inIllumArea) {
    const BsdfEval bsdfV = MaterialEval(matId, lambda, shadowRayDir, (-1.0f)*ray_dir, hit.norm, hit.tang, hit.uv);
    float cosThetaOut = std::max(dot(shadowRayDir, hit.norm), 0.0f);
    // was dLightEvalPDF
    float lgtPdfW = (1.0f / m_lights.size()) * LightEvalPDF(lightId, shadowRayPos, shadowRayDir, _m, to_float3(_ls.norm));
    float4 lightColor = GetLightSourceIntensity(lightId, wavelengths, shadowRayDir);

    return to_float3(lightColor * bsdfV.val * (cosThetaOut / lgtPdfW));
  }
  return {};

  // // for SPHERE light source - start here
  // const LightSample lSam = LightSampleRev(lightId, { 0.5f, 0.5f }, hit.pos);
  // const float  hitDist   = std::sqrt(dot(hit.pos - lSam.pos, hit.pos - lSam.pos));

  // const bool   inShadow     = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  // const bool   inIllumArea  = (dot(shadowRayDir, lSam.norm) < 0.0f) || lSam.isOmni;

  // if(!inShadow && inIllumArea) {
  //   const BsdfEval bsdfV = MaterialEval(matId, lambda, shadowRayDir, (-1.0f)*ray_dir, hit.norm, hit.tang, hit.uv);
  //   float cosThetaOut    = std::max(dot(shadowRayDir, hit.norm), 0.0f);
    
  //   float lgtPdfW   = LightPdfSelectRev(lightId) * LightEvalPDF(lightId, shadowRayPos, shadowRayDir, lSam.pos, lSam.norm);
  //   float misWeight = 1.0f;

  //   const float4 lightColor = GetLightSourceIntensity(lightId, wavelengths, shadowRayDir);
  //   return to_float3(lightColor * bsdfV.val * (cosThetaOut / lgtPdfW));
  // }
  // return {};
}

// float Integrator::dLightEvalPDF(int a_lightId, float3 illuminationPoint, float3 ray_dir, const float3 lpos, const float3 lnorm) {
//   const float hitDist2 = dot(illuminationPoint - lpos, illuminationPoint - lpos);
//   float cosVal = std::max(dot(ray_dir, -1.0f*lnorm), 0.0f);

//   if (m_lights[a_lightId].geomType == LIGHT_GEOM_SPHERE) {
//       const float3 dirToV  = normalize(lpos - illuminationPoint);
//       cosVal = std::abs(dot(dirToV, lnorm));
//   }
  
//   return (m_lights[a_lightId].pdfA * hitDist2) / std::max(cosVal, 1e-30f);
// }

// void Integrator::dNextBounce(uint tid, uint bounce, const float4* in_hitPart1, const float4* in_hitPart2, const float4* in_hitPart3, const uint* in_instId,
//                                    const float4* in_shadeColor, float4* rayPosAndNear, float4* rayDirAndFar, const float4* wavelengths,
//                                    float4* accumColor, float4* accumThoroughput, RandomGen* a_gen, MisData* misPrev, uint* rayFlags) {
//   const uint currRayFlags = *rayFlags;
//   const uint32_t matId = extractMatId(currRayFlags);

//   if(m_materials[matId].mtype == MAT_TYPE_LIGHT_SOURCE) {
//     printf("Error: somehow got MATERIAL_LIGHT_SOURCE\n");
//     return;
//   }

//   *accumColor = (*accumColor) + (*in_shadeColor);
// }


// bool Integrator::dRaySqLightIntersect(const float3 &_pos, const float3 &_dir,
//                                       const float3 &_center, const float2 &_size, float3 &_res) {
//   float3 _width{ _size.y, 0.f, 0.f }, _height{ 0.f, 0.f, _size.x };
//   float3 _norm = normalize(cross(_width, _height));
//   float denom = dot(_norm, _dir);
//   if (abs(denom) > 1e-4f) {
//       float t = dot((_center - _pos), _norm) / denom;
//       if (t >= 0) {
//         _res = _pos + t * _dir;
//         return true;
//       }
//   }
//   return false;
//   _res = _pos + _dir * dot((_center - _pos), float3{0, 1, 0}) /
//                        dot(           _dir , normalize(cross(_width, _height)));
// }


bool dRayTrigIntersect(float3 _pos, float3 _dir, 
                        const float3 &a, const float3 &b, const float3 &c,
                        float3 &_res, float3 &_dRes_dC) {
  // printf("a: [%1.2f, %1.2f, %1.2f], b: [%1.2f, %1.2f, %1.2f], c: [%1.2f, %1.2f, %1.2f]\n",
  //         a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
  // float inv_det = 1.0 / dot(b - a, cross(_dir, c - a));
  // float u = inv_det * dot(_pos - a, cross(_dir, c - a));
  // float v = inv_det * dot( _dir, cross(_pos - a, b - a));
  // float t = dot(c - a, cross(_pos - a, b - a)) /
  //           dot(b - a, cross(_dir, c - a));
  
  // 1
  float3 _e1 = b - a,
         _e2 = c - a,
         _l = _pos - a;

  // 2
  float3 _cr2 = cross(_l, _e1),
         _cr1 = cross(_dir, _e2);
  
  // 3
  float _numer = dot(_e2, _cr2),
        _denom = dot(_e1, _cr1);
  
  // 4
  float t = _numer / _denom;

  // 5
  _res = _pos + _dir * t;

  // bonus
  float u = dot(_l, _cr1) / _denom;
  float v = dot(_dir, _cr2) / _denom;

  // printf("LEN: %f, uv: [%1.2f, %1.2f, %1.2f]\n", length(_res - (u * b + v * c + (1.f - u - v) * a)), u, v, 1.f - u - v);

// DIFF
  // -5
  float3 _dRes_dt = _dir;

  // -4 (prep)
  float3 _dNum_da, _dNum_db, _dNum_dc;
  float3 _dDen_da, _dDen_db, _dDen_dc;
  {
    // -3
    float3 _dNum_de2 = _cr2;
    float3 _dNum_dcr2 = _e2;
    float3 _dDen_de1 = _cr1;
    float3 _dDen_dcr1 = _e1;

    // -2
    float3 _dcr2_dlx = DCross::crossD0X(_l, _e1);
    float3 _dcr2_dly = DCross::crossD0Y(_l, _e1);
    float3 _dcr2_dlz = DCross::crossD0Z(_l, _e1);
    float3 _dNum_dl = { dot(_dNum_dcr2, _dcr2_dlx),
                        dot(_dNum_dcr2, _dcr2_dly),
                        dot(_dNum_dcr2, _dcr2_dlz) };

    float3 _dcr2_de1x = DCross::crossD1X(_l, _e1);
    float3 _dcr2_de1y = DCross::crossD1Y(_l, _e1);
    float3 _dcr2_de1z = DCross::crossD1Z(_l, _e1);
    float3 _dNum_de1 = { dot(_dNum_dcr2, _dcr2_de1x),
                         dot(_dNum_dcr2, _dcr2_de1y),
                         dot(_dNum_dcr2, _dcr2_de1z) };

    // float3 _dcr1_dDirx = DCross::crossD0X(_dir, _e2);
    // float3 _dcr1_dDiry = DCross::crossD0Y(_dir, _e2);
    // float3 _dcr1_dDirz = DCross::crossD0Z(_dir, _e2);
    float3 _dcr1_de2x = DCross::crossD1X(_dir, _e2);
    float3 _dcr1_de2y = DCross::crossD1Y(_dir, _e2);
    float3 _dcr1_de2z = DCross::crossD1Z(_dir, _e2);
    float3 _dDen_de2 = { dot(_dDen_dcr1, _dcr1_de2x),
                         dot(_dDen_dcr1, _dcr1_de2y),
                         dot(_dDen_dcr1, _dcr1_de2z) };

    // -1
    // de1_db = 1, de1_da = -1,
    // de2_dc = 1, de2_da = -1,
    // dl_da = -1
    _dNum_da = -1.f * (_dNum_de1 + _dNum_de2 + _dNum_dl);
    _dNum_db = _dNum_de1;
    _dNum_dc = _dNum_de2;
    _dDen_da = -1.f * (_dDen_de1 + _dDen_de2);
    _dDen_db = _dDen_de1;
    _dDen_dc = _dDen_de2;
  }



  // -4 (res)
  float3 _dt_da = (_denom * _dNum_da - _numer * _dDen_da) / (_denom * _denom);
  float3 _dt_db = (_denom * _dNum_db - _numer * _dDen_db) / (_denom * _denom);
  float3 _dt_dc = (_denom * _dNum_dc - _numer * _dDen_dc) / (_denom * _denom);


// taking diagonals of these matrices?
  float3 _dRes_dax = _dRes_dt * _dt_da.x;
  float3 _dRes_day = _dRes_dt * _dt_da.y;
  float3 _dRes_daz = _dRes_dt * _dt_da.z;

  // printf("dRes/da:\n[%1.2f, %1.2f, %1.2f]\n[%1.2f, %1.2f, %1.2f]\n[%1.2f, %1.2f, %1.2f]\n",
  //         _dRes_dax.x, _dRes_dax.y, _dRes_dax.z, _dRes_day.x, _dRes_day.y, _dRes_day.z, _dRes_daz.x, _dRes_daz.y, _dRes_daz.z);

  float3 _dRes_dbx = _dRes_dt * _dt_db.x;
  float3 _dRes_dby = _dRes_dt * _dt_db.y;
  float3 _dRes_dbz = _dRes_dt * _dt_db.z;

  // printf("dRes/db:\n[%1.2f, %1.2f, %1.2f]\n[%1.2f, %1.2f, %1.2f]\n[%1.2f, %1.2f, %1.2f]\n",
  //         _dRes_dbx.x, _dRes_dbx.y, _dRes_dbx.z, _dRes_dby.x, _dRes_dby.y, _dRes_dby.z, _dRes_dbz.x, _dRes_dbz.y, _dRes_dbz.z);

  float3 _dRes_dcx = _dRes_dt * _dt_dc.x;
  float3 _dRes_dcy = _dRes_dt * _dt_dc.y;
  float3 _dRes_dcz = _dRes_dt * _dt_dc.z;

  // printf("dRes/dc:\n[%1.2f, %1.2f, %1.2f]\n[%1.2f, %1.2f, %1.2f]\n[%1.2f, %1.2f, %1.2f]\n",
  //         _dRes_dcx.x, _dRes_dcx.y, _dRes_dcx.z, _dRes_dcy.x, _dRes_dcy.y, _dRes_dcy.z, _dRes_dcz.x, _dRes_dcz.y, _dRes_dcz.z);

// should be it (diagonals), because
  // da/dC -> Identity 3x3. Example:
  // float _dResx_dCx = _dRes_dt.x * dot(_dt_da, DLightRectParams::dcenterx());
  float3 _dRes_da = _dRes_dt * _dt_da;
  float3 _dRes_db = _dRes_dt * _dt_db;
  float3 _dRes_dc = _dRes_dt * _dt_dc;

  // printf("da: [%1.2f, %1.2f, %1.2f], db: [%1.2f, %1.2f, %1.2f], dc: [%1.2f, %1.2f, %1.2f]\n",
  //         _dRes_da.x, _dRes_da.y, _dRes_da.z, _dRes_db.x, _dRes_db.y, _dRes_db.z, _dRes_dc.x, _dRes_dc.y, _dRes_dc.z);
  _dRes_dC = _dRes_da + _dRes_db + _dRes_dc;



  // "bonus"
  float3 _duNum_da, _duNum_db, _duNum_dc;
  float3 _dvNum_da, _dvNum_db, _dvNum_dc;
  {
    // float u = dot(_l, _cr1) / _denom;
    // float v = dot(_dir, _cr2) / _denom;

    // -3'
    float3 _duNum_dl = _cr1;
    float3 _duNum_dcr1 = _l;
    // float3 _dvNum_dl = _cr1;
    float3 _dvNum_dcr2 = _dir;

    // -2'
    float3 _dcr2_dlx = DCross::crossD0X(_l, _e1);
    float3 _dcr2_dly = DCross::crossD0Y(_l, _e1);
    float3 _dcr2_dlz = DCross::crossD0Z(_l, _e1);
    float3 _dvNum_dl = { dot(_dvNum_dcr2, _dcr2_dlx),
                         dot(_dvNum_dcr2, _dcr2_dly),
                         dot(_dvNum_dcr2, _dcr2_dlz) };

    float3 _dcr2_de1x = DCross::crossD1X(_l, _e1);
    float3 _dcr2_de1y = DCross::crossD1Y(_l, _e1);
    float3 _dcr2_de1z = DCross::crossD1Z(_l, _e1);
    float3 _dvNum_de1 = { dot(_dvNum_dcr2, _dcr2_de1x),
                          dot(_dvNum_dcr2, _dcr2_de1y),
                          dot(_dvNum_dcr2, _dcr2_de1z) };


    float3 _dcr1_de2x = DCross::crossD1X(_dir, _e2);
    float3 _dcr1_de2y = DCross::crossD1Y(_dir, _e2);
    float3 _dcr1_de2z = DCross::crossD1Z(_dir, _e2);
    float3 _duNum_de2 = { dot(_duNum_dcr1, _dcr1_de2x),
                          dot(_duNum_dcr1, _dcr1_de2y),
                          dot(_duNum_dcr1, _dcr1_de2z) };

    // -1'
    _dvNum_da = -1.f * (_dvNum_de1 + _dvNum_dl);
    _dvNum_db = _dvNum_de1;
    _dvNum_dc = {0,0,0};
    _duNum_da = -1.f * (_duNum_de2 + _duNum_dl);
    _duNum_db = {0,0,0};
    _duNum_dc = _duNum_de2;
  }

  // -4' (res)
  float3 _du_da = (_denom * _duNum_da - dot(_l,   _cr1) * _dDen_da) / (_denom * _denom);
  float3 _du_db = (_denom * _duNum_db - dot(_l,   _cr1) * _dDen_db) / (_denom * _denom);
  float3 _du_dc = (_denom * _duNum_dc - dot(_l,   _cr1) * _dDen_dc) / (_denom * _denom);
  float3 _dv_da = (_denom * _dvNum_da - dot(_dir, _cr2) * _dDen_da) / (_denom * _denom);
  float3 _dv_db = (_denom * _dvNum_db - dot(_dir, _cr2) * _dDen_db) / (_denom * _denom);
  float3 _dv_dc = (_denom * _dvNum_dc - dot(_dir, _cr2) * _dDen_dc) / (_denom * _denom);

  _res = u * b + v * c + (1.f - u - v) * a;

  // float3 _dRes_dax2 = b * _du_da.x + c * _dv_da.x + float3{1.f - u - v, 0.f, 0.f} + a * ;
  // float3 _dRes_day2 = _dRes_dt * _dt_da.y;
  // float3 _dRes_daz2 = _dRes_dt * _dt_da.z;
  float3 _dRes_da2 = b * _du_da + c * _dv_da + float3{1.f} * (1.f - u - v) - a * (_du_da + _dv_da);

  // float3 _dRes_dbx2 = _dRes_dt * _dt_db.x;
  // float3 _dRes_dby2 = _dRes_dt * _dt_db.y;
  // float3 _dRes_dbz2 = _dRes_dt * _dt_db.z;
  float3 _dRes_db2 = b * _du_db + float3{1.f} * u + c * _dv_db - a * (_du_db + _dv_db);

  // float3 _dRes_dcx2 = _dRes_dt * _dt_dc.x;
  // float3 _dRes_dcy2 = _dRes_dt * _dt_dc.y;
  // float3 _dRes_dcz2 = _dRes_dt * _dt_dc.z;
  float3 _dRes_dc2 = b * _du_dc + c * _dv_dc + float3{1.f} * v - a * (_du_dc + _dv_dc);

  printf("da: [%1.2f, %1.2f, %1.2f], db: [%1.2f, %1.2f, %1.2f], dc: [%1.2f, %1.2f, %1.2f]\n",
          _dRes_da2.x, _dRes_da2.y, _dRes_da2.z, _dRes_db2.x, _dRes_db2.y, _dRes_db2.z, _dRes_dc2.x, _dRes_dc2.y, _dRes_dc2.z);
  return true;
}

void Integrator::simpleSampler() {
  DLightSourceUpdater _li = m_lightInst[0];
  LightSource _ls =  m_lights[_li.lightID];
  float3 _center = to_float3(_ls.pos);

  float3 _v[4] = {_center + float3(-_ls.size.y, 0.f, -_ls.size.x),
                  _center + float3(-_ls.size.y, 0.f,  _ls.size.x),
                  _center + float3( _ls.size.y, 0.f,  _ls.size.x),
                  _center + float3( _ls.size.y, 0.f, -_ls.size.x)};

  const uint N = 128u;
  float3 _meanDres{};
  for (uint i = 0u; i < N; ++i) {
    int3 _rands{ 0 };
    _rands.x = std::rand();
    _rands.y = std::rand();
    _rands.z = std::rand();
    printf("RANDS: %i, %i, %i\n", _rands.x, _rands.y, _rands.z);
    uint _ind = (float(_rands.x) /  2147483647) > 0.5f ? 2u : 0u;
    float u =   (float(_rands.y) / (2147483647));
    float v =   (float(_rands.z) / (2147483647)) * (1.f - u);
    printf("UV: [%1.2f, %1.2f, %1.2f]\n", u, v, (1.f - u - v));
    float3 _hit = _v[_ind] * u + _v[1] * v + _v[3] * (1.f - u - v);
    float3 _pos = mul4x3(m_worldViewInv, float3(0,0,0));
    float3 _dir = normalize(_hit - _pos);

    float3 _res{}, _dRes_dC{};
    dRayTrigIntersect(_pos, _dir, _v[_ind], _v[1], _v[3], _res, _dRes_dC);
    printf("Sample: [%1.2f, %1.2f, %1.2f], hit: [%1.2f, %1.2f, %1.2f], dist = %1.2f\n", _dRes_dC.x, _dRes_dC.y, _dRes_dC.z, _hit.x, _hit.y, _hit.z, length(_res - _hit));
    _meanDres += (1.f / N) * _dRes_dC;

    // float2 _sscoords{getImageSScoords(_pos, _dir)};
    // uint2 __ind2d{ uint(floorf(_sscoords.x)), uint(floorf(_sscoords.y)) };
    // uint  __ind = (__ind2d.y * m_winWidth + __ind2d.x) << 2;
  }
  printf("Sample: [%1.2f, %1.2f, %1.2f]\n", _meanDres.x, _meanDres.y, _meanDres.z);
}

// void Integrator::EstimateWarp(const float3 &_pos, const float3 &_dir, uint _N) {

// }

// void Integrator::WASRadiance(const float3 &_pos, const float3 &_dirIn) {

// }

// void Integrator::warpedAreaSamplingStep(float* out_color, const float* a_refImg, float* a_DerivPosImg,
//                                         float* a_DerivNegImg, uint _iter) {

// } 


void DLightSourceUpdater::update(AdamOptimizer2<float> &_opt, std::shared_ptr<ISceneObject> _accel_struct,
                                  std::vector<LightSource>&_m_lights, const DLightSource &_deriv, int iter) {
  float c3s2[5]{0.f, 0.f, 0.f, 0.f, 0.f};
  _opt.step(c3s2, &_deriv.dI_dCx, iter);
  float4 _dpos {c3s2[0], c3s2[1], c3s2[2], 0.f};
  float2 _dsize{c3s2[3], c3s2[4]};

  // update both instance matrix and lightSource parameters
  // pos
  printf("Position before: %f, %f, %f, derivative: %f, %f, %f\n", _m_lights[lightID].pos.x,
                                                                  _m_lights[lightID].pos.y,
                                                                  _m_lights[lightID].pos.z,
                                                                  _deriv.dI_dCx, _deriv.dI_dCy, _deriv.dI_dCz);
  // _dpos.y = 0.f;
  _m_lights[lightID].pos += _dpos;
  instMat.m_col[3] += _dpos;

  printf("Position updated: %f, %f, %f, derivative: %f, %f, %f\n", _m_lights[lightID].pos.x,
                                                                   _m_lights[lightID].pos.y,
                                                                   _m_lights[lightID].pos.z,
                                                                   c3s2[0], c3s2[1], c3s2[2]);

  // size
  printf("Size before: %f, %f, derivative: %f, %f\n", _m_lights[lightID].size.x,
                                                      _m_lights[lightID].size.y,
                                                      _deriv.dI_dSx, _deriv.dI_dSy);

  // _dsize += _m_lights[lightID].size;
  // float2 _dscale = _dsize / _m_lights[lightID].size;
  // _m_lights[lightID].size = _dsize;
  // instMat.m_col[2] *= _dscale.x;
  // instMat.m_col[0] *= _dscale.y;

  printf("Size updated: %f, %f, derivative: %f, %f\n", _m_lights[lightID].size.x,
                                                       _m_lights[lightID].size.y,
                                                       c3s2[3], c3s2[4]);

  // update instance transform matrix
  _accel_struct->UpdateInstance(instID, instMat);
}

void Integrator::LightEdgeSamplingInit() {
  AdamOptimizer2<float>* __ptr = new AdamOptimizer2<float>[m_lightInst.size()];
  const float _lr = 0.02f;

  for (uint i = 0; i < m_lightInst.size(); ++i) {
    __ptr[i].setParamsCount(5, _lr);
    m_adams.push_back(&(__ptr[i]));
  }
}

void Integrator::paramsIOinit(bool _do_io_stuff, const char *_fname, bool _read_write, uint _iters_to_skip) {
  if (_do_io_stuff) {
    param_io.open(_fname);

    if (!param_io) {
      if (_read_write == false) {
        throw std::runtime_error("Error: params IO init - cannot read from file that doesn't exist\n");
      }
      param_io.open(_fname, std::fstream::trunc | std::fstream::in | std::fstream::out);
    }
    read_write = _read_write;
    if (_read_write == false && _iters_to_skip) {
      const uint _lim = _iters_to_skip * m_lightInst.size();
      uint  _iter{0u};
      float _tmp{0.f};
      for (uint i = 0u; i < _lim; ++i)
        param_io >> _iter >> _tmp >> _tmp >> _tmp >> _tmp >> _tmp;
    }
  }
}

bool Integrator::loadParamsFromFile() {
  if (!param_io) return false;

  for (int i = 0; i < m_lightInst.size(); ++i) {
    DLightSourceUpdater &_tmp = m_lightInst[i];
    LightSource &_ls = m_lights[_tmp.lightID];
    float4 _newpos{0.f, 0.f, 0.f, 1.f};
    float2 _newsize{0.f};
    uint _iter{0u};

    param_io >> _iter;
    if (_iter == 0) return false;

    param_io >> _newpos.x >> _newpos.y >> _newpos.z >> _newsize.x >> _newsize.y;

    _tmp.instMat.m_col[2] *= _newsize.x / _ls.size.x;
    _tmp.instMat.m_col[0] *= _newsize.y / _ls.size.y;
    _tmp.instMat.m_col[3]  = _newpos;
    _ls.pos = _newpos;
    _ls.size = _newsize;

    m_pAccelStruct->UpdateInstance(_tmp.instID, _tmp.instMat);
  }
  return true;
}

void Integrator::saveParamsToFile(uint _iter) {
  if (!param_io) return;

  _iter += 1;
  for (int i = 0; i < m_lightInst.size(); ++i) {
    LightSource &_ls = m_lights[m_lightInst[i].lightID];
    param_io << _iter << " " << _ls.pos .x << " " << _ls.pos .y << " " << _ls.pos.z
                      << " " << _ls.size.x << " " << _ls.size.y << std::endl;
  }
}

float Integrator::PathTraceDR(uint tid, uint channels, uint a_passNum, float* out_color, const float* a_refImg) {
  m_disableImageContrib = 1;

  // todo if needed

  m_disableImageContrib = 0;
  return 0.f;
}


void Integrator::PathTraceFromInputRays(uint tid, uint channels, const RayPosAndW* in_rayPosAndNear, const RayDirAndT* in_rayDirAndFar, float* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  float4 wavelengths;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags;
  kernel_InitEyeRayFromInput(tid, in_rayPosAndNear, in_rayDirAndFar, 
                             &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &gen, &rayFlags, &mis, &wavelengths);
  
  for(uint depth = 0; depth < m_traceDepth; depth++) 
  {
    float4   shadeColor, hitPart1, hitPart2, hitPart3;
    uint instId;
    kernel_RayTrace2(tid, depth, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &hitPart3, &instId, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_SampleLightSource(tid, &rayPosAndNear, &rayDirAndFar, &wavelengths, &hitPart1, &hitPart2, &hitPart3, &rayFlags, depth,
                             &gen, &shadeColor);

    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &hitPart3, &instId, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &wavelengths, &accumColor, &accumThroughput, &gen, &mis, &rayFlags);

    if(isDeadRay(rayFlags))
      break;
  }

  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
                        &accumColor);
  
  //////////////////////////////////////////////////// same as for PathTrace

  kernel_CopyColorToOutput(tid, channels, &accumColor, &gen, out_color);
}


void Integrator::getImageIndicesCheck() {
  bool __error = false;
  for (int tid = 0; tid < m_winHeight * m_winWidth; ++tid) {
    const uint XY = m_packedXY[tid];

    const uint x = (XY & 0x0000FFFF);
    const uint y = (XY & 0xFFFF0000) >> 16;

    // forward (canon)
    const float2 pixelOffsets{0.01f, 0.01f};

    float3 rayDir = EyeRayDirNormalized((float(x) + pixelOffsets.x)/float(m_winWidth), 
                                        (float(y) + pixelOffsets.y)/float(m_winHeight), m_projInv);
    float3 rayPos = float3(0,0,0);

    transform_ray3f(m_worldViewInv, &rayPos, &rayDir);

    // forward simplified new
    float __x = x, __y = y;
    const float3 rayPos2 = mul4x3(m_worldViewInv, float3(0,0,0));

    // float4 __pos = float4(2.f * (__x + pixelOffsets.x) / m_winWidth  - 1.f,
    //                       2.f * (__y + pixelOffsets.y) / m_winHeight - 1.f, 0.f, 1.f );
    // __pos = m_projInv * __pos;
    // float3 rayDir2 = normalize(float3{ __pos.x / __pos.w, __pos.y / __pos.w, __pos.z / __pos.w });
    // float3 _pos2 = mul4x3(m_worldViewInv, 100.f*rayDir2);
    // rayDir2  = normalize(_pos2 - rayPos2);
    float3 rayDir2 = dirFromSScoords(__x + pixelOffsets.x, __y + pixelOffsets.y);


    // print forward errors
    if (rayPos.x != rayPos2.x || rayPos.y != rayPos2.y || rayPos.z != rayPos2.z)
      printf("Position is wrong: %f, %f, %f\n", rayPos2.x - rayPos.x, rayPos2.y - rayPos.y, rayPos2.z - rayPos.z);
    if (rayDir.x != rayDir2.x || rayDir.y != rayDir2.y || rayDir.z != rayDir2.z)
      printf("Direction is wrong: %e, %e, %e\n", rayDir2.x - rayDir.x, rayDir2.y - rayDir.y, rayDir2.z - rayDir.z);


    // back, copy of getImageIndicesSomehow
    uint2 _pixels = getImageIndicesSomehow(rayPos2, rayDir2);

    // print inversion errors
    if (x != _pixels.x || y != _pixels.y) {
      __error = true;
      std::cout << "Correct: " << x << ", " << y <<
                    "; got: " << _pixels.x << ", " << _pixels.y << "\n";
    }
  }
  std::cout << "Inverse ind check done" << std::endl;
  if (__error) {
    std::cout << "Index error occured, print matrices.\nProj:\n";
    std::cout << m_proj[0][0] << ", " << m_proj[0][1] << ", " << m_proj[0][2] << ", " << m_proj[0][3] << ",\n"
              << m_proj[1][0] << ", " << m_proj[1][1] << ", " << m_proj[1][2] << ", " << m_proj[1][3] << ",\n"
              << m_proj[2][0] << ", " << m_proj[2][1] << ", " << m_proj[2][2] << ", " << m_proj[2][3] << ",\n"
              << m_proj[3][0] << ", " << m_proj[3][1] << ", " << m_proj[3][2] << ", " << m_proj[3][3] << ",\nView:\n";
    std::cout << m_worldView[0][0] << ", " << m_worldView[0][1] << ", " << m_worldView[0][2] << ", " << m_worldView[0][3] << ",\n"
              << m_worldView[1][0] << ", " << m_worldView[1][1] << ", " << m_worldView[1][2] << ", " << m_worldView[1][3] << ",\n"
              << m_worldView[2][0] << ", " << m_worldView[2][1] << ", " << m_worldView[2][2] << ", " << m_worldView[2][3] << ",\n"
              << m_worldView[3][0] << ", " << m_worldView[3][1] << ", " << m_worldView[3][2] << ", " << m_worldView[3][3] << std::endl;
  }
}