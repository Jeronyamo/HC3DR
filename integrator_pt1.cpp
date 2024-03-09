#include "integrator_pt.h"
#include "include/crandom.h"
#include "diff_render/adam.h"

#include <chrono>
#include <string>

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
        return _center + make_float3(-_size.x, 0.f, -_size.y);
      }
      case (1): {
        //-+
        return _center + make_float3(-_size.x, 0.f,  _size.y);
      }
      case (2): {
        //++
        return _center + make_float3( _size.x, 0.f,  _size.y);
      }
      case (3): {
        //+-
        return _center + make_float3( _size.x, 0.f, -_size.y);
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
  static float3 dsizex(int _edge) {
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
  static float3 dsizey(int _edge) {
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
    return DDot::dot(_m - _p, DCross::cross(v0 - _p, v1 - _p));
  }
};

// Parametrized edge equation, returns a point on edge
struct DEdgeEquationPrim3D {
  static float3 a(float _t, const float3 &v0, const float3 &v1) {
    return v0 + (v1 - v0) * _t; // or  v0*(1-_t) + v1*_t
  }
  static float3 da_dv0(float _t) { return { 1.f-_t, 1.f-_t, 1.f-_t }; }
  static float3 da_dv1(float _t) { return { _t, _t, _t }; }


  // for rectangle light sources
  static float3 a_rect(float _t, const float3 &_center, const float2 &_size, int _edge_ind) {
    float3 v0 = _center + float3{  _edge_ind & 2 ? _size.x : -_size.x, 0.f,
                                   (_edge_ind == 1 || _edge_ind == 2) ? _size.y : -_size.y };
    float3 v1 = _center + float3{ (_edge_ind + 1) & 2 ? _size.x : -_size.x, 0.f,
                                  (_edge_ind == 0 || _edge_ind == 1) ? _size.y : -_size.y };
    return v0 + (v1 - v0) * _t; // or  <v0*(1-_t) + v1*_t>
  }
  static float3 da_rect_dcenter(float _t, const float3 &_center, const float2 &_size, int _edge_ind) {
    // float3 v0 = _center + float3{_size0.x, 0.f, _size0.y};
    // float3 v1 = _center + float3{_size1.x, 0.f, _size1.y};
    return { 1.f, 1.f, 1.f }; // because <center * (1-_t)  +  center * _t>
  }
  static float2 da_rect_dsize(float _t, int _edge_ind) {
    float2 _sign0 = {  _edge_ind & 2 ? 1 : -1, (_edge_ind == 1 || _edge_ind == 2) ? 1 : -1 };
    float2 _sign1 = { (_edge_ind + 1) & 2 ? 1 : -1, (_edge_ind == 0 || _edge_ind == 1) ? 1 : -1 };
    return { _sign0.x * (1.f - _t) + _sign1.x * _t, _sign0.y * (1.f - _t) + _sign1.y * _t };
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




static const float _ss_compute_coef = 100.f;

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


float3 Integrator::sampleImage(const float2 &_coords, const float *_image) {
  uint2 _ind{roundf(_coords.x), roundf(_coords.y)};

  return {_image[uint(_ind.y * m_winWidth + _ind.x)    ],
          _image[uint(_ind.y * m_winWidth + _ind.x) + 1],
          _image[uint(_ind.y * m_winWidth + _ind.x) + 2]};
}

float3 Integrator::sampleImageBilinear(const float2 &_coords, const float *_image) {
  float2 _min{0.f, 0.f};
  float2 _t{modff(_coords.x, &_min.x), modff(_coords.y, &_min.y)};
  uint2 _max{_min + 1};

  float3 _00{_image[uint(_min.y * m_winWidth + _min.x)    ],
             _image[uint(_min.y * m_winWidth + _min.x) + 1],
             _image[uint(_min.y * m_winWidth + _min.x) + 2]},
         _01{_image[uint(_min.y * m_winWidth + _max.x)    ],
             _image[uint(_min.y * m_winWidth + _max.x) + 1],
             _image[uint(_min.y * m_winWidth + _max.x) + 2]},
         _10{_image[uint(_max.y * m_winWidth + _min.x)    ],
             _image[uint(_max.y * m_winWidth + _min.x) + 1],
             _image[uint(_max.y * m_winWidth + _min.x) + 2]},
         _11{_image[uint(_max.y * m_winWidth + _max.x)    ],
             _image[uint(_max.y * m_winWidth + _max.x) + 1],
             _image[uint(_max.y * m_winWidth + _max.x) + 2]};

  return        _t.x  *        _t.y  * _00 +
                _t.x  * (1.f - _t.y) * _01 +
         (1.f - _t.x) *        _t.y  * _10 +
         (1.f - _t.x) * (1.f - _t.y) * _11;
}


float3 Integrator::projectSSperspective(const float3 &_pt, const float2 &_dxy) {
  return {0.f, 0.f, 0.f};
}
float3 Integrator::projectSSdmatmul(const float4x4 &_mat, const float3 &_v, const float3 &_dv_local) {
    float4 tpt = mul4x4x4(_mat, to_float4(_v, 1.f));
    float inv_w = 1.f / tpt[3];
    float3 d_tpt03 = _dv_local * inv_w;
    float d_inv_w = dot(_dv_local, to_float3(tpt));
    float4 d_tpt = to_float4(d_tpt03, -d_inv_w * inv_w * inv_w);

    return { dot(d_tpt, _mat.col(0)), dot(d_tpt, _mat.col(1)), dot(d_tpt, _mat.col(2))};
}

// _v - point in 3D, _dv_ss - derivative of its screen space projection
float3 Integrator::projectSSderivatives(const float3 & _v, const float2 &_dv_ss) {
  const float3 _pos = mul4x3(m_worldViewInv, float3(0,0,0));
// 1
  float3 _dir = normalize(_v - _pos);
// 2
  float _coef = _ss_compute_coef;
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

// add parameter - reference image,
//                 num_pixels(? as N for mean(MSE), but maybe some other coef)
float Integrator::LightEdgeSamplingStep(float* out_color, const float* a_refImg,
                                        float* a_DerivPosImg, float* a_DerivNegImg, uint a_passNum) {
  const uint num_samples = 128u;
  const float norm_coef = 1.f / (num_samples);
  float _loss = 0.f;

  for (uint i = 0u; i < m_lightInst.size(); ++i) {
    DLightSourceUpdater _lsu = m_lightInst[i];
    LightSource _ls = m_lights[_lsu.lightID];
    DLightSource _deriv;
    float3 _p = mul4x3(m_worldViewInv, float3(0,0,0));
    // printf("CamPos: %f, %f, %f\n", _p.x, _p.y, _p.z);
    float3 _center = to_float3(_ls.pos);
    // printf("LightCenter: %f, %f, %f\n", _center.x, _center.y, _center.z);


    if (_ls.geomType == LIGHT_GEOM_RECT) {
      // Rectangle. Edges 01-12-23-30

      // v0 -> v1 -> v2 -> v3 -> v0
      float3 _v[4] = {_center + float3(-_ls.size.x, 0.f, -_ls.size.y),
                      _center + float3(-_ls.size.x, 0.f,  _ls.size.y),
                      _center + float3( _ls.size.x, 0.f,  _ls.size.y),
                      _center + float3( _ls.size.x, 0.f, -_ls.size.y)};
      float2 _v_ss[4] = {getImageSScoords(_p, normalize(_v[0] - _p)),
                         getImageSScoords(_p, normalize(_v[1] - _p)),
                         getImageSScoords(_p, normalize(_v[2] - _p)),
                         getImageSScoords(_p, normalize(_v[3] - _p))};
      // for (int i = 0; i < 4; ++i)
      //   printf("v%d: %f, %f, %f\n", i, _v[i].x, _v[i].y, _v[i].z);

      // for (int _edge_ind = 0; _edge_ind < 4; ++_edge_ind) {
      for (int n = 0; n < num_samples; ++n) {
        int _edge_ind = 0;
        float _t = sampleSSfrom2Dpoints(_v_ss, 4, _edge_ind);
        // double _t = double(std::rand()) / RAND_MAX;
        float3 v0 = _center + float3{  _edge_ind &  2 ? _ls.size.x : -_ls.size.x, 0.f,
                                      (_edge_ind == 1 || _edge_ind == 2) ? _ls.size.y : -_ls.size.y };
        float3 v1 = _center + float3{ (_edge_ind +  1) & 2 ? _ls.size.x : -_ls.size.x, 0.f,
                                      (_edge_ind == 0 || _edge_ind == 1) ? _ls.size.y : -_ls.size.y };

        // printf("V0: %f, %f, %f; V1: %f, %f, %f\n", v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
        // float3 _m = v0 + (v1 - v0) * _t;
        float3 _d0 = normalize(v0 - _p), _d1 = normalize(v1 - _p);
        float3 _edge_center = 0.5f * (v1 + v0);
        // printf("DirToEdge: %f, %f, %f; t: %f\n", _d.x, _d.y, _d.z, _t);

        float2 v0ss = getImageSScoords(_p, _d0),
               v1ss = getImageSScoords(_p, _d1),
               _center_ss = getImageSScoords(_p, normalize(_center - _p)),
               _edgec_ss  = getImageSScoords(_p, normalize(_edge_center - _p));
        float2 _n{v1ss.y - v0ss.y, v0ss.x - v1ss.x}, _m_ss = v0ss + (v1ss - v0ss) * _t;
        _n = normalize(_n);
        if (dot(_n, normalize(_edgec_ss - _center_ss)) < 0)
          _n *= -1;
        float3 f_in = getColor2(_p, dirFromSScoords((_m_ss - _n * 1.9f).x, (_m_ss - _n * 1.9f).y)),
              f_out = getColor2(_p, dirFromSScoords((_m_ss + _n * 1.9f).x, (_m_ss + _n * 1.9f).y));
        float3 f_diff{f_in - f_out};

        // float2 _dv0_ss{v1ss.y - _m_ss.y, _m_ss.x - v1ss.x}, _dv1_ss{_m_ss.y - v0ss.y, v0ss.x - _m_ss.x};
        float2 _dv0_ss{_n.x * _t, _n.y * _t}, _dv1_ss{_n.x * (1.f - _t), _n.y * (1.f - _t)}; // screen space derivatives
        // backprop from dv0ss, dv1ss to dv0, dv1
        float3 _dv0 = projectSSderivatives(v0, _dv0_ss), _dv1 = projectSSderivatives(v1, _dv1_ss);

        // image derivative and loss:
        uint __ind = ((uint)floorf(_m_ss.y) * m_winWidth + (uint)floorf( _m_ss.x)) << 2;
        // 4-channel image, but only need 3 components
        float3 _colRef = {a_refImg[__ind], a_refImg[__ind+1], a_refImg[__ind+2]},
               _col = {out_color[__ind], out_color[__ind+1], out_color[__ind+2]};
        float3 _colDiff{_col - _colRef};
        // show samples on the images
        // out_color[__ind] = 1.f;
        // out_color[__ind+1] = 0.f;
        // out_color[__ind+2] = 0.f;

        _loss += dot(_colDiff, _colDiff);

// v0
        float3 _dIdv0x = _dv0.x * f_diff,
               _dIdv0y = _dv0.y * f_diff,
               _dIdv0z = _dv0.z * f_diff;
        float3 _dMSEdv0 = { dot(_colDiff, _dIdv0x),
                            dot(_colDiff, _dIdv0y),
                            dot(_colDiff, _dIdv0z)};
        float3 _dMSEdcenter = { dot(_dMSEdv0, DLightRectParams::dcenterx()),
                                dot(_dMSEdv0, DLightRectParams::dcentery()),
                                dot(_dMSEdv0, DLightRectParams::dcenterz()) };
        _dMSEdcenter *= 1.f - _t; // v0*(1-t) + v1*t, this is for v0
        float2 _dMSEdsize = { dot(_dMSEdv0, DLightRectParams::dsizex(_edge_ind)),
                              dot(_dMSEdv0, DLightRectParams::dsizey(_edge_ind)) };
        _dMSEdsize *= 1.f - _t; // v0*(1-t) + v1*t, this is for v0

// v1
        float3 _dIdv1x = _dv1.x * f_diff,
               _dIdv1y = _dv1.y * f_diff,
               _dIdv1z = _dv1.z * f_diff;
        float3 _dMSEdv1 = { dot(_colDiff, _dIdv1x),
                            dot(_colDiff, _dIdv1y),
                            dot(_colDiff, _dIdv1z)};
        _dMSEdcenter += float3{ dot(_dMSEdv1, DLightRectParams::dcenterx()),
                                dot(_dMSEdv1, DLightRectParams::dcentery()),
                                dot(_dMSEdv1, DLightRectParams::dcenterz()) } * _t;
        _dMSEdsize += float2{ dot(_dMSEdv1, DLightRectParams::dsizex(_edge_ind)),
                              dot(_dMSEdv1, DLightRectParams::dsizey(_edge_ind)) } * _t;


        // update parameters and loss
        float _s1 = _dMSEdsize.x;
        float _s2 = _dMSEdsize.y;
        float _s3 = f_diff.z * 0.f;
        // _s1 = _dv0.x * (f_diff).x + _dv0.x * (f_diff).y + _dv0.x * (f_diff).z;
        // _s2 = _dv0.y * (f_diff).x + _dv0.y * (f_diff).y + _dv0.y * (f_diff).z;
        // _s3 = _dv0.z * (f_diff).x + _dv0.z * (f_diff).y + _dv0.z * (f_diff).z;
        if (_s1 < 0) a_DerivNegImg[__ind  ] += _s1 * -3; else a_DerivPosImg[__ind  ] += _s1 * 3;
        if (_s2 < 0) a_DerivNegImg[__ind+1] += _s2 * -3; else a_DerivPosImg[__ind+1] += _s2 * 3;
        if (_s3 < 0) a_DerivNegImg[__ind+2] += _s3 * -3; else a_DerivPosImg[__ind+2] += _s3 * 3;

        _deriv.dI_dCx += _dMSEdcenter.x;
        _deriv.dI_dCy += _dMSEdcenter.y;
        _deriv.dI_dCz += _dMSEdcenter.z;
        _deriv.dI_dSx += _dMSEdsize.x;
        _deriv.dI_dSy += _dMSEdsize.y;
      }
    }
    // Circle. TODO
    // Sphere. TODO

    float _mse_coef = 2.f / (m_winHeight * m_winWidth);
    _deriv.dI_dCx *= norm_coef * _mse_coef;
    _deriv.dI_dCy *= norm_coef * _mse_coef;
    _deriv.dI_dCz *= norm_coef * _mse_coef;
    _deriv.dI_dSx *= norm_coef * _mse_coef;
    _deriv.dI_dSy *= norm_coef * _mse_coef;
    _lsu.update(*m_adams[i], m_pAccelStruct, m_lights, _deriv, a_passNum);
    m_lightInst[i] = _lsu;
  }
  m_pAccelStruct->CommitScene();
  return _loss *= norm_coef;
}

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

  _dsize += _m_lights[lightID].size;
  float2 _dscale = _dsize / _m_lights[lightID].size;
  _m_lights[lightID].size = _dsize;
  instMat.m_col[0] *= _dscale.x;
  instMat.m_col[2] *= _dscale.y;

  printf("Size updated: %f, %f, derivative: %f, %f\n", _m_lights[lightID].size.x,
                                                       _m_lights[lightID].size.y,
                                                       c3s2[3], c3s2[4]);

  // update instance transform matrix
  _accel_struct->UpdateInstance(instID, instMat);
}

void Integrator::LightEdgeSamplingInit() {
  AdamOptimizer2<float>* __ptr = new AdamOptimizer2<float>[m_lightInst.size()];

  for (uint i = 0; i < m_lightInst.size(); ++i) {
    // m_dlights.push_back({});
    __ptr[i].setParamsCount(5, 0.003f);
    m_adams.push_back(&(__ptr[i]));
  }
  // std::cout << "Lights: " << m_lights.size() << "\n";
  // std::cout << "Light1 type = " << m_lights[1].geomType << "\n";
  // std::cout << "Pos: "  << m_lights[1].pos.x  << ", " << m_lights[1].pos.y  << ", "
  //                       << m_lights[1].pos.z  << ", " << m_lights[1].pos.w  << "\n";
  // std::cout << "Size: " << m_lights[1].size.x << ", " << m_lights[1].size.y << std::endl;
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
      uint  _iter{0u};
      float _tmp{0.f};
      for (uint i = 0u; i < _iters_to_skip; ++i)
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

    _tmp.instMat.m_col[0] *= _newsize.x / _ls.size.x;
    _tmp.instMat.m_col[2] *= _newsize.y / _ls.size.y;
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