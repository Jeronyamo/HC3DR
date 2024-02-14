#include "integrator_pt_scene.h"

std::vector<float> CreateSphericalTextureFromIES(const std::string& a_iesData, int* pW, int* pH);

LightSource LoadLightSourceFromNode(hydra_xml::LightInstance lightInst, const std::string& sceneFolder, bool a_spectral_mode,
                                    std::vector< std::shared_ptr<ICombinedImageSampler> >& a_textures, 
                                    float4& a_envColor, 
                                    std::vector<unsigned int>& a_actualFeatures)
{
  const std::wstring ltype = lightInst.lightNode.attribute(L"type").as_string();
  const std::wstring shape = lightInst.lightNode.attribute(L"shape").as_string();
  const std::wstring ldist = lightInst.lightNode.attribute(L"distribution").as_string();

  const float sizeX        = lightInst.lightNode.child(L"size").attribute(L"half_width").as_float();
  const float sizeZ        = lightInst.lightNode.child(L"size").attribute(L"half_length").as_float();
  float power              = lightInst.lightNode.child(L"intensity").child(L"multiplier").text().as_float();
  if (power == 0.0f) power = lightInst.lightNode.child(L"intensity").child(L"multiplier").attribute(L"val").as_float();
  if (power == 0.0f) power = 1.0f;

  float4 color     = GetColorFromNode(lightInst.lightNode.child(L"intensity").child(L"color"), a_spectral_mode != 0);
  auto matrix      = lightInst.matrix;
  auto lightSpecId = GetSpectrumIdFromNode(lightInst.lightNode.child(L"intensity").child(L"color"));  

  LightSource lightSource{};
  lightSource.specId   = lightSpecId;
  lightSource.mult     = power;
  lightSource.distType = LIGHT_DIST_LAMBERT;
  lightSource.iesId    = uint(-1);
  lightSource.texId    = uint(-1);
  lightSource.flags    = 0;
  if(ltype == std::wstring(L"sky"))
  {
    a_envColor = color;
  }
  else if(ltype == std::wstring(L"directional"))
  {
    lightSource.pos       = lightInst.matrix * float4(0.0f, 0.0f, 0.0f, 1.0f);
    lightSource.norm      = normalize(lightInst.matrix * float4(0.0f, -1.0f, 0.0f, 0.0f));
    lightSource.intensity = color;
    lightSource.geomType  = LIGHT_GEOM_DIRECT;
  }
  else if(shape == L"rect" || shape == L"disk")
  {
    lightSource.pos       = lightInst.matrix * float4(0.0f, 0.0f, 0.0f, 1.0f);
    lightSource.norm      = normalize(lightInst.matrix * float4(0.0f, -1.0f, 0.0f, 0.0f));
    lightSource.intensity = color;
    lightSource.geomType  = (shape == L"rect") ? LIGHT_GEOM_RECT : LIGHT_GEOM_DISC;
    // extract scale and rotation from transformation matrix
    float3 scale;
    for(int i = 0; i < 3; ++i)
    {
      float4 vec = matrix.col(i);
      scale[i] = length3f(vec);
    }
    lightSource.matrix = matrix;
    lightSource.matrix.set_col(3, float4(0,0,0,1));
    lightSource.size = float2(sizeZ, sizeX);       ///<! Please note tha we HAVE MISTAKEN with ZX order in Hydra2 implementation
    if(shape == L"disk")
    {
      lightSource.size.x = lightInst.lightNode.child(L"size").attribute(L"radius").as_float();
      lightSource.pdfA   = 1.0f / (LiteMath::M_PI * lightSource.size.x *lightSource.size.x * scale.x * scale.z);
    }
    else
      lightSource.pdfA   = 1.0f / (4.0f * lightSource.size.x * lightSource.size.y * scale.x * scale.z);
  }
  else if (shape == L"sphere")
  {
    float radius = lightInst.lightNode.child(L"size").attribute(L"radius").as_float();
    float3 scale; 
    for(int i = 0; i < 3; ++i)
    {
      float4 vec = matrix.col(i);
      scale[i] = length3f(vec);
    }
    radius = radius*scale.x; // support for uniform scale, assume scale.x == scale.y == scale.z
    if(std::abs(scale.x - scale.y) > 1e-5f || std::abs(scale.x - scale.z) > 1e-5f)
    {
      std::cout << "[Integrator::LoadScene]: ALERT!" << std::endl;
      std::cout << "[Integrator::LoadScene]: non uniform scale for spherical light instance matrix is not supported: (" << scale.x << ", " << scale.y << ", " << scale.z << ")" << std::endl; 
    }
    lightSource.pos       = lightInst.matrix * float4(0.0f, 0.0f, 0.0f, 1.0f);
    lightSource.norm      = float4(0.0f, -1.0f, 0.0f, 0.0f);
    lightSource.intensity = color;
    lightSource.geomType  = LIGHT_GEOM_SPHERE;
    lightSource.matrix    = float4x4{};
    lightSource.size      = float2(radius, radius);
    lightSource.pdfA      = 1.0f / (4.0f*LiteMath::M_PI*radius*radius);
  }
  else if (shape == L"point")
  {
    lightSource.pos       = lightInst.matrix * float4(0.0f, 0.0f, 0.0f, 1.0f);
    lightSource.norm      = normalize(lightInst.matrix * float4(0.0f, -1.0f, 0.0f, 0.0f));
    lightSource.intensity = color;
    lightSource.geomType  = LIGHT_GEOM_POINT;
    lightSource.distType  = (ldist == L"uniform" || ldist == L"omni" || ldist == L"ies") ? LIGHT_DIST_OMNI : LIGHT_DIST_LAMBERT;
    lightSource.pdfA      = 1.0f;
    lightSource.size      = float2(0,0);
    lightSource.matrix    = float4x4{};
  }
    
  auto iesNode = lightInst.lightNode.child(L"ies");
  if(iesNode != nullptr)
  {
    const std::wstring iesFileW = std::wstring(sceneFolder.begin(), sceneFolder.end()) + L"/" + iesNode.attribute(L"loc").as_string();\
    const std::string  iesFileA = hydra_xml::ws2s(iesFileW);
    
    int w,h;
    std::vector<float> sphericalTexture = CreateSphericalTextureFromIES(iesFileA.c_str(), &w, &h);
    
    // normalize ies texture
    //
    float maxVal = 0.0f;
    for (auto i = 0; i < sphericalTexture.size(); i++)
      maxVal = std::max(maxVal, sphericalTexture[i]);

    if(maxVal == 0.0f)
    {
      std::cerr << "[ERROR]: broken IES file (maxVal = 0.0): " << iesFileA.c_str() << std::endl;
      maxVal = 1.0f;
    }

    float invMax = 1.0f / maxVal;
    for (auto i = 0; i < sphericalTexture.size(); i++)
    {
      float val = invMax*sphericalTexture[i];
      sphericalTexture[i] = val;
    }
    ////
    auto pTexture = std::make_shared< Image2D<float> >(w, h, sphericalTexture.data());
    pTexture->setSRGB(false);

    Sampler sampler;
    sampler.filter   = Sampler::Filter::LINEAR; 
    sampler.addressU = Sampler::AddressMode::CLAMP;
    sampler.addressV = Sampler::AddressMode::CLAMP;
    
    a_textures.push_back(MakeCombinedTexture2D(pTexture, sampler));
    lightSource.iesId = uint(a_textures.size()-1);
    
    auto matrixAttrib = iesNode.attribute(L"matrix");
    if(matrixAttrib != nullptr)
    {
      float4x4 mrot           = LiteMath::rotate4x4Y(DEG_TO_RAD*90.0f);
      float4x4 matrixFromNode = hydra_xml::float4x4FromString(matrixAttrib.as_string());
      float4x4 instMatrix     = matrix;
      instMatrix.set_col(3, float4(0,0,0,1));
      lightSource.iesMatrix = mrot*transpose(transpose(matrixFromNode)*instMatrix);
      lightSource.iesMatrix.set_col(3, float4(0,0,0,1));
    }
    
    int pointArea = iesNode.attribute(L"point_area").as_int();
    if(pointArea != 0)
      lightSource.flags |= LIGHT_FLAG_POINT_AREA;

    a_actualFeatures[Integrator::KSPEC_LIGHT_IES] = 1;
  }

  return lightSource;
}