# Hydra Renderer XML format for scenes
Hydra Renderer uses XML representation for scenes. Older (used by [HydraCore2](https://github.com/Ray-Tracing-Systems/HydraCore) but still mostly valid) and detailed description can be found in [HydraAPI repo](https://github.com/Ray-Tracing-Systems/HydraAPI/blob/master/doc/doc_xml/hydra_xml.tex).

Here we will mainly describe scene components specific for HydraCore3.

## Material models
All materials in the scene are specified inside ```<materials_lib>``` parent node as ```<material>``` child nodes. Material type is set using *type* attribute.

For example:

```
<materials_lib>
  <material id="0" name="gray_material" type="diffuse">
    <bsdf type="lambert" />
    <reflectance val="0.5 0.5 0.5" />
  </material> 
</materials_lib>
```

Some material parameters can be set with a texture or spectrum. To do this add a child node (`<texture>` or `<spectrum>`) to the respective parameter. This child node should contain a reference by id to a previously declared texture or spectrum in `<texture_lib>` or `<spectra_lib>` parts of the XML description. 

For example:
```
<textures_lib>
  <texture id="0" name="Map#0" loc="Tests/data/chunk_00000.image4ub" bytesize="16" width="2" height="2" />
  <texture id="1" name="my_texture" loc="data/chunk_00000.image4ub" bytesize="262144" width="256" height="256" />
</textures_lib>
<spectra_lib>
  <spectrum id="0" name="Au.eta" loc="data/spd/Au.eta.spd" />
  <spectrum id="1" name="Au.k" loc="data/spd/Au.k.spd" />
  <spectrum id="2" name="d50" loc="data/spd/cie.stdillum.D5000.spd" />
</spectra_lib>
<materials_lib>
  <material id="0" name="conductor_texture" type="rough_conductor">
    <bsdf type="ggx" />
    <alpha val="0.1" >
      <texture id="1" type="texref" matrix="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1" addressing_mode_u="wrap" addressing_mode_v="wrap" input_gamma="1.0" input_alpha="rgb" />
    </alpha>
    <eta val="1.5">
      <spectrum id="0" type="ref"/>
    </eta>
    <k val="1.0">
      <spectrum id="1" type="ref"/>
    </k>
  </material>
</materials_lib>
```

HydraCore3 impelements several material models:
* lambertian diffuse
* conductor 
* dielectric with ideal reflection and refraction (glass)
* plastic (diffuse material with thin dielectric coat)
* blend material (allows to combine two materials using a mask)
* gltf-like material - mix of metallic, dielectric and diffuse reflections

### Diffuse

| Node | Attributes | Texture | Spectrum |
| --- | --- | --- | --- |
| bsdf  | `type` - bsdf model, possible values - `lambert` | - | - |
| reflectance | `val` - color, possible values - 1, 3 or 4 floats | yes | yes |

Examples:

```
<materials_lib>
  <material id="0" name="red_material" type="diffuse">
    <bsdf type="lambert" />
    <reflectance val="0.5 0.0 0.0" />
  </material>
  <material id="1" name="gray_material" type="diffuse">
    <bsdf type="lambert" />
    <reflectance val="0.5" />
  </material>
  <material id="2" name="texture_material" type="diffuse">
    <bsdf type="lambert" />
    <reflectance val="0.5">
      <texture id="1" type="texref" />
    </reflectance>
  </material>  
  <material id="3" name="spectral_diffuse" type="diffuse">
    <bsdf type="lambert" />
    <reflectance val="1.0">
      <spectrum id="1" type="ref"/>
    </reflectance>
  </material> 
</materials_lib>
```

### Conductor

Models conductive materials with complex Fresnel IOR: eta (refractive index) and k (extinction coefficient). 
Should be used in *spectral* rendering mode (pass `--spectral` in command line).

Spectra can be found, for example, [here](https://refractiveindex.info/)

| Node | Attributes | Texture | Spectrum |
| --- | --- | --- | --- |
| bsdf  | `type` - bsdf model, possible values - `ggx` | - | - |
| alpha  | `val` - microfacet roughness, used to set `alpha_u` and `alpha_v` to the same value, possible values - 1 float | yes | no |
| alpha_u  | `val` - microfacet roughness in u direction, possible values - 1 float | no | no |
| alpha_v  | `val` - microfacet roughness in v direction, possible values - 1 float | no | no |
| eta  | `val` - refractive index, possible values - 1 float | no | yes |
| k  | `val` - extinction coefficient, possible values - 1 float | no | yes |

Examples:

```
<materials_lib>
  <material id="0" name="conductor" type="rough_conductor">
    <bsdf type="ggx" />
    <alpha val="0.1" />
    <eta val="1.5">
      <spectrum id="1" type="ref"/>
    </eta>
    <k val="1.0">
      <spectrum id="2" type="ref"/>
    </k>
  </material>
  <material id="1" name="conductor_anisotropic" type="rough_conductor">
    <bsdf type="ggx" />
    <alpha_u val="0.25" />
    <alpha_v val="0.01" />
    <eta val="0.0" />
    <k val="1.0" />
  </material>
  <material id="2" name="conductor_texture" type="rough_conductor">
    <bsdf type="ggx" />
    <alpha val="0.1" >
      <texture id="1" type="texref" matrix="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1" addressing_mode_u="wrap" addressing_mode_v="wrap" input_gamma="1.0" input_alpha="rgb" />
    </alpha>
    <eta val="1.5" />
    <k val="1.0" />
  </material>
</materials_lib>
```

### Ideal dielectric (glass)

Material with ideal specular reflection and refraction. Can be used with spectral refraction coefficient.


| Node | Attributes | Texture | Spectrum |
| --- | --- | --- | --- |
| int_ior  | `val` - internal index of refraction, possible values - 1 float | no | yes |
| ext_ior  | `val` - external index of refraction, possible values - 1 float | no | no |
Examples:

```
<materials_lib>
  <material id="0" name="glass" type="dielectric">
    <int_ior val="1.5" />
    <ext_ior val="1.0" />
  </material>
  <material id="1" name="spectral-dielectric" type="dielectric">
    <int_ior val="1.4585">
      <spectrum id="1" type="ref"/>
    </int_ior>
    <ext_ior val="1.00028" />
  </material>
</materials_lib>
```

### Plastic

This material model was ported from Mitsuba3 render. Essentially it represents a diffuse surface coated a thin dielectric layer.

| Node | Attributes | Texture | Spectrum |
| --- | --- | --- | --- |
| reflectance  | `val` - diffuse color, possible values - 1, 3 or 4 floats | yes | yes |
| int_ior  | `val` - internal index of refraction, possible values - 1 float | no | no |
| ext_ior  | `val` - external index of refraction, possible values - 1 float | no | no |
| alpha  | `val` - microfacet roughness, possible values - 1 float | no | no |
| nonlinear  | `val` - turn on/off nonlinear color shifts due to internal scattering, possible values - 1 or 0 | no | no |

Examples:

```
<materials_lib>
  <material id="0" name="plastic_smooth" type="plastic">
    <reflectance val="0.1 0.27 0.36" />
    <int_ior val="1.9"/>
    <ext_ior val="1.000277"/>
    <alpha val="0.0001"/>
    <nonlinear val="0"/>
  </material>
  <material id="1" name="plastic_rough" type="plastic">
    <reflectance val="1 1 1">
      <texture id="1" type="texref" matrix="-1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1" addressing_mode_u="wrap" addressing_mode_v="wrap" input_alpha="rgb" />
    </reflectance>
    <int_ior val="1.9"/>
    <ext_ior val="1.000277"/>
    <alpha val="0.2"/>
    <nonlinear val="0"/>
  </material>
</materials_lib>
```

### Blend

Meta-material used to combine two child materials (which can also be blends) using a mask.
Materials are combined as:

`weight * mat2 + (1 - weight) * mat1` 

| Node | Attributes | Texture | Spectrum |
| --- | --- | --- | --- |
| weight  | `val` - mixing weight, possible values - 1 float | yes | no |
| bsdf_1  | `id` - id of the first material (already defined in the XML description), possible values - 1 integer, | no | no |
| bsdf_2  | `id` - id of the first material (already defined in the XML description), possible values - 1 integer, | no | no |

Examples:

```
<materials_lib>
  <material id="1" name="cuprum" type="rough_conductor">
    <bsdf type="ggx" />
    <alpha val="0.1" />
    <eta val="1.5">
      <spectrum id="1" type="ref"/>
    </eta>
    <k val="1.0">
      <spectrum id="2" type="ref"/>
    </k>
  </material>
  <material id="2" name="argentum" type="rough_conductor">
    <bsdf type="ggx" />
    <alpha val="0.025" />
    <eta val="1.5">
      <spectrum id="3" type="ref"/>
    </eta>
    <k val="1.0">
      <spectrum id="4" type="ref"/>
    </k>
  </material>
  <material id="3" name="blend" type="blend">
    <weight val="1.0">
      <texture id="1" type="texref" matrix="8 0 0 0 0 8 0 0 0 0 1 0 0 0 0 1" addressing_mode_u="wrap" addressing_mode_v="wrap" input_gamma="1.0" input_alpha="rgb" />
    </weight>
    <bsdf_1 id="1" />
    <bsdf_2 id="2" />
  </material>
</materials_lib>
```