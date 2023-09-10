#version 450
#extension GL_KHR_vulkan_glsl:enable

//specify the precision of floating point
precision highp float;

layout(location = 0) in vec2 texCoords;
layout(location = 1) in vec3 worldPos;
layout(location = 2) in vec3 cameraPos;
layout(location = 3) in mat3 tbn;

layout(set = 1, binding = 0) uniform sampler2D albedoMap;
layout(set = 1, binding = 1) uniform sampler2D metallicMap;
layout(set = 1, binding = 2) uniform sampler2D roughnessMap;   
layout(set = 1, binding = 3) uniform sampler2D normalMap;
layout(set = 1, binding = 4) uniform sampler2D aoMap;  
                
layout(set = 2, binding = 0) uniform LIGHT
{
	vec4  lightPos;
	vec3  color;
    float padding;
    vec4  viewPos;
}light;

layout(location = 0) out vec4 oColor;

const float PI = 3.14159265359;
const vec3 dielectric_constant = vec3(0.04);
const float gamma = 2.2;

vec3 calculateNormal()
{
    vec3 tangentspace_normal = normalize(texture(normalMap, texCoords).xyz) * 2.0 - 1.0;

    return normalize(tbn * tangentspace_normal);
}

// Frensel_Term
vec3 Frensel_Term(vec3 F0, vec3 albedo, float metallic, float HdotV)
{
    return F0 + (1 - F0) * pow( (1.0 - HdotV), 5.0 );
}

// normal distribution function D
float Distribution_BlinnPhong(float ap, float NdotH)
{
    return ((ap + 2) / (2 * PI) ) * max(pow( NdotH, ap),0);
}

// GGX/Towbridge-Reitz normal distribution function.
// Uses Disney's reparametrization of alpha = roughness^2
float Distribution_ndfGGX(float cosLh, float roughness)
{
	float alpha   = roughness * roughness;
	float alphaSq = alpha * alpha;
	
	float  denom = (cosLh * cosLh) * (alphaSq - 1.0) + 1.0;
	return alphaSq / (PI * denom * denom);
}

vec3 Lambertian(vec3 F, float metallic)
{
    return ( 1 - F ) * ( 1 - metallic );
}

float GaSchlickGGX_Item(float ndoth, float ndotv, float vdoth, float ndotl)
{
    return min( 1 , min( 2 * ( ndoth * ndotv ) / vdoth , 2 * ( ndoth * ndotl)  / vdoth ) );
}

// implement the light square-law falloff, where I setup the radius of the light source to be 2.0
float falloff()
{
    vec3 L = normalize(light.lightPos.xyz - worldPos);

    float dist = length(L);

    L = normalize(L);

    float atten = (2.) / (pow(dist, 2.0) + 1.0);

    return atten;
}

// alpha_p = 2 / ( pow(roughness, 4) + 0.0001 ) - 2;
float shininess(float roughness)
{
    return 2 / ( pow(roughness, 4) + 0.0001 ) - 2;
}

void main()
{
    vec3  albedo    = texture(albedoMap, texCoords).rgb;
    float metallic  = texture(metallicMap, texCoords).r;

    float roughness = texture(roughnessMap, texCoords).r;

    vec3 tangentNormal = normalize(texture(normalMap, texCoords).xyz);

    vec3 n = calculateNormal();
    

    float alpha_p = shininess(roughness);

    // view direction, point to the camera
    vec3 v = normalize(light.viewPos.xyz - worldPos);
    vec3 l = normalize(light.lightPos.xyz - worldPos);
    vec3 h = normalize(v + l);  

    // 
    float HdotV = dot(h, v);
    float NdotH = max( dot(n, h), 0.0 );
    float NdotV = max( dot(n, v), 0.0 );
    float NdotL = max( dot(n, l), 0.0 );
    float VdotH = dot(v, h);

    vec3  F0 = (1 - metallic) * dielectric_constant + metallic * albedo;
    
    // F = k_s in the reflection equation
    vec3  F = Frensel_Term(F0, albedo, metallic, HdotV); 
    //float D = Distribution_BlinnPhong(alpha_p, NdotH);
    float D = Distribution_ndfGGX(NdotH, roughness);

    vec3 diffuseBRDF = Lambertian(F, metallic) * (albedo/ PI);
   
   
    float cosLi = max(0.0, dot(n, l));
   
    float G = GaSchlickGGX_Item(NdotH, NdotV, VdotH, NdotL);
   
    vec3 specularBRDF = (F * D * G) / max(0.000001, 4.0 * NdotL * NdotV);
    vec3 lightContribution = (specularBRDF + diffuseBRDF ) * vec3(light.color) * NdotL * falloff(); 
    vec3 ambientContribution = albedo * 0.02;
    vec3 color = lightContribution + ambientContribution;
    color = pow(color, vec3(1.0 * gamma));
    oColor = vec4(n, 1.0);
}