#version 450
precision highp float;

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inPosition;
layout(location = 2) in vec3 inNormal;


layout(push_constant) uniform PUSHCONSTANT
{
    vec4 cameraPos;
} p_Camera;

layout(set = 1, binding = 0) uniform UCOLOR 
{
    vec4 basecolor;
    vec4 emissive;
    float roughness;
    float metalness;
} u_Colors;

layout(set = 2,binding = 0) uniform ULIGHT
{
	vec4 position;
	vec4 color;
}u_Light;

layout(set = 3,binding = 0) uniform sampler2D albedoMap;
layout(set = 3,binding = 1) uniform sampler2D metallicMap;
layout(set = 3,binding = 2) uniform sampler2D roughnessMap;   

const float PI = 3.14159265359;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outPosition;
layout(location = 3) out vec4 outBrightness;

float ndGGX(float NdotH, float roughness)
{
	float shiness = pow(roughness, 4);
	float denom = (NdotH * NdotH * (shiness- 1.0) + 1.0);
	denom = PI * denom * denom;
	return shiness / denom;
}

vec3 Fresnel(vec3 F0, float VdotH)
{
	return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

void main()
{
    vec3 N = normalize(inNormal);
    vec3 V = normalize(vec3(p_Camera.cameraPos) - inPosition);
    vec3 L = normalize(u_Light.position.xyz - inPosition);
    vec3 H = normalize(V + L);  

    float NdotH = clamp(dot(N, H), 0.0 ,1.0);
    float NdotV = clamp(dot(N, V), 0.0 ,1.0);
    float NdotL = clamp(dot(N, L), 0.0 ,1.0);
    float LdotH = clamp(dot(L, H), 0.0, 1.0);
    float VdotH = dot(V, H);
    
    vec3 albedo = vec3(texture(albedoMap,inUV)) * vec3(u_Colors.basecolor);
    float metallic = texture(metallicMap,inUV).r * u_Colors.metalness;
    float roughness= texture(roughnessMap,inUV).r * u_Colors.roughness;

    vec3 F0 = mix(vec3(0.04), vec3(albedo), u_Colors.metalness);

    vec3 color = vec3(0.0);

    if (NdotL > 0.0)
    {
        vec3  F = Fresnel(F0, VdotH);
	    float D = ndGGX(NdotH, roughness);
        float G = min( 1 , min( 2 * ( NdotH * NdotV ) / VdotH , 2 * ( NdotH * NdotL)  / VdotH ) );

        vec3 diffuseBRDF = ( vec3(albedo)/ PI ) * ( vec3(1) - F ) * ( 1 - metallic);
   
        vec3 specularBRDF = F * D * G / max(0.000001, 4.0 * NdotL * NdotV);
   
        color += (specularBRDF +  diffuseBRDF) * NdotL + albedo * 0.02 ;
    }

   	const float luminancePBR = dot(color, vec3(0.2126, 0.7152, 0.0722));
	vec3 scaled = color / (luminancePBR+ 1.0);
    vec3 mapped = scaled/ (scaled + 1.0);
    outColor = vec4(mapped , 1.0);
     
    float luminanceEmssive = dot(colors.emissive.rgb, vec3(0.2126, 0.7152, 0.0722));
	scaled = colors.emissive.rgb / (luminanceEmssive+ 1.0);
    mapped = scaled / (scaled + 1.0);
    outBrightness = vec4(mapped , 1.0);
    outNormal = vec4(N,1.0);
    outPosition = vec4(inPosition,1.0);
}