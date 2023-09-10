#version 450

//specify the precision of floating point
precision highp float;

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in mat3 inTBN;
layout(location = 5) in vec4 inLightSpacePosition;

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
	vec4  direction;
	float cutoff;
	vec3  paddings;
}light;

layout(set = 3, binding = 0) uniform sampler2DShadow shadowmap;  

layout(location = 0) out vec4 oColor;

const float PI = 3.14159265359;
const vec3 dielectric_constant = vec3(0.04);
const float gamma = 2.2;

// basic projection shadow
float MytextureProj(vec4 shadowCoord, vec2 off)
{
	float shadow = 1.0;
	float bias = 0.00005f;

	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
	{
	    vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
        projCoords.y += off.y;
        projCoords.x += off.x;

		float dist = textureProj(shadowmap, vec4(projCoords, shadowCoord.z - bias));
		if ( shadowCoord.w > 0.0 && dist < shadowCoord.z - bias) 
		{
			shadow = 0.f;
		}
	}
	return shadow;
}

float filterPCF(vec4 sc)
{
	ivec2 texDim = textureSize(shadowmap, 0).xy;
	float scale = 0.75;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;
	
	for (int x = -range; x <= range; x++) {
		for (int y = -range; y <= range; y++) {
			shadowFactor += MytextureProj(sc, vec2(dx*x, dy*y));
			count++;
		}
	}
	return shadowFactor / count;
}

vec3 calculateNormal()
{
	vec3 n = texture(normalMap, inUV).rgb * 2.0 - 1.0;
	n = inTBN * n;  // convert to [0, 1]
	return normalize(n);    
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
	vec3 L = normalize(light.lightPos.xyz - inWorldPosition);

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

float LinearizeDepth(float depth)
{
  float n = 0.1f;
  float f = 100.f;
  float z = depth;
  return (2.0 * n) / (f + n - z * (f - n));	
}


void main()
{
	vec3  albedo    = texture(albedoMap, inUV).rgb;
	float metallic  = texture(metallicMap, inUV).r;

	float roughness = texture(roughnessMap, inUV).r;
	float ao = texture(albedoMap, inUV).a;
	
	if(ao < 0.1)
	{
		discard;
	}
	else
	{
		vec3 n = calculateNormal();
	
		float alpha_p = shininess(roughness);

		// view direction, point to the camera
		vec3 v = normalize(light.viewPos.xyz - inWorldPosition);
		vec3 l = normalize(light.lightPos.xyz - inWorldPosition);
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

   
		// float shadow = textureProj(inLightSpacePosition / inLightSpacePosition.w, vec2(0.f));  // ShadowCalculation(inLightSpacePosition,n, l);
		float shadow = filterPCF(inLightSpacePosition / inLightSpacePosition.w);

		vec3 lightContribution = vec3(0.f);
	
		vec3 ambientContribution = vec3(0.f);

		lightContribution += (specularBRDF + diffuseBRDF ) * vec3(light.color) * NdotL * ( shadow ) ; 

		ambientContribution = albedo * ao * 0.1;

		vec3 color = lightContribution + ambientContribution;
		
		color = pow(color, vec3(1.0 * gamma));
	  
		oColor = vec4(color, 1.0);

	}
}