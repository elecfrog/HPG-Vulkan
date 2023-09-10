#version 450
#extension GL_KHR_vulkan_glsl:enable
#extension GL_EXT_debug_printf : enable

layout(location = 0) in vec2 v2fTexCoord;
layout(location = 1) in vec4 v2fColor;

layout(set = 1,binding = 0) uniform sampler2D uTexColor;

layout(location = 0) out vec4 oColor;

layout(push_constant) uniform MeshDensity {
	float density;
    float min_density;
    float max_density;
} meshDensity;


void main()
{
	float density = meshDensity.density * 20.f;
	vec3 mixColor ;
	if(density > 1)
		mixColor = mix(vec3(1.0f, .0f, 0.f), vec3(1.0f,1.f,1.f), density);
	if(density > 0.5f)
		mixColor = mix(vec3(1.0f, 0.0f,0.f), vec3(1.0f, 1.0f,1.f), density);
	else
		mixColor = mix(vec3(0.0f, 1.0f,0.f), vec3(1.0f, 1.0f,1.f), density);
	oColor = vec4(mixColor, 1.0f);	
}
