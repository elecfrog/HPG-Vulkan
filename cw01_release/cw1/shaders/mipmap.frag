#version 450
#extension GL_KHR_vulkan_glsl	: enable
#extension GL_EXT_debug_printf	: enable

layout(location = 0) in vec2 v2fTexCoord;
layout(location = 1) in vec4 v2fColor;

layout(set = 1,binding = 0) uniform sampler2D uTexColor;

layout(location = 0) out vec4 oColor;

void main()
{
	vec4 tmpColor = texture(uTexColor,v2fTexCoord).rgba;
	if(tmpColor.a == 0.0f)
	{
		oColor = v2fColor;
	}
	else
	{
		float mipmapLevel = textureQueryLod(uTexColor, v2fTexCoord).x;
		vec4 nearest     = vec4(1.0, 0.1, 0.1, 1.0); 
		vec4 secondNearest  = vec4(0.5, 0.7, 0.1, 1.0); 
		vec4 secondFarestLevel = vec4(0.1, 0.4, 1.0, 1.0); 
		vec4 farest      = vec4(0.1, 0.1, 0.1, 1.0); 

		vec4 color = mix(nearest, secondNearest, smoothstep(0.0, 1.0, mipmapLevel));
		color = mix(color, secondFarestLevel, smoothstep(1.0, 2.0, mipmapLevel));
		color = mix(color, farest, smoothstep(2.0, 3.0, mipmapLevel));
		oColor =color;
	}
}
