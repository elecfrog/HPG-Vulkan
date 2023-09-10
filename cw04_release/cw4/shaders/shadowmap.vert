#version 450 core

#extension GL_EXT_debug_printf : enable
layout (location = 0) in vec3 inPosition;
/*layout (location = 1) in vec3 aNormals;
layout (location = 2) in vec3 aTangents;
layout (location = 2) in vec2 aTexCoords;*/

layout(set = 0, binding = 0) uniform MVP_MATRICES
{
	mat4 lightSpaceMatrix;
}u_MVP;

// uniform mat4 M; // Model Matrix
// uniform mat4 lightSpaceMatrix;	// Light Space Matrix ( in light view space)

void main()
{
	gl_Position = u_MVP.lightSpaceMatrix * vec4(inPosition, 1.0);
}



//std140
