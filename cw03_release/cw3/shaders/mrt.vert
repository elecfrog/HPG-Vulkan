#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormals;

//std140
layout(set = 0,binding = 0) uniform VIEW
{
	mat4 camera;
	mat4 projection;
	mat4 projCamera;
} u_View;

layout(location = 0) out vec2 outTexCoords;
layout(location = 1) out vec3 outPosition;
layout(location = 2) out vec3 outNormal;

void main()
{
	outTexCoords = inUV;
	outPosition = inPosition;
	outNormal = inNormals;

	gl_Position = u_View.projCamera * vec4(inPosition,1.f);

}

