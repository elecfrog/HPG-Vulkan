#version 450

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexcoord;

layout(set = 0, binding = 0) uniform UScene{
	mat4 camera;
	mat4 projection;
	mat4 projCamera;
}uScene;

layout(location = 0) out vec2 v2fTexCoord;

void main()
{
	v2fTexCoord = iTexcoord;

	gl_Position = uScene.projCamera * vec4(iPosition,1.f);
}
