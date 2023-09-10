#version 450
#extension GL_EXT_debug_printf : enable
layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec3 iColor;
layout(location = 2) in vec2 iTexcoord;

layout(set = 0, binding = 0) uniform UScene{
	mat4 camera;
	mat4 projection;
	mat4 projCamera;
}uScene;

layout(location = 0) out vec2 v2fTexCoord;
layout(location = 1) out vec4 v2fColor;


void main()
{
	v2fTexCoord = iTexcoord;
	v2fColor = vec4(iColor,1.f);
	gl_Position = uScene.projCamera * vec4(iPosition,1.f);


}
