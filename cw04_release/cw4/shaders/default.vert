#version 450
// #extension GL_EXT_debug_printf : enable
layout(location = 0) in vec3 inPositions;
layout(location = 1) in vec3 inNormals;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inTangents;

//std140
layout(set = 0, binding = 0, std140) uniform MVP_MATRICES
{
	mat4 VP;
}u_MVP;

layout(set = 4, binding = 0, std140) uniform LIGHT_SPACE
{
	mat4 lightSpaceMatrix;

}lightSpace;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec2 outUV;
layout(location = 2) out mat3 outTBN;
layout(location = 5) out vec4 outLightSpacePosition;


const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );


void main()
{
	outWorldPosition = inPositions;
	outUV = inUV;
	
	vec3 bitangent  = cross(inTangents, inNormals);
	outTBN = mat3(inTangents, bitangent, inNormals);						// TBN Matrix in world space

	//bias matrix
	outLightSpacePosition = biasMat * lightSpace.lightSpaceMatrix * vec4(inPositions,1.f); 
	// outLightSpacePosition = LightSpacePosition.xyz;
	
	gl_Position = u_MVP.VP * vec4(inPositions,1.f);

}

