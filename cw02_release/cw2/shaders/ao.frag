#version 450
#extension GL_KHR_vulkan_glsl:enable

layout(location = 0) in vec2 texCoords;
layout(location = 1) in vec3 worldPos;
layout(location = 2) in vec3 cameraPos;
layout(location = 3) in mat3 tbn;

layout(set = 1, binding = 0) uniform sampler2D aoMap;  
                
layout(location = 0) out vec4 oColor;

const float gamma = 2.2f;
void main()
{
    vec4 texColor = texture(aoMap, texCoords);
    texColor = pow(texColor, vec4(1.0 * gamma));

    oColor = vec4(texColor);
}