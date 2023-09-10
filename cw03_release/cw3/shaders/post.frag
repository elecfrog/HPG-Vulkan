#version 450

precision highp float;

layout (set = 0, binding = 0) uniform BLOOMFACTORS
{
    float scale;
    float strength;
} u_BloomFactors;

layout (set = 0, binding = 1) uniform sampler2D samplerColor;

layout (constant_id = 0) const int blurdirection = 1;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

const float weights[9] = float[]( 0.0093, 0.028002, 0.065984, 0.121703, 0.175713, 0.121703, 0.065984, 0.028002, 0.0093);

void main() 
{
//    vec3 color = texture(samplerColor, inUV).rgb;
    vec2 tex_offset = 1.0 / textureSize(samplerColor, 0) * u_BloomFactors.scale; // gets size of single texel
    vec3 color = texture(samplerColor, inUV).rgb * weights[4]; // current fragment's contribution

    for (int j = 1; j <= 4; ++j)
    {
        for (int i = 1; i <= 3; ++i)
        {
            vec2 offset = tex_offset * float(i * j);
            if (blurdirection == 1)
            {
                // H
                color += texture(samplerColor, inUV + vec2(offset.x, 0.0)).rgb * weights[4 - j] * u_BloomFactors.strength;
                color += texture(samplerColor, inUV - vec2(offset.x, 0.0)).rgb * weights[4 - j] * u_BloomFactors.strength;
            }
            else
            {
                // V
                color += texture(samplerColor, inUV + vec2(0.0, offset.y)).rgb * weights[4 - j] * u_BloomFactors.strength;
                color += texture(samplerColor, inUV - vec2(0.0, offset.y)).rgb * weights[4 - j] * u_BloomFactors.strength;
            }
        }
    }

    outColor = vec4(color, 1.0);
}
