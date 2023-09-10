#version 450
#extension GL_EXT_debug_printf : enable
layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexcoord;
layout(location = 2) in vec4 inTBNQuat;
layout(location = 3) in uint inCompressedTBNQuat;

//std140
layout(set = 0, binding = 0) uniform MVP_MATRICES
{
	mat4 VP;
}mvp;

layout(location = 0) out vec2 texCoords;
layout(location = 1) out vec3 worldPos;
layout(location = 2) out vec3 cameraPos;
layout(location = 3) out mat3 tbn;

vec3 unpackNormal(vec4 packedNormal)
{
    return normalize(packedNormal.xyz * 2.0 - 1.0);
}

mat3 unpackTBN(vec4 packedTBN) {
    vec4 quat = normalize(vec4(packedTBN.x, packedTBN.y, packedTBN.z, 0.0f));
    bool mirrored = (packedTBN.w < 0.0f);

    float x = quat.x;
    float y = quat.y;
    float z = quat.z;
    float w = quat.w;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float xw = x * w;
    float yy = y * y;
    float yz = y * z;
    float yw = y * w;
    float zz = z * z;
    float zw = z * w;

    mat3 tbnMatrix;

    tbnMatrix[2] = vec3(1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (xz - yw));
    tbnMatrix[1] = vec3(2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw));
    tbnMatrix[0] = vec3(2 * (xz + yw), 2 * (yz - xw), 1 - 2 * (xx + yy));

    if (mirrored) {
        tbnMatrix[0] = -tbnMatrix[0];
    }

    return tbnMatrix;
}

mat3 quaternionToTBNMatrix(vec4 quat)
{
    float x = quat.x;
    float y = quat.y;
    float z = quat.z;
    float w = quat.w;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float xw = x * w;
    float yy = y * y;
    float yz = y * z;
    float yw = y * w;
    float zz = z * z;
    float zw = z * w;

    mat3 tbnMatrix;

    tbnMatrix[0] = vec3(1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (xz - yw));
    tbnMatrix[1] = vec3(2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw));
    tbnMatrix[2] = vec3(2 * (xz + yw), 2 * (yz - xw), 1 - 2 * (xx + yy));

    return tbnMatrix;
}	

//
// Helper: remap floating number in range [min1..max1] to range [min2..max2]
float remap( float x, float min1, float max1, float min2, float max2 ) 
{
    return ( ( ( x - min1 ) / ( max1 - min1 ) ) * ( max2 - min2 ) ) + min2;
}

float decode_unorm(uint x, uint N) 
{
    return float(x) / float((1 << N) - 1);
}

float decode_snorm(uint x, uint N) 
{
    float value = ( x & 1 ) > 0 ? -1.0 : 1.0;
    return decode_unorm( x >> 1, N - 1 ) * value;
}

float rsqrt( float number ) {
    return float(1 / sqrt(number));
}


vec4 decode_quaternion()
{
    const float rmax = 1 / sqrt(2); 
    const float rmin = -1 * rmax;
  
    vec4 ret; 
    switch( inCompressedTBNQuat >> 30 ) 
    {
        default: 
        case 0:
            ret.y = decode_snorm( ( inCompressedTBNQuat >> 20 )& 0x3FF, 10 );
            ret.z = decode_snorm( ( inCompressedTBNQuat >> 10 )& 0x3FF, 10 );
            ret.w = decode_snorm( ( inCompressedTBNQuat >>  0 )& 0x3FF, 10 );
            ret.y = remap( ret.y, -1, 1, rmin, rmax );
            ret.z = remap( ret.z, -1, 1, rmin, rmax );
            ret.w = remap( ret.w, -1, 1, rmin, rmax );
            ret.x = sqrt( 1 - ret.y * ret.y - ret.z * ret.z - ret.w * ret.w );
        break; 
        case 1:
             ret.x = decode_snorm( ( inCompressedTBNQuat >> 20 ) & 0x3FF, 10  );
             ret.z = decode_snorm( ( inCompressedTBNQuat >> 10 ) & 0x3FF, 10  );
             ret.w = decode_snorm( ( inCompressedTBNQuat >>  0 ) & 0x3FF, 10  );
             ret.x = remap(  ret.x, -1, 1, rmin, rmax );
             ret.z = remap(  ret.z, -1, 1, rmin, rmax );
             ret.w = remap(  ret.w, -1, 1, rmin, rmax );
             ret.y = sqrt( 1 -  ret.x *  ret.x -  ret.z *  ret.z -  ret.w * ret. w );
        break; 
        case 2:
             ret.x = decode_snorm( ( inCompressedTBNQuat >> 20 ) & 0x3FF, 10  );
             ret.y = decode_snorm( ( inCompressedTBNQuat >> 10 ) & 0x3FF, 10  );
             ret.w = decode_snorm( ( inCompressedTBNQuat >>  0 ) & 0x3FF, 10  );
             ret.x = remap( ret.x, -1, 1, rmin, rmax );
             ret.y = remap( ret.y, -1, 1, rmin, rmax );
             ret.w = remap( ret.w, -1, 1, rmin, rmax );
             ret.z = sqrt( 1 -  ret.x * ret. x -  ret.y *  ret.y -  ret.w *  ret.w );
        break; 
        case 3:
             ret.x = decode_snorm( ( inCompressedTBNQuat >> 20 ) & 0x3FF, 10 );
             ret.y = decode_snorm( ( inCompressedTBNQuat >> 10 ) & 0x3FF, 10 );
             ret.z = decode_snorm( ( inCompressedTBNQuat >>  0 ) & 0x3FF, 10 );
             ret.x = remap(  ret.x, -1, 1, rmin, rmax );
             ret.y = remap(  ret.y, -1, 1, rmin, rmax );
             ret.z = remap(  ret.z, -1, 1, rmin, rmax );
             ret.w = sqrt( 1 -  ret.x *  ret.x -  ret.y *  ret.y -  ret.z * ret.z );
    }
    return ret;
}



void main()
{
	texCoords = iTexcoord;
	worldPos = iPosition;

    tbn = quaternionToTBNMatrix(inTBNQuat);
    //tbn = quaternionToTBNMatrix(decode_quaternion());

	// cameraPos = vec3(uScene.camera[3]);
	gl_Position = mvp.VP * vec4(iPosition,1.f);
}

