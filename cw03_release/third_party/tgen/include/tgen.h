/**
 * TGen - Simple Tangent Generator
 *
 * 2016 by Max Limper, Fraunhofer IGD
 *
 * This code is public domain.
 * 
 */

#ifndef TGEN_H
#define TGEN_H

#include <cstdint>
#include <vector>
#include <cmath>

namespace tgen
{

    auto lengthVec3 = [](const float* vec) {
        return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    };
    
    //-------------------------------------------------------------------------

    //typedef std::size_t VIndexT;
    //typedef double      RealT;
    
    //-------------------------------------------------------------------------

    /**
     * Computes tangents and bitangents for each corner of a triangle.
     * In an indexed triangle list, each entry corresponds to one corner.
     *
     * Requirements for input:
     * - triIndicesPos and triIndicesUV must be of the same size
     * - triIndicesPos refers to (at maximum) num3DVertices different elements
     * - triIndicesUV  refers to (at maximum) numUVVertices different elements
     * - positions3D must have a size of num3DVertices*3
     * - uvs2D       must have a size of numUVVertices*2
     *
     * Output:
     * - cTangents3D   has numTriIndices*3 entries, contains per-corner tangents
     * - cBitangents3D has numTriIndices*3 entries, contains per-corner bitangents
     */
    void computeCornerTSpace(const std::vector<uint32_t> & triIndicesPos,
                             const std::vector<uint32_t> & triIndicesUV,
                             const std::vector<float>   & positions3D,
                             const std::vector<float>   & uvs2D,
                             std::vector<float>         & cTangents3D,
                             std::vector<float>         & cBitangents3D);

    //-------------------------------------------------------------------------

    /**
     * Computes per-vertex tangents and bitangents, for each UV vertex.
     * This is done by averaging vectors across each wedge (all vertex instances
     * sharing a common UV vertex).
     *
     * The basic method used here currently makes the assumption that UV
     * vertices are not being re-used across multiple 3D vertices.
     * However, the multi-indexed structure used here allows a single 3D vertex
     * to be split in UV space (to enable usage of UV charts without explicitly
     * cutting / splitting the 3D mesh).
     *
     * Requirements about input:
     * - triIndicesUV refers to (at maximum) numUVVertices different elements
     * - cTangents3D   has numTriIndices*3 entries, contains per-corner tangents
     * - cBitangents3D has numTriIndices*3 entries, contains per-corner bitangents     
     *
     * Output:
     * - vTangents3D   has numUVVertices*3 entries
     * - vBitangents3D has numUVVertices*3 entries
     */
    void computeVertexTSpace(const std::vector<uint32_t> & triIndicesUV,
                             const std::vector<float>   & cTangents3D,
                             const std::vector<float>   & cBitangents3D,
                             std::size_t                  numUVVertices,
                             std::vector<float>         & vTangents3D,
                             std::vector<float>         & vBitangents3D);

    //-------------------------------------------------------------------------

    /**
     * Makes the given tangent frames orthogonal.
     *
     * Input arrays must have the same number of (numUVVertices*3) elements.
     */
    void orthogonalizeTSpace(const std::vector<float> & normals3D,
                             std::vector<float>       & tangents3D,
                             std::vector<float>       & bitangents3D);

    //-------------------------------------------------------------------------

    /**
     * Makes the given tangent frames orthogonal.
     *
     * Input arrays must have the same number of (numUVVertices*3) elements.
     *
     * The output will be an array with 4-component versions of the tangents,
     * where the first three components are equivalent to the input tangents
     * and the fourth component contains a factor for flipping a computed
     * bitangent, if the original tangent frame was right-handed.
     * Concretely speaking, the 3D bitangent can be obtained as:
     *   bitangent = tangent4.w * (normal.cross(tangent4.xyz))
     */
    void computeTangent4D(const std::vector<float> & normals3D,
                          const std::vector<float> & tangents3D,
                          const std::vector<float> & bitangents3D,
                          std::vector<float>       & tangents4D);
    
    //-------------------------------------------------------------------------

}

/**
 * TGen - Simple Tangent Generator
 *
 * 2016 by Max Limper, Fraunhofer IGD
 *
 * This code is public domain.
 * 
 */

//#include <tgen.h>


// local utility definitions
namespace
{
    const float DenomEps = 1e-10;

    //-------------------------------------------------------------------------

    inline void addVec3(const float * a,
                        const float * b,
                        float       * result)
    {
        result[0] = a[0] + b[0];
        result[1] = a[1] + b[1];
        result[2] = a[2] + b[2];
    }

    //-------------------------------------------------------------------------

    inline void subVec3(const float * a,
                        const float * b,
                        float       * result)
    {
        result[0] = a[0] - b[0];
        result[1] = a[1] - b[1];
        result[2] = a[2] - b[2];
    }
    
    //-------------------------------------------------------------------------

    inline void multVec3(const float * a,
                         const float   s,
                         float       * result)
    {
        result[0] = a[0] * s;
        result[1] = a[1] * s;
        result[2] = a[2] * s;
    }

    //-------------------------------------------------------------------------

    void normalizeVec3(float * v)
    {
        float len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (len > 1e-6f) {
            multVec3(v, 1.0 / len, v);
        }
    }

    //-------------------------------------------------------------------------

    inline float dotProd(const float * a,
                               const float * b )
    {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    //-------------------------------------------------------------------------

    inline void crossProd(const float * a,
                          const float * b,
                          float       * result)
    {
        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];
    }
    
    //-------------------------------------------------------------------------

    inline void subVec2(const float * a,
                        const float * b,
                        float       * result)
    {
        result[0] = a[0] - b[0];
        result[1] = a[1] - b[1];
    }

} //anonymous namespace


namespace tgen
{

    //-------------------------------------------------------------------------

    void computeCornerTSpace(const std::vector<uint32_t> & triIndicesPos,
                             const std::vector<uint32_t> & triIndicesUV,
                             const std::vector<float>   & positions3D,
                             const std::vector<float>   & uvs2D,
                             std::vector<float>         & cTangents3D,
                             std::vector<float>         & cBitangents3D)
    {
        const std::size_t numCorners = triIndicesPos.size();

        cTangents3D.resize(  numCorners * 3);
        cBitangents3D.resize(numCorners * 3);

        float edge3D[3][3], edgeUV[3][2],
              tmp0[3],      tmp1[3];

        for (std::size_t i = 0; i < triIndicesPos.size(); i += 3)
        {            
            const uint32_t vertexIndicesPos[3]  = { triIndicesPos[i  ],
                                                   triIndicesPos[i+1],
                                                   triIndicesPos[i+2]  };

            const uint32_t vertexIndicesUV[3]   = { triIndicesUV[i  ],
                                                   triIndicesUV[i+1],
                                                   triIndicesUV[i+2]  };
            
            // compute derivatives of positions and UVs along the edges
            for (std::size_t j = 0; j < 3; ++j)
            {                
                const std::size_t next = (j + 1) % 3;

                const uint32_t v0PosIdx = vertexIndicesPos[j];
                const uint32_t v1PosIdx = vertexIndicesPos[next];
                const uint32_t v0UVIdx  = vertexIndicesUV[j];
                const uint32_t v1UVIdx  = vertexIndicesUV[next];

                subVec3(&positions3D[v1PosIdx * 3],
                        &positions3D[v0PosIdx * 3],
                        edge3D[j]);

                subVec2(&uvs2D[v1UVIdx * 2],
                        &uvs2D[v0UVIdx * 2],
                        edgeUV[j]);
            }
            
            // compute per-corner tangent and bitangent (not normalized),
            // using the derivatives of the UVs
            // http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
            for (std::size_t j = 0; j < 3; ++j)
            {
                const std::size_t prev = (j + 2) % 3;

                const float * dPos0    = edge3D[j];
                const float * dPos1Neg = edge3D[prev];
                const float * dUV0     = edgeUV[j];
                const float * dUV1Neg  = edgeUV[prev];

                float * resultTangent   = &cTangents3D[  (i + j) * 3];
                float * resultBitangent = &cBitangents3D[(i + j) * 3];

                float denom = (dUV0[0] * -dUV1Neg[1] - dUV0[1] * -dUV1Neg[0]);
                float r     = std::abs(denom) > DenomEps ? 1.0 / denom : 0.0;

                multVec3(dPos0,    -dUV1Neg[1] * r, tmp0);
                multVec3(dPos1Neg, -dUV0[1]    * r, tmp1);
                subVec3(tmp0, tmp1, resultTangent);
                //resultTangent = glm::normalize(resultTangent);

                multVec3(dPos1Neg, -dUV0[0]    * r, tmp0);
                multVec3(dPos0,    -dUV1Neg[0] * r, tmp1);
                subVec3(tmp0, tmp1, resultBitangent);
            }
        }
    }

    //-------------------------------------------------------------------------

     void computeVertexTSpace(const std::vector<uint32_t> & triIndicesUV,
                              const std::vector<float>   & cTangents3D,
                              const std::vector<float>   & cBitangents3D,
                              std::size_t                  numUVVertices,
                              std::vector<float>         & vTangents3D,
                              std::vector<float>         & vBitangents3D )
     {
         vTangents3D.resize(  numUVVertices * 3, 0.0);
         vBitangents3D.resize(numUVVertices * 3, 0.0);


         // average tangent vectors for each "wedge" (UV vertex)
         // this assumes that we do not use different vertex positions
         // for the same UV coordinate (example: mirrored parts)

         for (std::size_t i = 0; i < triIndicesUV.size(); ++i)
         {
             const uint32_t uvIdx = triIndicesUV[i];

             float * cornerTangent   = &vTangents3D[  uvIdx*3];
             float * cornerBitangent = &vBitangents3D[uvIdx*3];

             addVec3(&cTangents3D[  i*3], cornerTangent,   cornerTangent  );
             addVec3(&cBitangents3D[i*3], cornerBitangent, cornerBitangent);
         }


         // normalize results

         for (uint32_t i = 0; i < numUVVertices; ++i)
         {
             if (lengthVec3(&vTangents3D[i * 3]) > 1e-6)
             {
                 normalizeVec3(&vTangents3D[i * 3]);
             }
             if (lengthVec3(&vBitangents3D[i * 3]) > 1e-6)
             {
                 normalizeVec3(&vBitangents3D[i * 3]);
             }
         }        
     }

     //-------------------------------------------------------------------------
     
     void orthogonalizeTSpace(const std::vector<float> & normals3D,
                              std::vector<float>       & tangents3D,
                              std::vector<float>       & bitangents3D)
     {
         const std::size_t numVertices = normals3D.size() / 3;

         float correction[3];
         for (uint32_t i = 0; i < numVertices; ++i)
         {
             const float * nV = &normals3D[   i*3];

             float * bV = &bitangents3D[i*3];
             float * tV = &tangents3D[i*3];
             
             float d = dotProd(nV, tV);

             multVec3(nV, d, correction);
             subVec3(tV, correction, tV);
             normalizeVec3(tV);

             crossProd(nV, tV, bV);
         }
     }

     //-------------------------------------------------------------------------

     void computeTangent4D(const std::vector<float> & normals3D,
                           const std::vector<float> & tangents3D,
                           const std::vector<float> & bitangents3D,
                           std::vector<float>       & tangents4D)
     {
         const std::size_t numVertices = normals3D.size() / 3;

         tangents4D.resize(numVertices * 4);

         float cross[3];
         for (uint32_t i = 0; i < numVertices; ++i)
         {
             crossProd(&normals3D[i*3], &tangents3D[i*3], cross);

             float sign = dotProd(cross, &bitangents3D[i*3]) > 0.0 ? 1.0 : -1.0;

             tangents4D[i*4  ] = tangents3D[i*3+0];
             tangents4D[i*4+1] = tangents3D[i*3+1];
             tangents4D[i*4+2] = tangents3D[i*3+2];
             tangents4D[i*4+3] = sign;
         }
     }

     //-------------------------------------------------------------------------

} //namespace tgen

#endif //TGEN_H
