#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace elf
{
	namespace tgen
	{
		using real_t = float;
		using index_t = std::uint32_t;
		const real_t eps = real_t(1e-8);

		//-------------------------------------------------------------------------

		struct TangentCaculation
		{
			std::vector<glm::vec3> ctangents3d;
			std::vector<glm::vec3> cbitangents3d;
			std::vector<glm::vec3> tangents3d;
			std::vector<glm::vec3> bitangents3d;

			TangentCaculation() = default;
		};

		inline void computeCornerTSpace(
			TangentCaculation& ret,
			const std::vector<uint32_t>& indices,
			const std::vector<glm::vec3>& positions3D,
			const std::vector<glm::vec2>& uvs2D)
		{
			const auto& triangleCounts = indices.size();

			ret.ctangents3d.resize(triangleCounts);
			ret.cbitangents3d.resize(triangleCounts);

			std::array<glm::vec3, 3> edge3D;
			std::array<glm::vec2, 3> edgeUV;

			for (std::size_t i = 0; i < triangleCounts; i += 3)
			{
				const glm::uvec3 faceIndices = glm::vec3(indices[i], indices[i + 1], indices[i + 2]);

				// compute derivatives of positions and UVs along the edges
				for (std::size_t idx = 0; idx < 3; ++idx)
				{
					const std::size_t next = (idx + 1) % 3;

					const size_t curr_idx = faceIndices[idx];
					const size_t next_idx = faceIndices[next];

					edge3D[idx] = positions3D[next_idx] - positions3D[curr_idx];
					edgeUV[idx] = uvs2D[next_idx] - uvs2D[curr_idx];
				}

				// compute per-corner tangent and bitangent (not normalized),
				// using the derivatives of the UVs
				// http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
				for (std::size_t j = 0; j < 3; ++j)
				{
					const std::size_t prev = (j + 2) % 3;

					const glm::vec3 dPos0 = edge3D[j];
					const glm::vec3 dPos1Neg = edge3D[prev];
					const glm::vec2 dUV0 = edgeUV[j];
					const glm::vec2 dUV1Neg = edgeUV[prev];

					real_t denom = (dUV0[0] * -dUV1Neg[1] - dUV0[1] * -dUV1Neg[0]);
					real_t r = std::fabs(denom) > eps ? (real_t)1.0 / denom : (real_t)0.0;

					glm::vec3 tmp0 = dPos0 * (-dUV1Neg[1] * r);
					glm::vec3 tmp1 = dPos1Neg * (-dUV0[1] * r);

					ret.ctangents3d[i + j] = tmp0 - tmp1;

					tmp0 = dPos1Neg * (-dUV0[0] * r);
					tmp1 = dPos0 * (-dUV1Neg[0] * r);

					ret.cbitangents3d[i + j] = tmp0 - tmp1;
				}
			}
		}

		//-------------------------------------------------------------------------
		inline void computeVertexTSpace(TangentCaculation& ret,
		                                const std::vector<uint32_t>& triIndicesUV,
		                                std::size_t numUVVertices)
		{
			// std::vector<glm::vec3> vTangents3D;
			ret.tangents3d.resize(numUVVertices);

			// std::vector<glm::vec3> vBitangents3D;
			ret.bitangents3d.resize(numUVVertices);

			// average tangent vectors for each "wedge" (UV vertex)
			// this assumes that we do not use different vertex positions
			// for the same UV coordinate (example: mirrored parts)

			for (std::size_t i = 0; i < triIndicesUV.size(); ++i)
			{
				const auto& uvIdx = triIndicesUV[i];

				ret.tangents3d[uvIdx] += ret.ctangents3d[i];
				ret.bitangents3d[uvIdx] += ret.cbitangents3d[i];
			}

			// normalize results
			for (uint32_t i = 0; i < numUVVertices; ++i)
			{
				if (glm::length(ret.tangents3d[i]) > eps)
				{
					glm::normalize(ret.tangents3d[i]);
				}
				if (glm::length(ret.bitangents3d[i]) > eps)
				{
					glm::normalize(ret.bitangents3d[i]);
				}
			}
		}

		//-------------------------------------------------------------------------

		inline void orthogonalizeTSpace(TangentCaculation& ret, const std::vector<glm::vec3>& normals3D)
		{
			// Gram-Schmidt
			for (uint32_t i = 0; i < normals3D.size(); ++i)
			{
				real_t d = glm::dot(normals3D[i], ret.tangents3d[i]);

				glm::vec3 correction = normals3D[i] * d;
				ret.tangents3d[i] = glm::normalize(ret.tangents3d[i] - correction);
				ret.bitangents3d[i] = glm::cross(normals3D[i], ret.tangents3d[i]);
			}

			// Mirror
			for (uint32_t i = 0; i < normals3D.size(); ++i)
			{
				glm::vec3 cross = glm::cross(normals3D[i], ret.tangents3d[i]);
				real_t sign = (real_t)glm::dot(cross, ret.bitangents3d[i]) > (real_t)0.0 ? (real_t)1.0 : (real_t)-1.0;
				if (sign < 0)
				{
					ret.tangents3d[i] *= -1.0f;
				}
			}
		}

		inline std::vector<glm::vec3> CalculateTangents(
			const std::vector<uint32_t>& indices,
			const std::vector<glm::vec3>& vertices,
			const std::vector<glm::vec3>& normals,
			const std::vector<glm::vec2>& uvs)
		{
			TangentCaculation ret;
			computeCornerTSpace(ret, indices, vertices, uvs);
			computeVertexTSpace(ret, indices, vertices.size());
			orthogonalizeTSpace(ret, normals);
		
			return ret.tangents3d;
		}

		inline std::vector<glm::quat> CalculateTBNQuats(
			const std::vector<uint32_t>& indices,
			const std::vector<glm::vec3>& vertices,
			const std::vector<glm::vec3>& normals,
			const std::vector<glm::vec2>& uvs)
		{
			TangentCaculation ret;
			computeCornerTSpace(ret, indices, vertices, uvs);
			computeVertexTSpace(ret, indices, vertices.size());
			orthogonalizeTSpace(ret, normals);

			std::vector<glm::quat> tbnQuaternions{};
			for (size_t n = 0; n < normals.size(); ++n)
			{
				glm::mat3 tbnMatrix(ret.tangents3d[n], ret.bitangents3d[n], normals[n]);
				tbnQuaternions.emplace_back(glm::normalize(glm::quat_cast(tbnMatrix)));
			}

			return tbnQuaternions;
		}

        //-------------------------------------------------------------------------

        inline void addVec3(const tgen::real_t* a,
            const tgen::real_t* b,
            tgen::real_t* result)
        {
            result[0] = a[0] + b[0];
            result[1] = a[1] + b[1];
            result[2] = a[2] + b[2];
        }

        //-------------------------------------------------------------------------

        inline void subVec3(const tgen::real_t* a,
            const tgen::real_t* b,
            tgen::real_t* result)
        {
            result[0] = a[0] - b[0];
            result[1] = a[1] - b[1];
            result[2] = a[2] - b[2];
        }

        //-------------------------------------------------------------------------

        inline void multVec3(const tgen::real_t* a,
            const tgen::real_t   s,
            tgen::real_t* result)
        {
            result[0] = a[0] * s;
            result[1] = a[1] * s;
            result[2] = a[2] * s;
        }

        //-------------------------------------------------------------------------

        void normalizeVec3(tgen::real_t* v)
        {
            tgen::real_t len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

            multVec3(v, 1.0 / len, v);
        }

        //-------------------------------------------------------------------------

        inline tgen::real_t dotProd(const tgen::real_t* a,
            const tgen::real_t* b)
        {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }

        //-------------------------------------------------------------------------

        inline void crossProd(const tgen::real_t* a,
            const tgen::real_t* b,
            tgen::real_t* result)
        {
            result[0] = a[1] * b[2] - a[2] * b[1];
            result[1] = a[2] * b[0] - a[0] * b[2];
            result[2] = a[0] * b[1] - a[1] * b[0];
        }

        //-------------------------------------------------------------------------

        inline void subVec2(const tgen::real_t* a,
            const tgen::real_t* b,
            tgen::real_t* result)
        {
            result[0] = a[0] - b[0];
            result[1] = a[1] - b[1];
        }

        //-------------------------------------------------------------------------

        inline void computeCornerTSpace(const std::vector<index_t>& triIndicesPos,
            const std::vector<index_t>& triIndicesUV,
            const std::vector<real_t>& positions3D,
            const std::vector<real_t>& uvs2D,
            std::vector<real_t>& cTangents3D,
            std::vector<real_t>& cBitangents3D)
        {
            const std::size_t numCorners = triIndicesPos.size();

            cTangents3D.resize(numCorners * 3);
            cBitangents3D.resize(numCorners * 3);

            real_t edge3D[3][3], edgeUV[3][2],
                tmp0[3], tmp1[3];

            for (std::size_t i = 0; i < triIndicesPos.size(); i += 3)
            {
                const index_t vertexIndicesPos[3] = { triIndicesPos[i],
                                                       triIndicesPos[i + 1],
                                                       triIndicesPos[i + 2] };

                const index_t vertexIndicesUV[3] = { triIndicesUV[i],
                                                       triIndicesUV[i + 1],
                                                       triIndicesUV[i + 2] };

                // compute derivatives of positions and UVs along the edges
                for (std::size_t j = 0; j < 3; ++j)
                {
                    const std::size_t next = (j + 1) % 3;

                    const index_t v0PosIdx = vertexIndicesPos[j];
                    const index_t v1PosIdx = vertexIndicesPos[next];
                    const index_t v0UVIdx = vertexIndicesUV[j];
                    const index_t v1UVIdx = vertexIndicesUV[next];

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

                    const real_t* dPos0 = edge3D[j];
                    const real_t* dPos1Neg = edge3D[prev];
                    const real_t* dUV0 = edgeUV[j];
                    const real_t* dUV1Neg = edgeUV[prev];

                    real_t* resultTangent = &cTangents3D[(i + j) * 3];
                    real_t* resultBitangent = &cBitangents3D[(i + j) * 3];

                    real_t denom = (dUV0[0] * -dUV1Neg[1] - dUV0[1] * -dUV1Neg[0]);
                    real_t r = std::abs(denom) > eps ? 1.0 / denom : 0.0;

                    multVec3(dPos0, -dUV1Neg[1] * r, tmp0);
                    multVec3(dPos1Neg, -dUV0[1] * r, tmp1);
                    subVec3(tmp0, tmp1, resultTangent);

                    multVec3(dPos1Neg, -dUV0[0] * r, tmp0);
                    multVec3(dPos0, -dUV1Neg[0] * r, tmp1);
                    subVec3(tmp0, tmp1, resultBitangent);
                }
            }
        }

        //-------------------------------------------------------------------------

        inline void computeVertexTSpace(const std::vector<index_t>& triIndicesUV,
            const std::vector<real_t>& cTangents3D,
            const std::vector<real_t>& cBitangents3D,
            std::size_t                  numUVVertices,
            std::vector<real_t>& vTangents3D,
            std::vector<real_t>& vBitangents3D)
        {
            vTangents3D.resize(numUVVertices * 3, 0.0);
            vBitangents3D.resize(numUVVertices * 3, 0.0);


            // average tangent vectors for each "wedge" (UV vertex)
            // this assumes that we do not use different vertex positions
            // for the same UV coordinate (example: mirrored parts)

            for (std::size_t i = 0; i < triIndicesUV.size(); ++i)
            {
                const index_t uvIdx = triIndicesUV[i];

                real_t* cornerTangent = &vTangents3D[uvIdx * 3];
                real_t* cornerBitangent = &vBitangents3D[uvIdx * 3];

                addVec3(&cTangents3D[i * 3], cornerTangent, cornerTangent);
                addVec3(&cBitangents3D[i * 3], cornerBitangent, cornerBitangent);
            }


            // normalize results

            for (index_t i = 0; i < numUVVertices; ++i)
            {
                normalizeVec3(&vTangents3D[i * 3]);
                normalizeVec3(&vBitangents3D[i * 3]);
            }
        }

        //-------------------------------------------------------------------------

        inline void orthogonalizeTSpace(const std::vector<real_t>& normals3D,
            std::vector<real_t>& tangents3D,
            std::vector<real_t>& bitangents3D)
        {
            const std::size_t numVertices = normals3D.size() / 3;

            real_t correction[3];
            for (index_t i = 0; i < numVertices; ++i)
            {
                const real_t* nV = &normals3D[i * 3];

                real_t* bV = &bitangents3D[i * 3];
                real_t* tV = &tangents3D[i * 3];

                real_t d = dotProd(nV, tV);

                multVec3(nV, d, correction);
                subVec3(tV, correction, tV);
                normalizeVec3(tV);

                crossProd(nV, tV, bV);
            }
        }

        //-------------------------------------------------------------------------

        inline void computeTangent4D(const std::vector<real_t>& normals3D,
            const std::vector<real_t>& tangents3D,
            const std::vector<real_t>& bitangents3D,
            std::vector<real_t>& tangents4D)
        {
            const std::size_t numVertices = normals3D.size() / 3;

            tangents4D.resize(numVertices * 4);

            real_t cross[3];
            for (index_t i = 0; i < numVertices; ++i)
            {
                crossProd(&normals3D[i * 3], &tangents3D[i * 3], cross);

                real_t sign = dotProd(cross, &bitangents3D[i * 3]) > 0.0 ? 1.0 : -1.0;

                tangents4D[i * 4] = tangents3D[i * 3 + 0];
                tangents4D[i * 4 + 1] = tangents3D[i * 3 + 1];
                tangents4D[i * 4 + 2] = tangents3D[i * 3 + 2];
                tangents4D[i * 4 + 3] = sign;
            }
        }

        //-------------------------------------------------------------------------



        inline std::vector<glm::vec3> calculateTangents(
            const std::vector<uint32_t>& indices,
            const std::vector<glm::vec3>& vertices,
            const std::vector<glm::vec3>& normals,
            const std::vector<glm::vec2>& uvs)
        {

            std::vector<glm::vec3> ret;
            std::vector<real_t> raw_position3Ds;
            for(auto& data : vertices)
            {
                raw_position3Ds.emplace_back(data.x);
                raw_position3Ds.emplace_back(data.y);
                raw_position3Ds.emplace_back(data.z);
            }

			std::vector<real_t> raw_normals;
            for(auto& data : normals)
            {
                raw_normals.emplace_back(data.x);
                raw_normals.emplace_back(data.y);
                raw_normals.emplace_back(data.z);
            }

			std::vector<real_t> raw_uvs;
            for(auto& data : uvs)
            {
                raw_uvs.emplace_back(data.x);
                raw_uvs.emplace_back(data.y);
            }

            std::vector<real_t> cTangents3D;
            std::vector<real_t> cBitangents3D;

			std::vector<real_t> vTangents3D;
            std::vector<real_t> vBitangents3D;

            std::vector<real_t> tangents4D;


            computeCornerTSpace(indices, indices, raw_position3Ds, raw_uvs, cTangents3D, cBitangents3D);
            computeVertexTSpace(indices, cTangents3D, cBitangents3D, indices.size(), vTangents3D, vBitangents3D);
            orthogonalizeTSpace(raw_normals, vTangents3D, vBitangents3D);
            computeTangent4D(raw_normals, vTangents3D, vBitangents3D, tangents4D);

            for(size_t idx = 0 ; idx < tangents4D.size(); idx+=4)
            {
                auto& x = tangents4D[idx    ];
                auto& y = tangents4D[idx + 1];
                auto& z = tangents4D[idx + 2];
                auto& w = tangents4D[idx + 3];

                if(w < 0 )
                {
	                x = -x;
					y = -y;
					z = -z;
				}

                glm::vec3 tangent(x,y,z);
                ret.emplace_back(tangent);
            }

            return ret;
        }
    } //namespace tgen


}
