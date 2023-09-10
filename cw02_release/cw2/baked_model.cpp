#include "baked_model.hpp"

#include <cstdio>
#include <cstring>
#include <glm/gtc/quaternion.hpp>

#include "../labutils/error.hpp"
#include "../labutils/elf_tgen.hpp"
namespace lut = labutils;

namespace
{
	// See cw2-bake/main.cpp for more info
	constexpr char kFileMagic[16] = "\0\0COMP5822Mmesh";
	constexpr char kFileVariant[16] = "default";

	constexpr std::uint32_t kMaxString = 32 * 1024;

	// functions
	BakedModel LoadBakedModelDataWithQuat(FILE*, char const*);
}

namespace quat
{
	const float minValue = -ReverseSquareRoot(2);
	const float maxValue = ReverseSquareRoot(2);

	// Encode 9-bit unsigned integer
	uint16_t EncodeUnsigned9Btyes(float factor, uint16_t length)
	{
		return uint16_t(int(factor * ((1 << (length)) - 1) + 0.5f));
	}

	// Encode a float into a 10-bit unsigned integer -> 1-bit sign and 9-bit unsigned integer
	uint16_t EncodeSignedData(float factor, uint16_t length)
	{
		return (factor < 0) | (EncodeUnsigned9Btyes((factor < 0 ? -factor : factor), (length - 1)) << 1);
	}

	// Map data into the limited range
	void MapIntoRange(float& factor)
	{
		factor = (factor - minValue) / (maxValue - minValue) * (2.f) + -1.f;
	}

	// Calucalte the reverse square root of a float
	float ReverseSquareRoot(float factor) { return 1.0f / sqrtf(factor); }

	// Encode quaternion( vec4 ) to 32-bit unsgined integer
	void EncodeQuatTo32Int(uint32_t& ret, glm::vec4& quat)
	{
		// get from raw data
		auto& x = quat.x;
		auto& y = quat.y;
		auto& z = quat.z;
		auto& w = quat.w;

		const auto xx = std::pow(x, 2.f);
		const auto yy = std::pow(y, 2.f);
		const auto zz = std::pow(z, 2.f);
		const auto ww = std::pow(w, 2.f);

		// abs(x) is the largest component
		if (xx >= yy && xx >= zz && xx >= ww)
		{
			MapIntoRange(y);
			MapIntoRange(z);
			MapIntoRange(w);

			ret = x >= 0
				      ? uint32_t(
					      (0 << 30) | (EncodeSignedData(y, 10) << 20) | (EncodeSignedData(z, 10) << 10) | (
						      EncodeSignedData(w, 10) << 10))
				      : uint32_t(
					      (0 << 30) | (EncodeSignedData(-y, 10) << 20) | (EncodeSignedData(-z, 10) << 10) | (
						      EncodeSignedData(-w, 10) << 10));
		}
		else if (yy >= zz && yy >= ww)
		{
			MapIntoRange(x);
			MapIntoRange(z);
			MapIntoRange(w);
			ret = y >= 0
				      ? uint32_t(
					      (1 << 30) | (EncodeSignedData(x, 10) << 20) | (EncodeSignedData(z, 10) << 10) | (
						      EncodeSignedData(w, 10) << 0))
				      : uint32_t(
					      (1 << 30) | (EncodeSignedData(-x, 10) << 20) | (EncodeSignedData(-z, 10) << 10) | (
						      EncodeSignedData(-w, 10) << 0));
		}
		else if (zz >= ww)
		{
			MapIntoRange(x);
			MapIntoRange(y);
			MapIntoRange(w);
			ret = z >= 0
				      ? uint32_t(
					      (2 << 30) | (EncodeSignedData(x, 10) << 20) | (EncodeSignedData(y, 10) << 10) | (
						      EncodeSignedData(w, 10) << 0))
				      : uint32_t(
					      (2 << 30) | (EncodeSignedData(-x, 10) << 20) | (EncodeSignedData(-y, 10) << 10) | (
						      EncodeSignedData(-w, 10) << 0));
		}
		else
		{
			MapIntoRange(x);
			MapIntoRange(y);
			MapIntoRange(z);
			ret = w >= 0
				      ? uint32_t(
					      (3 << 30) | (EncodeSignedData(x, 10) << 20) | (EncodeSignedData(y, 10) << 10) | (
						      EncodeSignedData(z, 10) << 0))
				      : uint32_t(
					      (3 << 30) | (EncodeSignedData(-x, 10) << 20) | (EncodeSignedData(-y, 10) << 10) | (
						      EncodeSignedData(-z, 10) << 0));
		}
	}
}

BakedModel LoadBakedModel(char const* aModelPath)
{
	FILE* fin = std::fopen(aModelPath, "rb");
	if (!fin)
		throw lut::Error("LoadBakedModel(): unable to open '%s' for reading", aModelPath);

	try
	{
		auto ret = LoadBakedModelDataWithQuat(fin, aModelPath);
		std::fclose(fin);
		return ret;
	}
	catch (...)
	{
		std::fclose(fin);
		throw;
	}
}

namespace
{
	void checked_read_(FILE* aFin, std::size_t aBytes, void* aBuffer)
	{
		auto ret = std::fread(aBuffer, 1, aBytes, aFin);

		if (aBytes != ret)
			throw lut::Error("checked_read_(): expected %zu bytes, got %zu", aBytes, ret);
	}

	std::uint32_t read_uint32_(FILE* aFin)
	{
		std::uint32_t ret;
		checked_read_(aFin, sizeof(std::uint32_t), &ret);
		return ret;
	}

	std::string read_string_(FILE* aFin)
	{
		auto const length = read_uint32_(aFin);

		if (length >= kMaxString)
			throw lut::Error("read_string_(): unexpectedly long string (%u bytes)", length);

		std::string ret;
		ret.resize(length);

		checked_read_(aFin, length, ret.data());
		return ret;
	}

	BakedModel LoadBakedModelDataWithQuat(FILE* aFin, char const* aInputName)
	{
		BakedModel ret;

		// Figure out base path
		char const* pathBeg = aInputName;
		char const* pathEnd = std::strrchr(pathBeg, '/');

		std::string const prefix = pathEnd
			                           ? std::string(pathBeg, pathEnd + 1)
			                           : "";

		// Read header and verify file magic and variant
		char magic[16];
		checked_read_(aFin, 16, magic);

		if (0 != std::memcmp(magic, kFileMagic, 16))
			throw lut::Error("LoadBakedModelDataWithQuat(): %s: invalid file signature!", aInputName);

		char variant[16];
		checked_read_(aFin, 16, variant);

		if (0 != std::memcmp(variant, kFileVariant, 16))
			throw lut::Error("LoadBakedModelDataWithQuat(): %s: file variant is '%s', expected '%s'", aInputName,
			                 variant,
			                 kFileVariant);

		// Read texture info
		auto const textureCount = read_uint32_(aFin);
		for (std::uint32_t i = 0; i < textureCount; ++i)
		{
			BakedTextureInfo info;
			info.path = prefix + read_string_(aFin);

			std::uint8_t channels;
			checked_read_(aFin, sizeof(std::uint8_t), &channels);
			info.channels = channels;

			ret.textures.emplace_back(std::move(info));
		}

		// Read material info
		auto const materialCount = read_uint32_(aFin);
		for (std::uint32_t i = 0; i < materialCount; ++i)
		{
			BakedMaterialInfo info;
			info.baseColorTextureId = read_uint32_(aFin);
			info.roughnessTextureId = read_uint32_(aFin);
			info.metalnessTextureId = read_uint32_(aFin);
			info.alphaMaskTextureId = read_uint32_(aFin);
			info.normalMapTextureId = read_uint32_(aFin);

			assert(info.baseColorTextureId < ret.textures.size());
			assert(info.roughnessTextureId < ret.textures.size());
			assert(info.metalnessTextureId < ret.textures.size());

			ret.materials.emplace_back(std::move(info));
		}

		// Read mesh data
		auto const meshCount = read_uint32_(aFin);
		for (std::uint32_t i = 0; i < meshCount; ++i)
		{
			BakedMeshData data;
			data.materialId = read_uint32_(aFin);
			assert(data.materialId < ret.materials.size());

			auto const V = read_uint32_(aFin);
			auto const I = read_uint32_(aFin);

			data.positions.resize(V);
			checked_read_(aFin, V * sizeof(glm::vec3), data.positions.data());

			data.texcoords.resize(V);
			checked_read_(aFin, V * sizeof(glm::vec2), data.texcoords.data());

			data.tbnQuats.resize(V);
			checked_read_(aFin, V * sizeof(glm::vec4), data.tbnQuats.data());

			for (size_t q = 0; q < V; ++q)
			{
				uint32_t out;
				quat::EncodeQuatTo32Int(out, data.tbnQuats[q]);
				data.compressedInt.emplace_back(out);
			}

			data.indices.resize(I);
			checked_read_(aFin, I * sizeof(std::uint32_t), data.indices.data());


			ret.meshes.emplace_back(std::move(data));
		}

		// Check
		char byte;
		auto const check = std::fread(&byte, 1, 1, aFin);

		if (0 != check)
			std::fprintf(stderr, "Note: '%s' contains trailing bytes\n", aInputName);

		return ret;
	}
}
