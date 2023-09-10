#pragma once

#include <cstdint>
#include <vector> 

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp"
#include "../cw4/baked_model.hpp"

struct objMesh
{
	// vec3
	labutils::Buffer positions;
	// vec3
	labutils::Buffer normals;
	// vec2
	labutils::Buffer texcoords;
	// vec3
	labutils::Buffer tangents;

	// uint32_t
	labutils::Buffer indices;

	objMesh() = default;

	objMesh(objMesh&& other) noexcept :
		positions(std::move(other.positions)),
		normals(std::move(other.normals)),
		texcoords(std::move(other.texcoords)),
		tangents(std::move(other.tangents)),
		indices(std::move(other.indices))
	{}
};


objMesh create_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, BakedMeshData aModel);

