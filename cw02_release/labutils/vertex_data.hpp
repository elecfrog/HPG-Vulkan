#pragma once

#include <cstdint>
#include <vector> 

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp"
#include "../cw2/baked_model.hpp"

struct objMesh
{
	labutils::Buffer positions;
	labutils::Buffer texcoords;
	// labutils::Buffer normals;
	labutils::Buffer indices;
	labutils::Buffer tbnQuats;
	labutils::Buffer compressedInts;

	objMesh() = default;

	objMesh(objMesh&& other) noexcept :
		positions(std::move(other.positions)),
		texcoords(std::move(other.texcoords)),
		// normals(std::move(other.normals)),
		indices(std::move(other.indices)),
		tbnQuats(std::move(other.tbnQuats)),
		compressedInts(std::move(other.compressedInts))
	{}
};


objMesh create_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, BakedMeshData aModel);

