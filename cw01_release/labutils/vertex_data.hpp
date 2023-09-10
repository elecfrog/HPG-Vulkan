#pragma once

#include <cstdint>
#include <memory>
#include <vector> 

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
#include "../cw1/simple_model.hpp"

struct VerticesMesh
{
	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> colors;
	std::vector<glm::vec2> texcoord;
	bool isTextured;
	std::uint32_t vertexCount;
};

struct VertexBuffer
{
	std::vector<labutils::Buffer> positionBuffer;
	std::vector<labutils::Buffer> colorBuffer;
	std::vector<labutils::Buffer> uvBuffer;
	std::vector<std::uint32_t>    indicesCount;

};

std::vector<VerticesMesh> CreateMeshes(const SimpleModel& _model);
VertexBuffer CreateVerticesBuffer(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, std::vector<VerticesMesh>&& meshes);

