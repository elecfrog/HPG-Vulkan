#include "vertex_data.hpp"

#include <limits>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
namespace lut = labutils;

std::vector<VerticesMesh> CreateMeshes(const SimpleModel& _model)
{
	std::vector<VerticesMesh> ret;
	for (const auto& mesh : _model.meshes)
	{
		VerticesMesh tmp;

		tmp.vertexCount = mesh.vertexCount;

		for (auto index = mesh.vertexStartIndex; index < mesh.vertexStartIndex + mesh.vertexCount; index++)
		{ 
			if (mesh.textured)
			{
				tmp.positions.emplace_back(_model.dataTextured.positions[index]);
				tmp.texcoord.emplace_back(_model.dataTextured.texcoords[index]);
			}
			else
			{
				tmp.positions.emplace_back(_model.dataUntextured.positions[index]);
				tmp.texcoord.emplace_back(0.f);
			}
			tmp.colors.emplace_back(_model.materials[mesh.materialIndex].diffuseColor);
		}
		ret.emplace_back(std::move(tmp));
	}
	return ret;
}

VertexBuffer CreateVerticesBuffer(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, std::vector<VerticesMesh>&& meshes)
{

	VertexBuffer vertexbuffer;
	for (const auto& mesh : meshes)
	{
		vertexbuffer.indicesCount.emplace_back(mesh.vertexCount);

		const auto& meshPositions = mesh.positions;
		const auto& meshColors = mesh.colors;
		const auto& meshUvs = mesh.texcoord;

		lut::Buffer VertexPosGPU = lut::create_buffer(
			aAllocator,
			meshPositions.size() * sizeof(glm::vec3),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY     
		);
		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			meshPositions.size() * sizeof(glm::vec3),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);
		void* posPtr = nullptr;
		if (const auto res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str()
			);
		}
		std::memcpy(posPtr, meshPositions.data(), mesh.positions.size() * sizeof(glm::vec3));
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);


		lut::Buffer VertexColGPU = lut::create_buffer(
			aAllocator,
			meshColors.size() * sizeof(glm::vec3), // <-- Use the correct size
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
			VMA_MEMORY_USAGE_GPU_ONLY
		);
		lut::Buffer colStaging = lut::create_buffer(
			aAllocator,
			meshColors.size() * sizeof(glm::vec3), // <-- Use the correct size
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		void* colPtr = nullptr;
		if (const auto res = vmaMapMemory(aAllocator.allocator, colStaging.allocation, &colPtr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str()
			);
		}
		std::memcpy(colPtr, meshColors.data(), meshColors.size() * sizeof(glm::vec3)); // <-- Use the correct size
		vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);


		lut::Buffer VertexTexGPU{};
		lut::Buffer texStaging{};
		if (!meshUvs.empty())
		{
			VertexTexGPU = lut::create_buffer(
				aAllocator,
				meshPositions.size() * sizeof(glm::vec2),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_GPU_ONLY
			);

			texStaging = lut::create_buffer(
				aAllocator,
				meshPositions.size() * sizeof(glm::vec2),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_MEMORY_USAGE_CPU_TO_GPU
			);

			void* texPtr = nullptr;
			if (const auto res = vmaMapMemory(aAllocator.allocator, texStaging.allocation, &texPtr);
				VK_SUCCESS != res)
			{
				throw lut::Error("Mapping memory for writing\n"
					"vmaMapMemory() returned %s", lut::to_string(res).c_str()
				);
			}
			std::memcpy(texPtr, meshUvs.data(), meshUvs.size() * sizeof(glm::vec2)); // <-- Use the correct size
			vmaUnmapMemory(aAllocator.allocator, texStaging.allocation);
		}


		//prepare for issuing the transfer commands that copy data from the staging buffers to
		//the final on - GPU buffers
		lut::Fence uploadComplete = create_fence(aContext);

		// Queue data uploads from staging buffers to the final buffers 
		// This uses a separate command pool for simplicity. 
		lut::CommandPool uploadPool = lut::CreateGraphicsCommandPool(aContext);
		VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aContext, uploadPool.handle);

		//record copy commands into command buffer
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (const auto res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}

		VkBufferCopy pcopy{};
		pcopy.size = meshPositions.size() * sizeof(glm::vec3);

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, VertexPosGPU.buffer, 1, &pcopy);

		lut::buffer_barrier(uploadCmd,
			VertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ccopy{};
		ccopy.size = meshPositions.size() * sizeof(glm::vec3);

		vkCmdCopyBuffer(uploadCmd, colStaging.buffer, VertexColGPU.buffer, 1, &ccopy);

		lut::buffer_barrier(uploadCmd,
			VertexColGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		if(texStaging.buffer != VK_NULL_HANDLE && VertexTexGPU.buffer != VK_NULL_HANDLE)
		{
			VkBufferCopy tcopy{};
			tcopy.size = meshUvs.size() * sizeof(glm::vec2); // <-- Use the correct size

			vkCmdCopyBuffer(uploadCmd, texStaging.buffer, VertexTexGPU.buffer, 1, &tcopy);

			lut::buffer_barrier(uploadCmd,
				VertexTexGPU.buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
			);

		}

		if (const auto res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}

		//submit transfer commands
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (const auto res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle);
			VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
			);
		}

		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
			VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n"
				"vkWaitForFences() returned %s", lut::to_string(res).c_str()
			);
		}

		vertexbuffer.positionBuffer.emplace_back(std::move(VertexPosGPU));
		vertexbuffer.colorBuffer.emplace_back(std::move(VertexColGPU));
		vertexbuffer.uvBuffer.emplace_back(std::move(VertexTexGPU));

	}
	return vertexbuffer;
}
