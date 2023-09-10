#include <volk/volk.h>

#include <tuple>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>
#include <stb_image_write.h>
#include <glm/gtc/quaternion.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <map>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../elfbase/Marocs.hpp"
#include "../elfbase/VulkanInitalizers.hpp"
#include "../elfbase/FrameBufferAttachment.hpp"

namespace lut = labutils;

#include "../labutils/vertex_data.hpp"

#include "baked_model.hpp"

namespace
{
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
#		define SHADERDIR_ "assets/cw2/shaders/"

		// pbr pipeline
		constexpr const char* pbrVSPath = SHADERDIR_ "pbr.vert.spv";
		constexpr const char* pbrFSPath = SHADERDIR_ "pbr.frag.spv";

		// ao pipeline
		constexpr const char* aoFSPath = SHADERDIR_ "ao.frag.spv";

		//baked obj file
		constexpr const char* MODEL_PATH = "assets/cw2/sponza-pbr.comp5822mesh";

#		undef SHADERDIR_

		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;

		constexpr auto kCameraFov = 60.0_degf;

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;

		constexpr float kCameraBaseSpeed = 1.7f;
		constexpr float kCameraFastMult = 1.7f;
		constexpr float kCameraSlowMult = 1.7f;

		constexpr float kCameraMouseSensitivity = 0.1f;

		constexpr float k_LightSens = 0.1f;
	}

	using clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;


	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);
	void glfw_callback_srcoll(GLFWwindow* aWin, double aX, double aY);


	namespace glsl
	{
		struct ViewUniform
		{
			glm::mat4 projCamera;
		};

		struct LightUniform
		{
			glm::vec4 position;
			glm::vec4 color;
			glm::vec4 viewPos;
		};
	}

	struct UniformBlock
	{
		struct UniformBuffers
		{
			lut::Buffer viewUBO{};
			lut::Buffer lightUBO{};
		} uniformBuffers;

		struct UniformData
		{
			glsl::ViewUniform viewData{};
			glsl::LightUniform lightData{};
		} uniformData;
	};

	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		lightmove,
		max,
	};

	struct UserState
	{
		bool inputMap[static_cast<std::size_t>(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;

		glm::mat4 camera2world = glm::identity<glm::mat4>();

		glm::vec3 lightPos = {0, 2, 0};
		glm::vec3 lightStartPos = {0, 2, 0};
		float angleVelocity;

		bool is_dragging = false;
		bool is_moving = false;
		double start_x = 0, start_y = 0;
		double move_x = 0, move_y = 0;
	};

	void update_user_state(UserState&, float elapsedTime);

	lut::RenderPass create_render_pass(const lut::VulkanWindow&);

	lut::Pipeline CreateRenderPipeline(const lut::VulkanWindow& aWindow,
	                                   const std::unordered_map<std::string, lut::ShaderModule>& shaders,
	                                   VkRenderPass aRenderPass,
	                                   VkPipelineLayout aPipelineLayout);


	void create_swapchain_framebuffers(
		const lut::VulkanWindow&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView depthView
	);

	void updateViewUniform(glsl::ViewUniform& viewUniform, const VkExtent2D& ext, UserState& aState);

	struct PipelineLayouts
	{
		lut::PipelineLayout pbr;
		lut::PipelineLayout ao;
	};

	struct DescriptorSets
	{
		VkDescriptorSet view{};
		VkDescriptorSet light{};
		std::vector<VkDescriptorSet> pbrTextures;
		std::vector<VkDescriptorSet> aoTextures;
	};

	struct GraphicsPipelines
	{
		lut::Pipeline pbr;
		lut::Pipeline ao;
	};

	struct InputData
	{
		std::vector<objMesh> sponzaVertexBuffers;
		BakedModel sponza;
	};

	void record_commands(
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		const VkExtent2D&,
		const UniformBlock& uniformBlock,
		const PipelineLayouts& pipelineLayouts,
		const DescriptorSets& descriptorSets,
		const GraphicsPipelines& pipelines,
		const InputData&
	);

	void submit_commands(
		const lut::VulkanWindow&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore);

	void present_results(
		VkQueue,
		VkSwapchainKHR,
		std::uint32_t aImageIndex,
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);


	elf::FrameBufferAttachment create_depth_buffer(const lut::VulkanWindow&, const lut::Allocator&);
}


int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	UserState state{};

	glfwSetWindowUserPointer(window.window, &state);
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);
	glfwSetScrollCallback(window.window, &glfw_callback_srcoll);

	// Configure the GLFW window
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);

	// Create a unique resource managment (VMA allocator)
	lut::Allocator allocator = create_allocator(window);

	// Intialize renderpass
	lut::RenderPass renderPass = create_render_pass(window);

	struct DescriptorSetLayouts
	{
		lut::DescriptorSetLayout view;
		lut::DescriptorSetLayout pbrs;
		lut::DescriptorSetLayout ao;
		lut::DescriptorSetLayout light;
	} descLayouts;

	PipelineLayouts pipeLayouts;

	descLayouts.view = enit::createDescLayout(window, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT);
	descLayouts.pbrs = enit::createDescLayout(window, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 5);
	descLayouts.ao = enit::createDescLayout(window, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	descLayouts.light = enit::createDescLayout(window, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);

	pipeLayouts.pbr = enit::createPipelineLayout(window, {
		                                             descLayouts.view.handle, descLayouts.pbrs.handle,
		                                             descLayouts.light.handle
	                                             });
	pipeLayouts.ao = enit::createPipelineLayout(window, {descLayouts.view.handle, descLayouts.ao.handle});

	//load shader
	GraphicsPipelines pipelines;

	std::unordered_map<std::string, lut::ShaderModule> pbrShaders;
	pbrShaders["vert"] = load_shader_module(window, cfg::pbrVSPath);
	pbrShaders["frag"] = load_shader_module(window, cfg::pbrFSPath);

	pipelines.pbr = CreateRenderPipeline(window, pbrShaders, renderPass.handle, pipeLayouts.pbr.handle);

	std::unordered_map<std::string, lut::ShaderModule> aoShaders;
	aoShaders["vert"] = load_shader_module(window, cfg::pbrVSPath);
	aoShaders["frag"] = load_shader_module(window, cfg::aoFSPath);
	pipelines.ao = CreateRenderPipeline(window, aoShaders, renderPass.handle, pipeLayouts.ao.handle);

	// depth buffer attachment
	elf::FrameBufferAttachment depthBuffer = elf::createFBAttachment(window, allocator, cfg::kDepthFormat,
	                                                                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

	std::vector<lut::Framebuffer> framebuffers;
	//similar to an image stored in the swap chain
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBuffer.view.handle);

	//create descriptor pool(all the descriptor set)
	lut::DescriptorPool descriptorPool = CreateDescriptorPool(window);

	// command pool
	lut::CommandPool cmdPool = create_command_pool(
		window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	std::vector<VkCommandBuffer> cmdBuffs;
	std::vector<lut::Fence> cbfences;

	//There are as many framebuffers as there are command buffer and as many fence
	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cmdBuffs.emplace_back(alloc_command_buffer(window, cmdPool.handle));
		cbfences.emplace_back(create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	lut::Semaphore imageAvailable = create_semaphore(window);
	lut::Semaphore renderFinished = create_semaphore(window);


	//------------------------------------------------------------------------------------------------------------------------------
	InputData inputData;
	inputData.sponza = LoadBakedModel(cfg::MODEL_PATH);

	//1.Create and load textures.This gives a list of Images(which includes a
	//VkImage + VmaAllocation) and VkImageViews.We only need to keep these
	//around -- place them in a vector.

	//load textures into image
	// lut::CommandPool cmdPool = create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	std::vector<lut::Image> sponzaImages;
	std::vector<lut::ImageView> sponzaImageViews;
	for (size_t idx = 0; idx < inputData.sponza.textures.size(); ++idx)
	{
		lut::Image img = LoadImage(inputData.sponza.textures[idx].path.c_str(), window, cmdPool.handle, allocator);
		sponzaImages.emplace_back(std::move(img));

		lut::ImageView view = LoadImageView(window, sponzaImages[idx].image, VK_FORMAT_R8G8B8A8_UNORM);
		sponzaImageViews.emplace_back(std::move(view));
	}

	//create default texture sampler
	lut::Sampler defaultSampler = CreateDefaultSampler(window);


	UniformBlock uniformBlock;
	//-----------------------------------------------------------------------------------------------------
	//scene uniform buffer in vertex shader
	uniformBlock.uniformBuffers.viewUBO = createBuffer(
		allocator,
		sizeof(glsl::ViewUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	uniformBlock.uniformBuffers.lightUBO = createBuffer(
		allocator,
		sizeof(glsl::LightUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	DescriptorSets descriptorSets;


	descriptorSets.view = alloc_desc_set(window, descriptorPool.handle, descLayouts.view.handle);
	{
		VkDescriptorBufferInfo bufferInfo{uniformBlock.uniformBuffers.viewUBO.buffer, 0, VK_WHOLE_SIZE};
		VkWriteDescriptorSet desc = enit::writeDescriptorSet(descriptorSets.view, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
		                                                     &bufferInfo);
		vkUpdateDescriptorSets(window.device, 1, &desc, 0, nullptr);
	}

	descriptorSets.light = alloc_desc_set(window, descriptorPool.handle, descLayouts.light.handle);
	{
		VkDescriptorBufferInfo bufferInfo{uniformBlock.uniformBuffers.lightUBO.buffer, 0, VK_WHOLE_SIZE};
		VkWriteDescriptorSet desc = enit::writeDescriptorSet(descriptorSets.light, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
		                                                     &bufferInfo);
		vkUpdateDescriptorSets(window.device, 1, &desc, 0, nullptr);
	}

	labutils::Image dummy_image = DummyImage(window, cmdPool.handle, allocator);
	labutils::ImageView defaultNormalView = LoadImageView(window, dummy_image.image, VK_FORMAT_R8G8B8A8_UNORM);

	// std::vector<objMesh> sponza_Meshes;
	for (const auto& mesh : inputData.sponza.meshes)
	{
		const auto& currMaterial = inputData.sponza.materials[mesh.materialId];

		{
			VkDescriptorSet ret = alloc_desc_set(window, descriptorPool.handle, descLayouts.pbrs.handle);

			VkWriteDescriptorSet wDesc[5]{};
			VkDescriptorImageInfo imageInfos[5]{};

			// base color
			imageInfos[0] = enit::descriptorImageInfo(defaultSampler.handle,
			                                          sponzaImageViews[currMaterial.baseColorTextureId].handle,
			                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			// metalness
			imageInfos[1] = enit::descriptorImageInfo(defaultSampler.handle,
			                                          sponzaImageViews[currMaterial.metalnessTextureId].handle,
			                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			// roughenss
			imageInfos[2] = enit::descriptorImageInfo(defaultSampler.handle,
			                                          sponzaImageViews[currMaterial.roughnessTextureId].handle,
			                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			// normals
			if (currMaterial.normalMapTextureId > 0)
				imageInfos[3] = enit::descriptorImageInfo(defaultSampler.handle, defaultNormalView.handle,
				                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			else
				imageInfos[3] = enit::descriptorImageInfo(defaultSampler.handle,
				                                          sponzaImageViews[currMaterial.normalMapTextureId].handle,
				                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			// aos
			uint32_t aoid;
			currMaterial.alphaMaskTextureId > 0 ? aoid = 0 : aoid = currMaterial.alphaMaskTextureId;
			imageInfos[4] = enit::descriptorImageInfo(defaultSampler.handle, sponzaImageViews[aoid].handle,
			                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

			for (size_t idx = 0; idx < std::size(imageInfos); ++idx)
				wDesc[idx] = enit::writeDescriptorSet(ret, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (uint32_t)idx,
				                                      &imageInfos[idx]);

			constexpr auto numSets = std::size(wDesc);
			vkUpdateDescriptorSets(window.device, numSets, wDesc, 0, nullptr);
			descriptorSets.pbrTextures.emplace_back(ret);
		}


		{
			VkDescriptorSet ret = alloc_desc_set(window, descriptorPool.handle, descLayouts.ao.handle);

			VkWriteDescriptorSet desc[1]{};
			VkDescriptorImageInfo imageInfo[1]{};

			uint32_t aoid;
			currMaterial.alphaMaskTextureId == -1 ? aoid = 0 : aoid = currMaterial.alphaMaskTextureId;

			imageInfo[0] = enit::descriptorImageInfo(defaultSampler.handle, sponzaImageViews[aoid].handle,
			                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			desc[0] = enit::writeDescriptorSet(ret, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageInfo[0]);

			constexpr auto numSets = std::size(desc);
			vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
			descriptorSets.aoTextures.emplace_back(ret);
		}

		{
			objMesh meshBuffer = create_mesh(window, allocator, mesh);
			inputData.sponzaVertexBuffers.emplace_back(std::move(meshBuffer));
		}
	}

	// Application main loop
	bool recreateSwapchain = false;
	auto previousClock = clock_::now();
	while (!glfwWindowShouldClose(window.window))
	{
		glfwPollEvents(); // or: glfwWaitEvents()

		if (recreateSwapchain)
		{
			//re-create swapchain and associated resources
			vkDeviceWaitIdle(window.device);

			//recreate them
			const auto changes = recreate_swapchain(window);

			if (changes.changedFormat)
				renderPass = create_render_pass(window);

			if (changes.changedSize)
			{
				depthBuffer = elf::createFBAttachment(window, allocator, cfg::kDepthFormat,
				                                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
				pipelines.pbr = CreateRenderPipeline(window, pbrShaders, renderPass.handle, pipeLayouts.pbr.handle);
				pipelines.ao = CreateRenderPipeline(window, aoShaders, renderPass.handle, pipeLayouts.ao.handle);
			}

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBuffer.view.handle);

			recreateSwapchain = false;
			continue;
		}

		//acquire swapchain image
		std::uint32_t imageIndex = 0;
		const auto acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire enxt swapchain image\n"
			                 "vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str()
			);
		}
		// wait for command buffer to be available
		assert(static_cast<std::size_t>(imageIndex) < cbfences.size());
		VK_CHECK_RESULT(
			vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>
				::max()))

		VK_CHECK_RESULT(vkResetFences(window.device, 1, &cbfences[imageIndex].handle))

		const auto now = clock_::now();
		const auto dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(state, dt);

		//record and submit commands(have done)
		assert(static_cast<std::size_t>(imageIndex) < cmdBuffs.size());
		assert(static_cast<std::size_t>(imageIndex) < framebuffers.size());

		updateViewUniform(uniformBlock.uniformData.viewData, window.swapchainExtent, state);

		uniformBlock.uniformData.lightData = {glm::vec4(state.lightPos, 1), glm::vec4(1.f), state.camera2world[3]};

		static_assert(sizeof(uniformBlock.uniformData.viewData) <= 65536,
		              "ViewUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(uniformBlock.uniformData.viewData) % 4 == 0,
		              "ViewUniform size must be a multiple of 4 bytes");

		record_commands(
			cmdBuffs[imageIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			window.swapchainExtent,
			uniformBlock,
			pipeLayouts,
			descriptorSets,
			pipelines,
			inputData
		);

		submit_commands(
			window,
			cmdBuffs[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);
	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle(window.device);

	return 0;
}
catch (const std::exception& eErr)
{
	std::fprintf(stderr, "\n");
	std::fprintf(stderr, "Error: %s\n", eErr.what());
	return 1;
}

namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
		}

		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		const bool isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[static_cast<std::size_t>(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[static_cast<std::size_t>(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[static_cast<std::size_t>(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[static_cast<std::size_t>(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[static_cast<std::size_t>(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[static_cast<std::size_t>(EInputState::sink)] = !isReleased;
			break;
		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[static_cast<std::size_t>(EInputState::fast)] = !isReleased;
			break;
		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[static_cast<std::size_t>(EInputState::slow)] = !isReleased;
			break;
		case GLFW_KEY_SPACE:
			if (GLFW_PRESS == aAction)
			{
				state->inputMap[static_cast<std::size_t>(EInputState::lightmove)] = !state->inputMap[static_cast<
					std::size_t>(EInputState::lightmove)];
			}
			break;
		default:
			break;
		}
	}

	void glfw_callback_button(GLFWwindow* window, int button, int action, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(window));
		assert(state);

		if (GLFW_MOUSE_BUTTON_MIDDLE == button && GLFW_PRESS == action)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}

		if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
		{
			state->is_dragging = true;
			glfwGetCursorPos(window, &state->start_x, &state->start_y);
		}
		else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
		{
			state->is_dragging = false;
		}

		if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
		{
			state->is_moving = true;
			glfwGetCursorPos(window, &state->move_x, &state->move_y);
		}
		else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
		{
			state->is_moving = false;
		}
	}

	void glfw_callback_srcoll(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		auto& cam = state->camera2world;

		cam = cam * glm::translate(glm::vec3(0.f, 0.f, -aY * 0.5f));
	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);

		auto curr_x = aX;
		auto curr_y = aY;

		double dx = curr_x - state->start_x;
		double dy = curr_y - state->start_y;

		auto& cam = state->camera2world;

		const auto sens = cfg::kCameraMouseSensitivity;

		if (state->is_dragging)
		{
			if (std::fabs(dx) > std::fabs(dy))
			{
				cam = cam * glm::rotate(float(dx) * sens * 0.1f, glm::vec3(0.f, 1.f, 0.f));
			}
			else
			{
				cam = cam * glm::rotate(float(dy) * sens * 0.1f, glm::vec3(1.f, 0.f, 0.f));
			}
		}

		if (state->is_moving)
		{
			cam = cam * glm::translate(glm::vec3(-float(dx) * sens * 0.1f, float(dy) * sens * 0.1f, 0.f));
		}

		state->start_x = curr_x;
		state->start_y = curr_y;
	}

	void update_user_state(UserState& state, float elapsedTime)
	{
		auto& cam = state.camera2world;

		const auto move = elapsedTime * cfg::kCameraBaseSpeed *
			(state.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(state.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f);

		if (state.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (state.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, move));
		if (state.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		if (state.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));
		if (state.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (state.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));
		if (state.inputMap[std::size_t(EInputState::lightmove)])
		{
			state.angleVelocity += glm::radians(90.f) * elapsedTime;

			float lightDistance = (state.lightPos - state.lightStartPos).length();
			float lightYaw = lightDistance * cos(state.angleVelocity);
			float lightPitch = lightDistance * sin(state.angleVelocity);
			state.lightPos = glm::vec3(lightYaw, state.lightPos.y, lightPitch);
		}
	}
}

namespace
{
	void updateViewUniform(glsl::ViewUniform& viewUniform, const VkExtent2D& ext, UserState& aState)
	{
		const float aspect = (float)ext.width / (float)ext.height;
		//The RH indicates a right handed clip space, and the ZO indicates
		//that the clip space extends from zero to one along the Z - axis.
		auto projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		projection[1][1] *= -1.f;
		auto camera = inverse(aState.camera2world);
		viewUniform.projCamera = projection * camera;
	}
}

namespace
{
	lut::RenderPass create_render_pass(const lut::VulkanWindow& aWindow)
	{
		//Create Render Pass Attachments
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0;
		passInfo.pDependencies = nullptr;

		VkRenderPass rpass = VK_NULL_HANDLE;
		VK_CHECK_RESULT(vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass));

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::Pipeline CreateRenderPipeline(const lut::VulkanWindow& aWindow,
	                                   const std::unordered_map<std::string, lut::ShaderModule>& shaders,
	                                   VkRenderPass aRenderPass,
	                                   VkPipelineLayout aPipelineLayout)
	{
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = shaders.at("vert").handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = shaders.at("frag").handle;
		stages[1].pName = "main";

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		std::vector vertexInputs
		{
			// position
			enit::vertexInputBindingDescription(0, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX),
			// texcoords
			enit::vertexInputBindingDescription(1, sizeof(float) * 2, VK_VERTEX_INPUT_RATE_VERTEX),
			// uncompressed tbn quaternions
			enit::vertexInputBindingDescription(2, sizeof(float) * 4, VK_VERTEX_INPUT_RATE_VERTEX),
			// packed TBN quateraions
			enit::vertexInputBindingDescription(3, sizeof(uint32_t), VK_VERTEX_INPUT_RATE_VERTEX)
		};

		std::vector vertexAttributes
		{
			enit::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),
			enit::vertexInputAttributeDescription(1, 1, VK_FORMAT_R32G32_SFLOAT, 0),
			enit::vertexInputAttributeDescription(2, 2, VK_FORMAT_R32G32B32A32_SFLOAT, 0),
			enit::vertexInputAttributeDescription(3, 3, VK_FORMAT_R32_UINT, 0)
		};

		VkPipelineVertexInputStateCreateInfo inputInfo = enit::pipelineVertexInputStateCreateInfo(
			vertexInputs, vertexAttributes);

		//assembled for rasterization
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo = enit::pipelineInputAssemblyStateCreateInfo(
			VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE);

		//tessellation state:
		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = static_cast<float>(aWindow.swapchainExtent.width);
		viewport.height = static_cast<float>(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		//scissor can be used to restrict drawing to a part of the frame buffer without changing the coordinate system
		VkRect2D scissor{};
		scissor.offset = VkOffset2D{0, 0};
		scissor.extent = VkExtent2D{aWindow.swapchainExtent.width, aWindow.swapchainExtent.height};

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterInfo = enit::pipelineRasterizationStateCreateInfo(
			VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);

		VkPipelineMultisampleStateCreateInfo samplingInfo = enit::pipelineMultisampleStateCreateInfo(
			VK_SAMPLE_COUNT_1_BIT);

		// define blned State:
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_TRUE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		//In Vulkan, logical operations are used to perform bitwise operations on color data in a framebuffer
		//attachment during blending.(e.g. AND, OR, XOR)
		VkPipelineColorBlendStateCreateInfo blendInfo = enit::pipelineColorBlendStateCreateInfo(1, blendStates);

		//dynamic state:none

		//Assembling the VkGraphicsPipelineCreateInfo structure
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe))

		return lut::Pipeline(aWindow.device, pipe);
	}


	void create_swapchain_framebuffers(const lut::VulkanWindow& window, VkRenderPass renderPass, std::vector<lut::Framebuffer>& frameBuf, VkImageView depthView)
	{
		assert(frameBuf.empty());

		for (std::size_t i = 0; i < window.swapViews.size(); i++)
		{
			const VkImageView attachments[2] = 
			{
				window.swapViews[i],
				depthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; 
			fbInfo.renderPass = renderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = window.swapchainExtent.width;
			fbInfo.height = window.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			VK_CHECK_RESULT (vkCreateFramebuffer(window.device, &fbInfo, nullptr, &fb))

			frameBuf.emplace_back(window.device, fb);
		}

		assert(window.swapViews.size() == frameBuf.size());
	}


	void record_commands(VkCommandBuffer comdBuf,
	                     VkRenderPass renderPass,
	                     VkFramebuffer surfaceFrameBuf,
	                     const VkExtent2D& aImageExtent,
	                     const UniformBlock& uniformBlock,
	                     const PipelineLayouts& pipelineLayouts,
	                     const DescriptorSets& descriptorSets,
	                     const GraphicsPipelines& pipelines,
	                     const InputData& inputData
	)
	{
		//begin recording commands
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		VK_CHECK_RESULT(vkBeginCommandBuffer(comdBuf, &beginInfo))

		lut::buffer_barrier(comdBuf,
		                    uniformBlock.uniformBuffers.viewUBO.buffer,
		                    VK_ACCESS_UNIFORM_READ_BIT,
		                    VK_ACCESS_TRANSFER_WRITE_BIT,
		                    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
		                    VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(comdBuf, uniformBlock.uniformBuffers.viewUBO.buffer, 0, sizeof(glsl::ViewUniform),
		                  &uniformBlock.uniformData.viewData);

		lut::buffer_barrier(comdBuf,
		                    uniformBlock.uniformBuffers.viewUBO.buffer,
		                    VK_ACCESS_TRANSFER_WRITE_BIT,
		                    VK_ACCESS_UNIFORM_READ_BIT,
		                    VK_PIPELINE_STAGE_TRANSFER_BIT,
		                    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		lut::buffer_barrier(comdBuf,
		                    uniformBlock.uniformBuffers.lightUBO.buffer,
		                    VK_ACCESS_UNIFORM_READ_BIT,
		                    VK_ACCESS_TRANSFER_WRITE_BIT,
		                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		                    VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(comdBuf, uniformBlock.uniformBuffers.lightUBO.buffer, 0, sizeof(glsl::LightUniform),
		                  &uniformBlock.uniformData.lightData);

		lut::buffer_barrier(comdBuf,
		                    uniformBlock.uniformBuffers.lightUBO.buffer,
		                    VK_ACCESS_TRANSFER_WRITE_BIT,
		                    VK_ACCESS_UNIFORM_READ_BIT,
		                    VK_PIPELINE_STAGE_TRANSFER_BIT,
		                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);


		//begin render pass
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.f;

		clearValues[1].depthStencil.depth = 1.f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = renderPass;
		passInfo.framebuffer = surfaceFrameBuf;
		passInfo.renderArea.offset = VkOffset2D{0, 0};
		passInfo.renderArea.extent = VkExtent2D{aImageExtent.width, aImageExtent.height};
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(comdBuf, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		for (int i = 0; i < inputData.sponzaVertexBuffers.size(); i++)
		{
			if (inputData.sponza.materials[inputData.sponza.meshes[i].materialId].alphaMaskTextureId == -1)
			{
				vkCmdBindPipeline(comdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.pbr.handle);
				vkCmdBindDescriptorSets(comdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.pbr.handle, 0, 1, &descriptorSets.view, 0, nullptr);
				vkCmdBindDescriptorSets(comdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.pbr.handle, 1, 1, &descriptorSets.pbrTextures[i], 0, nullptr);
				vkCmdBindDescriptorSets(comdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.pbr.handle, 2, 1, &descriptorSets.light, 0, nullptr);

				//Bind vertex input
				VkBuffer vao[4] = 
				{
					inputData.sponzaVertexBuffers[i].positions.buffer,
					inputData.sponzaVertexBuffers[i].texcoords.buffer,
					inputData.sponzaVertexBuffers[i].tbnQuats.buffer,
					inputData.sponzaVertexBuffers[i].compressedInts.buffer
				};
				VkDeviceSize offsets[4]{};

				vkCmdBindVertexBuffers(comdBuf, 0, 4, vao, offsets);
				vkCmdBindIndexBuffer(comdBuf, inputData.sponzaVertexBuffers[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdDrawIndexed(comdBuf, static_cast<uint32_t>(inputData.sponza.meshes[i].indices.size()), 1, 0, 0, 0);
			}
		}

		for (int i = 0; i < inputData.sponzaVertexBuffers.size(); i++)
		{
			if (inputData.sponza.materials[inputData.sponza.meshes[i].materialId].alphaMaskTextureId != -1)
			{
				vkCmdBindPipeline(comdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.ao.handle);
				vkCmdBindDescriptorSets(comdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.ao.handle, 0, 1, &descriptorSets.view, 0, nullptr);
				vkCmdBindDescriptorSets(comdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.ao.handle, 1, 1, &descriptorSets.aoTextures[i], 0, nullptr);

				// Bind vertex input
				VkBuffer vao[4] =
				{
					inputData.sponzaVertexBuffers[i].positions.buffer,
					inputData.sponzaVertexBuffers[i].texcoords.buffer,
					inputData.sponzaVertexBuffers[i].tbnQuats.buffer,
					inputData.sponzaVertexBuffers[i].compressedInts.buffer
				};
				VkDeviceSize offsets[4]{};

				vkCmdBindVertexBuffers(comdBuf, 0, 4, vao, offsets);
				vkCmdBindIndexBuffer(comdBuf, inputData.sponzaVertexBuffers[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(comdBuf, static_cast<uint32_t>(inputData.sponza.meshes[i].indices.size()), 1, 0, 0, 0);
			}
		}

		vkCmdEndRenderPass(comdBuf);

		// end command recording
		VK_CHECK_RESULT(vkEndCommandBuffer(comdBuf))
	}

	void submit_commands(const lut::VulkanWindow& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		constexpr VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		VK_CHECK_RESULT(vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence))
	}

	void present_results(VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain)
	{
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		const auto presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
			aNeedToRecreateSwapchain = true;

		// VK_CHECK_RESULT(!presentRes)
	}
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
