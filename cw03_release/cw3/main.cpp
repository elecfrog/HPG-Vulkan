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
#include <array>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
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
namespace lut = labutils;

#include "../labutils/vertex_data.hpp"
#include "../elfbase/Marocs.hpp"
#include "../elfbase/VulkanInitalizers.hpp"
#include "../elfbase/FrameBufferAttachment.hpp"
#include "baked_model.hpp"


namespace
{
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
#		define SHADERDIR_ "assets/cw3/shaders/"

		constexpr char const* mrtVSPath = SHADERDIR_ "mrt.vert.spv";
		constexpr char const* mrtFSPath = SHADERDIR_ "mrt.frag.spv";

		constexpr char const* postVSPath = SHADERDIR_ "post.vert.spv";
		constexpr char const* postFSPath = SHADERDIR_ "post.frag.spv";

		constexpr char const* surfaceVSPath = SHADERDIR_ "screen.vert.spv";
		constexpr char const* surfaceFSPath = SHADERDIR_ "screen.frag.spv";

		//baked obj file
		constexpr char const* MODEL_PATH = "assets/cw3/ship.comp5822mesh";

#		undef SHADERDIR_

		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 100.f;

		constexpr auto kCameraFov    = 60.0_degf;

		constexpr VkFormat k_DepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
		constexpr VkFormat k_ColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

		constexpr float kCameraBaseSpeed = 1.7f;
		constexpr float kCameraFastMult = 1.7f;
		constexpr float kCameraSlowMult = 1.7f;

		constexpr float kCameraMouseSensitivity = 0.1f;

		constexpr float kLightRotationSpeed = 0.1f;
	}

	using clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);
	void glfw_callback_srcoll(GLFWwindow* aWin, double aX, double aY);

	// Uniform data
	namespace glsl
	{
		struct ViewUniform
		{
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCamera;
		};

		struct LightUniform
		{
			glm::vec4 position;
			glm::vec4 color;
		};

		struct ColorUniform
		{
			glm::vec4 basecolor;
			glm::vec4 emissive;
			float roughness;
			float metalness;
		};

		struct BloomUniform
		{
			float blurScale;
			float blurStrength;
		};

	}

	struct UniformBlock
	{
		struct UniformBuffers
		{
			lut::Buffer viewUBO{};
			lut::Buffer lightUBO{};
			lut::Buffer postUBO{};

			std::vector<lut::Buffer> colorUBOs;
		} uniformBuffers;

		struct UniformData
		{
			glsl::ViewUniform viewData{};
			glsl::LightUniform lightData{};

			glsl::BloomUniform bloomUniform{};
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
		max
	};

	struct  UserState
	{
		bool inputMap[static_cast<std::size_t>(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;

		glm::mat4 camera2world = glm::identity<glm::mat4>();

		glm::vec3 lightPos = { 0, 2, 0 };
		glm::vec3 lightStartPos = { 0, 2, 0 };
		float angleVelocity;

		bool is_dragging = false;
		bool is_moving = false;
		double start_x = 0, start_y = 0;
		double move_x = 0, move_y = 0;
	};

	void updateState(UserState&, float);


	struct SurfaceFrameBuffer
	{
		std::vector<lut::Framebuffer> swapChains;
		elf::FrameBufferAttachment depth;
		lut::RenderPass renderPass;
	};

	struct OffScreenFrameBuffer
	{
		uint32_t width, height;
		lut::Framebuffer framebuffer;
		elf::FrameBufferAttachment position, normal, albedo, brightness;
		elf::FrameBufferAttachment depth;
		lut::RenderPass renderPass;
	};

	struct FrameBuffer
	{
		lut::Framebuffer framebuffer;
		elf::FrameBufferAttachment color, depth;
		VkDescriptorImageInfo descriptor;
	};
	struct PostFrameBuffer
	{
		int32_t width, height;
		lut::RenderPass renderPass;
		lut::Sampler sampler;
		FrameBuffer v;
		FrameBuffer h;
		// std::array<FrameBuffer,2> framebuffers;
	};
	struct InputData
	{
		std::vector<objMesh> VBOs;
		BakedModel modelData;

		std::vector<lut::Image> modelImages;
		std::vector<lut::ImageView> modelImageViews;
	};


	lut::DescriptorSetLayout createCustomedDescLayout(lut::VulkanWindow const&);
	std::vector<lut::Buffer> createColorUBOs(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator, std::vector<glsl::ColorUniform>& colorUniforms, BakedModel& model);

	lut::PipelineLayout createOffScreenPipeLayout( lut::VulkanContext const& , VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout, VkDescriptorSetLayout aLightLayout, VkDescriptorSetLayout aTextureLayout);
	lut::PipelineLayout createSurfacePipelineLayout(lut::VulkanContext const& , VkDescriptorSetLayout aLayout);

	lut::Pipeline createOffScreenPipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline createPostPipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout, uint32_t direction = 0);
	lut::Pipeline createSurfacePipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	lut::RenderPass createSurfaceRenderPass(lut::VulkanWindow const&);
	std::vector<lut::Framebuffer> createSwapchainFramebuffers( lut::VulkanWindow const&, VkRenderPass, VkImageView aDepthView );
	OffScreenFrameBuffer createOffscreenFramebuffer(lut::VulkanWindow const&, labutils::Allocator const&);
	PostFrameBuffer createPostProcessingFrameBuffer(lut::VulkanWindow const&, labutils::Allocator const&);

	void updateViewUniform(glsl::ViewUniform& viewUniform, const VkExtent2D& ext, UserState& aState);

	struct DescriptorSets
	{
		VkDescriptorSet view{}, light{}, mrt{};
		VkDescriptorSet bloomVertical{}, bloomHorizontal{};
		std::vector<VkDescriptorSet> colors, textures;
	};

	struct PipelineLayouts
	{
		lut::PipelineLayout offscreenPipeLayout, surfacePipeLayout, postProcessLayout;
	};

	struct GraphicsPipelines
	{
		lut::Pipeline offscreenPipe, surfacePipeline;
		lut::Pipeline bloomVerticalPipe, bloomHorizontalPipe;
	};

	void recordCommands(
		VkCommandBuffer,  VkExtent2D const&,  VkFramebuffer, UserState,
		// framebuffer
		OffScreenFrameBuffer&, PostFrameBuffer&, SurfaceFrameBuffer& ,
		// input data
		InputData&,
		// pipeline and uniform resources
		UniformBlock&, DescriptorSets&, PipelineLayouts&,
		// pipelines
		GraphicsPipelines&
	);
	void submitCommands( lut::VulkanWindow const&, VkCommandBuffer, VkFence, VkSemaphore, VkSemaphore );
	void presentResults( VkQueue, VkSwapchainKHR, std::uint32_t aImageIndex, VkSemaphore, bool& aNeedToRecreateSwapchain );

}


int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();
	
	UserState state{};

	// Configure the GLFW window
	glfwSetWindowUserPointer(window.window, &state);
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);
	glfwSetScrollCallback(window.window, &glfw_callback_srcoll);
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);

	// Create a unique resource managment (VMA allocator)
	lut::Allocator allocator = lut::create_allocator( window );

	// descriptorSetLayouts
	struct DescriptorSetLayouts
	{
		lut::DescriptorSetLayout view;
		lut::DescriptorSetLayout color;
		lut::DescriptorSetLayout texture;
		lut::DescriptorSetLayout mrt;
		lut::DescriptorSetLayout light;
		lut::DescriptorSetLayout blur;
	} descLayouts;
	descLayouts.view = enit::createDescLayout(window, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT);
	descLayouts.color = enit::createDescLayout(window, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);
	descLayouts.texture = enit::createDescLayout(window, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,3);
	descLayouts.light = enit::createDescLayout(window, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
	descLayouts.mrt = enit::createMRTDescLayout(window, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4);
	descLayouts.blur = createCustomedDescLayout(window);

	PipelineLayouts pipelineLayouts;
	pipelineLayouts.offscreenPipeLayout = createOffScreenPipeLayout(window, descLayouts.view.handle, descLayouts.color.handle, descLayouts.light.handle, descLayouts.texture.handle);
	pipelineLayouts.postProcessLayout = enit::createPipelineLayout(window, { descLayouts.blur.handle });
	pipelineLayouts.surfacePipeLayout = createSurfacePipelineLayout(window, descLayouts.mrt.handle);

	// ---- OFFSCREEN STAGE PROCESS ---- 
	OffScreenFrameBuffer osFrameBuffer{};
	osFrameBuffer = createOffscreenFramebuffer(window, allocator);

	// ---- POST PROCESSING STAGE PROCESS ----
	PostFrameBuffer postFrameBuffer = createPostProcessingFrameBuffer(window, allocator);

	// ---- FINAL SURFACE DISPLAY STAGE ----
	SurfaceFrameBuffer sfFrameBuffer{};
	sfFrameBuffer.renderPass = createSurfaceRenderPass(window);
	sfFrameBuffer.depth = elf::createFBAttachment(window, allocator, cfg::k_DepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
	sfFrameBuffer.swapChains = createSwapchainFramebuffers( window, sfFrameBuffer.renderPass.handle, sfFrameBuffer.depth.view.handle);

	GraphicsPipelines pipelines;
	pipelines.offscreenPipe = createOffScreenPipeline(window, osFrameBuffer.renderPass.handle, pipelineLayouts.offscreenPipeLayout.handle);
	pipelines.bloomVerticalPipe = createPostPipeline(window, postFrameBuffer.renderPass.handle, pipelineLayouts.postProcessLayout.handle);
	pipelines.bloomHorizontalPipe = createPostPipeline(window, postFrameBuffer.renderPass.handle, pipelineLayouts.postProcessLayout.handle, 1);
	pipelines.surfacePipeline = createSurfacePipeline(window, sfFrameBuffer.renderPass.handle, pipelineLayouts.surfacePipeLayout.handle);

	// descriptor pool
	lut::DescriptorPool descriptorPool = lut::create_descriptor_pool(window);

	// command pool
	lut::CommandPool cmdPool = lut::create_command_pool( window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT );

	std::vector<VkCommandBuffer> cmdBuffs;
	std::vector<lut::Fence> cbfences;

	//There are as many framebuffers as there are command buffer and as many fence
	for( std::size_t i = 0; i < sfFrameBuffer.swapChains.size(); ++i )
	{
		cmdBuffs.emplace_back( lut::alloc_command_buffer( window, cmdPool.handle ) );
		cbfences.emplace_back( lut::create_fence( window, VK_FENCE_CREATE_SIGNALED_BIT ) );
	}

	lut::Semaphore imageAvailable = lut::create_semaphore( window );
	lut::Semaphore renderFinished = lut::create_semaphore( window );

	InputData inputData;
	inputData.modelData = load_baked_model(cfg::MODEL_PATH);

	//load textures into image
	for (size_t idx = 0; idx < inputData.modelData.textures.size(); ++idx)
	{
		lut::Image img; 
		img = load_image_texture2d(inputData.modelData.textures[idx].path.c_str(), window, cmdPool.handle, allocator);
		inputData.modelImages.emplace_back(std::move(img));

		lut::ImageView view;
		view = create_image_view_texture2d(window, inputData.modelImages[idx].image, VK_FORMAT_R8G8B8A8_SRGB);
		inputData.modelImageViews.emplace_back(std::move(view));
	}

	//create texture sampler
	lut::Sampler imageSampler  = createTexturesampler(window);
	lut::Sampler screenSampler = createScreenSampler(window);

	UniformBlock uniformBlock;
	uniformBlock.uniformBuffers.viewUBO  = create_buffer(allocator, sizeof(glsl::ViewUniform),  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY );
	uniformBlock.uniformBuffers.lightUBO = create_buffer(allocator, sizeof(glsl::LightUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY );
	uniformBlock.uniformBuffers.postUBO  = create_buffer(allocator, sizeof(glsl::BloomUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY );

	std::vector<glsl::ColorUniform> colorData;
	for (uint32_t i = 0; i < inputData.modelData.meshes.size(); i++) 
	{
		const auto& currMaterial = inputData.modelData.materials[inputData.modelData.meshes[i].materialId];
		glm::vec4 basecolor = glm::vec4(currMaterial.baseColor,1.0f);
		glm::vec4 emissive  = glm::vec4(currMaterial.emissiveColor, 1.0f);
		float metalness = currMaterial.metalness;
		float roughness = currMaterial.roughness;

		colorData.emplace_back(glsl::ColorUniform{ basecolor, emissive, roughness, metalness });
	}
	uniformBlock.uniformBuffers.colorUBOs =  createColorUBOs(window, allocator, colorData, inputData.modelData);

	DescriptorSets descriptorSets;
	descriptorSets.view = alloc_desc_set(window, descriptorPool.handle, descLayouts.view.handle);
	{
		VkDescriptorBufferInfo bufferInfo{ uniformBlock.uniformBuffers.viewUBO.buffer, 0, VK_WHOLE_SIZE };
		VkWriteDescriptorSet desc = enit::writeDescriptorSet(descriptorSets.view, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &bufferInfo);
		vkUpdateDescriptorSets(window.device, 1, &desc, 0, nullptr);
	}
	descriptorSets.light = alloc_desc_set(window, descriptorPool.handle, descLayouts.light.handle);
	{
		VkDescriptorBufferInfo bufferInfo{ uniformBlock.uniformBuffers.lightUBO.buffer, 0, VK_WHOLE_SIZE };
		VkWriteDescriptorSet desc = enit::writeDescriptorSet(descriptorSets.light, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &bufferInfo);
		vkUpdateDescriptorSets(window.device, 1, &desc, 0, nullptr);
	}

	//--deferred texture descriptor--------------------------------------------------------------------------------
	descriptorSets.mrt = alloc_desc_set(window, descriptorPool.handle, descLayouts.mrt.handle);
	VkWriteDescriptorSet mrtDesc[4]{};
	{
		VkDescriptorImageInfo imageInfos[3]{};
		imageInfos[0] = enit::descriptorImageInfo(screenSampler.handle, osFrameBuffer.albedo.view.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		imageInfos[1] = enit::descriptorImageInfo(screenSampler.handle, osFrameBuffer.normal.view.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		imageInfos[2] = enit::descriptorImageInfo(screenSampler.handle, osFrameBuffer.position.view.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		mrtDesc[0] = enit::writeDescriptorSet(descriptorSets.mrt, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageInfos[0]);
		mrtDesc[1] = enit::writeDescriptorSet(descriptorSets.mrt, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &imageInfos[1]);
		mrtDesc[2] = enit::writeDescriptorSet(descriptorSets.mrt, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &imageInfos[2]);
		mrtDesc[3] = enit::writeDescriptorSet(descriptorSets.mrt, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3, &postFrameBuffer.v.descriptor);
	}	

	VkDescriptorImageInfo  bloomImagesInfo  = enit::descriptorImageInfo(screenSampler.handle, osFrameBuffer.brightness.view.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	VkDescriptorBufferInfo bloomBufferInfo{ uniformBlock.uniformBuffers.postUBO.buffer, 0, VK_WHOLE_SIZE };

	// vertical
	descriptorSets.bloomVertical = lut::alloc_desc_set(window, descriptorPool.handle, descLayouts.blur.handle);
	{
		VkWriteDescriptorSet desc[2]{};
		desc[0] = enit::writeDescriptorSet(descriptorSets.bloomVertical, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &bloomBufferInfo);
		desc[1] = enit::writeDescriptorSet(descriptorSets.bloomVertical, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &bloomImagesInfo);

		vkUpdateDescriptorSets(window.device, 2, desc, 0, nullptr);
	}

	// horizontal
	descriptorSets.bloomHorizontal = lut::alloc_desc_set(window, descriptorPool.handle, descLayouts.blur.handle);
	{
		VkWriteDescriptorSet desc[2]{};
		desc[0] = enit::writeDescriptorSet(descriptorSets.bloomHorizontal, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &bloomBufferInfo);
		desc[1] = enit::writeDescriptorSet(descriptorSets.bloomHorizontal, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &postFrameBuffer.h.descriptor);

		vkUpdateDescriptorSets(window.device, 2, desc, 0, nullptr);
	}

	for (uint32_t i = 0; i < inputData.modelData.meshes.size(); i++)
	{
		{
			VkDescriptorBufferInfo colorUboInfo{};
			colorUboInfo.buffer = uniformBlock.uniformBuffers.colorUBOs[i].buffer;
			colorUboInfo.range  = VK_WHOLE_SIZE;

			VkDescriptorSet color = lut::alloc_desc_set(window, descriptorPool.handle, descLayouts.color.handle);
			VkWriteDescriptorSet desc = enit::writeDescriptorSet(color, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &colorUboInfo);
			vkUpdateDescriptorSets(window.device, 1, &desc, 0, nullptr);
			descriptorSets.colors.emplace_back(color);
		}

		{
			VkDescriptorSet ret = lut::alloc_desc_set(window, descriptorPool.handle, descLayouts.texture.handle);

			VkWriteDescriptorSet desc[3]{};
			VkDescriptorImageInfo imageInfos[3]{};
			const auto& currMatieral = inputData.modelData.materials[inputData.modelData.meshes[i].materialId];

			imageInfos[0] = enit::descriptorImageInfo(imageSampler.handle, inputData.modelImageViews[currMatieral.baseColorTextureId].handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			imageInfos[1] = enit::descriptorImageInfo(imageSampler.handle, inputData.modelImageViews[currMatieral.metalnessTextureId].handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			imageInfos[2] = enit::descriptorImageInfo(imageSampler.handle, inputData.modelImageViews[currMatieral.roughnessTextureId].handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

			desc[0] = enit::writeDescriptorSet(ret, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageInfos[0]);
			desc[1] = enit::writeDescriptorSet(ret, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &imageInfos[1]);
			desc[2] = enit::writeDescriptorSet(ret, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &imageInfos[2]);

			vkUpdateDescriptorSets(window.device, 3, desc, 0, nullptr);
			descriptorSets.textures.emplace_back(ret);
		}

		{
			objMesh meshBuffer = create_mesh(window, allocator, inputData.modelData.meshes[i]);
			inputData.VBOs.emplace_back(std::move(meshBuffer));
		}
	}

	// Application main loop
	bool recreateSwapchain = false;
	auto previousClock = clock_::now();
	while( !glfwWindowShouldClose( window.window ) )
	{
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if( recreateSwapchain )
		{
			//re-create swapchain and associated resources
			vkDeviceWaitIdle(window.device);

			//recreate them
			const auto changes = recreate_swapchain(window);

			if (changes.changedFormat) {
				sfFrameBuffer.renderPass = createSurfaceRenderPass(window);
				osFrameBuffer = createOffscreenFramebuffer(window, allocator);
				postFrameBuffer = createPostProcessingFrameBuffer(window, allocator);
			}

			if (changes.changedSize) 
			{
				sfFrameBuffer.depth = elf::createFBAttachment(window, allocator, cfg::k_DepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
				pipelines.offscreenPipe = createOffScreenPipeline(window, osFrameBuffer.renderPass.handle, pipelineLayouts.offscreenPipeLayout.handle);
				pipelines.surfacePipeline = createSurfacePipeline(window, sfFrameBuffer.renderPass.handle, pipelineLayouts.surfacePipeLayout.handle);
				pipelines.bloomVerticalPipe = createPostPipeline(window, postFrameBuffer.renderPass.handle, pipelineLayouts.postProcessLayout.handle);
				pipelines.bloomHorizontalPipe = createPostPipeline(window, postFrameBuffer.renderPass.handle, pipelineLayouts.postProcessLayout.handle, 1 );

				// osFrameBuffer = createOffscreenFramebuffer(window, allocator);
				// postFrameBuffer = createPostProcessingFrameBuffer(window, allocator);
			}

			sfFrameBuffer.swapChains.clear();
			sfFrameBuffer.swapChains = createSwapchainFramebuffers(window, sfFrameBuffer.renderPass.handle, sfFrameBuffer.depth.view.handle);

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

		VK_CHECK_RESULT(acquireRes)

		// wait for command buffer to be available
		assert(static_cast<std::size_t>(imageIndex) < cbfences.size());
		VK_CHECK_RESULT(vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()))
		VK_CHECK_RESULT(vkResetFences(window.device, 1, &cbfences[imageIndex].handle))

		auto const now = clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		updateState(state, dt);

		//record and submit commands(have done)
		assert(std::size_t(imageIndex) < cmdBuffs.size());
		assert(std::size_t(imageIndex) < sfFrameBuffer.swapChains.size());

		updateViewUniform(uniformBlock.uniformData.viewData, window.swapchainExtent, state);

		uniformBlock.uniformData.lightData = { glm::vec4(state.lightPos, 1), glm::vec4(1.f) };
		uniformBlock.uniformData.bloomUniform = { 1.f, 1.5f};

		static_assert(sizeof(uniformBlock.uniformData.viewData) <= 65536, "ViewUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(uniformBlock.uniformData.viewData) % 4 == 0, "ViewUniform size must be a multiple of 4 bytes");

		vkUpdateDescriptorSets(window.device, 4, mrtDesc, 0, nullptr);

		recordCommands(
			cmdBuffs[imageIndex],
			window.swapchainExtent,
			sfFrameBuffer.swapChains[imageIndex].handle,
			state,
			// framebuffer
			osFrameBuffer, postFrameBuffer, sfFrameBuffer,
			// input data
			inputData,
			// pipeline and uniform resources
			uniformBlock, descriptorSets, pipelineLayouts,
			// pipelines
			pipelines
		);

		submitCommands(
			window,
			cmdBuffs[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		presentResults(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);
	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle( window.device );

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
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

	void updateState(UserState& state, float elapsedTime)
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
	lut::RenderPass createSurfaceRenderPass(lut::VulkanWindow const& aWindow)
	{
		//Create Render Pass Attachments
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		//Create Subpass Definition
		//The reference means it uses the attachment above as a color attachment
		VkAttachmentReference subpassAttachments[1]{};
		//the zero refers to the 0th render pass attachment declared earlier
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		attachments[1].format = cfg::k_DepthFormat;
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

		//create render pass information
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0;
		passInfo.pDependencies = nullptr;

		//create render pass and see if the render pass is created successfully
		VkRenderPass rpass = VK_NULL_HANDLE;
		VK_CHECK_RESULT (vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass))

		//return the wrapped value
		return lut::RenderPass(aWindow.device, rpass);
	}

	OffScreenFrameBuffer createOffscreenFramebuffer(lut::VulkanWindow const& aWindow, labutils::Allocator const& aAllocator)
	{
		OffScreenFrameBuffer ret{};

		ret.width = aWindow.swapchainExtent.width;
		ret.height = aWindow.swapchainExtent.height;

		// albedo & emissive
		ret.albedo = elf::createFBAttachment(aWindow, aAllocator, cfg::k_ColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		ret.normal = elf::createFBAttachment(aWindow, aAllocator, cfg::k_ColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		ret.position = elf::createFBAttachment(aWindow, aAllocator, cfg::k_ColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		ret.brightness = elf::createFBAttachment(aWindow, aAllocator, cfg::k_ColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
		ret.depth = elf::createFBAttachment(aWindow, aAllocator, cfg::k_DepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

		// Set up separate renderpass with references to the color and depth attachments
		VkAttachmentDescription attachDescs[5]{};

		// Init attachment properties
		for (uint32_t i = 0; i < 5; ++i)
		{
			attachDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
			attachDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			if (i == 4)
			{
				attachDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachDescs[i].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			}
			else
			{
				attachDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachDescs[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}
		}

		// Formats
		attachDescs[0].format = ret.albedo.format;
		attachDescs[1].format = ret.normal.format;
		attachDescs[2].format = ret.position.format;
		attachDescs[3].format = ret.brightness.format;
		attachDescs[4].format = ret.depth.format;

		// Create Subpass Definition
		VkAttachmentReference colorAttachs[4]{};
		// Color attachment for albedo and emissive
		colorAttachs[0].attachment = 0;
		colorAttachs[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		// Color attachment for normals
		colorAttachs[1].attachment = 1;
		colorAttachs[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		// Color attachment for position
		colorAttachs[2].attachment = 2;
		colorAttachs[2].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		// Color attachment for brightness
		colorAttachs[3].attachment = 3;
		colorAttachs[3].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 4;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses{};
		subpasses.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses.colorAttachmentCount = 4;
		subpasses.pColorAttachments = colorAttachs;
		subpasses.pDepthStencilAttachment = &depthAttachment;

		// Use subpass dependencies for attachment layout transitions
		VkSubpassDependency deps[2];

		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].dstSubpass = 0;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		deps[1].srcSubpass = 0;
		deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create Render Pass Information
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 5;
		passInfo.pAttachments = attachDescs;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = &subpasses;
		passInfo.dependencyCount = 2;
		passInfo.pDependencies = deps;

		// Create Render Pass
		VK_CHECK_RESULT ( vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &ret.renderPass.handle) )
		
		VkImageView attachImageViews[5];
		attachImageViews[0] = ret.albedo.view.handle;
		attachImageViews[1] = ret.normal.view.handle;
		attachImageViews[2] = ret.position.view.handle;
		attachImageViews[3] = ret.brightness.view.handle;
		attachImageViews[4] = ret.depth.view.handle;

		VkFramebufferCreateInfo fbufCreateInfo = enit::framebufferCreateInfo();
		fbufCreateInfo.renderPass = ret.renderPass.handle;
		fbufCreateInfo.pAttachments = attachImageViews;
		fbufCreateInfo.attachmentCount = 5;
		fbufCreateInfo.width = ret.width;
		fbufCreateInfo.height = ret.height;
		fbufCreateInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(aWindow.device, &fbufCreateInfo, nullptr, &ret.framebuffer.handle))

		return std::move(ret);
	}

	PostFrameBuffer createPostProcessingFrameBuffer(lut::VulkanWindow const& window, labutils::Allocator const& allocator)
	{
		PostFrameBuffer ret{};
		ret.width = window.swapchainExtent.width;
		ret.height = window.swapchainExtent.height;

		// Create a separate render pass for the offscreen rendering as it may differ from the one used for scene rendering

		VkAttachmentDescription attchmentDescriptions[2]{};
		// Color attachment
		attchmentDescriptions[0].format = cfg::k_ColorFormat;
		attchmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attchmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attchmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attchmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attchmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attchmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		// Depth attachment
		attchmentDescriptions[1].format = cfg::k_DepthFormat;
		attchmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attchmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attchmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attchmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attchmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
		VkAttachmentReference depthReference = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		subpassDescription.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for layout transitions
		VkSubpassDependency deps[2];

		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].dstSubpass = 0;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		deps[1].srcSubpass = 0;
		deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 2;
		renderPassInfo.pAttachments = attchmentDescriptions;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = 2;
		renderPassInfo.pDependencies = deps;

		// Create Render Pass
		VK_CHECK_RESULT ( vkCreateRenderPass(window.device, &renderPassInfo, nullptr, &ret.renderPass.handle) )

		// Create sampler to sample from the color attachments
		auto sampler = lut::createTexturesampler(window);
		ret.sampler = std::move(sampler);

		// H and V frameBuffers
		{
			auto& V = ret.h;
			V.color = elf::createFBAttachment(window, allocator, cfg::k_ColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
			V.depth = elf::createFBAttachment(window, allocator, cfg::k_DepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

			VkImageView attachments[2];
			attachments[0] = V.color.view.handle;
			attachments[1] = V.depth.view.handle;

			VkFramebufferCreateInfo fbufCreateInfo = enit::framebufferCreateInfo();
			fbufCreateInfo.renderPass = ret.renderPass.handle;
			fbufCreateInfo.attachmentCount = 2;
			fbufCreateInfo.pAttachments = attachments;
			fbufCreateInfo.width = window.swapchainExtent.width;
			fbufCreateInfo.height = window.swapchainExtent.height;
			fbufCreateInfo.layers = 1;

			VK_CHECK_RESULT(vkCreateFramebuffer(window.device, &fbufCreateInfo, nullptr, &V.framebuffer.handle))

			V.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			V.descriptor.imageView = V.color.view.handle;
			V.descriptor.sampler = ret.sampler.handle;
		}

		{
			auto& V = ret.v;
			V.color = elf::createFBAttachment(window, allocator, cfg::k_ColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
			V.depth = elf::createFBAttachment(window, allocator, cfg::k_DepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

			VkImageView attachments[2];
			attachments[0] = V.color.view.handle;
			attachments[1] = V.depth.view.handle;

			VkFramebufferCreateInfo fbufCreateInfo = enit::framebufferCreateInfo();
			fbufCreateInfo.renderPass = ret.renderPass.handle;
			fbufCreateInfo.attachmentCount = 2;
			fbufCreateInfo.pAttachments = attachments;
			fbufCreateInfo.width = window.swapchainExtent.width;
			fbufCreateInfo.height = window.swapchainExtent.height;
			fbufCreateInfo.layers = 1;

			VK_CHECK_RESULT(vkCreateFramebuffer(window.device, &fbufCreateInfo, nullptr, &V.framebuffer.handle))

			V.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			V.descriptor.imageView = V.color.view.handle;
			V.descriptor.sampler = ret.sampler.handle;
		}

		return std::move(ret);
	}



	lut::PipelineLayout createOffScreenPipeLayout( lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout,
		VkDescriptorSetLayout aLightLayout, VkDescriptorSetLayout aTextureLayout)
	{
		//in two shader state,there are two layouts about what uniforms it has
		VkDescriptorSetLayout layouts[] = { 
			// Order must match the set = N in the shaders 
			aSceneLayout,   //set 0
			aObjectLayout,  //set 1
			aLightLayout,    //set 2
			aTextureLayout   //set 3
		};

		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(glm::vec4);

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		VK_CHECK_RESULT(vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout))

		//return wrapped info
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout createSurfacePipelineLayout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aLayout)
	{
		VkDescriptorSetLayout layouts[] = {
			aLayout   //set 0
		};

		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(glm::vec4);

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		VK_CHECK_RESULT( vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout))

		//return wrapped info
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::Pipeline createOffScreenPipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		//load shader
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::mrtVSPath);

		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::mrtFSPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.back.compareOp = VK_COMPARE_OP_ALWAYS;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		VkVertexInputBindingDescription vertexInputs[3]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0;		//must match binding above
		vertexAttributes[0].location = 0;		//must match shader;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1;		//must match binding above
		vertexAttributes[1].location = 1;		//must match shader;
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2;		//must match binding above
		vertexAttributes[2].location = 2;		//must match shader;
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;
		                                                                                                                                
		inputInfo.vertexBindingDescriptionCount = 3;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		//input assembly state:define which primitive(point,line,triangle) the input is
		//assembled for rasterization
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//tessellation state:


		//scissor can be used to restrict drawing to a part of the frame buffer without changing the coordinate system
		// dynamic:
		VkDynamicState dynamicStates[] =
		{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		// define viewport and scissor regions in dynamic state
		VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
		dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateInfo.dynamicStateCount = 2;
		dynamicStateInfo.pDynamicStates = dynamicStates;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = nullptr;

		//rasterization state:
		//define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasClamp = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Multisample State��
		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[4]{};
		for (int i = 0; i < 4; i++) {
			blendStates[i].colorWriteMask = 0xf;
			blendStates[i].blendEnable = VK_FALSE;
		}

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.attachmentCount = 4;
		blendInfo.pAttachments = blendStates;

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
		pipeInfo.pDynamicState = &dynamicStateInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;
		
		VkPipeline pipe = VK_NULL_HANDLE;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe))

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline createSurfacePipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		//load shader
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::surfaceVSPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::surfaceFSPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		//input assembly state:define which primitive(point,line,triangle) the input is
		//assembled for rasterization
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkDynamicState dynamicStates[] =
		{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		// define viewport and scissor regions in dynamic state
		VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
		dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateInfo.dynamicStateCount = 2;
		dynamicStateInfo.pDynamicStates = dynamicStates;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = nullptr;

		//rasterization state:
		//define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasClamp = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Multisample State��
		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_TRUE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		//In Vulkan, logical operations are used to perform bitwise operations on color data in a framebuffer
		//attachment during blending.(e.g. AND, OR, XOR)
		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;
		
		//dynamic state:none

		//Assembling the VkGraphicsPipelineCreateInfo structure
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = &dynamicStateInfo;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		pipeInfo.pVertexInputState = &inputInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe))

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline createPostPipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, uint32_t direction)
	{
		//load shader
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::postVSPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::postFSPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		// Use specialization constants to select between horizontal and vertical blur
	// 	uint32_t blurdirection = direction;
		VkSpecializationMapEntry specializationMapEntry{};
		specializationMapEntry.constantID = 0;
		specializationMapEntry.offset = 0;
		specializationMapEntry.size = sizeof(uint32_t);

		VkSpecializationInfo specializationInfo{};
		specializationInfo.mapEntryCount = 1;
		specializationInfo.pMapEntries = &specializationMapEntry;
		specializationInfo.dataSize = sizeof(uint32_t);
		specializationInfo.pData = &direction;

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";
		stages[1].pSpecializationInfo= &specializationInfo;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		//input assembly state:define which primitive(point,line,triangle) the input is
		//assembled for rasterization
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkDynamicState dynamicStates[] =
		{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		// define viewport and scissor regions in dynamic state
		VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
		dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateInfo.dynamicStateCount = 2;
		dynamicStateInfo.pDynamicStates = dynamicStates;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = nullptr;

		//rasterization state:
		//define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasClamp = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Multisample State��
		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendAttachmentState{};
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

		//In Vulkan, logical operations are used to perform bitwise operations on color data in a framebuffer
		//attachment during blending.(e.g. AND, OR, XOR)
		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = &blendAttachmentState;

		//dynamic state:none

		//Assembling the VkGraphicsPipelineCreateInfo structure
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = &dynamicStateInfo;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		pipeInfo.pVertexInputState = &inputInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		//the second arguement means whether to use VkPipelineCache which can keep the cost down
		VK_CHECK_RESULT( vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe))

		return lut::Pipeline(aWindow.device, pipe);
	}

	std::vector<lut::Framebuffer> createSwapchainFramebuffers( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkImageView aDepthView )
	{
		std::vector<lut::Framebuffer> ret;

		for (std::size_t i = 0; i < aWindow.swapViews.size(); i++) {
			VkImageView attachments[2] = {
				aWindow.swapViews[i],

				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; //normal frame buffer
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			VK_CHECK_RESULT (vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb))

			ret.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == ret.size());

		return std::move(ret);
	}

	lut::DescriptorSetLayout createCustomedDescLayout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorBindingFlags binding_flags[2] = 
		{
			VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
			VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		};

		VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
		binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
		binding_flags_info.bindingCount = sizeof(binding_flags) / sizeof(binding_flags[0]);
		binding_flags_info.pBindingFlags = binding_flags;

		VkDescriptorSetLayoutBinding bindings[2]{};
		bindings[0] = enit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
		bindings[1] = enit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1);

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT; 
		layoutInfo.pNext = &binding_flags_info;
		layoutInfo.bindingCount = 2;
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		VK_CHECK_RESULT (vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout))

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	void recordCommands(VkCommandBuffer cmdBuf, 
		const VkExtent2D & ext,
		VkFramebuffer swapchainFB,
		UserState state,
		// framebuffer
		OffScreenFrameBuffer& offScreenFrameBuffer, PostFrameBuffer& postFrameBuffer, SurfaceFrameBuffer& surfaceFrameBuffer,
		// input data
		InputData& inputData,
		// pipeline and uniform resources
		UniformBlock& uniformBlock, DescriptorSets& descriptorSets, PipelineLayouts& pipelineLayouts,
		// pipelines
		GraphicsPipelines& pipelines
	)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuf, &beginInfo))

		lut::buffer_barrier(cmdBuf,
			uniformBlock.uniformBuffers.viewUBO.buffer,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(cmdBuf, uniformBlock.uniformBuffers.viewUBO.buffer, 0, sizeof(glsl::ViewUniform), &uniformBlock.uniformData.viewData);

		lut::buffer_barrier(cmdBuf,
			uniformBlock.uniformBuffers.viewUBO.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		lut::buffer_barrier(cmdBuf,
			uniformBlock.uniformBuffers.lightUBO.buffer,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(cmdBuf, uniformBlock.uniformBuffers.lightUBO.buffer, 0, sizeof(glsl::LightUniform), &uniformBlock.uniformData.lightData);

		lut::buffer_barrier(cmdBuf,
			uniformBlock.uniformBuffers.lightUBO.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		lut::buffer_barrier(cmdBuf,
			uniformBlock.uniformBuffers.postUBO.buffer,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(cmdBuf, uniformBlock.uniformBuffers.postUBO.buffer, 0, sizeof(glsl::BloomUniform), &uniformBlock.uniformData.bloomUniform);

		lut::buffer_barrier(cmdBuf,
			uniformBlock.uniformBuffers.postUBO.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		// setup dynamic viewport and scissor
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(ext.width);
		viewport.height = float(ext.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ ext.width, ext.height };

		// offscreen mrt
		VkClearValue clearValues[5]{};
		for (int i = 0; i < 4; i++) {
			clearValues[i].color.float32[0] = 0.0f;
			clearValues[i].color.float32[1] = 0.0f;
			clearValues[i].color.float32[2] = 0.0f;
			clearValues[i].color.float32[3] = 1.0f;
		}

		clearValues[4].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo offscreenPass{};
		offscreenPass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		offscreenPass.renderPass = offScreenFrameBuffer.renderPass.handle;
		offscreenPass.framebuffer = offScreenFrameBuffer.framebuffer.handle;
		offscreenPass.renderArea.offset = VkOffset2D{ 0,0 };
		offscreenPass.renderArea.extent = VkExtent2D{ ext.width,ext.height };
		offscreenPass.clearValueCount = 5;
		offscreenPass.pClearValues = clearValues;

		vkCmdBeginRenderPass(cmdBuf, &offscreenPass, VK_SUBPASS_CONTENTS_INLINE);
		glm::vec4 cameraPos = state.camera2world[3];
		vkCmdPushConstants(cmdBuf, pipelineLayouts.offscreenPipeLayout.handle, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec4), &cameraPos);
		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.offscreenPipe.handle);
		vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
		vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.offscreenPipeLayout.handle, 0, 1, &descriptorSets.view, 0, nullptr);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.offscreenPipeLayout.handle, 2, 1, &descriptorSets.light, 0, nullptr);
		for (uint32_t i = 0; i < inputData.VBOs.size(); i++) 
		{
			vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.offscreenPipeLayout.handle, 1, 1, &descriptorSets.colors[i], 0, nullptr);
			vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.offscreenPipeLayout.handle, 3, 1, &descriptorSets.textures[i], 0, nullptr);

			//Bind vertex input
			VkBuffer objBuffers[3] = { inputData.VBOs[i].positions.buffer, inputData.VBOs[i].texcoords.buffer, inputData.VBOs[i].normals.buffer };
			VkDeviceSize objOffsets[3]{};

			vkCmdBindVertexBuffers(cmdBuf, 0, 3, objBuffers, objOffsets);
			vkCmdBindIndexBuffer(cmdBuf, inputData.VBOs[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(cmdBuf, static_cast<uint32_t>(inputData.modelData.meshes[i].indices.size()), 1, 0, 0, 0);
		}
		//end the render pass
		vkCmdEndRenderPass(cmdBuf);

		// vertical pass
		VkClearValue postProcessClearColor[2]{};
		postProcessClearColor[0].color.float32[0] = 0.0f;
		postProcessClearColor[0].color.float32[1] = 0.0f;
		postProcessClearColor[0].color.float32[2] = 0.0f;
		postProcessClearColor[0].color.float32[3] = 1.0f;

		postProcessClearColor[1].depthStencil.depth = 1.0f;

		VkRenderPassBeginInfo postProcessPass{};
		postProcessPass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		postProcessPass.renderPass = postFrameBuffer.renderPass.handle;
		postProcessPass.framebuffer = postFrameBuffer.h.framebuffer.handle;
		postProcessPass.renderArea.offset = VkOffset2D{ 0,0 };
		postProcessPass.renderArea.extent = VkExtent2D{ ext.width,ext.height };
		postProcessPass.clearValueCount = 2;
		postProcessPass.pClearValues = postProcessClearColor;

		vkCmdBeginRenderPass(cmdBuf, &postProcessPass, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.bloomVerticalPipe.handle);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.postProcessLayout.handle, 0, 1, &descriptorSets.bloomVertical, 0, nullptr);




		vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
		vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

		vkCmdDraw(cmdBuf, 3, 1, 0, 0);
		//end the render pass
		vkCmdEndRenderPass(cmdBuf);

		// horizontal pass
		VkRenderPassBeginInfo postProcessPass2{};
		postProcessPass2.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		postProcessPass2.renderPass = postFrameBuffer.renderPass.handle;
		postProcessPass2.framebuffer = postFrameBuffer.v.framebuffer.handle;
		postProcessPass2.renderArea.offset = VkOffset2D{ 0,0 };
		postProcessPass2.renderArea.extent = VkExtent2D{ ext.width,ext.height };
		postProcessPass2.clearValueCount = 2;
		postProcessPass2.pClearValues = postProcessClearColor;

		// postProcessPass.framebuffer = postFrameBuffer.framebuffers[1].framebuffer.handle;
		// postProcessPass.renderArea.offset = VkOffset2D{ 0,0 };
		// postProcessPass.renderArea.extent = VkExtent2D{ ext.width,ext.height };

		vkCmdBeginRenderPass(cmdBuf, &postProcessPass2, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.bloomHorizontalPipe.handle);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.postProcessLayout.handle, 0, 1, &descriptorSets.bloomHorizontal, 0, nullptr);
		vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
		vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
		vkCmdDraw(cmdBuf, 3, 1, 0, 0);
		//end the render pass
		vkCmdEndRenderPass(cmdBuf);

		// surface pass
		VkClearValue surfaceClearColor[2]{};
		surfaceClearColor[0].color.float32[0] = 0.0f;
		surfaceClearColor[0].color.float32[1] = 0.0f;
		surfaceClearColor[0].color.float32[2] = 0.0f;
		surfaceClearColor[0].color.float32[3] = 1.0f;

		surfaceClearColor[1].depthStencil.depth = 1.0f;

		VkRenderPassBeginInfo surfacePass{};
		surfacePass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		surfacePass.renderPass = surfaceFrameBuffer.renderPass.handle;
		surfacePass.framebuffer = swapchainFB;
		surfacePass.renderArea.offset = VkOffset2D{ 0,0 };
		surfacePass.renderArea.extent = VkExtent2D{ ext.width,ext.height };
		surfacePass.clearValueCount = 2;
		surfacePass.pClearValues = surfaceClearColor;

		vkCmdBeginRenderPass(cmdBuf, &surfacePass, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.surfacePipeline.handle);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.surfacePipeLayout.handle, 0, 1, &descriptorSets.mrt, 0, nullptr);
		vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
		vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
		vkCmdDraw(cmdBuf, 3, 1, 0, 0);
		//end the render pass
		vkCmdEndRenderPass(cmdBuf);

		//end command recording
		VK_CHECK_RESULT (vkEndCommandBuffer(cmdBuf))
	}

	void submitCommands( lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

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

	void presentResults( VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain )
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
		{
			aNeedToRecreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str()
			);
		}
	}

	std::vector<lut::Buffer> createColorUBOs(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator, std::vector<glsl::ColorUniform>& colorUniforms, BakedModel& model)
	{
		std::vector<lut::Buffer> ret;
		for (uint32_t i = 0; i < model.meshes.size(); i++) {
			lut::Buffer colorUBO = lut::create_buffer(
				aAllocator,
				sizeof(glsl::ColorUniform),
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_GPU_ONLY
			);

			lut::Buffer colorStaging = lut::create_buffer(
				aAllocator,
				sizeof(glsl::ColorUniform),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_MEMORY_USAGE_CPU_TO_GPU
			);

			void* texPtr = nullptr;
			VK_CHECK_RESULT (vmaMapMemory(aAllocator.allocator, colorStaging.allocation, &texPtr))

			std::memcpy(texPtr, &colorUniforms[i], sizeof(glsl::ColorUniform));
			vmaUnmapMemory(aAllocator.allocator, colorStaging.allocation);

			lut::Fence uploadComplete = labutils::create_fence(aWindow);

			// Queue data uploads from staging buffers to the final buffers 
			// This uses a separate command pool for simplicity. 
			lut::CommandPool uploadPool = lut::create_command_pool(aWindow);
			VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aWindow, uploadPool.handle);

			//record copy commands into command buffer
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = 0;
			beginInfo.pInheritanceInfo = nullptr;

			VK_CHECK_RESULT(vkBeginCommandBuffer(uploadCmd, &beginInfo))

			//texcoords
			VkBufferCopy ccopy{};
			ccopy.size = sizeof(glsl::ColorUniform);

			vkCmdCopyBuffer(uploadCmd, colorStaging.buffer, colorUBO.buffer, 1, &ccopy);

			lut::buffer_barrier(uploadCmd,
				colorUBO.buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
			);

			VK_CHECK_RESULT(vkEndCommandBuffer(uploadCmd))

			//submit transfer commands
			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &uploadCmd;

			VK_CHECK_RESULT( vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, uploadComplete.handle))

			VK_CHECK_RESULT(vkWaitForFences(aWindow.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()))

			ret.emplace_back(std::move(colorUBO));
		}

		return ret;
	}
	
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
