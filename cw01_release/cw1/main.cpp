#include <array>

#include "ApplicationPreComplied.h"
// #define MIPMAP			/* uncomment here for task 1.4 */
#define MESH_DENSITY		/* uncomment here for task 1.5 */

struct MeshDesities;

namespace shader
{
	enum class ShaderType
	{
		Vertex,
		Fragment,
		Geometry,
		Compute,
		TessellationControl,
		TessellationEvaluation,
		Count
	};

	struct Shader
	{
		ShaderType type;
		std::string path;
	};

	struct GraphicsShaders
	{
		Shader vertShader;
		Shader fragShader;
		// GraphicsShaders(Shader&& _vertShader, Shader&& _fragShader);
	};

	struct ExtenShaders : GraphicsShaders
	{
		Shader computeShader;
	};
}


namespace
{

	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
#define SHADERDIR_ "assets/cw1/shaders/"

		constexpr char const* kRenderVertShaderPath		= SHADERDIR_ "render.vert.spv";
		constexpr char const* kRenderFragShaderPath		= SHADERDIR_ "render.frag.spv";
		constexpr char const* kMipMapFragShaderPath		= SHADERDIR_ "mipmap.frag.spv";
		constexpr char const* kDensityFragShaderPath	= SHADERDIR_ "mesh_density.frag.spv";
		constexpr char const* kCompShaderPath			= SHADERDIR_ "mesh_density.comp.spv";


		//obj file
		constexpr char const* MODEL_PATH = "assets/cw1/sponza_with_ship.obj";

#undef SHADERDIR_

		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 100.f;

		constexpr auto kCameraFov    = 60.0_degf;
			
		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

		constexpr float kCameraBaseSpeed	= 1.0f;
		constexpr float kCameraFastMult		= 1.0f;
		constexpr float kCameraSlowMult		= 1.0f;

		constexpr float kCameraMouseSensitivity = 0.2f;
	}

	using clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	// GLFW callbacks
	void glfw_callback_key_press( GLFWwindow*, int, int, int, int );

	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	/*
	 * Uniform data
	 * MVP transformations
	 */ 
	namespace glsl
	{
		struct SceneUniform
		{
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCamera;
		};

	}

	// Helpers:

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
		max
	};

	struct  UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};

	void update_user_state(UserState&, float aElapsedTime);

	lut::RenderPass CreateRenderPass( lut::VulkanWindow const& );

	lut::DescriptorSetLayout CreateDescriptorSetLayout(lut::VulkanWindow const& _window, VkDescriptorType _type, VkShaderStageFlags _stageFlags)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = _type;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = _stageFlags;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = std::size(bindings);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout{};
		if (const auto& res = vkCreateDescriptorSetLayout(_window.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::DescriptorSetLayout(_window.device, layout);
	}


	lut::PipelineCache CreatePipelineCache(lut::VulkanContext const& aContext);

	/* pipeline layout creation */
	lut::PipelineLayout CreateGraphicsPipelineLayout( lut::VulkanContext const& , VkDescriptorSetLayout _vertLayout, VkDescriptorSetLayout _fragLayout);

#ifdef MESH_DENSITY
	lut::PipelineLayout CreateComputePipelineLayout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aDescriptorSetLayout);
	lut::DescriptorSetLayout CreateCompDescriptorLayout(lut::VulkanWindow const& _window);
	lut::Pipeline CreateComputePipeline(const lut::VulkanWindow&, VkPipelineLayout, const shader::Shader& _compShader);
#endif

	lut::Pipeline CreateGraphicsPipeline(lut::VulkanWindow const&, VkPipelineCache aPipelineCache, VkRenderPass, VkPipelineLayout, const shader::GraphicsShaders& aShaderWrapper);

	//be used to create different loaded obj pipeline

	void create_swapchain_framebuffers( 
		lut::VulkanWindow const&, 
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	void UpdateMVPMatrices(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState
	);

	void record_commands( 
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		VkExtent2D const&,
		VkBuffer aSceneUbo,
		glsl::SceneUniform const&,
		VkPipelineLayout,
		VkDescriptorSet aSceneDescriptors,
		VertexBuffer& aObjMesh,
		std::vector<VkDescriptorSet> aObjDescriptors,
		VkPipeline,
		MeshDesities _meshDensities
	);

	void submit_commands(
		lut::VulkanWindow const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
	void present_results( 
		VkQueue, 
		VkSwapchainKHR, 
		std::uint32_t aImageIndex, 
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

}

struct ArchitectureModel
{
	// active window pointer
	lut::VulkanWindow*			window;
	// shader layouts
	lut::DescriptorSetLayout	vertexShaderLayout{};
	lut::DescriptorSetLayout	fragmentshaderLayout{};
	// pipeline layout
	lut::PipelineLayout			graphicsPipelineLayout{};
	// pipeline cache
	lut::PipelineCache			pipelineCache{};
	// pipeline
	lut::Pipeline				graphicsPipeline{};
	// render pass
	lut::RenderPass				renderPass{};
	shader::GraphicsShaders		shaders{};

	struct TextureData_
	{
		std::vector<lut::Image> images;
		std::vector<lut::ImageView> imageViews;

		lut::Sampler activeSampler;
	}textures{};


	ArchitectureModel() = default;

	explicit ArchitectureModel(lut::VulkanWindow* _window)
		: window{ _window }
	{ }

	void SetActiveShaders(const shader::GraphicsShaders& _shaders)
	{
		shaders = _shaders;
		InitGraphicsPipeline(_shaders);
	}

	void InitGraphicsPipeline()
	{
		graphicsPipeline = CreateGraphicsPipeline(*window, pipelineCache.handle, renderPass.handle, graphicsPipelineLayout.handle, shaders);
	}

	void InitGraphicsPipeline(const shader::GraphicsShaders& _shaders)
	{
		graphicsPipeline = CreateGraphicsPipeline(*window, pipelineCache.handle, renderPass.handle, graphicsPipelineLayout.handle, _shaders);
	}

	void UpdateGraphicsPipleine()
	{
		graphicsPipeline = CreateGraphicsPipeline(*window, pipelineCache.handle, renderPass.handle, graphicsPipelineLayout.handle, shaders);
	}
};

struct CompProcess
{
	// active window pointer
	lut::VulkanWindow* window{};

	lut::DescriptorSetLayout	descriptorSetLayout{};
	lut::PipelineLayout			pipelineLayout{};
	lut::Pipeline				pipeline{};
	shader::Shader				shader{};

	lut::CommandPool			commandPool{};

	std::vector<VkDescriptorSet>	descriptorSets;
	std::vector<lut::Buffer>		outputBuffers;
	std::vector<lut::Buffer>		outputStagingBuffers;
};

struct MeshDesities
{
	float min;
	float max;
	std::vector<float> density;
};

int main() try
{
	/* Create a vulkan window */
	lut::VulkanWindow window = lut::make_vulkan_window();

	/* define a event object accept from user */
	UserState state{};

	/* glfw callback setup */
	glfwSetWindowUserPointer(window.window, &state);
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);

	/* Intialize shaders resouces */
	shader::GraphicsShaders model_Shaders {
		{shader::ShaderType::Vertex, cfg::kRenderVertShaderPath },
		{shader::ShaderType::Fragment, cfg::kRenderFragShaderPath } };

	shader::GraphicsShaders mipmap_Shaders {
		{shader::ShaderType::Vertex, cfg::kRenderVertShaderPath },
		{shader::ShaderType::Fragment, cfg::kMipMapFragShaderPath } };

	shader::GraphicsShaders density_Shaders {
		{shader::ShaderType::Vertex, cfg::kRenderVertShaderPath },
		{shader::ShaderType::Fragment, cfg::kDensityFragShaderPath } };

	/* create VMA allocator for resource managment */
	lut::Allocator allocator = lut::create_allocator(window);

	/* Intialize resources for model */
	ArchitectureModel model{&window};
	model.renderPass = CreateRenderPass(window);
	model.vertexShaderLayout = CreateDescriptorSetLayout(window, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT);
	model.fragmentshaderLayout = CreateDescriptorSetLayout(window, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	model.graphicsPipelineLayout = CreateGraphicsPipelineLayout(window, model.vertexShaderLayout.handle, model.fragmentshaderLayout.handle);
	model.pipelineCache = CreatePipelineCache(window);

#ifdef MIPMAP
	model.SetActiveShaders(mipmap_Shaders);
#else
	model.SetActiveShaders(model_Shaders);
#endif

#ifdef MESH_DENSITY
	model.SetActiveShaders(density_Shaders);
#endif

	/* create a depth buffer, return with depthbuffer(image), depthbufferview(operation). using structual binding to simplfy the code*/
	auto[depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	/*
	 * create and store frambuffer generated by swapchain
	 * similar to store an image stored in the swap chain
	 */
	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, model.renderPass.handle, framebuffers, depthBufferView.handle);

	/* command pool: create command pool, ready to give command buffer*/
	lut::CommandPool cpool = lut::CreateGraphicsCommandPool( window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT );

	/* create & fill command_buffers and fences for command buffers */
	std::vector<VkCommandBuffer> command_buffers;
	std::vector<lut::Fence> commandbuffer_fences;

	/*
	 * create command buffer and fence for each framebuffer
	 * and create semaphore for presenting a image and when render finished
	 */
	for( std::size_t i = 0; i < framebuffers.size(); ++i )
	{
		command_buffers.emplace_back( lut::alloc_command_buffer( window, cpool.handle ) );
		commandbuffer_fences.emplace_back( lut::create_fence( window, VK_FENCE_CREATE_SIGNALED_BIT ) );
	}

	lut::Semaphore imageAvailable = lut::create_semaphore( window );
	lut::Semaphore renderFinished = lut::create_semaphore( window );

	/* load vertices, attributes and indices */
	SimpleModel model_RawData = load_simple_wavefront_obj(cfg::MODEL_PATH);

	/* create a mesh container for attributed vertices */
	auto&& meshes = CreateMeshes(model_RawData);

	VertexBuffer model_VBO{ CreateVerticesBuffer(window, allocator, std::move(meshes)) };

	const auto& meshCount = model_VBO.indicesCount.size();

	/*
	 * create descriptor pool
	 * similar to glBindBuffer()
	 */
	lut::DescriptorPool descriptorPool = lut::CreateDescriptorPool(window);

	lut::CommandPool graphics_CommandPool = lut::CreateGraphicsCommandPool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

	/*
	 * uniform buffer in vertex shader, as a buffer, it is a resource...
	 */
	lut::Buffer model_UBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	/*
	 * To use the buffer, need to associate it with descriptor sets
	 */
	VkDescriptorSet model_VertexUniformDescriptorSet = lut::alloc_desc_set(window, descriptorPool.handle, model.vertexShaderLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};				/* specify write data*/

		VkDescriptorBufferInfo sceneUboInfo{};		/* specify buffer info*/
		sceneUboInfo.buffer = model_UBO.buffer;
		sceneUboInfo.range	= VK_WHOLE_SIZE;

		desc[0].sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet			= model_VertexUniformDescriptorSet;							/* data set is the target descriptor set */
		desc[0].dstBinding		= 0;										/* binding point in the shader  */
		desc[0].descriptorType	= VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;		/* descriptor type, here is UBO */
		desc[0].descriptorCount = 1;										/* how many descriptor should be writen in */
		desc[0].pBufferInfo		= &sceneUboInfo;							/* pass ubo in*/

		/* write in descriptor set */
		constexpr auto numSets = std::size(desc);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// ---------------------------Load Texture Resources------------------------------------------//
	/* load textures from disk into vkimage that could used by vulkan instance */
	// lut::Image defaultImage = lut::load_default_texture(window, graphics_CommandPool.handle, allocator);
	// lut::ImageView defaultImageView = lut::create_image_view_texture2d(window, defaultImage.image, VK_FORMAT_R8G8B8A8_SRGB);

	model.textures.images.resize(meshCount);
	for (size_t i = 0; i < meshCount; i++)
	{
		if (model_RawData.meshes[i].textured)
		{
			const auto& material = model_RawData.materials[model_RawData.meshes[i].materialIndex];
			const auto& texturePath = material.diffuseTexturePath;

			model.textures.images[i] = lut::load_image_texture2d(texturePath.c_str(), window, graphics_CommandPool.handle, allocator);
		}
		else
			model.textures.images[i] = lut::load_default_texture(window, graphics_CommandPool.handle, allocator);
	}

	/* create image view for texture image */
	// std::vector<lut::ImageView> model_ImageViewResources;
	model.textures.imageViews.resize(meshCount);
	for (size_t i = 0; i < meshCount; i++)
		model.textures.imageViews[i] = lut::create_image_view_texture2d(window, model.textures.images[i].image, VK_FORMAT_R8G8B8A8_SRGB);

	/*
	 * create image view for texture image,
	 * TODO and enable AF, whcih should be seen in Report*/
	model.textures.activeSampler = { lut::CreateTexture2DSampler(window) };

	/*
	 * set texture smapling for each textures, so using a loop to iterate each mesh,
	 * the core is simliar to UBO.
	 * But UBO in vertex shader, and all vertices use same MVP, and without bufferview, just from buffer to execuate 'vkUpdateDescriptorSets'
	 */
	std::vector<VkDescriptorSet> model_FragTextureSamplerDescriptorSet;
	model_FragTextureSamplerDescriptorSet.resize(meshCount);

	for (size_t i = 0; i < meshCount; i++)
	{
		VkDescriptorSet fragmentDescriptor = lut::alloc_desc_set(window, descriptorPool.handle, model.fragmentshaderLayout.handle);

		VkDescriptorImageInfo textureInfo{};
		textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo.sampler = model.textures.activeSampler.handle;
		textureInfo.imageView = model.textures.imageViews[i].handle;

		VkWriteDescriptorSet desc[1]{};

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = fragmentDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &textureInfo;

		constexpr auto numSets = std::size(desc);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
		model_FragTextureSamplerDescriptorSet[i] = fragmentDescriptor;
	}

#ifdef MESH_DENSITY

	// --------------------------------COMPUTE PROCESS DEALING -------------------------------------------------------//
	CompProcess compInstance;
	compInstance.window = { &window };
	compInstance.descriptorSetLayout = { CreateCompDescriptorLayout(window) };	/* remember to modity here if change the computer shader! */
	compInstance.pipelineLayout = { CreateComputePipelineLayout(window, compInstance.descriptorSetLayout.handle) };
	compInstance.shader = { shader::Shader{ shader::ShaderType::Compute, cfg::kCompShaderPath } };
	compInstance.pipeline = { CreateComputePipeline(window, compInstance.pipelineLayout.handle, compInstance.shader) };

	/* create descriptors from pool with compute layout */
	const auto& positionBufferSize = model_VBO.positionBuffer.size();
	compInstance.descriptorSets.resize(positionBufferSize);
	std::vector<lut::Buffer> outputBuffers;
	outputBuffers.resize(positionBufferSize);
	for (size_t i = 0; i < positionBufferSize; ++i)
	{
		VkDescriptorSet meshPosBufferDescriptorSet = lut::alloc_desc_set(window, descriptorPool.handle, compInstance.descriptorSetLayout.handle);

		VkWriteDescriptorSet writeSets[2]{};

		/* position buffer of each mesh*/
		VkDescriptorBufferInfo inputBufferInfo{};				
		inputBufferInfo.buffer = model_VBO.positionBuffer[i].buffer;
		inputBufferInfo.offset = 0;
		inputBufferInfo.range  = VK_WHOLE_SIZE;

		writeSets[0].sType				= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeSets[0].dstSet				= meshPosBufferDescriptorSet;
		writeSets[0].dstBinding			= 0;
		writeSets[0].descriptorType		= VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeSets[0].descriptorCount	= 1;
		writeSets[0].pBufferInfo		= &inputBufferInfo;

		/* specify output buffer info*/
		outputBuffers[i] = lut::create_buffer(
			allocator,
			model_VBO.indicesCount[i]/3 * sizeof(float),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VMA_MEMORY_USAGE_GPU_TO_CPU	/* access on CPU */
		);

		VkDescriptorBufferInfo outputBufferInfo{};
		outputBufferInfo.buffer = outputBuffers[i].buffer;
		outputBufferInfo.offset = 0;
		outputBufferInfo.range = VK_WHOLE_SIZE;

		writeSets[1].sType				= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeSets[1].dstSet				= meshPosBufferDescriptorSet;
		writeSets[1].dstBinding			= 1;
		writeSets[1].descriptorType		= VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeSets[1].descriptorCount	= 1;
		writeSets[1].pBufferInfo		= &outputBufferInfo;

		constexpr auto numSets = std::size(writeSets);
		vkUpdateDescriptorSets(window.device, numSets, writeSets, 0, nullptr);
		compInstance.descriptorSets[i] = meshPosBufferDescriptorSet;
	}

	compInstance.commandPool = lut::CreateComputeCommandPool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	VkCommandBuffer comp_command_buffer = lut::alloc_command_buffer(window, compInstance.commandPool.handle);
	lut::Fence comp_commandbuffer_fence = lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT);
#endif

	/* Application main loop */
	bool recreateSwapchain = false;
	auto previousClock = clock_::now();
	while( !glfwWindowShouldClose( window.window ) )
	{
		/* process events */
		glfwPollEvents(); // or: glfwWaitEvents()

		/*
		 * Recreate swap chain?
		 * for example: the window is resized, format of swapchain is changed...
		 * these events would trigger this code, if so, the application will wait for the device to be idle,
		 * then recreate the swapchain and related resources.
		 * recreate swapchain may need to recreate render pass, depth buffer, pipeline layout, pipeline and framebuffer.
		 */ 
		if( recreateSwapchain )
		{
			//re-create swapchain and associated resources
			vkDeviceWaitIdle(window.device);

			//recreate them
			const auto changes = recreate_swapchain(window);

			if (changes.changedFormat)	/* if format is changed, recreate the render pass*/
				model.renderPass = CreateRenderPass(window);

			if (changes.changedSize) 
			{
				/* if size is changed, recreate the depth buffer, pipeline and framebuffers */
				auto[depthBuffer,  depthBufferView] = create_depth_buffer(window, allocator);
				model.UpdateGraphicsPipleine();
			}

			/* clear all resouces and create, avoid any error*/
			framebuffers.clear();
			create_swapchain_framebuffers(window, model.renderPass.handle, framebuffers, depthBufferView.handle);

			/* stay false to keep logic right*/
			recreateSwapchain = false;
			continue;
		}

		/*
		 * acquire swapchain image
		 * acquire next swapchain available from vkAcquireNextImageKHR
		 * which will change the index of the swapchain
		 */
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

		/* wait for command buffer to be available */
		assert(static_cast<std::size_t>(imageIndex) < commandbuffer_fences.size());
		if (const auto res = 
			vkWaitForFences(window.device, 1, &commandbuffer_fences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); 
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str()
			);
		}

		if (const auto res = 
			vkResetFences(window.device, 1, &commandbuffer_fences[imageIndex].handle); 
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n"
				"vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str()
			);
		}

#ifdef MESH_DENSITY
		if (const auto res = 
			vkResetFences(window.device, 1, &comp_commandbuffer_fence.handle); 
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset compute command buffer fence %u\n"
				"vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str()
			);
		}
#endif

		/* update logic frame */
		auto const now = clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		/* compute shader part */
		MeshDesities meshDensity{};
#ifdef MESH_DENSITY
		meshDensity.min = 1.f;
		meshDensity.max = 0.f;
		{
			// Begin recording commands in the command buffer
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = 0; // Optional
			beginInfo.pInheritanceInfo = nullptr; // Optional

			if (vkBeginCommandBuffer(comp_command_buffer, &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			// Bind compute pipeline
			vkCmdBindPipeline(comp_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compInstance.pipeline.handle);

			// Bind descriptor sets for the compute shader
			for (std::size_t i = 0; i < model_VBO.positionBuffer.size(); ++i)
			{
				vkCmdBindDescriptorSets(comp_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compInstance.pipelineLayout.handle, 0, 1, &compInstance.descriptorSets[i], 0, nullptr);
				// Dispatch compute operation
				uint32_t num_triangles = model_VBO.indicesCount[i] / 3;
				uint32_t num_work_groups = (num_triangles + 255) / 256;
				vkCmdDispatch(comp_command_buffer, num_work_groups, 1, 1);
			}

			//end command recording
			if (auto const res = vkEndCommandBuffer(comp_command_buffer);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to end recording command buffer\n"
					"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()
				);
			}

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &comp_command_buffer;

			if (vkQueueSubmit(window.graphicsQueue, 1, &submitInfo, comp_commandbuffer_fence.handle) != VK_SUCCESS) {
				throw std::runtime_error("failed to submit compute command buffer!");
			}
			vkWaitForFences(window.device, 1, &comp_commandbuffer_fence.handle, VK_TRUE, UINT64_MAX);

			for (size_t bufferIndex = 0; bufferIndex < model_VBO.positionBuffer.size(); ++bufferIndex)
			{
				// get the size of buffer size
				uint32_t num_triangles = model_VBO.indicesCount[bufferIndex] / 3;
				VkDeviceSize bufferSize = num_triangles * sizeof(float);

				void* mappedData;
				vmaMapMemory(allocator.allocator, outputBuffers[bufferIndex].allocation, &mappedData);

				// access data
				float* triangleAreas = static_cast<float*>(mappedData);

				float totalArea = 0.0f;

				for (size_t i = 0; i < num_triangles; ++i) {
					totalArea += triangleAreas[i];
				}
				float density = float(bufferSize) / float(totalArea);
				meshDensity.density.emplace_back(density);
				if (density > meshDensity.max) meshDensity.max = density;
				if (density < meshDensity.min) meshDensity.min = density;

				vmaUnmapMemory(allocator.allocator, outputBuffers[bufferIndex].allocation);
			}
		}
#endif

		update_user_state(state, dt);

		/* record and submit commands */
		assert(static_cast<std::size_t>(imageIndex) < command_buffers.size());
		assert(static_cast<std::size_t>(imageIndex) < framebuffers.size());

		glsl::SceneUniform sceneUniforms{};
		UpdateMVPMatrices(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);

		static_assert(sizeof(sceneUniforms) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(sceneUniforms) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");
		record_commands(
			command_buffers[imageIndex],
			model.renderPass.handle,
			framebuffers[imageIndex].handle,
			window.swapchainExtent,
			model_UBO.buffer,
			sceneUniforms,
			model.graphicsPipelineLayout.handle,
			model_VertexUniformDescriptorSet,
			model_VBO,
			model_FragTextureSamplerDescriptorSet,
			model.graphicsPipeline.handle,
			meshDensity
		);

		submit_commands(
			window,
			command_buffers[imageIndex],
			commandbuffer_fences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		/* present rendered images (note: use the present_results() method) */
		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);

	}

	/* Cleanup */
	// but we sill need to ensure that all Vulkan commands have finished before that.
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
	void glfw_callback_key_press( GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/ )
	{
		if( GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction )
		{
			glfwSetWindowShouldClose( aWindow, GLFW_TRUE );
		}

		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		const bool isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;
		case GLFW_KEY_LEFT_SHIFT:			
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;
		case GLFW_KEY_LEFT_CONTROL: 			
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;
		default:
			;
		}
	}

	void glfw_callback_button(GLFWwindow* aWin,int aBut,int aAct,int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if(GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void glfw_callback_motion(GLFWwindow* aWin,double aX,double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if(aState.inputMap[std::size_t(EInputState::mousing)])
		{
			if(aState.wasMousing)
			{
				const auto sens = cfg::kCameraMouseSensitivity;
				const auto dx = sens * (aState.mouseX - aState.previousX);
				const auto dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}

		const auto move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f);

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, move));
		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move,0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move,0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));
	}

	void UpdateMVPMatrices(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState aState)
	{
		float const aspect = aFramebufferWidth / aFramebufferHeight;
		//The RH indicates a right handed clip space, and the ZO indicates
		//that the clip space extends from zero to one along the Z - axis.
		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		aSceneUniforms.projection[1][1] *= -1.f;
		aSceneUniforms.camera = glm::inverse(aState.camera2world);
		aSceneUniforms.projCamera = aSceneUniforms.projection * aSceneUniforms.camera;
	}
}

namespace
{
	/*
	* create a render pass object
	* renderpass:	a big struct contains all the information about how to render the scene, incluing lots of fileds.
	* inside the render pass,
	* attachments:	define attributes (color | depth | stencil)
	* subpass:		define behaviors / process / a sequence of pipeline
	*
	* one renderpass must has at least one subpass.
	*/
	lut::RenderPass CreateRenderPass(lut::VulkanWindow const& aWindow)
	{
		// create two attachments
		VkAttachmentDescription attachments[2]{};

		// the first one: color attachment
		attachments[0].format	= aWindow.swapchainFormat;
		attachments[0].samples	= VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp	= VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp	= VK_ATTACHMENT_STORE_OP_STORE;
		/* for color attachment, initial layout usually is undefined. */
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		/* it means when renderoass ended, color attachment will output on the window*/
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		// create a subpass for 0th attachment
		VkAttachmentReference color_attachment{};
		/* refers to the 0th render pass attachment declared earlier */
		color_attachment.attachment = 0;
		/* it is a color attachment */
		color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// second one: depth attachment
		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		/* for depth attachment, initial layout usually is also undefined. */
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		/* it means when renderoass ended, depth attachment will output on the window*/
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depth_attachment{};
		depth_attachment.attachment = 1;
		depth_attachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		/* define render subpasses */ 
		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint		 = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount	 = 1;
		subpasses[0].pColorAttachments		 = &color_attachment;
		subpasses[0].pDepthStencilAttachment = &depth_attachment;

		// grap attachments and subpass to create the render pass struct
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType				= VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount	= 2;
		passInfo.pAttachments		= attachments;
		passInfo.subpassCount		= 1;
		passInfo.pSubpasses			= subpasses;
		passInfo.dependencyCount	= 0;
		passInfo.pDependencies		= nullptr;

		// create render pass and see if the render pass is created successfully
		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str()
			);
		}

		//return the wrapped value
		return lut::RenderPass(aWindow.device, rpass);
	}

	/*
	 * create (shader) pipeline layout in vulkan
	 *
	 * here, create two layouts,
	 * one is for scene		-> MVP in vertex shader
	 * one is for object	-> texture sampler in fragment shader
	 */
	lut::PipelineLayout CreateGraphicsPipelineLayout( lut::VulkanContext const& _window, VkDescriptorSetLayout _vertLayout, VkDescriptorSetLayout _fragLayout)
	{
		const VkDescriptorSetLayout layouts[] = 
		{ 
			// Order must match the set = N in the shaders 
			_vertLayout,	// set 0
			_fragLayout,  // set 1
		};

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(float) * 3; // push_constant only contains one float

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType					= VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount			= std::size(layouts);
		layoutInfo.pSetLayouts				= layouts;
		layoutInfo.pushConstantRangeCount	= 1;
		layoutInfo.pPushConstantRanges		= &pushConstantRange;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(_window.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		//return wrapped info
		return lut::PipelineLayout(_window.device, layout);
	}

	/* using pipeline cache to accelebrate pipeline creation*/
	lut::PipelineCache CreatePipelineCache(lut::VulkanContext const& aContext)
	{
		VkPipelineCacheCreateInfo pipelineCacheCreateInfo{};
		pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		pipelineCacheCreateInfo.initialDataSize = 0;
		pipelineCacheCreateInfo.pInitialData = nullptr;

		VkPipelineCache cache = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineCache(aContext.device, &pipelineCacheCreateInfo, nullptr, &cache);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		//return wrapped info
		return lut::PipelineCache(aContext.device, cache);
	}


	lut::Pipeline CreateGraphicsPipeline(lut::VulkanWindow const& aWindow, VkPipelineCache aPipelineCache, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, const shader::GraphicsShaders& aShaderWrapper)
	{
		//load shader
		lut::ShaderModule vert = lut::load_shader_module(aWindow, aShaderWrapper.vertShader.path.c_str());
		lut::ShaderModule frag = lut::load_shader_module(aWindow, aShaderWrapper.fragShader.path.c_str());

		//define shader stage in the pipeline
		//shader stages:The pStages member points to an array of stageCount VkPipelineShaderStageCreateInfo structures.
		//this pipeline has just two shader stages :one for vertex shader,one for fragment shader
		//pName:We specify the name of each shader¡¯s entry point.In Exercise 2 this will be main, referring to the
		//main() function in each shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;	/*vertex shader*/
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;	/*fragment shader*/
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		/*
		 * simliar to glLinkAttributes(), layout of the vertices
		 *
		 */
		VkVertexInputBindingDescription vertexInputs[3]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 3;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 2;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0;		//must match binding above
		vertexAttributes[0].location = 0;		//must match shader;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1;		//must match binding above
		vertexAttributes[1].location = 1;		//must match shader;
		vertexAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2;		//must match binding above
		vertexAttributes[2].location = 2;		//must match shader;
		vertexAttributes[2].format	= VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[2].offset	= 0;

		inputInfo.vertexBindingDescriptionCount		= 3;
		inputInfo.pVertexBindingDescriptions		= vertexInputs;
		inputInfo.vertexAttributeDescriptionCount	= 3;
		inputInfo.pVertexAttributeDescriptions		= vertexAttributes;

		//input assembly state:define which primitive(point,line,triangle) the input is
		//assembled for rasterization
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//tessellation state:
		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		//scissor can be used to restrict drawing to a part of the frame buffer without changing the coordinate system
		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//rasterization state:
		//define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType					= VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable			= VK_FALSE;
		rasterInfo.rasterizerDiscardEnable	= VK_FALSE;
		rasterInfo.polygonMode				= VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode					= VK_CULL_MODE_BACK_BIT;		/* enable back face culling */
		rasterInfo.frontFace				= VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasClamp			= VK_FALSE;
		rasterInfo.lineWidth				= 1.f;

		//Multisample State£º
		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		//Depth/Stencil State:
		// Define blend state 
		// We define one blend state per color attachment - this example uses a 
		// single color attachment, so we only need one. Right now, we don¡¯t do any 
		// blending, so we can ignore most of the members.
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

		// dynamic state:none
		//TODO: window resized!

		//Assembling the VkGraphicsPipelineCreateInfo structure
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState		= &inputInfo;
		pipeInfo.pInputAssemblyState	= &assemblyInfo;
		pipeInfo.pTessellationState		= nullptr;
		pipeInfo.pViewportState			= &viewportInfo;
		pipeInfo.pRasterizationState	= &rasterInfo;
		pipeInfo.pMultisampleState		= &samplingInfo;
		pipeInfo.pDepthStencilState		= &depthInfo;
		pipeInfo.pColorBlendState		= &blendInfo;
		pipeInfo.pDynamicState			= nullptr;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		//the second arguement means whether to use VkPipelineCache which can keep the cost down
		if (auto const res = vkCreateGraphicsPipelines(
				aWindow.device,		/*device*/
				aPipelineCache,		/*pipelineCache*/
				1,					/*createInfoCount*/
				&pipeInfo,			/*pCreateInfos*/
				nullptr,			/*pAllocator*/
				&pipe				/*pPipelines*/
			); 
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::Pipeline(aWindow.device, pipe);
	}



	/*
	 * create a framebuffer container (many framebuffers),
	 * which size equals to the views size of swapchain */
	void create_swapchain_framebuffers( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView )
	{
		assert(aFramebuffers.empty());

		/*
		 * iterate through the swapchain views and create a framebuffer for each of them
		 * opertaion on each swapchain view:
		 * create a VkImageView array to hold the swapchain view and depth view
		 */
		for (std::size_t i = 0; i < aWindow.swapViews.size(); i++) 
		{
			VkImageView attachments[2] = 
			{
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
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer\n"
					"vkCreateFramebuffer() returned %s", lut::to_string(res).c_str()
				);
			}
			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

#ifdef MESH_DENSITY
	lut::Pipeline CreateComputePipeline(lut::VulkanWindow const& aWindow, VkPipelineLayout aPipelineLayout, const shader::Shader& _compShader)
	{
		/* load compute shader */
		lut::ShaderModule comp = lut::load_shader_module(aWindow, _compShader.path.c_str());

		/*
		 * compute shader stage info creation
		 * this part is almost same as vertex shade or fragment shader
		 */
		VkPipelineShaderStageCreateInfo compStage{};
		compStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		compStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;							/* compute shader */
		compStage.module = comp.handle;
		compStage.pName = "main";

		/* compute pipeline creation */
		VkComputePipelineCreateInfo computePipelineInfo{};
		computePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computePipelineInfo.stage = compStage;	/* pass compute stage in */
		computePipelineInfo.layout = aPipelineLayout;

		VkPipeline computePipeline = VK_NULL_HANDLE;
		//the second arguement means whether to use VkPipelineCache which can keep the cost down
		if (auto const res = vkCreateComputePipelines(
			aWindow.device,					/*device*/
			VK_NULL_HANDLE,					/*pipelineCache*/
			1,								/*createInfoCount*/
			&computePipelineInfo,			/*pCreateInfos*/
			nullptr,						/*pAllocator*/
			&computePipeline 				/*pPipelines*/
		); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create compute pipeline\n"
				"vkCreateComputePipelines() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::Pipeline(aWindow.device, computePipeline);
	}

	lut::DescriptorSetLayout CreateCompDescriptorLayout(lut::VulkanWindow const& _window)
	{
		VkDescriptorSetLayoutBinding bindings[2]{};

		// Input vertex buffer
		bindings[0].binding				= 0;
		bindings[0].descriptorType		= VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[0].descriptorCount		= 1;
		bindings[0].stageFlags			= VK_SHADER_STAGE_COMPUTE_BIT;

		// Output vertex buffer
		bindings[1].binding				= 1;
		bindings[1].descriptorType		= VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[1].descriptorCount		= 1;
		bindings[1].stageFlags			= VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = std::size(bindings);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout{};
		if (const auto& res = vkCreateDescriptorSetLayout(_window.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::DescriptorSetLayout(_window.device, layout);
	}
	lut::PipelineLayout CreateComputePipelineLayout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aDescriptorSetLayout)
	{
		// Set the descriptor set layout for the compute shader
		const VkDescriptorSetLayout layouts[] = {
			// Order must match the set = N in the shaders 
			aDescriptorSetLayout,  // set 0
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = std::size(layouts);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		// Return wrapped info
		return lut::PipelineLayout(aContext.device, layout);
	}
#endif

	void record_commands( 
		VkCommandBuffer aCmdBuff, 
		VkRenderPass aRenderPass, 
		VkFramebuffer aFramebuffer, 
		VkExtent2D const& aImageExtent, 
		VkBuffer aSceneUbo, 
		glsl::SceneUniform const& aSceneUniform,
		VkPipelineLayout aGraphicsLayout,
		VkDescriptorSet aSceneDescriptors, 
		VertexBuffer& aObjMesh,
		std::vector<VkDescriptorSet> aObjDescriptors, 
		VkPipeline aAlphaPipe,
		MeshDesities _meshDensities)
	{
		/*
		 * from VkCommandBufferBeginInfo
		 * to begin recording commands
		 */
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &beginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}

		lut::buffer_barrier(aCmdBuff,
			aSceneUbo,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUbo, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

		lut::buffer_barrier(aCmdBuff,
			aSceneUbo,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);


		//begin render pass
		VkClearValue clearValues[2]{};
		// Clear to a dark gray background. If we were debugging, this would potentially 
		// help us see whether the render pass took place, even if nothing else was drawn
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.f;

		clearValues[1].depthStencil.depth = 1.f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0,0 };
		passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width,aImageExtent.height };
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		/* step1: begin to render*/
		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		/* step2: bind the pipeline */
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aAlphaPipe);

		//bind the descriptor to specify the uniform inputs ahead of the draw call
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);

		for (std::size_t i = 0; i < aObjMesh.positionBuffer.size(); i++) 
		{
			//bind different textures
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, &aObjDescriptors[i], 0, nullptr);

			//Bind vertex input
			VkBuffer	 buffer[3] = { aObjMesh.positionBuffer[i].buffer, aObjMesh.colorBuffer[i].buffer, aObjMesh.uvBuffer[i].buffer };
			VkDeviceSize offsets[3]{};
			
			vkCmdBindVertexBuffers(aCmdBuff, 0, 3, buffer, offsets);

#ifdef MESH_DENSITY
			float constants[3] = { _meshDensities.density[i], _meshDensities.min, _meshDensities.max};

			vkCmdPushConstants(
				aCmdBuff,
				aGraphicsLayout,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				0, // offset
				sizeof(float),
				constants
			);
#endif
			/* step3: drawcall */
			vkCmdDraw(aCmdBuff, aObjMesh.indicesCount[i], 1, 0, 0);
		}


		/* step4: end the render pass*/
		vkCmdEndRenderPass(aCmdBuff);

		//end command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}
	}

	void submit_commands( lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore )
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
		
		if (const auto res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
			);
		}
	}

	void present_results( VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain )
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

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		if(const auto res = vmaCreateImage(aAllocator.allocator,&imageInfo,&allocInfo,&image,& allocation,nullptr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str()
			);
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0,1,
			0,1
		};

		VkImageView view = VK_NULL_HANDLE;
		if(const auto res = vkCreateImageView(aWindow.device,&viewInfo,nullptr,&view);VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", lut::to_string(res).c_str()
			);
		}

		return { std::move(depthImage),lut::ImageView(aWindow.device,view) };
	}
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
