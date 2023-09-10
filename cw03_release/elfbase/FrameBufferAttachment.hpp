#pragma once

#include "Marocs.hpp"
// #include "VulkanInitalizers.hpp"
#include "../labutils/vkobject.hpp"
// #include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vulkan_window.hpp"
// #include <volk/volk.h>


namespace elf
{
	struct FrameBufferAttachment
	{
		labutils::Image image{};
		labutils::ImageView view{};
		VkFormat format;
	};

	// [ref: code base of vulkan examples](https://github.com/SaschaWillems/Vulkan/blob/master/examples/)
	FrameBufferAttachment createFBAttachment(labutils::VulkanWindow const& _window, labutils::Allocator const& _allocator, VkFormat format, VkImageUsageFlags usage)
	{
		VkImageAspectFlags aspectMask = 0;
		uint32_t mipMapLevel = 0;

		if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
		{
			aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			mipMapLevel = VK_REMAINING_MIP_LEVELS;
		}

		if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			mipMapLevel = 1;
			aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
		}

		assert(aspectMask > 0);

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = format;
		imageInfo.extent.width = _window.swapchainExtent.width;
		imageInfo.extent.height = _window.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		// VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT flag is required for input attachments
		imageInfo.usage = usage /*| VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT*/;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		VK_CHECK_RESULT(vmaCreateImage(_allocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr))

		labutils::Image ret_image(_allocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = ret_image.image;
		// = vks::initializers::imageViewCreateInfo();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange
		{
			aspectMask,
			0, mipMapLevel,
			0, 1
		};

		VkImageView view = VK_NULL_HANDLE;
		VK_CHECK_RESULT(vkCreateImageView(_window.device, &viewInfo, nullptr, &view));

		return {std::move(ret_image), labutils::ImageView(_window.device, view), format};
	}

}
