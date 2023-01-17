#pragma once
#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>
#include "asset.hh"

namespace App {
namespace raii = vk::raii;


struct FrameData {
  raii::Semaphore availableSemaphore{nullptr};
  raii::Semaphore renderFinishedSemaphore{nullptr};
  raii::Fence inFlightFence{nullptr};

  raii::CommandPool commandPool{nullptr};
  raii::CommandBuffer commandBuffer{nullptr};

  VulkanBufferHandle cameraBuffer{nullptr};

  raii::DescriptorSet globalDescriptor{nullptr};

  VulkanBufferHandle objectBuffer{nullptr};
  raii::DescriptorSet objectDescriptorSet{nullptr};

};
}
