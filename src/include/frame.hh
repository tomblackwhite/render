#pragma once
#include "asset.hh"
#include <glm/glm.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace App {
namespace raii = vk::raii;

struct Frame {
  raii::Semaphore availableSemaphore{nullptr};
  raii::Semaphore renderFinishedSemaphore{nullptr};
  raii::Fence inFlightFence{nullptr};

  raii::CommandPool commandPool{nullptr};
  raii::CommandBuffer commandBuffer{nullptr};

  VulkanBufferHandle cameraBuffer{nullptr};

  raii::DescriptorSet globalDescriptor{nullptr};

  VulkanBufferHandle objectBuffer{nullptr};
  raii::DescriptorSet objectDescriptorSet{nullptr};

  static Frame createFrame(raii::Device *device,
                               vk::CommandPoolCreateInfo const &poolInfo,
                               App::VulkanMemory *memory,
                               vk::DescriptorSetLayout const &layout) {
    Frame frame{};
    frame.commandPool = device->createCommandPool(poolInfo);

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = *frame.commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;
    auto commandBuffers = device->allocateCommandBuffers(allocInfo);

    frame.commandBuffer = std::move(commandBuffers.at(0));

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    frame.availableSemaphore = device->createSemaphore(semaphoreInfo);
    frame.renderFinishedSemaphore = device->createSemaphore(semaphoreInfo);
    frame.inFlightFence = device->createFence(fenceInfo);


    return frame;
  }
};
} // namespace App
