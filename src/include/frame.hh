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

    // vk::BufferCreateInfo bufferInfo{};
    // bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
    // bufferInfo.setSize(sizeof(App::GPUCameraData));
    // VmaAllocationCreateInfo allocationInfo{};
    // allocationInfo.usage =
    // VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE; allocationInfo.flags
    // = VmaAllocationCreateFlagBits::
    //     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    // frame.cameraBuffer = memory->createBuffer(bufferInfo, allocationInfo);

    // auto sets = memory->createDescriptorSet(layout);

    // frame.globalDescriptor = std::move(sets[0]);

    // vk::DescriptorBufferInfo descriptorBufferInfo{};
    // descriptorBufferInfo.setBuffer(frame.cameraBuffer.get());
    // descriptorBufferInfo.setOffset(0);
    // descriptorBufferInfo.setRange(sizeof(App::GPUCameraData));

    // namespace init = App::VulkanInitializer;

    // std::array desCameraBufferInfos{descriptorBufferInfo};
    // auto writeDescriptorSet = init::getWriteDescriptorSet(
    //     vk::DescriptorType::eUniformBuffer, *frame.globalDescriptor,
    //     desCameraBufferInfos, 0);

    // constexpr int MAX_OBJECTS = 10000;
    // vk::BufferCreateInfo objectBufferInfo{};
    // objectBufferInfo.setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
    // objectBufferInfo.setSize(sizeof(App::GPUObjectData) * MAX_OBJECTS);
    // VmaAllocationCreateInfo objectAllocationInfo{};
    // objectAllocationInfo.usage =
    //     VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    // objectAllocationInfo.flags = VmaAllocationCreateFlagBits::
    //     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    // frame.objectBuffer =
    //     memory->createBuffer(objectBufferInfo, objectAllocationInfo);

    // std::array writeDes{writeDescriptorSet};
    // device->updateDescriptorSets(writeDes, {});

    return frame;
  }
};
} // namespace App
