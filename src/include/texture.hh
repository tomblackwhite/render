#pragma once
#include "tool.hh"
#include <fmt/format.h>
#include <memory>
#include <string>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <vk_mem_alloc.h>

#include <stb/stb_image.h>

using std::string;

// Texture own on vram
// can be used in vram
namespace App {

// Buffer Deleter
struct VulkanBufferDeleter {

  using pointer = VkBuffer;

  void operator()(VkBuffer pointer) {
    if (pointer != nullptr) {
      vmaDestroyBuffer(m_allocator, pointer, m_allocation);
    }
  }

  VmaAllocator m_allocator = {};
  VmaAllocation m_allocation = {};
};

using VulkanBufferHandle = std::unique_ptr<VkBuffer, VulkanBufferDeleter>;

class Texture {
public:
  Texture(vk::Device device, const string &path) : m_device(device) {}

private:
  void LoadImageFromFileToTexture(std::string const &path);

  vk::Device m_device;
};

class VulkanFactory {
public:
  // Interace
  VulkanBufferHandle createBuffer(vk::BufferCreateInfo const &createInfo) {
    VkBuffer buffer;
    VmaAllocation allocation;

    if(VkResult re=vmaCreateBuffer(m_allocator, &static_cast<VkBufferCreateInfo const&>(createInfo), &m_bufferAllocationCreateInfo,
                    &buffer, &allocation, nullptr);
       re!=VK_SUCCESS){
      App::ThrowException(fmt::format("create Buffer Error {}", re));
    }

    return VulkanBufferHandle(buffer,VulkanBufferDeleter{m_allocator,allocation});
  }

  VulkanFactory(VmaAllocatorCreateInfo const &createInfo) {

    if (VkResult re = vmaCreateAllocator(&createInfo, &m_allocator);
        re != VK_SUCCESS) {
      App::ThrowException(fmt::format("create VmaAllocator Error {}", re));
    }
  }

  ~VulkanFactory() {
    if (m_allocator != nullptr) {
      vmaDestroyAllocator(m_allocator);
    }
  }

private:
  VmaAllocator m_allocator = {};
  VmaAllocationCreateInfo m_bufferAllocationCreateInfo = {
      .usage = VMA_MEMORY_USAGE_AUTO};
};

} // namespace App
