#pragma once
#include "tool.hh"
#include <cstddef>
#include <cstring>
#include <fmt/format.h>
#include <memory>
#include <span>
#include <string>
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <glm/glm.hpp>
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

struct VertexInputDescription {
  std::vector<vk::VertexInputBindingDescription> bindings;
  std::vector<vk::VertexInputAttributeDescription> attributes;
  vk::PipelineVertexInputStateCreateFlags flags = {};
};

//顶点
struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec3 color;

  static VertexInputDescription getBindingDescription() {
    VertexInputDescription description;
    vk::VertexInputBindingDescription bindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex};
    description.bindings.push_back(bindingDescription);

    std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = ::vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[0].offset = offsetof(Vertex, position);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = ::vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = ::vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[2].offset = offsetof(Vertex, color);
    description.attributes.insert(
        description.attributes.end(),
        std::make_move_iterator(attributeDescriptions.begin()),
        std::make_move_iterator(attributeDescriptions.end()));

    return description;
  }

  // static std::array<vk::VertexInputAttributeDescription, 3>
  // getAttributeDescriptions() {
  //   std::array<vk::VertexInputAttributeDescription, 3>
  //   attributeDescriptions{};

  //   attributeDescriptions[0].binding = 0;
  //   attributeDescriptions[0].location = 0;
  //   attributeDescriptions[0].format = ::vk::Format::eR32G32Sfloat;
  //   attributeDescriptions[0].offset = offsetof(Vertex, pos);

  //   attributeDescriptions[1].binding = 0;
  //   attributeDescriptions[1].location = 1;
  //   attributeDescriptions[1].format = ::vk::Format::eR32G32B32Sfloat;
  //   attributeDescriptions[1].offset = offsetof(Vertex, color);

  //   attributeDescriptions[2].binding = 0;
  //   attributeDescriptions[2].location = 2;
  //   attributeDescriptions[2].format = ::vk::Format::eR32G32Sfloat;
  //   attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

  //   return attributeDescriptions;
  // }
};

// Mesh
struct Mesh {
  std::vector<Vertex> vertices;
  VulkanBufferHandle vertexBuffer;
};

class VulkanMemory {
public: // Inteface
  VulkanBufferHandle createBuffer(vk::BufferCreateInfo const &createInfo) {
    VkBuffer buffer;
    VmaAllocation allocation;

    if (VkResult re = vmaCreateBuffer(
            m_allocator, &static_cast<VkBufferCreateInfo const &>(createInfo),
            &m_bufferAllocationCreateInfo, &buffer, &allocation, nullptr);
        re != VK_SUCCESS) {
      App::ThrowException(fmt::format("create Buffer Error {}", re));
    }

    return VulkanBufferHandle(buffer,
                              VulkanBufferDeleter{m_allocator, allocation});
  }

  // uploadMesh
  void uploadMesh(Mesh &mesh) {
    auto deleter = mesh.vertexBuffer.get_deleter();
    upload(deleter.m_allocation, std::span(mesh.vertices));
  }

  // upload to gpu memory
  template <typename T>
  void upload(VmaAllocation alloction, std::span<T> buffer) {
    void *data = nullptr;
    vmaMapMemory(m_allocator, alloction, &data);

    std::memcpy(data, buffer.data(), buffer.size_bytes());
    vmaUnmapMemory(m_allocator, alloction);
  }

  explicit VulkanMemory(VmaAllocatorCreateInfo const &createInfo) {

    if (VkResult re = vmaCreateAllocator(&createInfo, &m_allocator);
        re != VK_SUCCESS) {
      App::ThrowException(fmt::format("create VmaAllocator Error {}", re));
    }
  }

  ~VulkanMemory() {
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
