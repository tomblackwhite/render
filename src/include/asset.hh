#pragma once
#include "tool.hh"
#include <concepts>
#include <cstddef>
#include <cstring>
#include <fmt/format.h>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <string>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <vk_mem_alloc.h>

#include <stb/stb_image.h>
#include <tiny_obj_loader.h>

using std::string;

// Texture own on vram
// can be used in vram
namespace App {


template <typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

template <typename T>
concept VulkanAssetObject = IsAnyOf<T, VkBuffer, VkImage>;
// Buffer Deleter
template <typename T> struct VmaDeleter {

  using pointer = T;

  void operator()(T pointer) {
    if (pointer != nullptr) {
      if constexpr (std::is_same_v<T, VkBuffer>) {
        vmaDestroyBuffer(m_allocator, pointer, m_allocation);
      } else if constexpr (std::is_same_v<T, VkImage>) {
        vmaDestroyImage(m_allocator, pointer, m_allocation);
      }else{
        //do nothing
      }
    }
  }

  VmaAllocator m_allocator = {};
  VmaAllocation m_allocation = {};
};


using VulkanBufferHandle = std::unique_ptr<VkBuffer, VmaDeleter<VkBuffer>>;
using VulkanImageHandle = std::unique_ptr<VkImage, VmaDeleter<VkImage>>;

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

  static VertexInputDescription getVertexDescription() {
    VertexInputDescription description;
    vk::VertexInputBindingDescription bindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex};
    description.bindings.push_back(bindingDescription);

    std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[0].offset = offsetof(Vertex, position);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[1].offset = offsetof(Vertex, normal);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[2].offset = offsetof(Vertex, color);
    description.attributes.insert(
        description.attributes.end(),
        std::make_move_iterator(attributeDescriptions.begin()),
        std::make_move_iterator(attributeDescriptions.end()));

    return description;
  }
};

struct MeshPushConstants {
  glm::vec4 data;
  glm::mat4 renderMatrix;
};

// Mesh
struct Mesh {
  std::vector<Vertex> vertices;
  VulkanBufferHandle vertexBuffer;

  bool loadFromOBJ(std::string const &path) {

    tinyobj::attrib_t attrib;

    std::vector<tinyobj::shape_t> shapes;

    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool re =
        tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path.c_str());

    if (!err.empty()) {
      std::cerr << "loadObject error " << err << '\n';
    }
    if (!re) {
      return false;
    }

    for (auto &shape : shapes) {

      // Loop over faces(polygon)
      size_t index_offset = 0;
      for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
        size_t fv = size_t(shape.mesh.num_face_vertices[f]);

        // Loop over vertices in the face.
        for (size_t v = 0; v < fv; v++) {
          // access to vertex
          tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

          Vertex vertex{};
          tinyobj::real_t vx =
              attrib.vertices[3 * size_t(idx.vertex_index) + 0];
          tinyobj::real_t vy =
              attrib.vertices[3 * size_t(idx.vertex_index) + 1];
          tinyobj::real_t vz =
              attrib.vertices[3 * size_t(idx.vertex_index) + 2];

          vertex.position = glm::vec3(vx, vy, vz);
          // Check if `normal_index` is zero or positive. negative = no normal
          // data
          if (idx.normal_index >= 0) {
            tinyobj::real_t nx =
                attrib.normals[3 * size_t(idx.normal_index) + 0];
            tinyobj::real_t ny =
                attrib.normals[3 * size_t(idx.normal_index) + 1];
            tinyobj::real_t nz =
                attrib.normals[3 * size_t(idx.normal_index) + 2];
            vertex.normal = glm::vec3(nx, ny, nz);
          }

          // Check if `texcoord_index` is zero or positive. negative = no
          // texcoord data
          if (idx.texcoord_index >= 0) {
            tinyobj::real_t tx =
                attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
            tinyobj::real_t ty =
                attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
          }

          vertex.color = vertex.normal;

          // Optional: vertex colors
          // tinyobj::real_t red   =
          // attrib.colors[3*size_t(idx.vertex_index)+0]; tinyobj::real_t green
          // = attrib.colors[3*size_t(idx.vertex_index)+1]; tinyobj::real_t blue
          // = attrib.colors[3*size_t(idx.vertex_index)+2];
          vertices.push_back(vertex);
        }
        index_offset += fv;
      }
    }

    return true;
  }
};

class VulkanMemory {
public: // Inteface
  [[nodiscard]] VulkanBufferHandle
  createBuffer(VkBufferCreateInfo &createInfo,
               VmaAllocationCreateInfo const &allocationInfo) {
    VkBuffer buffer = {};
    VmaAllocation allocation = {};

    VkResult re = vmaCreateBuffer(
        m_allocator, &static_cast<VkBufferCreateInfo const &>(createInfo),
        &allocationInfo, &buffer, &allocation, nullptr);

    if (re != VK_SUCCESS) {
      App::ThrowException(fmt::format("create Buffer Error {}", re));
    }

    return VulkanBufferHandle(buffer, VmaDeleter<VkBuffer>{m_allocator, allocation});
  }

 [[nodiscard]] VulkanImageHandle
  createImage(VkImageCreateInfo &createInfo,
               VmaAllocationCreateInfo const &allocationInfo) {
    VkImage image = {};
    VmaAllocation allocation = {};

    VkResult re = vmaCreateImage(
        m_allocator, &static_cast<VkImageCreateInfo const &>(createInfo),
        &allocationInfo, &image, &allocation, nullptr);

    if (re != VK_SUCCESS) {
      App::ThrowException(fmt::format("create Image Error {}", re));
    }

    return VulkanImageHandle(image, VmaDeleter<VkImage>{m_allocator, allocation});
  }


  // uploadMesh 后续可以改成模板
  void uploadMesh(Mesh &mesh) {
    upload(mesh.vertexBuffer, std::span(mesh.vertices));
  }

  // upload to gpu memory
  template <typename T>
  void upload(VulkanBufferHandle &handle, std::span<T> buffer) {

    vk::BufferCreateInfo bufferInfo = {};
    bufferInfo.setSize(buffer.size_bytes());
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eVertexBuffer);
    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.flags = VmaAllocationCreateFlagBits::
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    handle = createBuffer(static_cast<VkBufferCreateInfo &>(bufferInfo),
                          allocationInfo);

    auto deleter = handle.get_deleter();
    auto *alloction = deleter.m_allocation;
    void *data = nullptr;
    VULKAN_CHECK(vmaMapMemory(m_allocator, alloction, &data), "map error");

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
};

} // namespace App
