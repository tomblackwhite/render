#pragma once
#include "tool.hh"
#include <cassert>
#include <cmath>
#include <concepts>
// #include <boost/preprocessor/seq.hpp>
// #include <boost/preprocessor/variadic.hpp>
#include <cstddef>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fmt/format.h>
#include <glm/glm.hpp>
#include <memory>
#include <observer.hh>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <vk_mem_alloc.h>

#include <stb/stb_image.h>

#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#include "gltf_type.hh"
#include <tiny_gltf.h>
#include <tiny_obj_loader.h>

using std::string;

// Texture own on vram
// can be used in vram
namespace App {

namespace raii = vk::raii;
using key = std::string;

template <typename T>
concept VulkanAssetObject = IsAnyOf<T, VkBuffer, VkImage>;
// Buffer Deleter
template <typename T> struct VmaDeleter {

  using pointer = T;

  void operator()(T pointer) noexcept {
    if (pointer != nullptr) {
      if constexpr (std::is_same_v<T, VkBuffer>) {
        vmaDestroyBuffer(m_allocator, pointer, m_allocation);
      } else if constexpr (std::is_same_v<T, VkImage>) {
        vmaDestroyImage(m_allocator, pointer, m_allocation);
      } else {
        // do nothing
      }
    }
  }

  VmaAllocator m_allocator = {};
  VmaAllocation m_allocation = {};
  std::size_t m_size = {};
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

// GPUSceneData
struct GPUSceneData {
  glm::vec4 fogColor;
  glm::vec4 fogDistances;
  glm::vec4 ambientColor;
  glm::vec4 sunlightDirection;
  glm::vec4 sunlightColor;
};

// 顶点
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

struct GPUCameraData {
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 viewProj;
};

struct GPUObjectData {
  glm::mat4 modelMatrix;
};

// 现在假设所有mesh中的indexType全部为16位。如果不是会不加载。打log;
using IndexType = uint16_t;

// Mesh
struct Mesh {

  template <typename... ComponentTypeList>
  using VariantSpan = std::variant<std::span<ComponentTypeList>...>;

  using IndexVarSpanType = VariantSpan<glm::uint8_t,   // 5121
                                       glm::uint16_t,  // 5123
                                       glm::uint32_t>; // 5125
  using IndexSpanType = std::span<IndexType>;

  using PositionType = glm::tvec3<float>;
  using NormalType = glm::tvec3<float>;
  struct SubMesh {
    // 小端系统可以直接取值使用。大端需要做转换，主要是因为vulkan使用的小端。
    // gltf小端。这是直接把buffer里的对象解释为c++对象。所以大端解释其中标量的含义
    // 时需要转换 不过不需要解释标量含义，则可以直接传给gpu使用。

    // 位置
    std::span<PositionType> positions;

    // 法向
    std::span<NormalType> normals;

    // 顶点索引
    IndexSpanType indices;
    static VertexInputDescription getVertexDescription();
  };

  std::vector<SubMesh> subMeshs;

  // uint32_t indexCount =0;
  // uint32_t vertexCount=0;

  // bool loadFromOBJ(std::string const &path) {

  //   tinygltf::Model model;
  //   tinyobj::attrib_t attrib;

  //   std::vector<tinyobj::shape_t> shapes;

  //   std::vector<tinyobj::material_t> materials;

  //   std::string warn;
  //   std::string err;

  //   bool re =
  //       tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path.c_str());

  //   if (!err.empty()) {
  //     std::cerr << "loadObject error " << err << '\n';
  //   }
  //   if (!re) {
  //     return false;
  //   }

  //   for (auto &shape : shapes) {

  //     // Loop over faces(polygon)
  //     size_t index_offset = 0;
  //     for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
  //       size_t fv = size_t(shape.mesh.num_face_vertices[f]);

  //       // Loop over vertices in the face.
  //       for (size_t v = 0; v < fv; v++) {
  //         // access to vertex
  //         tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

  //         Vertex vertex{};
  //         tinyobj::real_t vx =
  //             attrib.vertices[3 * size_t(idx.vertex_index) + 0];
  //         tinyobj::real_t vy =
  //             attrib.vertices[3 * size_t(idx.vertex_index) + 1];
  //         tinyobj::real_t vz =
  //             attrib.vertices[3 * size_t(idx.vertex_index) + 2];

  //         vertex.position = glm::vec3(vx, vy, vz);
  //         // Check if `normal_index` is zero or positive. negative = no
  //         normal
  //         // data
  //         if (idx.normal_index >= 0) {
  //           tinyobj::real_t nx =
  //               attrib.normals[3 * size_t(idx.normal_index) + 0];
  //           tinyobj::real_t ny =
  //               attrib.normals[3 * size_t(idx.normal_index) + 1];
  //           tinyobj::real_t nz =
  //               attrib.normals[3 * size_t(idx.normal_index) + 2];
  //           vertex.normal = glm::vec3(nx, ny, nz);
  //         }

  //         // Check if `texcoord_index` is zero or positive. negative = no
  //         // texcoord data
  //         if (idx.texcoord_index >= 0) {
  //           tinyobj::real_t tx =
  //               attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
  //           tinyobj::real_t ty =
  //               attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
  //         }

  //         vertex.color = vertex.normal;

  //         // Optional: vertex colors
  //         // tinyobj::real_t red   =
  //         // attrib.colors[3*size_t(idx.vertex_index)+0]; tinyobj::real_t
  //         green
  //         // = attrib.colors[3*size_t(idx.vertex_index)+1]; tinyobj::real_t
  //         blue
  //         // = attrib.colors[3*size_t(idx.vertex_index)+2];
  //         vertices.push_back(vertex);
  //       }
  //       index_offset += fv;
  //     }
  //   }

  //   return true;
  // }
};

// VertexBuffer Struct
struct VertexBuffer {
  VulkanBufferHandle indexBuffer;
  std::vector<VulkanBufferHandle> buffers;
  VertexInputDescription inputDescription;
};

class VulkanMemory {
public: // Inteface
  [[nodiscard]] VulkanBufferHandle
  createBuffer(VkBufferCreateInfo &createInfo,
               VmaAllocationCreateInfo const &allocationInfo) {
    VkBuffer buffer = {};
    VmaAllocation allocation = {};

    VkResult re = vmaCreateBuffer(
        m_allocator.get(), &static_cast<VkBufferCreateInfo const &>(createInfo),
        &allocationInfo, &buffer, &allocation, nullptr);

    if (re != VK_SUCCESS) {
      App::ThrowException(fmt::format("create Buffer Error {}", re));
    }

    return VulkanBufferHandle(
        buffer,
        VmaDeleter<VkBuffer>{m_allocator.get(), allocation, createInfo.size});
  }

  [[nodiscard]] VulkanImageHandle
  createImage(VkImageCreateInfo &createInfo,
              VmaAllocationCreateInfo const &allocationInfo) const {
    VkImage image = {};
    VmaAllocation allocation = {};

    VkResult re = vmaCreateImage(
        m_allocator.get(), &static_cast<VkImageCreateInfo const &>(createInfo),
        &allocationInfo, &image, &allocation, nullptr);

    if (re != VK_SUCCESS) {
      App::ThrowException(fmt::format("create Image Error {}", re));
    }

    return VulkanImageHandle(
        image, VmaDeleter<VkImage>{m_allocator.get(), allocation});
  }

  // uploadMesh 后续可以改成模板
  // 创建device memory 同时提交到device memory
  // void uploadMesh(Mesh &mesh) {

  //   std::span buffer(mesh.vertices);
  //   vk::BufferCreateInfo bufferInfo = {};
  //   bufferInfo.setSize(buffer.size_bytes());
  //   bufferInfo.setUsage(vk::BufferUsageFlagBits::eVertexBuffer);
  //   VmaAllocationCreateInfo allocationInfo = {};
  //   allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  //   allocationInfo.flags = VmaAllocationCreateFlagBits::
  //       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  //   mesh.vertexBuffer = createBuffer(
  //       static_cast<VkBufferCreateInfo &>(bufferInfo), allocationInfo);
  //   // upload(mesh.vertexBuffer, buffer);
  // }

  // 现在假设所有mesh中的indexType全部为16位。如果不是会不加载。打log;
  template <ranges::view T>
    requires std::same_as<ranges::range_value_t<T>, Mesh>
  VertexBuffer uploadMeshes(T meshes) {

    // 用于绑定vertexBuffer
    std::vector<Mesh::IndexSpanType> indices;
    vk::DeviceSize indexBufferSize = 0;
    std::vector<std::span<Mesh::PositionType>> positions;
    vk::DeviceSize positionBufferSize = 0;
    std::vector<std::span<Mesh::NormalType>> normals;
    vk::DeviceSize normalBufferSize = 0;

    // 获取mesh大小。
    vk::DeviceSize meshSize = 0;
    for (auto &mesh : meshes) {
      for (auto &subMesh : mesh.subMeshs) {
        indexBufferSize += subMesh.indices.size_bytes();
        indices.push_back(subMesh.indices);
        positionBufferSize += subMesh.positions.size_bytes();
        positions.push_back(subMesh.positions);
        normalBufferSize += subMesh.normals.size_bytes();
        normals.push_back(subMesh.normals);
      }
    }
    meshSize = indexBufferSize + positionBufferSize + normalBufferSize;

    VertexBuffer vertexBuffer;
    vertexBuffer.inputDescription = Mesh::SubMesh::getVertexDescription();
    // 创建buffer
    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.flags = VmaAllocationCreateFlagBits::
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    vk::BufferCreateInfo indexBufferInfo = {
        .size = indexBufferSize,
        .usage = vk::BufferUsageFlagBits::eIndexBuffer};
    auto indexBuffer = createBuffer(indexBufferInfo, allocationInfo);
    vertexBuffer.indexBuffer = std::move(indexBuffer);

    upload(indexBuffer, ranges::views::all(indices));

    vk::BufferCreateInfo positionBufferInfo = {
        .size = positionBufferSize,
        .usage = vk::BufferUsageFlagBits::eVertexBuffer};
    auto positionBuffer = createBuffer(indexBufferInfo, allocationInfo);
    upload(positionBuffer, ranges::views::all(positions));
    vertexBuffer.buffers.push_back(std::move(indexBuffer));

    vk::BufferCreateInfo normalBufferInfo = {
        .size = positionBufferSize,
        .usage = vk::BufferUsageFlagBits::eVertexBuffer};
    auto normalBuffer = createBuffer(normalBufferInfo, allocationInfo);
    upload(normalBuffer, ranges::views::all(normals));
    vertexBuffer.buffers.push_back(std::move(normalBuffer));

    return vertexBuffer;
  }

  // maybe change element_type as std::span<byte>
  //  upload to gpu memory
  template <typename View>
    requires std::same_as<
        ranges::range_value_t<View>,
        std::span<typename ranges::range_value_t<View>::element_type>>
  void upload(VulkanBufferHandle const &handle, View buffers) {

    auto deleter = handle.get_deleter();
    auto *alloction = deleter.m_allocation;
    auto size = deleter.m_size;

    void *data = nullptr;

    App::VulkanCheck(vmaMapMemory(m_allocator.get(), alloction, &data),
                     "map error");

    for (auto &buffer : buffers) {
      std::memcpy(data, buffer.data(), buffer.size_bytes());
      auto *nextAddr = static_cast<unsigned char *>(data);
      nextAddr += buffer.size_bytes();
      data = nextAddr;
    }

    vmaUnmapMemory(m_allocator.get(), alloction);
  }

  std::vector<raii::DescriptorSet> createDescriptorSet(
      const vk::ArrayProxyNoTemporaries<const vk::DescriptorSetLayout>
          &setLayouts) {
    vk::DescriptorSetAllocateInfo setAllocInfo{};

    setAllocInfo.setDescriptorPool(*m_descriptorPool);
    setAllocInfo.setSetLayouts(setLayouts);

    return m_pDevice->allocateDescriptorSets(setAllocInfo);
  }

  auto createDescriptorSetLayout(auto &&...args) {
    return m_pDevice->createDescriptorSetLayout(
        std::forward<decltype(args)>(args)...);
  }
  auto updateDescriptorSets(auto &&...args) {
    return m_pDevice->updateDescriptorSets(
        std::forward<decltype(args)>(args)...);
  }

  explicit VulkanMemory(VmaAllocatorCreateInfo const &createInfo,
                        raii::Device *device)
      : m_allocator(createAllocator(createInfo)), m_pDevice(device),
        m_descriptorPool(createDescriptorPool()) {}

  VulkanMemory() = default;

  void clear() {
    m_descriptorPool.clear();
    m_allocator.reset();
  }

  struct VmaAllocatorDeleter {
    using pointer = VmaAllocator;

    void operator()(pointer allocator) noexcept {
      if (allocator != nullptr) {
        vmaDestroyAllocator(allocator);
      }
    }
  };
  using VmaAllocatorHandle = std::unique_ptr<VmaAllocator, VmaAllocatorDeleter>;

private:
  VmaAllocatorHandle m_allocator{nullptr};

  raii::Device *m_pDevice;

  raii::DescriptorPool m_descriptorPool{nullptr};

  raii::DescriptorPool createDescriptorPool() {

    auto poolSizes = std::to_array<vk::DescriptorPoolSize>(
        {{vk::DescriptorType::eUniformBuffer, 10},
         {vk::DescriptorType::eUniformBufferDynamic, 10},
         {vk::DescriptorType::eStorageBuffer, 10}});
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.setPoolSizes(poolSizes);
    poolInfo.setMaxSets(10);
    poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    auto pool = m_pDevice->createDescriptorPool(poolInfo);
    return pool;
  }

  static VmaAllocatorHandle
  createAllocator(VmaAllocatorCreateInfo const &info) {
    VmaAllocator allocator = nullptr;

    auto result = vmaCreateAllocator(&info, &allocator);
    VulkanCheck(result, "create allocator error");

    return VmaAllocatorHandle(allocator);
  }
};

// 资源管理职责负责加载各种资源,管理各种资源。
class AssetManager {
public:
  // void fieldChanged(Mesh &source, const string &fieldName) override {

  // }

  tinygltf::Model &getScene(const std::string &sceneKey) {
    if (m_modelMap.contains(sceneKey)) {
      return m_modelMap[sceneKey];
    } else {
      m_modelMap.insert({sceneKey, loadScene(sceneKey)});
      return m_modelMap[sceneKey];
    }
  }

  // 单例
  static AssetManager &instance() {
    static AssetManager manager;
    return manager;
  }

  // std::unordered_map<key, std::vector<Buffer>> &BufferMap() {
  //   return m_bufferMap;
  // }

private:
  tinygltf::Model loadScene(const std::string &sceneKey) {

    using std::filesystem::path;
    std::filesystem::path basePath{"asset/"};

    auto scenePath = basePath / sceneKey;

    tinygltf::Model model;
    std::string err;
    std::string warn;

    bool ret = m_loader.LoadASCIIFromFile(&model, &err, &warn, scenePath);

    if (!ret) {
      std::clog << "load scene error" << err << "\n";
    }
    std::clog << "load scene warn" << warn << "\n";

    return model;
  }

  tinygltf::TinyGLTF m_loader;
  // std::vector<Buffer> m_buffers;
  std::unordered_map<key, tinygltf::Model> m_modelMap;
};

// class Scene {
// public:
//   void play() {}
// };

// template <typename T>
// concept IScene = requires(T t) {
//                    requires App::IsAnyOf<T, Scene>;

//                    // play主要是用来表示整个场景需要动起来。
//                    { t.play() } -> std::same_as<void>;
//                  };

namespace VulkanInitializer {
vk::ImageCreateInfo getImageCreateInfo(vk::Format format,
                                       vk::ImageUsageFlags usage,
                                       vk::Extent3D const &extent);
vk::ImageViewCreateInfo getImageViewCreateInfo(vk::Format format,
                                               vk::Image image,
                                               vk::ImageAspectFlags aspect);

// vk::DescriptorSetLayoutBinding
// getDescriptorSetLayoutBinding(vk::DescriptorType type,
//                               vk::ShaderStageFlags stageFlag, uint32_t
//                               binding);

// vk::WriteDescriptorSet getWriteDescriptorSet(
//     vk::DescriptorType type, vk::DescriptorSet dstSet,
//     vk::ArrayProxyNoTemporaries<vk::DescriptorBufferInfo> bufferInfos,
//     uint32_t binding);
} // namespace VulkanInitializer

} // namespace App
