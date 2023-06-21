#pragma once
#include "tool.hh"
#include <cassert>
#include <cmath>
#include <concepts>
#include <numeric>
#include <set>
#include <unordered_set>
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

namespace views = std::views;
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

struct Image {

  vk::Format format;

  vk::Extent3D extent;

  std::span<unsigned char> data;
  VulkanImageHandle image;
};

struct Texture {
  // coordIndex in TexCoord
  using ImageIterator = std::vector<Image>::iterator;

  uint32_t coordIndex = 0;
  vk::Format format;
  vk::Filter magFilter = vk::Filter::eLinear;
  vk::Filter minFilter = vk::Filter::eLinear;

  ImageIterator imageIterator{};
  raii::ImageView imageView{nullptr};
  vk::Sampler sampler{nullptr};
};

struct Material {

  using TextureIterator = std::vector<Texture>::iterator;

  struct PBRMetallicRoughness {

    TextureIterator baseColorTexture{};
    glm::vec4 baseColorFactor{1.0, 1.0, 1.0, 1.0};
    int32_t baseColorCoordIndex = -1;

    TextureIterator metallicRoughnessTexture{};
    int32_t metallicRoughnessCoordIndex = -1;

    float metallicFactor = 1.0;
    float roughnessFactor = 1.0;
  };

  PBRMetallicRoughness pbr;

  raii::DescriptorSet textureSet{nullptr};
};

struct alignas(16) GPUPBR {
  alignas(4 * alignof(glm::float32)) glm::vec4 baseColorFactor;
  glm::float32 metallicFactor;
  glm::float32 roughnessFactor;
  int32_t baseColorCoordIndex;
  int32_t metallicRoughnessCoordIndex;
};

// VertexBuffer Struct
struct GPUMeshBlock {
  VulkanBufferHandle indexBuffer;
  std::vector<VulkanBufferHandle> buffers;
  VulkanBufferHandle texCoordinateBuffer;
  std::vector<vk::DeviceSize> texCoordinateOffets;

  VulkanBufferHandle objectBuffer;
  std::vector<vk::DeviceSize> objectOffsets;

  VulkanBufferHandle materialsBuffer;
  vk::DeviceSize perMaterialPadSize = 0;

  // VertexInputDescription inputDescription;
};
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
  using TextureCoordinate = glm::tvec2<float>;

  struct SubMesh {
    // 小端系统可以直接取值使用。大端需要做转换，主要是因为vulkan使用的小端。
    // gltf小端。这是直接把buffer里的对象解释为c++对象。所以大端解释其中标量的含义
    // 时需要转换 不过不需要解释标量含义，则可以直接传给gpu使用。

    // 位置
    std::span<PositionType> positions;

    // 法向
    std::span<NormalType> normals;

    // 材质坐标
    std::vector<std::span<TextureCoordinate>> texCoords;

    // 材质
    Material *material{nullptr};

    // 顶点索引
    IndexSpanType indices;
    static VertexInputDescription getVertexDescription();
  };

  std::vector<SubMesh> subMeshs;
  // VertexBuffer vertexBuffer;
};

using MeshShowMap = std::map<Mesh *, std::vector<glm::mat4>>;

struct UploadBufferInfo {
  VulkanBufferHandle const *handle;
  std::vector<std::span<std::byte>> const *buffers;
  std::vector<vk::DeviceSize> const *offsets;
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

  // 现在假设所有mesh中的indexType全部为16位。如果不是会不加载。打log;
  template <ranges::input_range T>
    requires std::same_as<ranges::range_value_t<T>, Mesh>
  GPUMeshBlock uploadMeshes(T &&meshes) {

    if (ranges::size(meshes) == 0) {
      return GPUMeshBlock{};
    }
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

    GPUMeshBlock vertexBuffer;
    // 创建buffer
    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
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
    auto positionBuffer = createBuffer(positionBufferInfo, allocationInfo);
    upload(positionBuffer, ranges::views::all(positions));
    vertexBuffer.buffers.push_back(std::move(positionBuffer));

    vk::BufferCreateInfo normalBufferInfo = {
        .size = positionBufferSize,
        .usage = vk::BufferUsageFlagBits::eVertexBuffer};
    auto normalBuffer = createBuffer(normalBufferInfo, allocationInfo);
    upload(normalBuffer, ranges::views::all(normals));
    vertexBuffer.buffers.push_back(std::move(normalBuffer));

    return vertexBuffer;
  }

  void uploadAll(std::invocable auto &fun) {
    fun();
    immediateSubmit();
  }

  // 现在假设所有mesh中的indexType全部为16位。如果不是会不加载。打log;
  // template <ranges::input_range U, ranges::input_range T>
  //   requires std::same_as<ranges::range_value_t<U>, Image *> &&
  //            std::same_as<ranges::range_value_t<T>, Material *>
  // void uploadAll(U &&images, MeshShowMap &meshShowMap,
  //                std::unique_ptr<GPUMeshBlock> &meshBlock, T &materials) {

  //   meshBlock = uploadMeshes(meshShowMap);
  //   std::vector<Image *> imageVector;
  //   imageVector.reserve(images.size());
  //   for (auto *image : images) {
  //     imageVector.emplace_back(image);
  //   }
  //   uploadImages(imageVector);

  //   std::vector<Material *> materialVec;
  //   materialVec.reserve(materials.size());
  //   materialVec.assign(materials.begin(), materials.end());
  //   meshBlock->materialsBuffer = uploadMaterials(materialVec);
  //   meshBlock->perMaterialPadSize = getUniformPadSize(sizeof(GPUPBR));

  //   immediateSubmit();
  // }
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

    vk::BufferCreateInfo info{};
    info.setSize(size);

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

  // 上传到transfer buffer
  template <ranges::view View>
    requires std::same_as<
        ranges::range_value_t<View>,
        std::span<typename ranges::range_value_t<View>::element_type>>
  [[nodiscard("transferBuffer return")]] VulkanBufferHandle
  uploadToTransfer(vk::DeviceSize size, View buffers,
                   std::optional<vk::DeviceSize> elemPadSize = std::nullopt,
                   std::vector<vk::DeviceSize> const *offsets = nullptr) {

    vk::BufferCreateInfo info{};
    info.setSize(size);
    info.setSharingMode(vk::SharingMode::eExclusive);
    info.setUsage(vk::BufferUsageFlagBits::eTransferSrc);

    // 创建 transferBuffer
    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags = VmaAllocationCreateFlagBits::
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    auto transferBuffer = createBuffer(info, allocationInfo);

    auto transferDeleter = transferBuffer.get_deleter();
    auto *transferAllocation = transferDeleter.m_allocation;

    void *data = nullptr;
#ifdef DEBUG
    std::vector<std::byte> bytes(size);
    void *debugData = bytes.data();
    unsigned char *preData = nullptr;
#endif

    App::VulkanCheck(vmaMapMemory(m_allocator.get(), transferAllocation, &data),
                     "map error");
#ifdef DEBUG
    preData = static_cast<unsigned char *>(data);
    auto *lastData = preData + size;
#endif
    if (elemPadSize.has_value()) {

      auto *addr = static_cast<unsigned char *>(data);
      std::size_t offset = 0;
      for (auto buffer : buffers) {
        auto *currentData = addr + offset;

        assert(currentData < lastData);
        std::memcpy(currentData, buffer.data(), buffer.size_bytes());
        offset += elemPadSize.value();
      }
    } else if (offsets != nullptr) {

      std::size_t offsetIndex = 0;
      for (auto buffer : buffers) {
        auto *addr = static_cast<unsigned char *>(data);
        addr += offsets->at(offsetIndex);
        assert(addr < lastData);
        std::memcpy(addr, buffer.data(), buffer.size_bytes());
        ++offsetIndex;
      }
    } else {
      auto *currentAddr = static_cast<unsigned char *>(data);
      for (auto buffer : buffers) {
        assert(currentAddr < lastData);
        std::memcpy(currentAddr, buffer.data(), buffer.size_bytes());
        currentAddr += buffer.size_bytes();
      }
    }
    vmaUnmapMemory(m_allocator.get(), transferAllocation);

    return transferBuffer;
  }

  // 通过transfer buffer上传到对应buffer
  template <typename View>
    requires std::same_as<
        ranges::range_value_t<View>,
        std::span<typename ranges::range_value_t<View>::element_type>>
  void uploadByTransfer(VulkanBufferHandle const &handle, View buffers) {

    auto deleter = handle.get_deleter();
    auto *alloction = deleter.m_allocation;
    auto size = deleter.m_size;
    auto transferBuffer = uploadToTransfer(size, buffers);

    std::vector<VulkanBufferHandle> transferBuffers;
    transferBuffers.emplace_back(std::move(transferBuffer));
    std::vector<vk::Buffer> vkBuffers;
    vkBuffers.emplace_back(handle.get());
    transferBuffersRecord(std::move(transferBuffers),
                          std::make_unique<std::vector<vk::Buffer>>(vkBuffers));
    immediateSubmit();
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
                        raii::Device *device, raii::Queue *transferQueue,
                        raii::Queue *graphicQueue,
                        uint32_t queueTransferFamilyIndex,
                        uint32_t queueGraphicFamilyIndex,
                        bool queueFamilyIndexSame, vk::DeviceSize alignment)
      : m_allocator(createAllocator(createInfo)), m_pDevice(device),
        m_descriptorPool(createDescriptorPool()),
        m_pTransferQueue(transferQueue), m_pGraphicQueue(graphicQueue),
        m_transferQueueFamilyIndex(queueTransferFamilyIndex),
        m_graphicQueueFamilyIndex(queueGraphicFamilyIndex),
        // m_commandPool(createTransferCommandPool(poolInfo)),
        // m_commandBuffer(createCommandBuffer()),
        m_isSameGraphicAndTransferQueue(queueFamilyIndexSame),
        m_minUniformBufferOffsetAlignment(alignment) {
    initCmdInfo();
  }

  VulkanMemory() = default;
  VulkanMemory(VulkanMemory &&) noexcept = default;
  VulkanMemory &operator=(VulkanMemory &&) noexcept = default;
  virtual ~VulkanMemory() noexcept { clear(); }

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

  std::vector<vk::SemaphoreSubmitInfo> signalSemaphoresSubmitInfos{};

  raii::Device *m_pDevice{nullptr};

protected:
  // 第二个引用会在submit中使用，所以需要保持生命周期。不要传递临时变量
  virtual void
  transferImagesRecord(VulkanBufferHandle imageBuffer,
                       std::unique_ptr<std::vector<Image *>> images) = 0;

  virtual void
  transferBuffersRecord(std::vector<VulkanBufferHandle> &&transferBuffers,
                        std::unique_ptr<std::vector<vk::Buffer>> buffers) = 0;
  virtual void transferMeshRecord(GPUMeshBlock &&transferMeshBuffer,
                                  GPUMeshBlock const *meshBuffer) = 0;

  virtual void immediateSubmit() = 0;

  VmaAllocatorHandle m_allocator{nullptr};

  raii::DescriptorPool m_descriptorPool{nullptr};

  raii::Queue *m_pTransferQueue{nullptr};
  raii::CommandPool m_commandPool{nullptr};
  raii::CommandBuffer m_commandBuffer{nullptr};

  raii::Queue *m_pGraphicQueue{nullptr};
  raii::CommandPool m_graphicCommandPool{nullptr};
  raii::CommandBuffer m_graphicCommandBuffer{nullptr};

  uint32_t m_graphicQueueFamilyIndex = 0;
  uint32_t m_transferQueueFamilyIndex = 0;

  bool m_isSameGraphicAndTransferQueue = false;

  vk::DeviceSize m_minUniformBufferOffsetAlignment = 0;

  raii::Semaphore m_transferSemaphore{nullptr};
  raii::Fence m_finishFence{nullptr};
  std::vector<raii::Semaphore> signalSemaphores{};

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

public:
  template <ranges::view View>
    requires std::same_as<
        ranges::range_value_t<View>,
        std::span<typename ranges::range_value_t<View>::element_type>>
  void uploadBuffers(VulkanBufferHandle const &handle, View buffers,
                     std::vector<vk::DeviceSize> const *offsets = nullptr) {
    auto deleter = handle.get_deleter();
    auto *alloction = deleter.m_allocation;
    auto size = deleter.m_size;
    auto transferBuffer =
        uploadToTransfer(size, views::all(buffers), std::nullopt, offsets);

    std::vector<VulkanBufferHandle> transferBuffers;
    transferBuffers.emplace_back(std::move(transferBuffer));
    std::vector<vk::Buffer> vkBuffers;
    vkBuffers.emplace_back(handle.get());
    transferBuffersRecord(std::move(transferBuffers),
                          std::make_unique<std::vector<vk::Buffer>>(vkBuffers));
    // if()
  }

  void uploadImages(std::vector<Image *> &images) {

    vk::DeviceSize imageBufferSize = std::accumulate(
        images.begin(), images.end(), static_cast<vk::DeviceSize>(0),
        [](vk::DeviceSize count, Image *right) {
          return count + right->data.size_bytes();
        });

    auto imageTransferBufferHandle = uploadToTransfer(
        imageBufferSize,
        ranges::views::all(images) |
            views::transform([](Image *image) { return image->data; }));

    vk::ImageCreateInfo imageInfo{};
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setSamples(vk::SampleCountFlagBits::e1);
    imageInfo.setTiling(vk::ImageTiling::eOptimal);
    imageInfo.setUsage(vk::ImageUsageFlagBits::eSampled |
                       vk::ImageUsageFlagBits::eTransferDst);
    imageInfo.setSharingMode(vk::SharingMode::eExclusive);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    VmaAllocationCreateInfo imageAllocationInfo{};
    imageAllocationInfo.usage =
        VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    auto vkImages = std::make_unique<std::vector<Image *>>();
    // std::unique_ptr<std::vector<Image *>> vkImages;
    vkImages->reserve(images.size());
    for (Image *image : images) {
      imageInfo.setFormat(image->format);
      imageInfo.setExtent(image->extent);
      image->image = createImage(imageInfo, imageAllocationInfo);
      vkImages->emplace_back(image);
    }

    transferImagesRecord(std::move(imageTransferBufferHandle),
                         std::move(vkImages));
  }

  [[nodiscard]] VulkanBufferHandle
  uploadMaterials(std::vector<Material *> &materials) {

    std::vector<GPUPBR> gpuMaterials;
    std::vector<std::span<GPUPBR>> gpuMaterialsSpan;
    gpuMaterialsSpan.reserve(materials.size());
    gpuMaterials.reserve(materials.size());
    auto padMaterialSize = getUniformPadSize(sizeof(GPUPBR));
    for (auto *material : materials) {
      auto &pbr = material->pbr;
      gpuMaterials.push_back(
          {.baseColorFactor = pbr.baseColorFactor,
           .metallicFactor = pbr.metallicFactor,
           .roughnessFactor = pbr.roughnessFactor,
           .baseColorCoordIndex = pbr.baseColorCoordIndex,
           .metallicRoughnessCoordIndex = pbr.metallicRoughnessCoordIndex});
      gpuMaterialsSpan.emplace_back(&gpuMaterials.back(), 1);
    }

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    auto materialsPadSize = padMaterialSize * gpuMaterials.size();
    vk::BufferCreateInfo bufferInfo = {
        .size = materialsPadSize,
        .usage = vk::BufferUsageFlagBits::eUniformBuffer |
                 vk::BufferUsageFlagBits::eTransferDst};
    auto materialbuffer = createBuffer(bufferInfo, allocationInfo);
    // auto materialSpan = std::span<GPUPBR>(gpuMaterials);
    auto materialTransferBuffer = uploadToTransfer(
        materialsPadSize, views::all(gpuMaterialsSpan), padMaterialSize);

    std::vector<VulkanBufferHandle> transferBuffers;
    transferBuffers.reserve(1);
    transferBuffers.emplace_back(std::move(materialTransferBuffer));

    std::initializer_list<vk::Buffer> buffers = {materialbuffer.get()};
    transferBuffersRecord(std::move(transferBuffers),
                          std::make_unique<std::vector<vk::Buffer>>(buffers));
    return materialbuffer;
  }

  [[nodiscard]] std::unique_ptr<GPUMeshBlock>
  uploadMeshes(MeshShowMap &meshShowMap) {

    auto gpuMeshBuffer = std::make_unique<GPUMeshBlock>();
    auto &vertexBuffer = *gpuMeshBuffer;
    GPUMeshBlock transferBuffer;
    // 用于绑定vertexBuffer
    std::vector<Mesh::IndexSpanType> indices;
    vk::DeviceSize indexBufferSize = 0;
    std::vector<std::span<Mesh::PositionType>> positions;
    vk::DeviceSize positionBufferSize = 0;
    std::vector<std::span<Mesh::NormalType>> normals;
    vk::DeviceSize normalBufferSize = 0;

    std::vector<std::span<Mesh::TextureCoordinate>> coordinates;
    vk::DeviceSize coordinatesBufferSize = 0;

    std::vector<std::span<glm::mat4>> objects;
    vk::DeviceSize objectBufferSize = 0;
    vk::DeviceSize objectOffset = 0;

    // std::vector<GPUPBR> materials;
    // materials.reserve(meshShowMap.size());
    // auto padMaterialSize = getUniformPadSize(sizeof(GPUPBR));
    // std::vector<vk::DeviceSize> materialOffsets;
    // materialOffsets.reserve(meshShowMap.size());
    // vk::DeviceSize materialOffset=0;

    // 获取mesh大小。
    vk::DeviceSize meshSize = 0;

    for (auto &[mesh, objectMatrice] : meshShowMap) {
      for (Mesh::SubMesh &subMesh : mesh->subMeshs) {
        indexBufferSize += subMesh.indices.size_bytes();
        indices.push_back(subMesh.indices);
        positionBufferSize += subMesh.positions.size_bytes();
        positions.push_back(subMesh.positions);
        normalBufferSize += subMesh.normals.size_bytes();
        normals.push_back(subMesh.normals);

        if (!subMesh.texCoords.empty()) {
          coordinates.push_back(subMesh.texCoords[0]);
          coordinatesBufferSize += subMesh.texCoords[0].size_bytes();
        }
      }

      std::span<glm::mat4> object{objectMatrice};
      // vertexBuffer.objectOffsets.push_back(objectOffset);
      auto objectSize = object.size_bytes();
      // auto padObjectSize = getUniformPadSize(objectSize);
      // objectBufferSize += padObjectSize;
      objectBufferSize += objectSize;
      objects.push_back(object);
    }

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    vk::BufferCreateInfo indexBufferInfo = {
        .size = indexBufferSize,
        .usage = vk::BufferUsageFlagBits::eIndexBuffer |
                 vk::BufferUsageFlagBits::eTransferDst};
    auto indexBuffer = createBuffer(indexBufferInfo, allocationInfo);
    vertexBuffer.indexBuffer = std::move(indexBuffer);
    transferBuffer.indexBuffer =
        uploadToTransfer(indexBufferSize, ranges::views::all(indices));

    vk::BufferCreateInfo positionBufferInfo = {
        .size = positionBufferSize,
        .usage = vk::BufferUsageFlagBits::eVertexBuffer |
                 vk::BufferUsageFlagBits::eTransferDst};
    auto positionBuffer = createBuffer(positionBufferInfo, allocationInfo);
    transferBuffer.buffers.push_back(
        uploadToTransfer(positionBufferSize, ranges::views::all(positions)));
    vertexBuffer.buffers.push_back(std::move(positionBuffer));

    vk::BufferCreateInfo normalBufferInfo = {
        .size = normalBufferSize,
        .usage = vk::BufferUsageFlagBits::eVertexBuffer |
                 vk::BufferUsageFlagBits::eTransferDst};
    auto normalBuffer = createBuffer(normalBufferInfo, allocationInfo);
    transferBuffer.buffers.push_back(
        uploadToTransfer(normalBufferSize, ranges::views::all(normals)));
    vertexBuffer.buffers.push_back(std::move(normalBuffer));

    vk::BufferCreateInfo texCoordBufferInfo = {
        .size = coordinatesBufferSize,
        .usage = vk::BufferUsageFlagBits::eVertexBuffer |
                 vk::BufferUsageFlagBits::eTransferDst};
    auto texCoordBuffer = createBuffer(texCoordBufferInfo, allocationInfo);
    transferBuffer.buffers.push_back(
        uploadToTransfer(coordinatesBufferSize, views::all(coordinates)));
    vertexBuffer.buffers.push_back(std::move(texCoordBuffer));

    vk::BufferCreateInfo objectBufferInfo = {
        .size = objectBufferSize,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                 vk::BufferUsageFlagBits::eTransferDst};
    auto objectBuffer = createBuffer(objectBufferInfo, allocationInfo);
    transferBuffer.objectBuffer =
        uploadToTransfer(objectBufferSize, ranges::views::all(objects));
    vertexBuffer.objectBuffer = std::move(objectBuffer);

    transferMeshRecord(std::move(transferBuffer), gpuMeshBuffer.get());

    return gpuMeshBuffer;
  }

  [[nodiscard]] vk::DeviceSize
  getUniformPadSize(vk::DeviceSize bufferSize) const {
    // M & (~N - 1)
    // 由于对齐大小一定是2的倍数，所以，上式一定等于(M/N)*N,计算出补全空白大小。
    return (bufferSize + m_minUniformBufferOffsetAlignment - 1) &
           ~(m_minUniformBufferOffsetAlignment - 1);
  }

private:
  // 设置command相关
  void initCmdInfo() {
    vk::CommandPoolCreateInfo transferPoolInfo{};
    transferPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    transferPoolInfo.queueFamilyIndex = m_transferQueueFamilyIndex;

    m_commandPool = m_pDevice->createCommandPool(transferPoolInfo);

    vk::CommandBufferAllocateInfo info{};
    info.setCommandPool(*m_commandPool);
    info.setCommandBufferCount(1);
    info.setLevel(vk::CommandBufferLevel::ePrimary);
    auto buffers = m_pDevice->allocateCommandBuffers(info);
    m_commandBuffer = std::move(buffers[0]);

    if (m_transferQueueFamilyIndex != m_graphicQueueFamilyIndex) {
      vk::CommandPoolCreateInfo graphicPoolInfo{};
      graphicPoolInfo.flags =
          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
      graphicPoolInfo.queueFamilyIndex = m_graphicQueueFamilyIndex;

      m_graphicCommandPool = m_pDevice->createCommandPool(graphicPoolInfo);

      vk::CommandBufferAllocateInfo graphicInfo{};
      graphicInfo.setCommandPool(*m_graphicCommandPool);
      graphicInfo.setCommandBufferCount(1);
      graphicInfo.setLevel(vk::CommandBufferLevel::ePrimary);
      auto graphicbuffers = m_pDevice->allocateCommandBuffers(graphicInfo);
      m_graphicCommandBuffer = std::move(graphicbuffers[0]);
    }

    vk::SemaphoreCreateInfo seamphoreInfo{};

    m_transferSemaphore = m_pDevice->createSemaphore(seamphoreInfo);
    vk::FenceCreateInfo fenceInfo{};
    m_finishFence = m_pDevice->createFence(fenceInfo);
  }
};

// 当tansfer family index和 graphic queue family index不相等时
class TransferVulkanMemory : public VulkanMemory {
public:
  using VulkanMemory::VulkanMemory;

protected:
  std::vector<std::move_only_function<void(vk::CommandBuffer) const>>
      m_transferCommands;
  std::vector<std::move_only_function<void(vk::CommandBuffer) const>>
      m_graphicCommands;
  void
  transferImagesRecord(VulkanBufferHandle imageBuffer,
                       std::unique_ptr<std::vector<Image *>> images) override {

    // auto imageBuf = std::move(imageBuffer);

    auto transferCommand = [imageTransferBufferHandle = std::move(imageBuffer),
                            &images = *images,
                            graphicFamilyIndex = m_graphicQueueFamilyIndex,
                            transferFamilyIndex = m_transferQueueFamilyIndex](
                               vk::CommandBuffer commandBuffer) {
      // image transfer
      vk::ImageSubresourceRange range{};
      range.setAspectMask(vk::ImageAspectFlagBits::eColor);
      range.setBaseMipLevel(0);
      range.setLevelCount(1);
      range.setBaseArrayLayer(0);
      range.setLayerCount(1);

      std::vector<vk::ImageMemoryBarrier2> imageTransitionBarriers{};
      imageTransitionBarriers.reserve(images.size());

      // std::vector<vk::BufferImageCopy2> bufferImageCopys{};
      // bufferImageCopys.reserve(images.size());

      for (auto *image : images) {
        vk::ImageMemoryBarrier2 imageTransitionBarrier{};
        imageTransitionBarrier.setSrcStageMask(
            vk::PipelineStageFlagBits2::eNone);
        imageTransitionBarrier.setSrcAccessMask(vk::AccessFlagBits2::eNone);
        imageTransitionBarrier.setDstStageMask(
            vk::PipelineStageFlagBits2::eTransfer);
        imageTransitionBarrier.setDstAccessMask(
            vk::AccessFlagBits2::eTransferWrite);
        imageTransitionBarrier.setOldLayout(vk::ImageLayout::eUndefined);
        imageTransitionBarrier.setNewLayout(
            vk::ImageLayout::eTransferDstOptimal);
        imageTransitionBarrier.setImage(image->image.get());
        imageTransitionBarrier.setSubresourceRange(range);
        imageTransitionBarriers.push_back(imageTransitionBarrier);
      }

      vk::DependencyInfo imageBarrierDencyInfo{};
      imageBarrierDencyInfo.setImageMemoryBarriers(imageTransitionBarriers);
      commandBuffer.pipelineBarrier2(imageBarrierDencyInfo);

      vk::BufferImageCopy2 imageCopy{};
      imageCopy.setBufferRowLength(0);
      imageCopy.setBufferImageHeight(0);
      vk::ImageSubresourceLayers layer{};
      layer.setAspectMask(vk::ImageAspectFlagBits::eColor);
      layer.setMipLevel(0);
      layer.setLayerCount(1);
      layer.setBaseArrayLayer(0);
      imageCopy.setImageSubresource(layer);
      imageCopy.setImageOffset({0, 0, 0});

      vk::CopyBufferToImageInfo2 copyBufferToImageInfo{};
      copyBufferToImageInfo.setSrcBuffer(imageTransferBufferHandle.get());
      copyBufferToImageInfo.setDstImageLayout(
          vk::ImageLayout::eTransferDstOptimal);
      copyBufferToImageInfo.setRegions(imageCopy);

      vk::DeviceSize imageBufferOffset = 0;
      std::vector<vk::ImageMemoryBarrier2> imageTransferBarriers{};

      for (auto *image : images) {
        imageCopy.setBufferOffset(imageBufferOffset);
        imageCopy.setImageExtent(image->extent);
        imageBufferOffset += image->data.size_bytes();
        copyBufferToImageInfo.setDstImage(image->image.get());
        commandBuffer.copyBufferToImage2(copyBufferToImageInfo);
        vk::ImageMemoryBarrier2 imageTransferBarrier{};
        imageTransferBarrier.setSrcAccessMask(
            vk::AccessFlagBits2::eTransferWrite);
        imageTransferBarrier.setSrcStageMask(
            vk::PipelineStageFlagBits2::eTransfer);
        imageTransferBarrier.setSrcQueueFamilyIndex(transferFamilyIndex);
        imageTransferBarrier.setDstQueueFamilyIndex(graphicFamilyIndex);
        imageTransferBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
        imageTransferBarrier.setNewLayout(
            vk::ImageLayout::eShaderReadOnlyOptimal);
        imageTransferBarrier.setImage(image->image.get());
        imageTransferBarrier.setSubresourceRange(range);
        imageTransferBarriers.push_back(imageTransferBarrier);
      }

      vk::DependencyInfo depencyInfo{};
      depencyInfo.setImageMemoryBarriers(imageTransferBarriers);
      commandBuffer.pipelineBarrier2(depencyInfo);
    };

    auto graphicCommand = [images = std::move(images),
                           graphicFamilyIndex = m_graphicQueueFamilyIndex,
                           transferFamilyIndex = m_transferQueueFamilyIndex](
                              vk::CommandBuffer commandBuffer) {
      std::vector<vk::ImageMemoryBarrier2> imageTransferBarriers{};

      vk::ImageSubresourceRange range{};
      range.setAspectMask(vk::ImageAspectFlagBits::eColor);
      range.setBaseMipLevel(0);
      range.setLevelCount(1);
      range.setBaseArrayLayer(0);
      range.setLayerCount(1);

      for (auto *image : *images) {
        vk::ImageMemoryBarrier2 imageTransferBarrier{};
        imageTransferBarrier.setDstAccessMask(vk::AccessFlagBits2::eShaderRead);
        imageTransferBarrier.setDstStageMask(
            vk::PipelineStageFlagBits2::eFragmentShader);
        imageTransferBarrier.setSrcQueueFamilyIndex(transferFamilyIndex);
        imageTransferBarrier.setDstQueueFamilyIndex(graphicFamilyIndex);
        imageTransferBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
        imageTransferBarrier.setNewLayout(
            vk::ImageLayout::eShaderReadOnlyOptimal);
        imageTransferBarrier.setImage(image->image.get());
        imageTransferBarrier.setSubresourceRange(range);
        imageTransferBarriers.push_back(imageTransferBarrier);
      }

      vk::DependencyInfo depencyInfo{};
      depencyInfo.setImageMemoryBarriers(imageTransferBarriers);
      commandBuffer.pipelineBarrier2(depencyInfo);
    };

    m_transferCommands.emplace_back(std::move(transferCommand));
    m_graphicCommands.emplace_back(std::move(graphicCommand));
    // m_transferCommands.emplace_back(transferCommand);
    // m_graphicCommands.emplace_back(graphicCommand);
  }

  void transferMeshRecord(GPUMeshBlock &&transferMeshBuffer,
                          GPUMeshBlock const *meshBuffer) override {

    auto transferCommand = [&vertexBuffer = *meshBuffer,
                            transferBuffer = std::move(transferMeshBuffer),
                            graphicFamilyIndex = m_graphicQueueFamilyIndex,
                            transferFamilyIndex = m_transferQueueFamilyIndex](
                               vk::CommandBuffer commandBuffer) {
      // vertex transfer
      std::vector<vk::BufferMemoryBarrier2> barriers{};

      vk::CopyBufferInfo2 copyInfo2{};
      copyInfo2.setSrcBuffer(transferBuffer.indexBuffer.get());
      copyInfo2.setDstBuffer(vertexBuffer.indexBuffer.get());
      auto indexSize = vertexBuffer.indexBuffer.get_deleter().m_size;
      vk::BufferCopy2 region{};
      region.setSize(indexSize);
      copyInfo2.setRegions(region);
      commandBuffer.copyBuffer2(copyInfo2);

      vk::CopyBufferInfo2 objectInfo2{};
      objectInfo2.setSrcBuffer(transferBuffer.objectBuffer.get());
      objectInfo2.setDstBuffer(vertexBuffer.objectBuffer.get());
      auto objectSize = vertexBuffer.objectBuffer.get_deleter().m_size;
      vk::BufferCopy2 objectRegion{.size = objectSize};
      objectInfo2.setRegions(objectRegion);
      commandBuffer.copyBuffer2(objectInfo2);

      vk::BufferMemoryBarrier2 barrier{};
      barrier.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite);
      barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer);
      barrier.setSrcQueueFamilyIndex(transferFamilyIndex);
      barrier.setDstQueueFamilyIndex(graphicFamilyIndex);
      barrier.buffer = vertexBuffer.indexBuffer.get();
      barrier.setSize(VK_WHOLE_SIZE);
      barriers.push_back(barrier);
      barrier.setBuffer(vertexBuffer.objectBuffer.get());
      barriers.push_back(barrier);

      for (auto i = 0uz; i < vertexBuffer.buffers.size(); ++i) {
        vk::CopyBufferInfo2 info{.srcBuffer = transferBuffer.buffers[i].get(),
                                 .dstBuffer = vertexBuffer.buffers[i].get()};
        auto bufferSize = vertexBuffer.buffers[i].get_deleter().m_size;
        vk::BufferCopy2 regionVertex{.size = bufferSize};
        info.setRegions(regionVertex);
        commandBuffer.copyBuffer2(info);

        vk::BufferMemoryBarrier2 barrier{};
        barrier.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite);
        barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer);
        barrier.setSrcQueueFamilyIndex(transferFamilyIndex);
        barrier.setDstQueueFamilyIndex(graphicFamilyIndex);
        barrier.buffer = vertexBuffer.buffers[i].get();
        barrier.setSize(VK_WHOLE_SIZE);
        barriers.push_back(barrier);
      }

      vk::DependencyInfo depencyInfo{};
      depencyInfo.setBufferMemoryBarriers(barriers);
      commandBuffer.pipelineBarrier2(depencyInfo);
    };

    auto graphicCommand = [&vertexBuffer = *meshBuffer,
                           graphicFamilyIndex = m_graphicQueueFamilyIndex,
                           transferFamilyIndex = m_transferQueueFamilyIndex](
                              vk::CommandBuffer commandBuffer) {
      // memory acquire
      std::vector<vk::BufferMemoryBarrier2> barriers{};
      vk::BufferMemoryBarrier2 barrier{};
      barrier.setDstAccessMask(vk::AccessFlagBits2::eMemoryRead);
      barrier.setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands);
      barrier.setSrcQueueFamilyIndex(transferFamilyIndex);
      barrier.setDstQueueFamilyIndex(graphicFamilyIndex);
      barrier.buffer = vertexBuffer.indexBuffer.get();
      barrier.setSize(VK_WHOLE_SIZE);
      barriers.push_back(barrier);
      barrier.setBuffer(vertexBuffer.objectBuffer.get());
      barriers.push_back(barrier);

      for (auto i = 0uz; i < vertexBuffer.buffers.size(); ++i) {
        vk::BufferMemoryBarrier2 barrier{};
        barrier.setDstAccessMask(vk::AccessFlagBits2::eMemoryRead);
        barrier.setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands);
        barrier.setSrcQueueFamilyIndex(transferFamilyIndex);
        barrier.setDstQueueFamilyIndex(graphicFamilyIndex);
        barrier.buffer = vertexBuffer.buffers[i].get();
        barrier.setSize(VK_WHOLE_SIZE);
        barriers.push_back(barrier);
      }

      vk::DependencyInfo depencyInfo{};
      depencyInfo.setBufferMemoryBarriers(barriers);
      commandBuffer.pipelineBarrier2(depencyInfo);
    };

    m_transferCommands.emplace_back(std::move(transferCommand));
    m_graphicCommands.emplace_back(std::move(graphicCommand));
  }

  void transferBuffersRecord(
      std::vector<VulkanBufferHandle> &&transferBuffers,
      std::unique_ptr<std::vector<vk::Buffer>> buffers) override {

    auto transferCommand = [transferBuffers = std::move(transferBuffers),
                            buffers = *(buffers),
                            transferFamilyIndex = m_transferQueueFamilyIndex,
                            graphicFamilyIndex = m_graphicQueueFamilyIndex](
                               vk::CommandBuffer commandBuffer) {
      std::vector<vk::BufferMemoryBarrier2> barriers{};
      for (auto i = 0uz; i < transferBuffers.size(); ++i) {
        vk::CopyBufferInfo2 info{.srcBuffer = transferBuffers[i].get(),
                                 .dstBuffer = buffers[i]};
        auto bufferSize = transferBuffers[i].get_deleter().m_size;
        vk::BufferCopy2 regionVertex{.size = bufferSize};
        info.setRegions(regionVertex);
        commandBuffer.copyBuffer2(info);

        vk::BufferMemoryBarrier2 barrier{};
        barrier.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite);
        barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer);
        barrier.setSrcQueueFamilyIndex(transferFamilyIndex);
        barrier.setDstQueueFamilyIndex(graphicFamilyIndex);
        barrier.buffer = buffers[i];
        barrier.setSize(VK_WHOLE_SIZE);
        barriers.push_back(barrier);
      }

      vk::DependencyInfo depencyInfo{};
      depencyInfo.setBufferMemoryBarriers(barriers);
      commandBuffer.pipelineBarrier2(depencyInfo);
    };

    auto graphicCommand = [buffers = std::move(buffers),
                           transferFamilyIndex = m_transferQueueFamilyIndex,
                           graphicFamilyIndex = m_graphicQueueFamilyIndex](
                              vk::CommandBuffer commandBuffer) {
      std::vector<vk::BufferMemoryBarrier2> barriers{};
      for (auto &buffer : *buffers) {

        vk::BufferMemoryBarrier2 barrier{};
        barrier.setDstAccessMask(vk::AccessFlagBits2::eMemoryRead);
        barrier.setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands);
        barrier.setSrcQueueFamilyIndex(transferFamilyIndex);
        barrier.setDstQueueFamilyIndex(graphicFamilyIndex);
        barrier.buffer = buffer;
        barrier.setSize(VK_WHOLE_SIZE);
        barriers.push_back(barrier);
      }
      vk::DependencyInfo depencyInfo{};
      depencyInfo.setBufferMemoryBarriers(barriers);
      commandBuffer.pipelineBarrier2(depencyInfo);
    };

    m_transferCommands.emplace_back(std::move(transferCommand));
    m_graphicCommands.emplace_back(std::move(graphicCommand));
  }

  void immediateSubmit() override {
    vk::CommandBufferBeginInfo info{};
    // command buffer submit and reset
    info.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    m_commandBuffer.begin(info);

    auto &transferFunctions = m_transferCommands;

    for (auto const &fun : transferFunctions) {
      fun(*m_commandBuffer);
    }

    m_commandBuffer.end();

    vk::SubmitInfo2 submitInfo{};

    vk::CommandBufferSubmitInfo comandSubmitInfo{.commandBuffer =
                                                     *m_commandBuffer};

    submitInfo.setCommandBufferInfos(comandSubmitInfo);

    vk::SemaphoreSubmitInfo semaphoreInfo{};
    semaphoreInfo.setSemaphore(*m_transferSemaphore);
    semaphoreInfo.setStageMask(vk::PipelineStageFlagBits2::eAllCommands);

    submitInfo.setSignalSemaphoreInfos(semaphoreInfo);

    m_pTransferQueue->submit2(submitInfo);

    vk::CommandBufferBeginInfo graphicInfo{};
    graphicInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    m_graphicCommandBuffer.begin(graphicInfo);
    for (auto &recordGraphicCommand : m_graphicCommands) {
      recordGraphicCommand(*m_graphicCommandBuffer);
    }
    m_graphicCommandBuffer.end();

    vk::SubmitInfo2 graphicSubmitInfo{};
    vk::CommandBufferSubmitInfo graphicComandSubmitInfo{
        .commandBuffer = *m_graphicCommandBuffer};

    graphicSubmitInfo.setCommandBufferInfos(graphicComandSubmitInfo);

    vk::SemaphoreSubmitInfo graphicSemaphoreInfo{};
    graphicSemaphoreInfo.setSemaphore(*m_transferSemaphore);
    graphicSemaphoreInfo.setStageMask(vk::PipelineStageFlagBits2::eAllCommands);
    graphicSubmitInfo.setWaitSemaphoreInfos(semaphoreInfo);
    m_pGraphicQueue->submit2(graphicSubmitInfo, *m_finishFence);

    using namespace std::chrono_literals;
    auto onesecond = std::chrono::nanoseconds(600s);

    auto result =
        m_pDevice->waitForFences(*m_finishFence, VK_TRUE, onesecond.count());
    if (result == vk::Result::eTimeout) {
      App::ThrowException("copy to device memory timeout");
    }
    m_pDevice->resetFences(*m_finishFence);

    (**m_pDevice).resetCommandPool(*m_commandPool);
    (**m_pDevice).resetCommandPool(*m_graphicCommandPool);
    m_transferCommands.clear();
    m_graphicCommands.clear();
  }
  // void transferBufferRecord();
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
  static AssetManager &instance(const fs::path &homePath) {
    static AssetManager manager(homePath);
    return manager;
  }

  // std::unordered_map<key, std::vector<Buffer>> &BufferMap() {
  //   return m_bufferMap;
  // }

private:
  explicit AssetManager(fs::path homePath) : m_homePath(std::move(homePath)) {}

  tinygltf::Model loadScene(const std::string &sceneKey) {

    using std::filesystem::path;
    std::filesystem::path basePath{"asset/"};

    auto scenePath = m_homePath / basePath / sceneKey;

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

  fs::path m_homePath;
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
