#include "asset.hh"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#include "tiny_gltf.h"
// void App::Texture::LoadImageFromFileToTexture(std::string const& path) {
//   int texWidth = 0;
//   int texHeight = 0;
//   int texChannels = 0;

//   auto *pixels =
//       stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels,
//       STBI_rgb_alpha);

//   std::unique_ptr<stbi_uc,
//                   decltype([](stbi_uc *stbi) { stbi_image_free(stbi); })>
//       raiiPixels{pixels};

//   auto imageSize = static_cast<vk::DeviceSize>(texWidth * texHeight) * 4;

//   // texWidth = 540;
//   // texHeight = 960;

//   // auto imageSize = m_softRender.getFrameBuffer().size() * sizeof(Pixel);

//   // auto const *pixelTest = m_softRender.getFrameBuffer().data();

//   raii::Buffer stagingBuffer{nullptr};
//   raii::DeviceMemory stagingBufferMemory{nullptr};

//   createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
//                vk::MemoryPropertyFlagBits::eHostVisible |
//                    vk::MemoryPropertyFlagBits::eHostCoherent,
//                stagingBuffer, stagingBufferMemory);

//   auto *data = (*m_device).mapMemory(*stagingBufferMemory, 0, imageSize);

//   std::memcpy(data, raiiPixels.get(), static_cast<std::size_t>(imageSize));
//   //std::memcpy(data, pixelTest, static_cast<std::size_t>(imageSize));

//   (*m_device).unmapMemory(*stagingBufferMemory);

//   createImage(static_cast<uint32_t>(texWidth),
//   static_cast<uint32_t>(texHeight),
//               vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
//               vk::ImageUsageFlagBits::eTransferDst |
//                   vk::ImageUsageFlagBits::eSampled,
//               vk::MemoryPropertyFlagBits::eDeviceLocal, m_textureImage,
//               m_textureImageMemory);

//   transitionImageLayout(*m_textureImage, vk::Format::eR8G8B8A8Srgb,
//                         vk::ImageLayout::eUndefined,
//                         vk::ImageLayout::eTransferDstOptimal);

//   copyBufferToImage(*stagingBuffer, *m_textureImage,
//                     static_cast<uint32_t>(texWidth),
//                     static_cast<uint32_t>(texHeight));

//   transitionImageLayout(*m_textureImage, vk::Format::eR8G8B8A8Srgb,
//                         vk::ImageLayout::eTransferDstOptimal,
//                         vk::ImageLayout::eShaderReadOnlyOptimal);

// }

namespace App {
VertexInputDescription Mesh::SubMesh::getVertexDescription() {
  vk::VertexInputBindingDescription positionBinding{
      .binding = 0,
      .stride = sizeof(PositionType),
      .inputRate = vk::VertexInputRate::eVertex};

  vk::VertexInputBindingDescription normalBinding{
      .binding = 1,
      .stride = sizeof(NormalType),
      .inputRate = vk::VertexInputRate::eVertex};

  vk::VertexInputBindingDescription texCoordBinding{
      .binding = 2,
      .stride = sizeof(TextureCoordinate),
      .inputRate = vk::VertexInputRate::eVertex};


  vk::VertexInputAttributeDescription positionAttr{
      .location = 0,
      .binding = 0,
      .format = vk::Format::eR32G32B32Sfloat,
      .offset = 0};
  vk::VertexInputAttributeDescription normalAttr{
      .location = 1,
      .binding = 1,
      .format = vk::Format::eR32G32B32Sfloat,
      .offset = 0};

  vk::VertexInputAttributeDescription texCoordAttr{
      .location = 2,
      .binding = 2,
      .format = vk::Format::eR32G32Sfloat,
      .offset = 0};

  VertexInputDescription des{{positionBinding, normalBinding,texCoordBinding},
                             {positionAttr, normalAttr,texCoordAttr}};

  return des;
}

namespace VulkanInitializer {

vk::ImageCreateInfo getImageCreateInfo(
    vk::Format format, vk::ImageUsageFlags usage, vk::Extent3D const &extent) {
  vk::ImageCreateInfo info{};
  info.setFormat(format);
  info.setUsage(usage);
  info.setExtent(extent);
  info.setImageType(vk::ImageType::e2D);
  info.setMipLevels(1);
  info.setArrayLayers(1);
  info.setSamples(vk::SampleCountFlagBits::e1);
  info.setTiling(vk::ImageTiling::eOptimal);
  return info;
}

vk::ImageViewCreateInfo
getImageViewCreateInfo(vk::Format format, vk::Image image,
                                          vk::ImageAspectFlags aspect) {
  vk::ImageViewCreateInfo info{};

  info.setImage(image);
  info.setFormat(format);
  info.subresourceRange.setAspectMask(aspect);
  info.setViewType(vk::ImageViewType::e2D);
  info.subresourceRange.setBaseMipLevel(0);
  info.subresourceRange.setLevelCount(1);
  info.subresourceRange.setBaseArrayLayer(0);
  info.subresourceRange.setLayerCount(1);

  return info;
}

// [[deprecated("don't use create directly")]]
// vk::DescriptorSetLayoutBinding getDescriptorSetLayoutBinding(
//     vk::DescriptorType type, vk::ShaderStageFlags stageFlag, uint32_t
//     binding) {
//   vk::DescriptorSetLayoutBinding setBind{};
//   setBind.setDescriptorType(type);
//   setBind.descriptorCount = 1;
//   setBind.setBinding(binding);
//   setBind.setStageFlags(stageFlag);
//   return setBind;
// }

// [[deprecated("don't use")]]
// vk::WriteDescriptorSet getWriteDescriptorSet(
//     vk::DescriptorType type, vk::DescriptorSet dstSet,
//     vk::ArrayProxyNoTemporaries<vk::DescriptorBufferInfo> bufferInfos,
//     uint32_t binding) {
//   vk::WriteDescriptorSet write{};

//   write.setDstBinding(binding);
//   write.setDstSet(dstSet);
//   write.setDescriptorCount(1);
//   write.setDescriptorType(type);
//   write.setBufferInfo(bufferInfos);

//   return write;
// }

} // namespace VulkanInitializer

} // namespace App
