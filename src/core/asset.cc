#include "asset.hh"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

// void App::Texture::LoadImageFromFileToTexture(std::string const& path) {
//   int texWidth = 0;
//   int texHeight = 0;
//   int texChannels = 0;

//   auto *pixels =
//       stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

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

//   createImage(static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight),
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
