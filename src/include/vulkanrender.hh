#pragma once
#include "asset.hh"
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <future>
#include <new>
#include <ratio>
#include <stb/stb_image.h>
#include <tool.hh>

#include <chrono>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace raii = vk::raii;
namespace chrono = std::chrono;

const int MAX_FRAMES_IN_FLIGHT = 2;

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
  [[nodiscard]] bool isComplete() const {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

struct SwapChainSupportDetails {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

// deleter
struct CommandBufferDeleter {

  const vk::Queue &m_queue;

  explicit CommandBufferDeleter(vk::Queue const &queue) : m_queue(queue) {}

  using pointer = raii::CommandBuffer *;
  void operator()(raii::CommandBuffer *point) {

    point->end();
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(**point);
    m_queue.submit(submitInfo);
    m_queue.waitIdle();
    point->clear();

    delete point;
  }
};

using CommandBufferPointer =
    std::unique_ptr<raii::CommandBuffer, CommandBufferDeleter>;

class VulkanRender {
public:
  explicit VulkanRender(std::string &&path, uint32_t renderWidth,
                        uint32_t renderHeight)
      : m_programRootPath(path), m_renderWidth(renderWidth),
        m_renderHeight(renderHeight) {}

  void initVulkanInstance(std::vector<std::string> &&extensions) {

    m_instanceExtensions = extensions;
    createInstance();
  }

  void initOthers(const VkSurfaceKHR &surface);
  VkInstance getVulkanInstance() { return *m_instance; }
  void waitDrawClean() { m_device.waitIdle(); };
  void cleanup();
  void drawFrame();

  void resize() { recreateSwapChain(); }

  std::optional<chrono::duration<float, std::milli>> getPerFrameTime() {
    return m_perFrameTime;
  }

  ~VulkanRender() {}

private:
  void initVulkan(const VkSurfaceKHR &surface);

  void initSwapChain();

  void initCommands();

  // void updateUniformBuffer(uint32_t currentImage);

  void createInstance();
  void createSyncObjects();

  void recordCommandBuffer(const raii::CommandBuffer &commandBuffer,
                           uint32_t imageIndex);

  void createCommandBuffers();

  void createCommandPool();

  void loadMeshs();

  void createTextureSampler();

  //创建commandBuffer记录命令
  // CommandBufferPointer beginSingleTimeCommands();

  // void transitionImageLayout(vk::Image image, vk::Format format,
  //                            vk::ImageLayout oldLayout,
  //                            vk::ImageLayout newLayout);

  // void createTextureImageView();

  // void createDescriptorPool();

  // void createDescriptorSets();

  // void createDescriptorSetLayout();

  // void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
  //                   vk::MemoryPropertyFlags properties, raii::Buffer &buffer,
  //                   raii::DeviceMemory &bufferMemory);

  // void createImage(uint32_t width, uint32_t height, vk::Format format,
  //                  vk::ImageTiling tiling, vk::ImageUsageFlags usage,
  //                  vk::MemoryPropertyFlagBits properties, raii::Image &image,
  //                  raii::DeviceMemory &imageMemory);

  // void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
  //                 vk::DeviceSize size);

  void createFramebuffers();

  void createRenderPass();

  void createGraphicsPipeline();

  raii::ShaderModule createShaderModule(const std::vector<char> &code);

  void createSwapChainImageViews();

  void createSwapChain();

  void createDepthImageAndView();

  void createSurface(const VkSurfaceKHR &surface);

  void createLogicalDevice();

  void pickPhysicalDevice();

  bool isDeviceSuitable(const vk::PhysicalDevice &device);

  bool checkDeviceExtensionSupport(const vk::PhysicalDevice &device);

  //寻找当前设备支持的队列列表 图形队列列表和presentFamily
  QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice &device);

  // get swapchainInfo
  SwapChainSupportDetails
  querySwapChainSupport(const vk::PhysicalDevice &device);

  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<vk::SurfaceFormatKHR> &availableFormats);

  vk::PresentModeKHR chooseSwapPresentMode(
      const std::vector<vk::PresentModeKHR> &availablePresentModes);

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities);

  uint32_t findMemoryType(uint32_t typeFilter,
                          const vk::MemoryPropertyFlags &properties);

  void recreateSwapChain() {
    m_device.waitIdle();
    createSwapChain();
    createSwapChainImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandBuffers();
  }

  void populateDebugMessengerCreateInfo(
      vk::DebugUtilsMessengerCreateInfoEXT &createInfo);

  void setupDebugMessenger();
  bool checkValidationLayerSupport();

  static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      App::ThrowException("failed to open file!");
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer;
    buffer.reserve(fileSize);
    file.seekg(0);
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(file),
                  std::istreambuf_iterator<char>());
    file.close();
    return buffer;
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
  }

private:
  raii::Context m_context;
  raii::Instance m_instance{nullptr};

  // raii::DebugUtilsMessengerEXT m_debugMessenger{nullptr};

  raii::SurfaceKHR m_surface{nullptr};
  raii::PhysicalDevice m_physicalDevice{nullptr};
  raii::Device m_device{nullptr};
  std::unique_ptr<App::VulkanMemory> m_vulkanMemory{nullptr};

  raii::Queue m_graphicsQueue{nullptr};
  raii::Queue m_presentQueue{nullptr};
  raii::SwapchainKHR m_swapChain{nullptr};

  App::Mesh m_mesh{};
  App::Mesh m_monkeyMesh{};

  std::vector<vk::Image> m_swapChainImages;
  vk::Format m_swapChainImageFormat;
  vk::Extent2D m_swapChainExtent;
  std::vector<raii::ImageView> m_swapChainImageViews;

  vk::Format m_depthFormat;
  App::VulkanImageHandle m_depthImage{};
  raii::ImageView m_depthImageView{nullptr};

  raii::PipelineLayout m_pipelineLayout{nullptr};
  raii::RenderPass m_renderPass{nullptr};

  raii::Pipeline m_graphicsPipeline{nullptr};
  std::vector<raii::Framebuffer> m_swapChainFramebuffers;
  std::vector<raii::CommandPool> m_commandPools;

  std::vector<raii::CommandBuffer> m_commandBuffers;
  std::vector<raii::Semaphore> m_imageAvailableSemaphores;
  std::vector<raii::Semaphore> m_renderFinishedSemaphores;
  std::vector<raii::Fence> m_inFlightFences;

  const std::vector<uint16_t> m_indices = {0, 1, 2, 2, 3, 0};
  uint32_t m_currentFrame = 0;
  uint32_t m_frameCount = 0;

  std::vector<std::string> m_instanceExtensions;

  std::optional<chrono::duration<float, std::milli>> m_perFrameTime;

  // render size property
  uint32_t m_renderWidth;
  uint32_t m_renderHeight;

  // shader path
  const std::string m_programRootPath;

  // render settings
#ifdef NDEBUG
  const bool m_enableValidationLayers = false;
#else
  const bool m_enableValidationLayers = true;
#endif
  const std::vector<const char *> m_validationLayers = {
      "VK_LAYER_KHRONOS_validation"};
  const std::vector<const char *> m_deviceExtensions{"VK_KHR_swapchain"};
};

struct VulkanInitializer {
  static vk::PipelineShaderStageCreateInfo
  getPipelineShaderStageCreateInfo(vk::ShaderStageFlagBits stage,
                                   vk::ShaderModule shaderModule);
  static vk::PipelineVertexInputStateCreateInfo
  getPipelineVertexInputStateCreateInfo(App::VertexInputDescription const &);
  static vk::PipelineInputAssemblyStateCreateInfo
  getPipelineInputAssemblyStateCreateInfo();
  static vk::PipelineRasterizationStateCreateInfo
  getPipelineRasterizationStateCreateInfo();
  static vk::PipelineMultisampleStateCreateInfo
  getPipelineMultisampleStateCreateInfo();
  static vk::PipelineColorBlendAttachmentState
  getPipelineColorBlendAttachmentState();
  static vk::PipelineLayoutCreateInfo getPipelineLayoutCreateInfo();

  static vk::PipelineDepthStencilStateCreateInfo getDepthStencilCreateInfo(bool depthTest,bool depthWrite,vk::CompareOp compareOp);

  static vk::ImageCreateInfo getImageCreateInfo(vk::Format format,
                                                vk::ImageUsageFlags usage,
                                                vk::Extent3D const &extent);
  static vk::ImageViewCreateInfo
  getImageViewCreateInfo(vk::Format format, vk::Image image,
                         vk::ImageAspectFlags aspect);
};

// build pipeLine Factory
class PipelineFactory {
public:
  std::vector<vk::PipelineShaderStageCreateInfo> m_shaderStages;
  vk::PipelineVertexInputStateCreateInfo m_vertexInputInfo;
  vk::PipelineInputAssemblyStateCreateInfo m_inputAssembly;
  vk::PipelineDepthStencilStateCreateInfo m_depthStencilCreateInfo;
  vk::Viewport m_viewPort;
  vk::Rect2D m_scissor;
  vk::PipelineRasterizationStateCreateInfo m_rasterizer;
  vk::PipelineColorBlendAttachmentState m_colorBlendAttachment;
  vk::PipelineMultisampleStateCreateInfo m_multisampling;
  vk::PipelineLayout m_pipelineLayout;

  raii::Pipeline buildPipeline(raii::Device const &device, vk::RenderPass pass);
};
