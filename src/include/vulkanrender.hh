#pragma once
#include "asset.hh"
#include "frame.hh"
#include "pipeline.hh"
#include "render_target.hh"
#include "scene.hh"
#include "tool.hh"
#include "vulkan_info.hh"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <fstream>
#include <future>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <iterator>
#include <new>
#include <optional>
#include <ratio>
#include <set>
#include <stb/stb_image.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace raii = vk::raii;
namespace chrono = std::chrono;

constexpr uint MAX_FRAMES_IN_FLIGHT = 2;

// struct UniformBufferObject {
//   glm::mat4 model;
//   glm::mat4 view;
//   glm::mat4 proj;
// };

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
  explicit VulkanRender(std::string path, App::Extent2D renderSize,
                        std::string appName, uint32_t appVersion,
                        std::string engineName, uint32_t engineVersion)
      : m_programRootPath(std::move(path)), m_renderSize(renderSize),
        m_appName(std::move(appName)), m_appVersion(appVersion),
        m_engineName(std::move(engineName)), m_engineVersion(engineVersion) {}

  VulkanRender(VulkanRender &&) noexcept = default;
  VulkanRender(VulkanRender const &) = delete;
  VulkanRender &operator=(const VulkanRender &other) = delete;
  VulkanRender &operator=(VulkanRender &&other) = delete;

  void initVulkanInstance(std::vector<std::string> &&extensions) {

    m_instanceExtensions = extensions;
    createInstance();
  }

  void initOthers(const VkSurfaceKHR &surface);
  VkInstance getVulkanInstance() { return *m_instance; }
  App::VulkanMemory *getVulkanMemory() { return m_vulkanMemory.get(); }
  App::PipelineFactory *getPipelineFactory() { return m_pipelineFactory.get(); }

  void waitDrawClean();
  void cleanup();
  void drawFrame(App::GPUScene const &scene);

  std::optional<chrono::duration<float, std::milli>> getPerFrameTime() {
    return m_perFrameTime;
  }

  ~VulkanRender() noexcept;

private:
  void initVulkan(const VkSurfaceKHR &surface);

  void initMemory();

  void initFrameDatas();

  void createInstance();

  // void drawObjects(uint32_t frameIndex);

  void recordCommandBuffer(
      vk::CommandBuffer commandBuffer, uint32_t imageIndex,
      App::GPUScene::NormalFunctions const &normalFuns,
      std::function<void(vk::CommandBuffer commandBuffer)> const &recordFun);

  // void createFrameDatas();

  // void loadMeshs();

  // void createGraphicsPipeline();

  void initPipelineFactory();

  void createRenderTarget();

  void createSurface(const VkSurfaceKHR &surface);

  std::unique_ptr<App::VulkanMemory> createVulkanMemory();
  void createLogicalDevice();

  void pickPhysicalDevice();

  [[nodiscard]] std::size_t
  getPadUniformBufferOffsetSize(std::size_t originSize) const;
  bool isDeviceSuitable(const vk::PhysicalDevice &device);

  bool checkDeviceExtensionSupport(const vk::PhysicalDevice &device);

  uint32_t findMemoryType(uint32_t typeFilter,
                          const vk::MemoryPropertyFlags &properties);

  static void populateDebugMessengerCreateInfo(
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

  raii::Context m_context;
  raii::Instance m_instance{nullptr};

  // raii::DebugUtilsMessengerEXT m_debugMessenger{nullptr};

  raii::SurfaceKHR m_surface{nullptr};
  raii::PhysicalDevice m_physicalDevice{nullptr};

  // to share
  std::unique_ptr<raii::Device> m_pDevice{nullptr};
  vk::PhysicalDeviceProperties m_gpuProperties{};

  App::QueueFamilyIndices m_queueFamilyIndices;
  std::unique_ptr<raii::Queue> m_graphicsQueue{nullptr};
  std::unique_ptr<raii::Queue> m_presentQueue{nullptr};
  std::unique_ptr<raii::Queue> m_transferQueue{nullptr};

  // App::GPUSceneData m_sceneParameters;
  // App::VulkanBufferHandle m_sceneParaBuffer{nullptr};
  // App::Mesh m_mesh{};
  // App::Mesh m_monkeyMesh{};

  std::unique_ptr<App::RenderTarget> m_renderTarget{nullptr};

  // raii::DescriptorSetLayout m_descriptorSetlayout{nullptr};
  // raii::DescriptorSetLayout m_objectSetLayout{nullptr};

  // raii::PipelineLayout m_pipelineLayout{nullptr};
  // raii::Pipeline m_graphicsPipeline{nullptr};
  std::unique_ptr<App::PipelineFactory> m_pipelineFactory{nullptr};

  std::unique_ptr<App::VulkanMemory> m_vulkanMemory;
  std::vector<App::Frame> m_frames;

  const std::vector<uint16_t> m_indices = {0, 1, 2, 2, 3, 0};
  uint32_t m_currentFrame = 0;
  uint32_t m_frameCount = 0;

  std::vector<std::string> m_instanceExtensions;

  std::optional<chrono::duration<float, std::milli>> m_perFrameTime;

  App::Extent2D m_renderSize;

  std::string m_appName;
  uint32_t m_appVersion;
  std::string m_engineName;
  uint32_t m_engineVersion;

  uint32_t m_vulkanApiVersion = VK_API_VERSION_1_3;

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
  const std::vector<const char *> m_deviceExtensions{"VK_KHR_swapchain",
                                                     "VK_KHR_synchronization2"};
};
