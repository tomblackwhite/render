#pragma once
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <gsl/gsl>
#include <ratio>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <stb/stb_image.h>

#define VULKAN_HPP_NO_CONSTRUCTORS
#define GLM_FORCE_RADIANS
#include <chrono>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <QMainWindow>
#include <QPlatformSurfaceEvent>
#include <QVulkanInstance>
#include <QWindow>

namespace raii = vk::raii;
namespace chrono = std::chrono;

const int MAX_FRAMES_IN_FLIGHT = 2;

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
  bool isComplete() {
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

//顶点
struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex};
    return bindingDescription;
  }

  static std::array<vk::VertexInputAttributeDescription, 3>
  getAttributeDescriptions() {
    std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = ::vk::Format::eR32G32Sfloat;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = ::vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = ::vk::Format::eR32G32Sfloat;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
  }
};

// deleter
struct CommandBufferDeleter {

  const vk::Queue &m_queue;

  explicit CommandBufferDeleter(vk::Queue const &queue) : m_queue(queue) {}

  using pointer = gsl::owner<raii::CommandBuffer *>;
  void operator()(gsl::owner<raii::CommandBuffer *> point) {

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

class VulkanWindow {
public:
  explicit VulkanWindow(std::string const &path) : m_shaderDirPath(path) {}

  void initInstance(std::vector<std::string> &&extensions) {

    m_instanceExtensions = extensions;
    createInstance();
  }

  //
  void initWindow(QWindow *window) { m_window = window; }

  void initVulkanOther(const VkSurfaceKHR &surface);
  VkInstance getVulkanInstance() { return *m_instance; }
  void waitDrawClean() { m_device.waitIdle(); };
  void cleanup();
  void drawFrame();

  void resize() { recreateSwapChain(); }

  std::optional<chrono::duration<float, std::milli>> getPerFrameTime() {
    return m_perFrameTime;
  }

  ~VulkanWindow() { spdlog::info("in Vulkan Window destructor"); }

private:
  void updateUniformBuffer(uint32_t currentImage);

  void initWindow();

  void createInstance();
  void createSyncObjects();

  void recordCommandBuffer(const raii::CommandBuffer &commandBuffer,
                           uint32_t imageIndex);

  void createCommandBuffers();

  void createCommandPool();

  void createVertexBuffer();

  void createIndexBuffer();

  void createUniformBuffers();

  void createTextureImage();

  void createTextureSampler();

  //创建commandBuffer记录命令
  CommandBufferPointer beginSingleTimeCommands();

  void transitionImageLayout(vk::Image image, vk::Format format,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout);

  void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width,
                         uint32_t height);

  void createTextureImageView();

  raii::ImageView createImageView(vk::Image, vk::Format format);

  void createDescriptorPool();

  void createDescriptorSets();

  void createDescriptorSetLayout();

  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties, raii::Buffer &buffer,
                    raii::DeviceMemory &bufferMemory);

  void createImage(uint32_t width, uint32_t height, vk::Format format,
                   vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                   vk::MemoryPropertyFlagBits properties, raii::Image &image,
                   raii::DeviceMemory &imageMemory);

  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                  vk::DeviceSize size);

  void createFramebuffers();

  void createRenderPass();

  void createGraphicsPipeline();

  raii::ShaderModule createShaderModule(const std::vector<char> &code);

  void createImageViews();

  void createSwapChain();

  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<vk::SurfaceFormatKHR> &availableFormats);

  vk::PresentModeKHR chooseSwapPresentMode(
      const std::vector<vk::PresentModeKHR> &availablePresentModes);

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities);
  void createSurface(const VkSurfaceKHR &surface);

  void createLogicalDevice();

  void pickPhysicalDevice();

  bool isDeviceSuitable(const vk::PhysicalDevice &device);

  bool checkDeviceExtensionSupport(const vk::PhysicalDevice &device);

  //寻找当前设备支持的队列列表 图形队列列表和presentFamily
  QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice &device);

  SwapChainSupportDetails
  querySwapChainSupport(const vk::PhysicalDevice &device);

  uint32_t findMemoryType(uint32_t typeFilter,
                          const vk::MemoryPropertyFlags &properties);

  // void mainLoop() {
  //   drawFrame();
  //   m_device.waitIdle();
  // }

  // std::vector<const char *> getRequiredExtensions();

  void recreateSwapChain() {
    m_device.waitIdle();
    createSwapChain();
    createImageViews();
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
      throw std::runtime_error("failed to open file!");
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

  raii::DebugUtilsMessengerEXT m_debugMessenger{nullptr};

  vk::SurfaceKHR m_surface{nullptr};
  raii::PhysicalDevice m_physicalDevice{nullptr};
  raii::Device m_device{nullptr};
  raii::Queue m_graphicsQueue{nullptr};
  raii::Queue m_presentQueue{nullptr};
  raii::SwapchainKHR m_swapChain{nullptr};

  raii::Buffer m_vertexBuffer{nullptr};
  raii::DeviceMemory m_vertexBufferMemory{nullptr};
  raii::Buffer m_indexBuffer{nullptr};
  raii::DeviceMemory m_indexBufferMemory{nullptr};

  std::vector<raii::Buffer> m_uniformBuffers;
  std::vector<raii::DeviceMemory> m_uniformBuffersMemory;

  raii::Image m_textureImage{nullptr};
  raii::DeviceMemory m_textureImageMemory{nullptr};
  raii::ImageView m_textureImageView{nullptr};
  raii::Sampler m_textureSampler{nullptr};

  std::vector<vk::Image> m_swapChainImages;
  vk::Format m_swapChainImageFormat;
  vk::Extent2D m_swapChainExtent;
  std::vector<raii::ImageView> m_swapChainImageViews;
  raii::PipelineLayout m_pipelineLayout{nullptr};
  raii::RenderPass m_renderPass{nullptr};

  raii::DescriptorSetLayout m_descriptorSetLayout{nullptr};
  raii::DescriptorPool m_descriptorPool{nullptr};
  raii::DescriptorSets m_descriptorSets{nullptr};

  raii::Pipeline m_graphicsPipeline{nullptr};
  std::vector<raii::Framebuffer> m_swapChainFramebuffers;
  raii::CommandPool m_commandPool{nullptr};

  std::vector<raii::CommandBuffer> m_commandBuffers;
  std::vector<raii::Semaphore> m_imageAvailableSemaphores;
  std::vector<raii::Semaphore> m_renderFinishedSemaphores;
  std::vector<raii::Fence> m_inFlightFences;

  const std::vector<Vertex> m_vertices = {
      {{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
      {{1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
      {{1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
      {{-1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}};

  const std::vector<uint16_t> m_indices = {0, 1, 2, 2, 3, 0};
  uint32_t m_currentFrame = 0;
  QWindow *m_window{nullptr};
  const uint32_t m_WIDTH = 800;
  const uint32_t m_HEIGHT = 450;
  const std::vector<const char *> m_validationLayers = {
      "VK_LAYER_KHRONOS_validation"};

  const std::string m_shaderDirPath;

  std::vector<std::string> m_instanceExtensions;

  std::optional<chrono::duration<float, std::milli>> m_perFrameTime;

  const std::vector<const char *> m_deviceExtensions{"VK_KHR_swapchain"};
#ifdef NDEBUG
  const bool m_enableValidationLayers = false;
#else
  const bool m_enableValidationLayers = true;
#endif
};

/*游戏窗口主要显示的window*/
class VulkanGameWindow : public QWindow {

public:
  VulkanGameWindow(QVulkanInstance *qVulkanInstance, std::string const &path,
                   QMainWindow *mainWindow)
      : m_qVulkanInstance(qVulkanInstance),
        m_vulkanWindow(new VulkanWindow(path)), m_mainWindow(mainWindow) {

    QWindow::setSurfaceType(QSurface::VulkanSurface);
  }

  void exposeEvent(QExposeEvent *) override {
    spdlog::info("exposeEvent");
    if (isExposed()) {
      if (!m_initialized) {
        m_initialized = true;
        init();
        m_vulkanWindow->drawFrame();

        auto optionTime = m_vulkanWindow->getPerFrameTime();
        if (optionTime.has_value()) {
          auto fps = 1000.0F / optionTime.value().count();
          auto resultTitle = fmt::format("测试，当前帧数：{:.2f}，帧生成时间：{:.2f}",
                                         fps, optionTime.value().count());
          m_mainWindow->setWindowTitle(QString::fromStdString(resultTitle));
        }


        requestUpdate();
      }
    }
  }

  void resizeEvent(QResizeEvent *ev) override {
    spdlog::info("resize");
    if (m_initialized) {
      m_vulkanWindow->resize();
    }
  }

  bool event(QEvent *e) override {
    // spdlog::info("inEvent {}", e->type());

    try {

      if (e->type() == QEvent::UpdateRequest) {

        m_vulkanWindow->drawFrame();

        auto optionTime = m_vulkanWindow->getPerFrameTime();
        if (optionTime.has_value()) {
          auto fps = 1000.0F / optionTime.value().count();
          auto resultTitle = fmt::format("测试，当前帧数：{:.1f}，帧生成时间：{:.1f}",
                                         fps, optionTime.value().count());
          m_mainWindow->setWindowTitle(QString::fromStdString(resultTitle));
        }

        requestUpdate();

      } else if (e->type() == QEvent::PlatformSurface) {

        auto *nowEvent = dynamic_cast<QPlatformSurfaceEvent *>(e);

        //删除surface 时清理和surface 相关的内容
        if (nowEvent->surfaceEventType() ==
            QPlatformSurfaceEvent::SurfaceEventType::
                SurfaceAboutToBeDestroyed) {
          m_vulkanWindow->waitDrawClean();
          m_vulkanWindow->cleanup();
          //删除suface 才能删除vulkaninstance
          // m_vulkanWindow.reset();
          // auto nowPointer = m_vulkanWindow.release();
        }
      } else {
        // do nothing
      }
    } catch (const std::exception &e) {
      spdlog::error(e.what());
    }

    return QWindow::event(e);
  }
  virtual ~VulkanGameWindow() { spdlog::info("in VulkanGameWindow"); }

private:
  //初始化vulkan 设置相关数据
  void init() {
    auto extensions = m_qVulkanInstance->supportedExtensions();
    std::vector<std::string> stdExtensions;
    stdExtensions.reserve(extensions.size());
    std::transform(
        extensions.constBegin(), extensions.constEnd(),
        std::back_insert_iterator<std::vector<std::string>>(stdExtensions),
        [](QVulkanExtension const &extension) {
          return extension.name.toStdString();
        });
    for (auto &extension : stdExtensions) {
      spdlog::info(extension);
    }
    m_vulkanWindow->initInstance(std::move(stdExtensions));

    auto vulkanInstance = m_vulkanWindow->getVulkanInstance();

    //设置VulkanInstance 以便创建QWindow
    m_qVulkanInstance->setVkInstance(vulkanInstance);
    if (!m_qVulkanInstance->create()) {
      throw "创建qVulkanInstance 失败";
    }
    QWindow::setVulkanInstance(m_qVulkanInstance);
    QWindow::create();

    //获取surface 以便初始化其他部分
    auto surface = QVulkanInstance::surfaceForWindow(this);
    m_vulkanWindow->initVulkanOther(surface);
  }

private:
  QVulkanInstance *m_qVulkanInstance;
  // VulkanWindow * m_vulkanWindow;

  std::unique_ptr<VulkanWindow> m_vulkanWindow;
  QMainWindow *m_mainWindow;
  bool m_initialized = false;
};
