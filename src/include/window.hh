#pragma once
#include <cstddef>
#include <cstdint>
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <fstream>
#include <glm/glm.hpp>
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

#include <QPlatformSurfaceEvent>
#include <QVulkanInstance>
#include <QWindow>

namespace raii = vk::raii;

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

//顶点
struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static vk::VertexInputBindingDescription getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex};
    return bindingDescription;
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = ::vk::Format::eR32G32Sfloat;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = ::vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    return attributeDescriptions;
  }
};

class VulkanWindow {
public:
  VulkanWindow(std::string const &path):m_shaderDirPath(path){

  }

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

  ~VulkanWindow() { spdlog::info("in Vulkan Window destructor"); }

private:
  void initWindow();

  void createInstance();
  void createSyncObjects();

  void recordCommandBuffer(const raii::CommandBuffer &commandBuffer,
                           uint32_t imageIndex);

  void createCommandBuffers();

  void createCommandPool();

  void createVertexBuffer();

  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties, raii::Buffer &buffer,
                    raii::DeviceMemory &bufferMemory);

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
  // raii::Buffer m_stagingBuffer{nullptr};
  // raii::DeviceMemory m_stagingBufferMemory{nullptr};

  std::vector<vk::Image> m_swapChainImages;
  vk::Format m_swapChainImageFormat;
  vk::Extent2D m_swapChainExtent;
  std::vector<raii::ImageView> m_swapChainImageViews;
  raii::PipelineLayout m_pipelineLayout{nullptr};
  raii::RenderPass m_renderPass{nullptr};
  raii::Pipeline m_graphicsPipeline{nullptr};
  std::vector<raii::Framebuffer> m_swapChainFramebuffers;
  raii::CommandPool m_commandPool{nullptr};

  std::vector<raii::CommandBuffer> m_commandBuffers;
  std::vector<raii::Semaphore> m_imageAvailableSemaphores;
  std::vector<raii::Semaphore> m_renderFinishedSemaphores;
  std::vector<raii::Fence> m_inFlightFences;

  const std::vector<Vertex> m_vertices = {{{0.0F, -0.5F}, {1.0F, 1.0F, 1.0F}},
                                          {{0.5F, 0.5F}, {0.0F, 1.0F, 0.0F}},
                                          {{-0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}}};

  uint32_t m_currentFrame = 0;
  QWindow *m_window{nullptr};
  const uint32_t m_WIDTH = 800;
  const uint32_t m_HEIGHT = 600;
  const std::vector<const char *> m_validationLayers = {
      "VK_LAYER_KHRONOS_validation"};


  const std::string m_shaderDirPath;

  std::vector<std::string> m_instanceExtensions;

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
  VulkanGameWindow(QVulkanInstance *qVulkanInstance,std::string const&path)
      : m_qVulkanInstance(qVulkanInstance),
        m_vulkanWindow(new VulkanWindow(path)) {
    QWindow::setSurfaceType(QSurface::VulkanSurface);
  }

  void exposeEvent(QExposeEvent *) override {
    spdlog::info("exposeEvent");
    if (isExposed()) {
      if (!m_initialized) {
        m_initialized = true;
        init();
        m_vulkanWindow->drawFrame();
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
        requestUpdate();
      } else if (e->type() == QEvent::PlatformSurface) {

        auto nowEvent = dynamic_cast<QPlatformSurfaceEvent *>(e);

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
  bool m_initialized = false;
};
