#include <window.hh>

// void VulkanWindow::run() {
//   initWindow();
//   initVulkan();
//   mainLoop();
//   cleanup();
// }

// void VulkanWindow::initWindow() {
//   glfwInit();
//   glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//   glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
//   m_window = glfwCreateWindow(m_WIDTH, m_HEIGHT, "Vulkan", nullptr, nullptr);
// }
using std::runtime_error;

void VulkanWindow::initVulkanOther(const VkSurfaceKHR &surface) {
  setupDebugMessenger();
  createSurface(surface);
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();
  createCommandPool();
  createVertexBuffer();
  createIndexBuffer();
  createCommandBuffers();
  createSyncObjects();
}

void VulkanWindow::createInstance() {

  if (m_enableValidationLayers && !checkValidationLayerSupport()) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  vk::ApplicationInfo appInfo{.pApplicationName = "Hello Triangle",
                              .applicationVersion = VK_MAKE_VERSION(1, 1, 0),
                              .pEngineName = "No Engine",
                              .engineVersion = VK_MAKE_VERSION(1, 1, 0),
                              .apiVersion = VK_API_VERSION_1_1};

  vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo};

  std::vector<const char *> extensions;
  extensions.reserve(m_instanceExtensions.size());

  for (auto &deviceExtensions : m_instanceExtensions) {
    extensions.push_back(deviceExtensions.c_str());
  }

  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

  if (m_enableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(m_validationLayers.size());
    createInfo.ppEnabledLayerNames = m_validationLayers.data();

    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = reinterpret_cast<vk::DebugUtilsMessengerCreateInfoEXT *>(
        &debugCreateInfo);
  } else {
    createInfo.enabledLayerCount = 0;
    createInfo.pNext = nullptr;
  }

  m_instance = raii::Instance(m_context, createInfo);
}

void VulkanWindow::populateDebugMessengerCreateInfo(
    vk::DebugUtilsMessengerCreateInfoEXT &createInfo) {

  createInfo.setMessageSeverity(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning);

  createInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
  createInfo.setPfnUserCallback(debugCallback);
}

void VulkanWindow::setupDebugMessenger() {

  // vk::DispatchLoaderDynamic dldy;
  // dldy.init(*m_instance);
  if (!m_enableValidationLayers) {
    return;
  }

  vk::DebugUtilsMessengerCreateInfoEXT createInfo;

  populateDebugMessengerCreateInfo(createInfo);

  m_debugMessenger = m_instance.createDebugUtilsMessengerEXT(createInfo);
}

void VulkanWindow::createSurface(const VkSurfaceKHR &surface) {

  m_surface = surface;
}

void VulkanWindow::pickPhysicalDevice() {
  auto devices = m_instance.enumeratePhysicalDevices();
  // m_instance.enumeratePhysicalDevices();

  if (devices.empty()) {
    throw runtime_error("failed to find GPUs with Vulkan support!");
  }

  for (auto &device : devices) {
    if (isDeviceSuitable(*device)) {

      m_physicalDevice = std::move(device);
      break;
    }
  }

  if ((*m_physicalDevice) == vk::PhysicalDevice(nullptr)) {
    throw runtime_error("failed to find a suitable GPU!");
  }
}

void VulkanWindow::createLogicalDevice() {
  QueueFamilyIndices indices = findQueueFamilies(*m_physicalDevice);

  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                            indices.presentFamily.value()};
  auto queuePriority = 1.0F;

  for (uint32_t queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCreateInfo{.queueFamilyIndex = queueFamily,
                                              .queueCount = 1,
                                              .pQueuePriorities =
                                                  &queuePriority};
    queueCreateInfos.push_back(queueCreateInfo);
  }

  vk::PhysicalDeviceFeatures deviceFeatures{};

  vk::DeviceCreateInfo createInfo{};
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pEnabledFeatures = &deviceFeatures;
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(m_deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = m_deviceExtensions.data();

  if (m_enableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(m_validationLayers.size());
    createInfo.ppEnabledLayerNames = m_validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  m_device = m_physicalDevice.createDevice(createInfo);

  m_graphicsQueue = m_device.getQueue(indices.graphicsFamily.value(), 0);
  m_presentQueue = m_device.getQueue(indices.presentFamily.value(), 0);
}

void VulkanWindow::createSwapChain() {

  SwapChainSupportDetails swapChainSupport =
      querySwapChainSupport(*m_physicalDevice);

  auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
  auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
  auto extent = chooseSwapExtent(swapChainSupport.capabilities);

  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  vk::SwapchainCreateInfoKHR createInfo{};
  createInfo.surface = m_surface;
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

  QueueFamilyIndices indices = findQueueFamilies(*m_physicalDevice);
  uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                   indices.presentFamily.value()};

  if (indices.graphicsFamily != indices.presentFamily) {
    createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    createInfo.queueFamilyIndexCount = 0;     // Optional
    createInfo.pQueueFamilyIndices = nullptr; // Optional
  }
  createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
  createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = *m_swapChain;

  m_swapChain = m_device.createSwapchainKHR(createInfo);
  m_swapChainImages = (*m_device).getSwapchainImagesKHR(*m_swapChain);

  m_swapChainImageFormat = surfaceFormat.format;
  m_swapChainExtent = extent;
}

void VulkanWindow::createImageViews() {

  m_swapChainImageViews.clear();
  m_swapChainImageViews.reserve(m_swapChainImages.size());
  for (std::size_t i = 0; i < m_swapChainImages.size(); i++) {
    vk::ImageViewCreateInfo createInfo{};
    createInfo.image = m_swapChainImages[i];
    createInfo.viewType = vk::ImageViewType::e2D;
    createInfo.format = m_swapChainImageFormat;
    createInfo.components.r = vk::ComponentSwizzle::eIdentity;
    createInfo.components.g = vk::ComponentSwizzle::eIdentity;
    createInfo.components.b = vk::ComponentSwizzle::eIdentity;
    createInfo.components.a = vk::ComponentSwizzle::eIdentity;
    createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;
    m_swapChainImageViews.emplace_back(m_device.createImageView(createInfo));
  }
}

void VulkanWindow::createRenderPass() {
  vk::AttachmentDescription colorAttachment{};
  colorAttachment.format = m_swapChainImageFormat;
  colorAttachment.samples = vk::SampleCountFlagBits::e1;
  colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
  colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
  colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
  colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

  vk::AttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

  vk::SubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

  vk::SubpassDescription subpass{};
  subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;

  vk::RenderPassCreateInfo renderPassInfo{};
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  m_renderPass = m_device.createRenderPass(renderPassInfo);
}

void VulkanWindow::createGraphicsPipeline() {
  std::string homePath = m_shaderDirPath;

  auto vertShaderCode = readFile(homePath + "/vert.spv");
  auto fragShaderCode = readFile(homePath + "/frag.spv");
  auto vertShaderModule = createShaderModule(vertShaderCode);
  auto fragShaderModule = createShaderModule(fragShaderCode);

  vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
  vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
  vertShaderStageInfo.module = *vertShaderModule;
  vertShaderStageInfo.pName = "main";

  vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
  fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
  fragShaderStageInfo.module = *fragShaderModule;
  fragShaderStageInfo.pName = "main";

  vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescription = Vertex::getAttributeDescriptions();
  vertexInputInfo.setVertexBindingDescriptions(bindingDescription);
  vertexInputInfo.setVertexAttributeDescriptions(attributeDescription);

  vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  vk::Viewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)m_swapChainExtent.width;
  viewport.height = (float)m_swapChainExtent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  vk::Rect2D scissor{.offset = {0, 0}, .extent = m_swapChainExtent};
  vk::PipelineViewportStateCreateInfo viewportState{};
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  vk::PipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = vk::PolygonMode::eFill;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = vk::CullModeFlagBits::eBack;
  rasterizer.frontFace = vk::FrontFace::eClockwise;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f; // Optional
  rasterizer.depthBiasClamp = 0.0f;          // Optional
  rasterizer.depthBiasSlopeFactor = 0.0f;    // Optional

  vk::PipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
  multisampling.minSampleShading = 1.0f;          // Optional
  multisampling.pSampleMask = nullptr;            // Optional
  multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
  multisampling.alphaToOneEnable = VK_FALSE;      // Optional

  vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
  colorBlendAttachment.blendEnable = VK_FALSE;
  colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eOne; // Optional
  colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eZero;
  colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
  colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne; // Optional
  colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eOne; // Optional
  colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;            // Optional

  vk::PipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = vk::LogicOp::eCopy; // Optional
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f; // Optional
  colorBlending.blendConstants[1] = 0.0f; // Optional
  colorBlending.blendConstants[2] = 0.0f; // Optional
  colorBlending.blendConstants[3] = 0.0f; // Optional

  std::vector<vk::DynamicState> dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eLineWidth,
  };

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};

  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutInfo);

  vk::GraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = nullptr; // Optional
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = nullptr; // Optional
  pipelineInfo.layout = *m_pipelineLayout;

  pipelineInfo.renderPass = *m_renderPass;
  pipelineInfo.subpass = 0;

  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
  pipelineInfo.basePipelineIndex = -1;

  m_graphicsPipeline = m_device.createGraphicsPipeline(nullptr, pipelineInfo);
}

void VulkanWindow::createFramebuffers() {
  m_swapChainFramebuffers.clear();
  m_swapChainFramebuffers.reserve(m_swapChainImageViews.size());

  for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
    vk::ImageView attachments[] = {*m_swapChainImageViews[i]};

    vk::FramebufferCreateInfo framebufferInfo{};
    framebufferInfo.renderPass = *m_renderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = m_swapChainExtent.width;
    framebufferInfo.height = m_swapChainExtent.height;
    framebufferInfo.layers = 1;

    m_swapChainFramebuffers.emplace_back(
        m_device.createFramebuffer(framebufferInfo));
  }
}

void VulkanWindow::createCommandPool() {
  QueueFamilyIndices queueFamilyIndices = findQueueFamilies(*m_physicalDevice);

  vk::CommandPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

  m_commandPool = m_device.createCommandPool(poolInfo);
}

void VulkanWindow::createCommandBuffers() {

  m_commandBuffers.clear();

  m_commandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
  auto bufferSize = MAX_FRAMES_IN_FLIGHT;
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = *m_commandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = static_cast<uint32_t>(bufferSize);

  m_commandBuffers = m_device.allocateCommandBuffers(allocInfo);
}

void VulkanWindow::createSyncObjects() {

  m_imageAvailableSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
  m_renderFinishedSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
  m_inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

  vk::SemaphoreCreateInfo semaphoreInfo{};
  vk::FenceCreateInfo fenceInfo{};
  fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    m_imageAvailableSemaphores.emplace_back(
        m_device.createSemaphore(semaphoreInfo));
    m_renderFinishedSemaphores.emplace_back(
        m_device.createSemaphore(semaphoreInfo));
    m_inFlightFences.emplace_back(m_device.createFence(fenceInfo));
  }
}

void VulkanWindow::drawFrame() {

  auto seconds = static_cast<uint64_t>(10e9);

  vk::FenceGetFdInfoKHR getInfo{};

  auto result = m_device.waitForFences(*m_inFlightFences[m_currentFrame],
                                       VK_TRUE, seconds);

  if (result == vk::Result::eTimeout) {
    spdlog::warn(" wait fences time out");
  }

  m_device.resetFences(*m_inFlightFences[m_currentFrame]);

  vk::AcquireNextImageInfoKHR acquireInfo{};

  // acquireInfo.swapchain = *m_swapChain;
  // acquireInfo.timeout = seconds;
  // acquireInfo.semaphore = *m_imageAvailableSemaphores[m_currentFrame];
  // acquireInfo.deviceMask = UINT32_MAX;

  auto [acquireResult, imageIndex] = (*m_device).acquireNextImageKHR(
      *m_swapChain, seconds, *m_imageAvailableSemaphores[m_currentFrame]);

  if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
    recreateSwapChain();
    return;
  } else if (acquireResult != vk::Result::eSuccess &&
             acquireResult != vk::Result::eSuboptimalKHR) {
    spdlog::error("failed to acquire swap chain image");
  }

  m_commandBuffers[m_currentFrame].reset();

  recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

  vk::SubmitInfo submitInfo{};

  vk::Semaphore waitSemaphores[] = {
      *m_imageAvailableSemaphores[m_currentFrame]};
  vk::PipelineStageFlags waitStages[] = {
      vk::PipelineStageFlagBits::eColorAttachmentOutput};
  vk::CommandBuffer commandBuffers[] = {*m_commandBuffers[m_currentFrame]};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = commandBuffers;

  vk::Semaphore signalSemaphores[] = {
      *m_renderFinishedSemaphores[m_currentFrame]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  m_graphicsQueue.submit(submitInfo, *m_inFlightFences[m_currentFrame]);

  vk::PresentInfoKHR presentInfo{};
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;
  vk::SwapchainKHR swapChains[] = {*m_swapChain};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &imageIndex;

  presentInfo.pResults = nullptr;

  try {
    auto presentQueueResult = m_presentQueue.presentKHR(presentInfo);

  } catch (const std::system_error &system) {
    auto code = system.code();
    auto errorCode = static_cast<vk::Result>(code.value());
    if (errorCode == vk::Result::eErrorOutOfDateKHR ||
        errorCode == vk::Result::eSuboptimalKHR) {
      recreateSwapChain();
    } else if (errorCode != vk::Result::eSuccess) {
      spdlog::error("failed to acquire swap chain image");
    }

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }
}
bool VulkanWindow::isDeviceSuitable(const vk::PhysicalDevice &device) {
  auto deviceProperties = device.getProperties();

  spdlog::info("{0},{1},{2}", deviceProperties.deviceName,
               deviceProperties.vendorID, deviceProperties.deviceID);

  QueueFamilyIndices indices = findQueueFamilies(device);

  bool extensionsSupported = checkDeviceExtensionSupport(device);

  bool swapChainAdequate = false;
  if (extensionsSupported) {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
    swapChainAdequate = !swapChainSupport.formats.empty() &&
                        !swapChainSupport.presentModes.empty();
  }

  return indices.isComplete() && extensionsSupported && swapChainAdequate &&
         (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ||
          deviceProperties.deviceType ==
              vk::PhysicalDeviceType::eIntegratedGpu);
}

bool VulkanWindow::checkDeviceExtensionSupport(
    const vk::PhysicalDevice &device) {

  auto availableExtensions = device.enumerateDeviceExtensionProperties();
  std::set<std::string> requiredExtensions(m_deviceExtensions.begin(),
                                           m_deviceExtensions.end());

  for (const auto &extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }

  return requiredExtensions.empty();
}

bool VulkanWindow::checkValidationLayerSupport() {

  auto availableLayers = vk::enumerateInstanceLayerProperties();
  for (const char *layerName : m_validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }
  return true;
}

// std::vector<const char *> VulkanWindow::getRequiredExtensions() {
//   uint32_t glfwExtensionCount = 0;
//   const char **glfwExtensions;
//   glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

//   std::vector<const char *> extensions(glfwExtensions,
//                                        glfwExtensions + glfwExtensionCount);

//   if (m_enableValidationLayers) {
//     extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
//   }

//   return extensions;
// }

void VulkanWindow::recordCommandBuffer(const raii::CommandBuffer &commandBuffer,
                                       uint32_t imageIndex) {
  vk::CommandBufferBeginInfo beginInfo{};

  // beginInfo.flags =0  ;                  // Optional
  beginInfo.pInheritanceInfo = nullptr; // Optional

  commandBuffer.begin(beginInfo);

  vk::RenderPassBeginInfo renderPassInfo{};
  renderPassInfo.renderPass = *m_renderPass;
  renderPassInfo.framebuffer = *m_swapChainFramebuffers[imageIndex];
  renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
  renderPassInfo.renderArea.extent = m_swapChainExtent;

  vk::ArrayWrapper1D<float, 4> array{{0.0f, 0.0f, 0.0f, 1.0f}};

  vk::ClearValue clearColor{.color = {std::move(array)}};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearColor;

  commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             *m_graphicsPipeline);

  vk::Buffer vertexBuffers[] = {*m_vertexBuffer};
  vk::DeviceSize offsets[] = {0};
  commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

  commandBuffer.bindIndexBuffer(*m_indexBuffer, 0, vk::IndexType::eUint16);

  commandBuffer.drawIndexed(static_cast<uint32_t>(m_indices.size()), 1, 0, 0, 0);

//  commandBuffer.draw(static_cast<uint32_t>(m_vertices.size()), 1, 0, 0);



  commandBuffer.endRenderPass();

  commandBuffer.end();
}

QueueFamilyIndices
VulkanWindow::findQueueFamilies(const vk::PhysicalDevice &device) {

  QueueFamilyIndices indices;

  auto queueFamilies = device.getQueueFamilyProperties();

  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.graphicsFamily = i;
    }

    auto presentSupport = device.getSurfaceSupportKHR(i, m_surface);

    spdlog::info("after get queue family{}", i);
    if (presentSupport) {
      indices.presentFamily = i;
    }
    i++;
  }
  return indices;
}

SwapChainSupportDetails
VulkanWindow::querySwapChainSupport(const vk::PhysicalDevice &device) {

  SwapChainSupportDetails details;
  details.capabilities = device.getSurfaceCapabilitiesKHR(m_surface);
  details.formats = device.getSurfaceFormatsKHR(m_surface);
  details.presentModes = device.getSurfacePresentModesKHR(m_surface);
  return details;
}

vk::SurfaceFormatKHR VulkanWindow::chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
  for (const auto &availableFormat : availableFormats) {
    if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
        availableFormat.colorSpace ==
            vk::ColorSpaceKHR::eVkColorspaceSrgbNonlinear) {
      return availableFormat;
    }
  }
  return availableFormats[0];
}

vk::PresentModeKHR VulkanWindow::chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  for (const auto &availablePresentMode : availablePresentModes) {
    if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
      return availablePresentMode;
    }
  }
  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D
VulkanWindow::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;

    height = m_window->height();
    width = m_window->width();
    vk::Extent2D actualExtent = {static_cast<uint32_t>(width),
                                 static_cast<uint32_t>(height)};

    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

    return actualExtent;
  }
}

raii::ShaderModule
VulkanWindow::createShaderModule(const std::vector<char> &code) {
  vk::ShaderModuleCreateInfo createInfo{};
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
  auto shaderModule = m_device.createShaderModule(createInfo);
  return shaderModule;
}

void VulkanWindow::cleanup() {

  m_imageAvailableSemaphores.clear();
  m_renderFinishedSemaphores.clear();
  m_inFlightFences.clear();

  m_commandBuffers.clear();

  m_commandPool.clear();
  m_swapChainFramebuffers.clear();
  m_graphicsPipeline.clear();
  m_pipelineLayout.clear();
  m_renderPass.clear();
  m_swapChainImageViews.clear();
  m_swapChain.clear();
  // m_device.clear();
  // m_debugMessenger.clear();
}

void VulkanWindow::createVertexBuffer() {
  auto size =
      static_cast<vk::DeviceSize>(sizeof(m_vertices[0]) * m_vertices.size());

  raii::Buffer stagingBuffer{nullptr};
  raii::DeviceMemory stagingBufferMemory{nullptr};

  createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  //填充buffer数据
  auto data = (*m_device).mapMemory(*stagingBufferMemory, 0, size);
  std::memcpy(data, m_vertices.data(), static_cast<size_t>(size));
  (*m_device).unmapMemory(*stagingBufferMemory);

  createBuffer(size,
               vk::BufferUsageFlagBits::eTransferDst |
                   vk::BufferUsageFlagBits::eVertexBuffer,
               vk::MemoryPropertyFlagBits::eDeviceLocal, m_vertexBuffer,
               m_vertexBufferMemory);

  copyBuffer(*stagingBuffer, *m_vertexBuffer, size);
}

uint32_t
VulkanWindow::findMemoryType(uint32_t typeFilter,
                             const vk::MemoryPropertyFlags &properties) {
  auto memProperties = m_physicalDevice.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (i << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  return 0;
  spdlog::error("failed to find suitable memory type!");
}

void VulkanWindow::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                vk::MemoryPropertyFlags properties,
                                raii::Buffer &buffer,
                                raii::DeviceMemory &bufferMemory) {
  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = vk::SharingMode::eExclusive;

  buffer = m_device.createBuffer(bufferInfo);

  //获取buffer 需要的类型;
  vk::DeviceBufferMemoryRequirements info{};
  info.setPCreateInfo(&bufferInfo);
  auto memRequirements = (*m_device).getBufferMemoryRequirements(*buffer);

  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.setAllocationSize(memRequirements.size);
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  //分配设备内存
  bufferMemory = m_device.allocateMemory(allocInfo);

  vk::BindBufferMemoryInfo bindInfo{};

  bindInfo.buffer = *buffer;
  bindInfo.memory = *bufferMemory;
  bindInfo.setMemoryOffset(0);

  m_device.bindBufferMemory2(bindInfo);
}

void VulkanWindow::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                              vk::DeviceSize size) {
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandPool = *m_commandPool;
  allocInfo.commandBufferCount = 1;
  auto commandBuffers = m_device.allocateCommandBuffers(allocInfo);

  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

  commandBuffers[0].begin(beginInfo);
  vk::BufferCopy copyRegion{};
  copyRegion.size = size;
  commandBuffers[0].copyBuffer(srcBuffer, dstBuffer, copyRegion);
  commandBuffers[0].end();

  vk::SubmitInfo submitInfo{};
  submitInfo.setCommandBuffers(*commandBuffers[0]);
  m_graphicsQueue.submit(submitInfo);
  m_graphicsQueue.waitIdle();
}

void VulkanWindow::createIndexBuffer() {
  auto size =
      static_cast<vk::DeviceSize>(sizeof(m_indices[0]) * m_indices.size());

  raii::Buffer stagingBuffer{nullptr};
  raii::DeviceMemory stagingBufferMemory{nullptr};

  createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  //填充buffer数据
  auto *data = (*m_device).mapMemory(*stagingBufferMemory, 0, size);
  std::memcpy(data, m_indices.data(), static_cast<size_t>(size));
  (*m_device).unmapMemory(*stagingBufferMemory);

  createBuffer(size,
               vk::BufferUsageFlagBits::eTransferDst |
                   vk::BufferUsageFlagBits::eIndexBuffer,
               vk::MemoryPropertyFlagBits::eDeviceLocal, m_indexBuffer,
               m_indexBufferMemory);

  copyBuffer(*stagingBuffer, *m_indexBuffer, size);
}
