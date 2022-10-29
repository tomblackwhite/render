#include <vulkanrender.hh>

using std::runtime_error;

void VulkanRender::initOthers(const VkSurfaceKHR &surface) {
  initVulkan(surface);
  // initSwapChain();
  initCommands();

  loadMeshs();
  createRenderTarget();
  createGraphicsPipeline();
  // createFramebuffers();

  createSyncObjects();
}

void VulkanRender::initVulkan(const VkSurfaceKHR &surface) {
  createSurface(surface);
  pickPhysicalDevice();
  createLogicalDevice();
  VmaAllocatorCreateInfo createInfo{};

  createInfo.vulkanApiVersion = VK_API_VERSION_1_3;
  createInfo.device = *m_device;
  createInfo.physicalDevice = *m_physicalDevice;
  createInfo.instance = *m_instance;
  m_vulkanMemory = std::make_unique<App::VulkanMemory>(createInfo);
}

// void VulkanRender::initSwapChain() {
//   createSwapChain();
//   createSwapChainImageViews();
//   createDepthImageAndView();
// }

void VulkanRender::initCommands() {
  createCommandPool();
  createCommandBuffers();
}

void VulkanRender::createInstance() {

  if (m_enableValidationLayers && !checkValidationLayerSupport()) {
    App::ThrowException("validation layers requested, but not available!");
  }

  vk::ApplicationInfo appInfo{.applicationVersion = VK_MAKE_VERSION(1u, 1u, 0u),
                              .engineVersion = VK_MAKE_VERSION(1u, 1u, 0u),
                              .apiVersion = VK_API_VERSION_1_3};

  vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo};

  std::vector<const char *> extensions(m_instanceExtensions.size());

  // using instance =
  std::transform(m_instanceExtensions.cbegin(), m_instanceExtensions.cend(),
                 extensions.begin(),
                 [](const std::string &str) { return str.c_str(); });

  createInfo.setPEnabledExtensionNames(extensions);

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

void VulkanRender::populateDebugMessengerCreateInfo(
    vk::DebugUtilsMessengerCreateInfoEXT &createInfo) {

  createInfo.setMessageSeverity(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo);

  createInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
  createInfo.setPfnUserCallback(debugCallback);
}

void VulkanRender::createSurface(const VkSurfaceKHR &surface) {

  //查看所有权之类删除
  m_surface = raii::SurfaceKHR(m_instance, surface);
}

void VulkanRender::pickPhysicalDevice() {
  auto devices = m_instance.enumeratePhysicalDevices();
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

void VulkanRender::createLogicalDevice() {
  App::QueueFamilyIndices indices =
      App::VulkanInitializer::findQueueFamilies(*m_physicalDevice, *m_surface);

  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                            indices.presentFamily.value()};

  // std::cerr << fmt::format("queueGraphic {}, presentFamily {}\n",
  //                          indices.graphicsFamily.value(),
  //                          indices.presentFamily.value());
  auto queuePriority = 1.0F;

  for (uint32_t queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCreateInfo{.queueFamilyIndex = queueFamily,
                                              .queueCount = 1,
                                              .pQueuePriorities =
                                                  &queuePriority};
    queueCreateInfos.push_back(queueCreateInfo);
  }

  vk::PhysicalDeviceFeatures deviceFeatures{};
  deviceFeatures.setSamplerAnisotropy(VK_TRUE);

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

// void VulkanRender::createSwapChain() {

//   SwapChainSupportDetails swapChainSupport =
//       querySwapChainSupport(*m_physicalDevice);

//   auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
//   auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
//   auto extent = chooseSwapExtent(swapChainSupport.capabilities);

//   uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
//   if (swapChainSupport.capabilities.maxImageCount > 0 &&
//       imageCount > swapChainSupport.capabilities.maxImageCount) {
//     imageCount = swapChainSupport.capabilities.maxImageCount;
//   }

//   vk::SwapchainCreateInfoKHR createInfo{};
//   createInfo.surface = *m_surface;
//   createInfo.minImageCount = imageCount;
//   createInfo.imageFormat = surfaceFormat.format;
//   createInfo.imageColorSpace = surfaceFormat.colorSpace;
//   createInfo.imageExtent = extent;
//   createInfo.imageArrayLayers = 1;
//   createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

//   QueueFamilyIndices indices = findQueueFamilies(*m_physicalDevice);
//   uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
//                                    indices.presentFamily.value()};

//   if (indices.graphicsFamily != indices.presentFamily) {
//     createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
//     createInfo.queueFamilyIndexCount = 2;
//     createInfo.pQueueFamilyIndices = queueFamilyIndices;
//   } else {
//     createInfo.imageSharingMode = vk::SharingMode::eExclusive;
//     createInfo.queueFamilyIndexCount = 0;     // Optional
//     createInfo.pQueueFamilyIndices = nullptr; // Optional
//   }
//   createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
//   createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
//   createInfo.presentMode = presentMode;
//   createInfo.clipped = VK_TRUE;
//   createInfo.oldSwapchain = *m_swapChain;

//   m_swapChain = m_device.createSwapchainKHR(createInfo);

//   m_swapChainImages = (*m_device).getSwapchainImagesKHR(*m_swapChain);

//   m_swapChainImageFormat = surfaceFormat.format;
//   m_swapChainExtent = extent;
// }

// void VulkanRender::createSwapChainImageViews() {

//   m_swapChainImageViews.clear();
//   m_swapChainImageViews.reserve(m_swapChainImages.size());
//   for (std::size_t i = 0; i < m_swapChainImages.size(); i++) {
//     vk::ImageViewCreateInfo createInfo{};
//     createInfo.image = m_swapChainImages[i];
//     createInfo.viewType = vk::ImageViewType::e2D;
//     createInfo.format = m_swapChainImageFormat;
//     createInfo.components.r = vk::ComponentSwizzle::eIdentity;
//     createInfo.components.g = vk::ComponentSwizzle::eIdentity;
//     createInfo.components.b = vk::ComponentSwizzle::eIdentity;
//     createInfo.components.a = vk::ComponentSwizzle::eIdentity;
//     createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
//     createInfo.subresourceRange.baseMipLevel = 0;
//     createInfo.subresourceRange.levelCount = 1;
//     createInfo.subresourceRange.baseArrayLayer = 0;
//     createInfo.subresourceRange.layerCount = 1;
//     m_swapChainImageViews.emplace_back(m_device.createImageView(createInfo));
//   }
// }

// void VulkanRender::createDepthImageAndView() {
//   vk::Extent3D depthImageExtent = {m_renderWidth, m_renderHeight, 1};
//   auto format = vk::Format::eD32Sfloat;
//   m_depthFormat = format;
//   auto imageInfo = VulkanInitializer::getImageCreateInfo(
//       format, vk::ImageUsageFlagBits::eDepthStencilAttachment,
//       depthImageExtent);
//   VmaAllocationCreateInfo allocationInfo{};
//   allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
//   allocationInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(
//       vk::MemoryPropertyFlagBits::eDeviceLocal);
//   m_depthImage = m_vulkanMemory->createImage(imageInfo, allocationInfo);

//   auto imageViewInfo = VulkanInitializer::getImageViewCreateInfo(
//       format, m_depthImage.get(), vk::ImageAspectFlagBits::eDepth);
//   m_depthImageView = m_device.createImageView(imageViewInfo);
// }

// void VulkanRender::createRenderPass() {
//   vk::AttachmentDescription colorAttachment{};
//   colorAttachment.format = m_swapChainImageFormat;
//   colorAttachment.samples = vk::SampleCountFlagBits::e1;
//   colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
//   colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
//   colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
//   colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
//   colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
//   colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

//   vk::AttachmentReference colorAttachmentRef{};
//   colorAttachmentRef.attachment = 0;
//   colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

//   vk::AttachmentDescription depthAttachment{};
//   depthAttachment.setFormat(m_depthFormat);
//   depthAttachment.setSamples(vk::SampleCountFlagBits::e1);
//   depthAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
//   depthAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
//   depthAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
//   depthAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
//   depthAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
//   depthAttachment.setFinalLayout(
//       vk::ImageLayout::eDepthStencilAttachmentOptimal);

//   vk::AttachmentReference depthAttachmentRef{};
//   depthAttachmentRef.attachment = 1;
//   depthAttachmentRef.layout =
//   vk::ImageLayout::eDepthStencilAttachmentOptimal;

//   vk::SubpassDependency dependency{};
//   dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
//   dependency.srcStageMask =
//   vk::PipelineStageFlagBits::eColorAttachmentOutput; dependency.dstStageMask
//   = vk::PipelineStageFlagBits::eColorAttachmentOutput;
//   dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

//   vk::SubpassDependency depthDependency{};
//   depthDependency.setSrcSubpass(VK_SUBPASS_EXTERNAL);
//   depthDependency.setSrcStageMask(
//       vk::PipelineStageFlagBits::eEarlyFragmentTests |
//       vk::PipelineStageFlagBits::eLateFragmentTests);
//   depthDependency.setDstStageMask(
//       vk::PipelineStageFlagBits::eEarlyFragmentTests |
//       vk::PipelineStageFlagBits::eLateFragmentTests);
//   depthDependency.setDstAccessMask(
//       vk::AccessFlagBits::eDepthStencilAttachmentWrite);

//   std::array dependencies{dependency, depthDependency};

//   vk::SubpassDescription subpass{};
//   subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
//   subpass.colorAttachmentCount = 1;
//   subpass.pColorAttachments = &colorAttachmentRef;
//   subpass.setPDepthStencilAttachment(&depthAttachmentRef);

//   std::array attachments{colorAttachment, depthAttachment};

//   vk::RenderPassCreateInfo renderPassInfo{};
//   renderPassInfo.setAttachments(attachments);
//   renderPassInfo.setSubpasses(subpass);
//   renderPassInfo.setDependencies(dependencies);

//   m_renderPass = m_device.createRenderPass(renderPassInfo);
// }

void VulkanRender::createGraphicsPipeline() {
  std::string homePath = m_programRootPath;

  auto vertShaderCode = readFile(homePath + "/shader/vert.spv");
  auto fragShaderCode = readFile(homePath + "/shader/frag.spv");
  auto vertShaderModule = createShaderModule(vertShaderCode);
  auto fragShaderModule = createShaderModule(fragShaderCode);

  using namespace App;
  PipelineFactory pipelineFactory;

  // pipelineFactory
  pipelineFactory.m_shaderStages.push_back(
      VulkanInitializer::getPipelineShaderStageCreateInfo(
          vk::ShaderStageFlagBits::eVertex, *vertShaderModule));

  pipelineFactory.m_shaderStages.push_back(
      VulkanInitializer::getPipelineShaderStageCreateInfo(
          vk::ShaderStageFlagBits::eFragment, *fragShaderModule));

  //保持生命周期
  auto inputDescriptor = App::Vertex::getVertexDescription();
  pipelineFactory.m_vertexInputInfo =
      VulkanInitializer::getPipelineVertexInputStateCreateInfo(inputDescriptor);

  pipelineFactory.m_inputAssembly =
      VulkanInitializer::getPipelineInputAssemblyStateCreateInfo();

  // let view port height negative , let ndc to left hand ,y is up

  auto height = m_renderTarget->m_swapChainExtent.height;
  auto width = m_renderTarget->m_swapChainExtent.width;
  pipelineFactory.m_viewPort = vk::Viewport{.x = 0.0F,
                                            .y = 0.0F + (float)height,
                                            .width = (float)width,
                                            .height = -((float)height),
                                            .minDepth = 0.0F,
                                            .maxDepth = 1.0F};
  pipelineFactory.m_scissor =
      vk::Rect2D{.offset = {0, 0}, .extent = m_renderTarget->m_swapChainExtent};

  pipelineFactory.m_rasterizer =
      VulkanInitializer::getPipelineRasterizationStateCreateInfo();

  pipelineFactory.m_multisampling =
      VulkanInitializer::getPipelineMultisampleStateCreateInfo();

  pipelineFactory.m_colorBlendAttachment =
      VulkanInitializer::getPipelineColorBlendAttachmentState();
  pipelineFactory.m_depthStencilCreateInfo =
      VulkanInitializer::getDepthStencilCreateInfo(true, true,
                                                   vk::CompareOp::eLessOrEqual);

  std::vector<vk::DynamicState> dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eLineWidth,
  };

  auto pipelineLayoutInfo = VulkanInitializer::getPipelineLayoutCreateInfo();

  vk::PushConstantRange pushConstant = {};
  pushConstant.setOffset(0);
  pushConstant.setSize(sizeof(App::MeshPushConstants));
  pushConstant.setStageFlags(vk::ShaderStageFlagBits::eVertex);

  pipelineLayoutInfo.setPushConstantRanges(pushConstant);

  // pipelineLayoutInfo.setSetLayouts(*m_descriptorSetLayout);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutInfo);

  pipelineFactory.m_pipelineLayout = *m_pipelineLayout;
  m_graphicsPipeline =
      pipelineFactory.buildPipeline(m_device, *(m_renderTarget->m_renderPass));
}


void VulkanRender::createCommandPool() {

  App::QueueFamilyIndices queueFamilyIndices =
      App::VulkanInitializer::findQueueFamilies(*m_physicalDevice, *m_surface);

  vk::CommandPoolCreateInfo poolInfo{};

  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    m_commandPools.push_back(m_device.createCommandPool(poolInfo));
  }
  // m_commandPool = m_device.createCommandPool(poolInfo);
}

void VulkanRender::createCommandBuffers() {

  m_commandBuffers.clear();

  m_commandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
  auto bufferSize = MAX_FRAMES_IN_FLIGHT;
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = *m_commandPools[i];
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;

    auto commandBuffers = m_device.allocateCommandBuffers(allocInfo);
    m_commandBuffers.insert(m_commandBuffers.end(),
                            std::make_move_iterator(commandBuffers.begin()),
                            std::make_move_iterator(commandBuffers.end()));
  }
}

void VulkanRender::createSyncObjects() {

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

void VulkanRender::drawFrame() {

  // auto startTime = chrono::high_resolution_clock().now();

  // updateUniformBuffer(m_currentFrame);
  using namespace std::literals::chrono_literals;
  auto onesecond=1ns;
  auto seconds = static_cast<uint64_t>(onesecond.count());

  vk::AcquireNextImageInfoKHR acquireInfo{};

  // acquireInfo.swapchain = *m_swapChain;
  // acquireInfo.timeout = seconds;
  // acquireInfo.semaphore = *m_imageAvailableSemaphores[m_currentFrame];
  // acquireInfo.deviceMask = UINT32_MAX;
  // std::vector<std::future<uint32_t>> funtures;
  auto swapChain = *(m_renderTarget->m_swapChain);
  auto [acquireResult, imageIndex] = (*m_device).acquireNextImageKHR(
      swapChain, UINT64_MAX, *m_imageAvailableSemaphores[m_currentFrame]);

  if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
    return;
  } else if (acquireResult != vk::Result::eSuccess &&
             acquireResult != vk::Result::eSuboptimalKHR) {
  }
  //    imageIndexs.push_back(drawFrameAsync(imageIndex, i));
  // funtures.push_back(std::async(std::mem_fn(&VulkanRender::drawFrameAsync),
  //                               this, imageIndex, i));

  // wait command buffer finish its work
  // because buffer has to be completed
  auto result = m_device.waitForFences(*m_inFlightFences[m_currentFrame],
                                       VK_TRUE, seconds);
  if (result == vk::Result::eTimeout) {
    App::ThrowException(" wait fences time out");
  }
  m_device.resetFences(*m_inFlightFences[m_currentFrame]);
  m_commandBuffers[m_currentFrame].reset();

  recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

  vk::SubmitInfo submitInfo{};

  std::array waitSemaphores = {*m_imageAvailableSemaphores[m_currentFrame]};
  auto waitStages = std::to_array<vk::PipelineStageFlags>(
      {vk::PipelineStageFlagBits::eColorAttachmentOutput});
  std::array commandBuffers = {*m_commandBuffers[m_currentFrame]};
  submitInfo.setWaitSemaphores(waitSemaphores);

  submitInfo.setWaitDstStageMask(waitStages);
  submitInfo.setCommandBuffers(commandBuffers);
  std::array signalSemaphores = {*m_renderFinishedSemaphores[m_currentFrame]};
  submitInfo.setSignalSemaphores(signalSemaphores);
  m_graphicsQueue.submit(submitInfo, *m_inFlightFences[m_currentFrame]);

  // auto imageIndex = imageIndexs[i];
  vk::PresentInfoKHR presentInfo{};
  presentInfo.setWaitSemaphores(signalSemaphores);
  std::array swapChains = {swapChain};
  presentInfo.setSwapchains(swapChains);
  presentInfo.setImageIndices(imageIndex);
  presentInfo.pResults = nullptr;

  try {
    auto presentQueueResult = m_presentQueue.presentKHR(presentInfo);

  } catch (const std::system_error &system) {
    auto code = system.code();
    auto errorCode = static_cast<vk::Result>(code.value());
    if (errorCode == vk::Result::eErrorOutOfDateKHR ||
        errorCode == vk::Result::eSuboptimalKHR) {
      //do nothing
    } else if (errorCode != vk::Result::eSuccess) {
      // do nothing
    }
  }

  m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  m_frameCount++;

  //  auto endTime = chrono::high_resolution_clock().now();

  // m_perFrameTime = endTime - startTime;
}

bool VulkanRender::isDeviceSuitable(const vk::PhysicalDevice &device) {
  auto deviceProperties = device.getProperties();

#ifdef DEBUG
  std::cout << fmt::format(
      "deviceInfo {0},{1},{2}\n", deviceProperties.deviceName,
      deviceProperties.vendorID, deviceProperties.deviceID);
#endif
  App::QueueFamilyIndices indices =
      App::VulkanInitializer::findQueueFamilies(device, *m_surface);

  bool extensionsSupported = checkDeviceExtensionSupport(device);

  bool swapChainAdequate = false;
  if (extensionsSupported) {
    App::SwapChainSupportDetails swapChainSupport =
        App::VulkanInitializer::querySwapChainSupport(device, *m_surface);
    swapChainAdequate = !swapChainSupport.formats.empty() &&
                        !swapChainSupport.presentModes.empty();
  }

  auto features = device.getFeatures();

  return indices.isComplete() && extensionsSupported && swapChainAdequate &&
         (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ||
          deviceProperties.deviceType ==
              vk::PhysicalDeviceType::eIntegratedGpu) &&
         static_cast<bool>(features.samplerAnisotropy);
}

bool VulkanRender::checkDeviceExtensionSupport(
    const vk::PhysicalDevice &device) {

  auto availableExtensions = device.enumerateDeviceExtensionProperties();
  std::set<std::string> requiredExtensions(m_deviceExtensions.begin(),
                                           m_deviceExtensions.end());

  for (const auto &extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }

  return requiredExtensions.empty();
}
bool VulkanRender::checkValidationLayerSupport() {

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

void VulkanRender::recordCommandBuffer(const raii::CommandBuffer &commandBuffer,
                                       uint32_t imageIndex) {
  vk::CommandBufferBeginInfo beginInfo{};

  // beginInfo.flags =0  ;                  // Optional
  beginInfo.pInheritanceInfo = nullptr; // Optional

  commandBuffer.begin(beginInfo);

  auto renderPass = *(m_renderTarget->m_renderPass);
  vk::RenderPassBeginInfo renderPassInfo{};
  renderPassInfo.renderPass = renderPass;
  renderPassInfo.framebuffer = *m_renderTarget->m_framebuffers[imageIndex];
  renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
  renderPassInfo.renderArea.extent = m_renderTarget->m_swapChainExtent;

  vk::ArrayWrapper1D<float, 4> array{{0.0f, 0.0f, 1.0f, 0.0f}};

  vk::ClearValue clearColor{.color = {std::move(array)}};
  vk::ClearValue depthClear{.depthStencil = {.depth = 1.0f}};

  std::array clearColors{clearColor, depthClear};

  renderPassInfo.setClearValues(clearColors);

  commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             *m_graphicsPipeline);

  auto vertexBuffers = std::to_array<vk::Buffer>({m_mesh.vertexBuffer.get()});
  auto monkeyVertexBuffers =
      std::to_array<vk::Buffer>({m_monkeyMesh.vertexBuffer.get()});
  auto offsets = std::to_array<vk::DeviceSize>({0});
  // commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
  commandBuffer.bindVertexBuffers(0, monkeyVertexBuffers, offsets);

  // commandBuffer.bindIndexBuffer(*m_indexBuffer, 0, vk::IndexType::eUint16);

  // commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
  //                                   *m_pipelineLayout, 0,
  //                                   *m_descriptorSets[m_currentFrame], {});

  // commandBuffer.drawIndexed(m_mesh.vertices.size(), 1, 0, 0, 0);

  // commandBuffer.drawIndexed(3, 1, 0, 0, 0);
  glm::vec3 camPos = {0.f, 0.f, -2.f};
  auto lookAtView =
      glm::lookAt(camPos, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0, -1, 0});

  auto width = m_renderTarget->m_swapChainExtent.width;
  auto height = m_renderTarget->m_swapChainExtent.height;
  // // camera projection
  glm::mat4 projection = glm::perspective(
      glm::radians(60.f), static_cast<float>(width) / height, 1.5f, 200.0f);
  // // model rotation
  glm::mat4 model =
      glm::rotate(glm::mat4{1.0f}, glm::radians(45.0f), glm::vec3(0, -1, 0));

  // auto big2 = projection * lookAtView ;
  auto big2 = projection * lookAtView;
  App::MeshPushConstants constants = {};

  constants.renderMatrix = big2;

  commandBuffer.pushConstants<App::MeshPushConstants>(
      *m_pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, constants);

  // commandBuffer.draw(m_mesh.vertices.size(), 1, 0, 0);
  commandBuffer.draw(m_monkeyMesh.vertices.size(), 1, 0, 0);

  commandBuffer.endRenderPass();

  commandBuffer.end();
}

// QueueFamilyIndices
// VulkanInitializer::findQueueFamilies(const vk::PhysicalDevice &device,
//                                      vk::SurfaceKHR surface) {

//   QueueFamilyIndices indices;

//   auto queueFamilies = device.getQueueFamilyProperties();

//   int i = 0;
//   for (const auto &queueFamily : queueFamilies) {
//     if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
//       indices.graphicsFamily = i;
//     }

//     auto presentSupport = device.getSurfaceSupportKHR(i, surface);

//     if (presentSupport) {
//       indices.presentFamily = i;
//     }
//     i++;
//   }
//   return indices;
// }

raii::ShaderModule
VulkanRender::createShaderModule(const std::vector<char> &code) {
  vk::ShaderModuleCreateInfo createInfo{};
  createInfo.codeSize = code.size();

  createInfo.pCode =
      std::launder(reinterpret_cast<const uint32_t *>(code.data()));
  auto shaderModule = m_device.createShaderModule(createInfo);
  return shaderModule;
}

void VulkanRender::cleanup() {

  m_imageAvailableSemaphores.clear();
  m_renderFinishedSemaphores.clear();
  m_inFlightFences.clear();

  m_commandBuffers.clear();

  m_commandPools.clear();
  // m_swapChainFramebuffers.clear();
   m_graphicsPipeline.clear();
   m_pipelineLayout.clear();
   m_renderTarget.reset();
  // m_renderPass.clear();
  // m_swapChainImageViews.clear();
  // m_swapChain.clear();
  // /
  // m_device.clear();
  // m_debugMessenger.clear();
}

void VulkanRender::loadMeshs() {
  m_mesh.vertices.resize(6);
  // vertex positions
  m_mesh.vertices[0].position = {1.f, 1.f, 0.0f};
  m_mesh.vertices[1].position = {-1.f, 1.f, 0.0f};
  m_mesh.vertices[2].position = {1.f, -1.f, 0.0f};
  m_mesh.vertices[3].position = {-1.f, -1.f, 0.0f};
  m_mesh.vertices[4].position = {-1.f, 1.f, 0.0f};
  m_mesh.vertices[5].position = {1.f, -1.f, 0.0f};
  // vertex colors, all green
  m_mesh.vertices[0].color = {1.f, 0.f, 0.0f}; // pure green
  m_mesh.vertices[1].color = {0.f, 1.f, 0.0f}; // pure green
  m_mesh.vertices[2].color = {0.f, 0.f, 1.0f}; // pure green
  m_mesh.vertices[3].color = {0.f, 1.f, 1.0f}; // pure green
  m_mesh.vertices[4].color = {0.f, 1.f, 0.0f}; // pure green
  m_mesh.vertices[5].color = {0.f, 0.f, 1.0f}; // pure green

  if (!m_monkeyMesh.loadFromOBJ(m_programRootPath + "/asset/houtou.obj")) {
    App::ThrowException("load obj error");
  }

  m_vulkanMemory->uploadMesh(m_mesh);
  m_vulkanMemory->uploadMesh(m_monkeyMesh);
}

void VulkanRender::createRenderTarget() {

  namespace init = App::VulkanInitializer;
  App::RenderTargetBuilder builder{m_device, *m_vulkanMemory};
  App::RenderTargetBuilder::CreateInfo info{};

  info.physicalDevice = *m_physicalDevice;
  info.renderExtent = vk::Extent2D{m_renderWidth, m_renderHeight};
  info.surface = *m_surface;

  m_renderTarget =
      builder.setCreateInfo(info).build();
}

uint32_t
VulkanRender::findMemoryType(uint32_t typeFilter,
                             const vk::MemoryPropertyFlags &properties) {
  auto memProperties = m_physicalDevice.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (i << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  return 0;
}
