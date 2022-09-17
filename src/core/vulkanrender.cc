#include <cstddef>
#include <vulkanrender.hh>

using std::runtime_error;

void VulkanRender::initOthers(const VkSurfaceKHR &surface) {
  initVulkan(surface);
  initSwapChain();
  initCommands();

  loadMeshs();
  // createTextureImage();
  // createTextureImageView();
  // createTextureSampler();

  // createDescriptorSetLayout();
  // createIndexBuffer();
  // createUniformBuffers();
  // createDescriptorPool();
  // createDescriptorSets();

  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();

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

void VulkanRender::initSwapChain() {
  createSwapChain();
  createSwapChainImageViews();
}

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
  QueueFamilyIndices indices = findQueueFamilies(*m_physicalDevice);

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

void VulkanRender::createSwapChain() {

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
  createInfo.surface = *m_surface;
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

void VulkanRender::createSwapChainImageViews() {

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

void VulkanRender::createRenderPass() {
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

void VulkanRender::createGraphicsPipeline() {
  std::string homePath = m_shaderDirPath;

  auto vertShaderCode = readFile(homePath + "/vert.spv");
  auto fragShaderCode = readFile(homePath + "/frag.spv");
  auto vertShaderModule = createShaderModule(vertShaderCode);
  auto fragShaderModule = createShaderModule(fragShaderCode);

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

  pipelineFactory.m_viewPort =
      vk::Viewport{.y = 0.0F,
                   .width = (float)m_swapChainExtent.width,
                   .height = (float)m_swapChainExtent.height,
                   .minDepth = 0.0F,
                   .maxDepth = 1.0F};
  pipelineFactory.m_scissor = {.offset = {0, 0}, .extent = m_swapChainExtent};

  pipelineFactory.m_rasterizer =
      VulkanInitializer::getPipelineRasterizationStateCreateInfo();

  pipelineFactory.m_multisampling =
      VulkanInitializer::getPipelineMultisampleStateCreateInfo();

  pipelineFactory.m_colorBlendAttachment =
      VulkanInitializer::getPipelineColorBlendAttachmentState();

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
  m_graphicsPipeline = pipelineFactory.buildPipeline(m_device, *m_renderPass);
}

void VulkanRender::createFramebuffers() {
  m_swapChainFramebuffers.clear();
  m_swapChainFramebuffers.reserve(m_swapChainImageViews.size());

  for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
    std::array attachments = {*m_swapChainImageViews[i]};

    vk::FramebufferCreateInfo framebufferInfo{};
    framebufferInfo.renderPass = *m_renderPass;
    framebufferInfo.setAttachments(attachments);
    framebufferInfo.width = m_swapChainExtent.width;
    framebufferInfo.height = m_swapChainExtent.height;
    framebufferInfo.layers = 1;

    m_swapChainFramebuffers.emplace_back(
        m_device.createFramebuffer(framebufferInfo));
  }
}

void VulkanRender::createCommandPool() {
  QueueFamilyIndices queueFamilyIndices = findQueueFamilies(*m_physicalDevice);

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
  auto seconds = static_cast<uint64_t>(10e9);

  vk::AcquireNextImageInfoKHR acquireInfo{};

  // acquireInfo.swapchain = *m_swapChain;
  // acquireInfo.timeout = seconds;
  // acquireInfo.semaphore = *m_imageAvailableSemaphores[m_currentFrame];
  // acquireInfo.deviceMask = UINT32_MAX;
  // std::vector<std::future<uint32_t>> funtures;
  auto [acquireResult, imageIndex] = (*m_device).acquireNextImageKHR(
      *m_swapChain, UINT64_MAX, *m_imageAvailableSemaphores[m_currentFrame]);

  if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
    recreateSwapChain();
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
  std::array swapChains = {*m_swapChain};
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
      recreateSwapChain();
    } else if (errorCode != vk::Result::eSuccess) {
    }
  }

  m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

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
  QueueFamilyIndices indices = findQueueFamilies(device);

  bool extensionsSupported = checkDeviceExtensionSupport(device);

  bool swapChainAdequate = false;
  if (extensionsSupported) {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
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

  vk::RenderPassBeginInfo renderPassInfo{};
  renderPassInfo.renderPass = *m_renderPass;
  renderPassInfo.framebuffer = *m_swapChainFramebuffers[imageIndex];
  renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
  renderPassInfo.renderArea.extent = m_swapChainExtent;

  vk::ArrayWrapper1D<float, 4> array{{0.0f, 0.0f, 1.0f, 1.0f}};

  vk::ClearValue clearColor{.color = {std::move(array)}};
  renderPassInfo.setClearValues(clearColor);

  commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                             *m_graphicsPipeline);

  auto vertexBuffers = std::to_array<vk::Buffer>({m_mesh.vertexBuffer.get()});
  auto offsets = std::to_array<vk::DeviceSize>({0});
  commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

  // commandBuffer.bindIndexBuffer(*m_indexBuffer, 0, vk::IndexType::eUint16);

  // commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
  //                                   *m_pipelineLayout, 0,
  //                                   *m_descriptorSets[m_currentFrame], {});

  // commandBuffer.drawIndexed(m_mesh.vertices.size(), 1, 0, 0, 0);

  // commandBuffer.drawIndexed(3, 1, 0, 0, 0);
  // glm::vec3 camPos = {0.f, 0.f, -2.f};
  // glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
  // // camera projection
  // glm::mat4 projection =
  //     glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);
  // projection[1][1] *= -1;
  // // model rotation
  // glm::mat4 model = glm::rotate(
  //     glm::mat4{1.0f}, glm::radians(6 * 0.4f), glm::vec3(0, 1, 0));

  // // calculate final mesh matrix
  // glm::mat4 meshMatrix = projection * view * model;


  glm::mat4 big2=glm::scale(glm::mat4(1.f), glm::vec3(0.5f,0.5f,0.5f));
   App::MeshPushConstants constants={};
  constants.renderMatrix = big2;

   commandBuffer.pushConstants<App::MeshPushConstants>(
       *m_pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, constants);

  commandBuffer.draw(m_mesh.vertices.size(), 1, 0, 0);

  commandBuffer.endRenderPass();

  commandBuffer.end();
}

QueueFamilyIndices
VulkanRender::findQueueFamilies(const vk::PhysicalDevice &device) {

  QueueFamilyIndices indices;

  auto queueFamilies = device.getQueueFamilyProperties();

  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.graphicsFamily = i;
    }

    auto presentSupport = device.getSurfaceSupportKHR(i, *m_surface);

    if (presentSupport) {
      indices.presentFamily = i;
    }
    i++;
  }
  return indices;
}

SwapChainSupportDetails
VulkanRender::querySwapChainSupport(const vk::PhysicalDevice &device) {

  SwapChainSupportDetails details;
  details.capabilities = device.getSurfaceCapabilitiesKHR(*m_surface);
  details.formats = device.getSurfaceFormatsKHR(*m_surface);
  details.presentModes = device.getSurfacePresentModesKHR(*m_surface);
  return details;
}

vk::SurfaceFormatKHR VulkanRender::chooseSwapSurfaceFormat(
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

vk::PresentModeKHR VulkanRender::chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  for (const auto &availablePresentMode : availablePresentModes) {
    if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
      return vk::PresentModeKHR::eImmediate;
    }
  }
  return vk::PresentModeKHR::eImmediate;
}

vk::Extent2D
VulkanRender::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {

    // spdlog::info("window daxiao{} {}", capabilities.currentExtent.width,
    //              capabilities.currentExtent.height);
    return capabilities.currentExtent;
  } else {
    int width, height;

    height = m_renderHeight;
    width = m_renderWidth;

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
  m_swapChainFramebuffers.clear();
  m_graphicsPipeline.clear();
  m_pipelineLayout.clear();
  m_renderPass.clear();
  m_swapChainImageViews.clear();
  m_swapChain.clear();
  // m_device.clear();
  // m_debugMessenger.clear();
}

void VulkanRender::loadMeshs() {
  m_mesh.vertices.resize(3);
  // vertex positions
  m_mesh.vertices[0].position = {1.f, 1.f, 0.0f};
  m_mesh.vertices[1].position = {-1.f, 1.f, 0.0f};
  m_mesh.vertices[2].position = {0.f, -1.f, 0.0f};

  // vertex colors, all green
  m_mesh.vertices[0].color = {0.f, 1.f, 0.0f}; // pure green
  m_mesh.vertices[1].color = {0.f, 1.f, 0.0f}; // pure green
  m_mesh.vertices[2].color = {0.f, 1.f, 0.0f}; // pure green

  m_vulkanMemory->uploadMesh(m_mesh);
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

void VulkanRender::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
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

void VulkanRender::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                              vk::DeviceSize size) {

  auto commandBufferPointer = beginSingleTimeCommands();
  vk::BufferCopy copyRegion{};
  copyRegion.size = size;
  (*commandBufferPointer).copyBuffer(srcBuffer, dstBuffer, copyRegion);
}


void VulkanRender::createDescriptorSetLayout() {
  vk::DescriptorSetLayoutBinding uboLayoutBinding{};
  uboLayoutBinding.setBinding(0);
  uboLayoutBinding.setDescriptorCount(1);
  uboLayoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
  uboLayoutBinding.setStageFlags(vk::ShaderStageFlagBits::eVertex);

  vk::DescriptorSetLayoutBinding samplerLayoutBinding{};

  samplerLayoutBinding.binding = 1;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType =
      vk::DescriptorType::eCombinedImageSampler;
  samplerLayoutBinding.pImmutableSamplers = nullptr;
  samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

  std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
      uboLayoutBinding, samplerLayoutBinding};

  vk::DescriptorSetLayoutCreateInfo layoutInfo{};

  layoutInfo.setBindingCount(bindings.size());
  layoutInfo.setBindings(bindings);

  m_descriptorSetLayout = m_device.createDescriptorSetLayout(layoutInfo);
}


void VulkanRender::updateUniformBuffer(uint32_t currentImage) {
  static auto startTime = chrono::high_resolution_clock::now();

  auto currentTime = chrono::high_resolution_clock::now();

  float time =
      chrono::duration<float, chrono::seconds::period>(currentTime - startTime)
          .count();

  UniformBufferObject ubo{};
  // ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
  //                         glm::vec3(0.0f, 0.0f, 1.0f));

  ubo.model = glm::mat4(1.0F);

  ubo.view =
      glm::lookAt(glm::vec3(2.0f, -2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                  glm::vec3(0.0f, 0.0f, 1.0f));

  // ubo.view=glm::mat4(1.0F);

  ubo.proj = glm::perspective(glm::radians(45.0f),
                              m_swapChainExtent.width /
                                  static_cast<float>(m_swapChainExtent.height),
                              0.1f, 10.0f);
  // ubo.proj=glm::mat4(1.0F);

  // ubo.proj[1][1] *= -1;

  auto *data = (*m_device).mapMemory(*m_uniformBuffersMemory[currentImage], 0,
                                     sizeof(ubo));

  std::memcpy(data, &ubo, sizeof(ubo));

  (*m_device).unmapMemory(*m_uniformBuffersMemory[currentImage]);
}

void VulkanRender::createDescriptorPool() {

  std::array<vk::DescriptorPoolSize, 2> poolSizes{};

  poolSizes[0].setType(vk::DescriptorType::eUniformBuffer);
  poolSizes[0].setDescriptorCount(static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT));
  poolSizes[1].setType(vk::DescriptorType::eCombinedImageSampler);
  poolSizes[1].setDescriptorCount(static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT));

  vk::DescriptorPoolCreateInfo poolInfo{};

  poolInfo.setPoolSizeCount(poolSizes.size());
  poolInfo.setPoolSizes(poolSizes);
  poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
  poolInfo.setMaxSets(static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT));
  m_descriptorPool = m_device.createDescriptorPool(poolInfo);
}

void VulkanRender::createDescriptorSets() {
  std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                               *m_descriptorSetLayout);

  vk::DescriptorSetAllocateInfo allocInfo{};
  allocInfo.descriptorPool = *m_descriptorPool;
  allocInfo.setSetLayouts(layouts);

  m_descriptorSets = raii::DescriptorSets(m_device, allocInfo);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = *m_uniformBuffers[i];
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    vk::DescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = *m_textureImageView;
    imageInfo.sampler = *m_textureSampler;

    std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
    descriptorWrites[0].setDstSet(*m_descriptorSets[i]);
    descriptorWrites[0].setDstBinding(0);
    descriptorWrites[0].setDstArrayElement(0);
    descriptorWrites[0].setDescriptorType(vk::DescriptorType::eUniformBuffer);
    descriptorWrites[0].setDescriptorCount(1);
    descriptorWrites[0].setBufferInfo(bufferInfo);

    descriptorWrites[1].setDstSet(*m_descriptorSets[i]);
    descriptorWrites[1].setDstBinding(1);
    descriptorWrites[1].setDstArrayElement(0);
    descriptorWrites[1].setDescriptorType(
        vk::DescriptorType::eCombinedImageSampler);
    descriptorWrites[1].setDescriptorCount(1);
    descriptorWrites[1].setImageInfo(imageInfo);

    m_device.updateDescriptorSets(descriptorWrites, {});
  }
}

void VulkanRender::createTextureImage() {
  int texWidth = 0;
  int texHeight = 0;
  int texChannels = 0;

  auto *pixels = stbi_load(
      "/home/tomblack/Pictures/aiyinsitan/c637138ab1c9a25e1a61f434a6e75ff6.png",
      &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

  std::unique_ptr<stbi_uc,
                  decltype([](stbi_uc *stbi) { stbi_image_free(stbi); })>
      raiiPixels{pixels};

  auto imageSize = static_cast<vk::DeviceSize>(texWidth * texHeight) * 4;

  // texWidth = 540;
  // texHeight = 960;

  // auto imageSize = m_softRender.getFrameBuffer().size() * sizeof(Pixel);

  // auto const *pixelTest = m_softRender.getFrameBuffer().data();

  raii::Buffer stagingBuffer{nullptr};
  raii::DeviceMemory stagingBufferMemory{nullptr};

  createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  auto *data = (*m_device).mapMemory(*stagingBufferMemory, 0, imageSize);

  // std::memcpy(data, raiiPixels.get(), static_cast<std::size_t>(imageSize));
  std::memcpy(data, pixels, static_cast<std::size_t>(imageSize));

  (*m_device).unmapMemory(*stagingBufferMemory);

  createImage(static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight),
              vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eSampled,
              vk::MemoryPropertyFlagBits::eDeviceLocal, m_textureImage,
              m_textureImageMemory);

  transitionImageLayout(*m_textureImage, vk::Format::eR8G8B8A8Srgb,
                        vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eTransferDstOptimal);

  copyBufferToImage(*stagingBuffer, *m_textureImage,
                    static_cast<uint32_t>(texWidth),
                    static_cast<uint32_t>(texHeight));

  transitionImageLayout(*m_textureImage, vk::Format::eR8G8B8A8Srgb,
                        vk::ImageLayout::eTransferDstOptimal,
                        vk::ImageLayout::eShaderReadOnlyOptimal);
}

void VulkanRender::createImage(uint32_t width, uint32_t height,
                               vk::Format format, vk::ImageTiling tiling,
                               vk::ImageUsageFlags usage,
                               vk::MemoryPropertyFlagBits properties,
                               raii::Image &image,
                               raii::DeviceMemory &imageMemory) {
  vk::ImageCreateInfo imageInfo{};
  imageInfo.setImageType(vk::ImageType::e2D);
  vk::Extent3D extend{.width = static_cast<uint32_t>(width),
                      .height = static_cast<uint32_t>(height),
                      .depth = 1};
  imageInfo.setExtent(extend);
  imageInfo.setMipLevels(1);
  imageInfo.setArrayLayers(1);
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
  imageInfo.usage = usage;
  imageInfo.sharingMode = vk::SharingMode::eExclusive;
  imageInfo.samples = vk::SampleCountFlagBits::e1;
  image = m_device.createImage(imageInfo);

  auto memRequirements = (*m_device).getImageMemoryRequirements(*image);

  vk::MemoryAllocateInfo allocInfo{};

  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  imageMemory = m_device.allocateMemory(allocInfo);

  (*m_device).bindImageMemory(*image, *imageMemory, 0);
}

CommandBufferPointer VulkanRender::beginSingleTimeCommands() {
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
  allocInfo.setCommandPool(*m_commandPools[0]);
  allocInfo.setCommandBufferCount(1);
  auto commandBuffers = m_device.allocateCommandBuffers(allocInfo);

  vk::CommandBufferBeginInfo beginInfo{};

  beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  commandBuffers[0].begin(beginInfo);

  CommandBufferDeleter deleter(*m_graphicsQueue);

  return {new raii::CommandBuffer(std::move(commandBuffers[0])), deleter};
}

void VulkanRender::transitionImageLayout(vk::Image image, vk::Format format,
                                         vk::ImageLayout oldLayout,
                                         vk::ImageLayout newLayout) {
  auto commandBufferPointer = beginSingleTimeCommands();

  vk::ImageMemoryBarrier barrier{};

  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;

  vk::ImageSubresourceRange range{};
  range.aspectMask = vk::ImageAspectFlagBits::eColor;
  range.baseMipLevel = 0;
  range.levelCount = 1;
  range.baseArrayLayer = 0;
  range.layerCount = 1;
  barrier.setSubresourceRange(range);

  vk::PipelineStageFlagBits sourceStage;
  vk::PipelineStageFlagBits destinationStage;

  if (oldLayout == vk::ImageLayout::eUndefined &&
      newLayout == vk::ImageLayout::eTransferDstOptimal) {
    barrier.srcAccessMask = {};
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
    sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    destinationStage = vk::PipelineStageFlagBits::eTransfer;
  }

  else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
           newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    sourceStage = vk::PipelineStageFlagBits::eTransfer;
    destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
  } else {
    throw std::invalid_argument("unsupported layout transition!");
  }

  (*commandBufferPointer)
      .pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
}

void VulkanRender::copyBufferToImage(vk::Buffer buffer, vk::Image image,
                                     uint32_t width, uint32_t height) {
  auto comomandPointer = beginSingleTimeCommands();

  vk::BufferImageCopy region{};
  region.setBufferOffset(0);
  region.setBufferRowLength(0);
  region.setBufferImageHeight(0);

  vk::ImageSubresourceLayers subresource{.aspectMask =
                                             vk::ImageAspectFlagBits::eColor,
                                         .mipLevel = 0,
                                         .baseArrayLayer = 0,
                                         .layerCount = 1};

  region.setImageSubresource(subresource);
  region.setImageOffset({0, 0, 0});
  region.setImageExtent({width, height, 1});

  (*comomandPointer)
      .copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal,
                         region);
}

raii::ImageView VulkanRender::createImageView(vk::Image image,
                                              vk::Format format) {

  vk::ImageViewCreateInfo viewInfo{
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};

  return m_device.createImageView(viewInfo);
}

void VulkanRender::createTextureImageView() {
  m_textureImageView =
      createImageView(*m_textureImage, vk::Format::eR8G8B8A8Srgb);
}

void VulkanRender::createTextureSampler() {
  vk::SamplerCreateInfo samplerInfo{};

  samplerInfo.magFilter = vk::Filter::eLinear;
  samplerInfo.minFilter = vk::Filter::eLinear;
  samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
  samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
  samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;

  samplerInfo.anisotropyEnable = VK_TRUE;
  auto properties = m_physicalDevice.getProperties();
  samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

  samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;

  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = vk::CompareOp::eAlways;

  samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
  samplerInfo.mipLodBias = 0.0F;
  samplerInfo.minLod = 0.0F;
  samplerInfo.maxLod = 0.0F;

  m_textureSampler = m_device.createSampler(samplerInfo);
}

vk::PipelineShaderStageCreateInfo
VulkanInitializer::getPipelineShaderStageCreateInfo(
    vk::ShaderStageFlagBits stage, vk::ShaderModule shaderModule) {

  vk::PipelineShaderStageCreateInfo shaderStageInfo{};
  shaderStageInfo.stage = stage;
  shaderStageInfo.module = shaderModule;
  shaderStageInfo.pName = "main";
  return shaderStageInfo;
}
vk::PipelineVertexInputStateCreateInfo
VulkanInitializer::getPipelineVertexInputStateCreateInfo(
    App::VertexInputDescription const &des) {

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.setVertexBindingDescriptions(des.bindings);
  vertexInputInfo.setVertexAttributeDescriptions(des.attributes);
  return vertexInputInfo;
}
vk::PipelineInputAssemblyStateCreateInfo
VulkanInitializer::getPipelineInputAssemblyStateCreateInfo() {
  vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}
vk::PipelineRasterizationStateCreateInfo
VulkanInitializer::getPipelineRasterizationStateCreateInfo() {
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
  return rasterizer;
}
vk::PipelineMultisampleStateCreateInfo
VulkanInitializer::getPipelineMultisampleStateCreateInfo() {

  vk::PipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
  multisampling.minSampleShading = 1.0f;          // Optional
  multisampling.pSampleMask = nullptr;            // Optional
  multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
  multisampling.alphaToOneEnable = VK_FALSE;      // Optional
  return multisampling;
}
vk::PipelineColorBlendAttachmentState
VulkanInitializer::getPipelineColorBlendAttachmentState() {
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
  return colorBlendAttachment;
}
vk::PipelineLayoutCreateInfo VulkanInitializer::getPipelineLayoutCreateInfo() {
  vk::PipelineLayoutCreateInfo info{};
  return info;
}

raii::Pipeline PipelineFactory::buildPipeline(const raii::Device &device,
                                              vk::RenderPass pass) {
  vk::PipelineViewportStateCreateInfo viewportState = {};
  viewportState.setViewports(m_viewPort);
  viewportState.setScissors(m_scissor);

  vk::PipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = vk::LogicOp::eCopy; // Optional
  colorBlending.setAttachments(m_colorBlendAttachment);
  colorBlending.blendConstants[0] = 0.0f; // Optional
  colorBlending.blendConstants[1] = 0.0f; // Optional
  colorBlending.blendConstants[2] = 0.0f; // Optional
  colorBlending.blendConstants[3] = 0.0f; // Optional

  vk::GraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.setStages(m_shaderStages);
  pipelineInfo.pVertexInputState = &m_vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &m_inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &m_rasterizer;
  pipelineInfo.pMultisampleState = &m_multisampling;
  pipelineInfo.pDepthStencilState = nullptr; // Optional
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = nullptr; // Optional
  pipelineInfo.layout = m_pipelineLayout;

  pipelineInfo.renderPass = pass;
  pipelineInfo.subpass = 0;

  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
  pipelineInfo.basePipelineIndex = -1;

  auto graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo);
  return graphicsPipeline;
}
