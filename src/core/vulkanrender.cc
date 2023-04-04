#include <vulkanrender.hh>

using std::runtime_error;

void VulkanRender::initOthers(const VkSurfaceKHR &surface) {
  initVulkan(surface);

  initDescriptors();
  initFrameDatas();
  loadMeshs();
  createRenderTarget();
  createGraphicsPipeline();
}

void VulkanRender::initVulkan(const VkSurfaceKHR &surface) {
  createSurface(surface);
  pickPhysicalDevice();
  createLogicalDevice();

  m_gpuProperties = m_physicalDevice.getProperties();

  std::clog << fmt::format(
      "The GPU has a minimum uniform buffer alignment of {}\n",
      m_gpuProperties.limits.minUniformBufferOffsetAlignment);
  VmaAllocatorCreateInfo createInfo{};

  createInfo.vulkanApiVersion = VK_API_VERSION_1_3;
  createInfo.device = *m_device;
  createInfo.physicalDevice = *m_physicalDevice;
  createInfo.instance = *m_instance;
  m_vulkanMemory = std::make_unique<App::VulkanMemory>(createInfo);
}

void VulkanRender::initDescriptors() {

  // binding camera data at 0
  auto cameraBufferBinding =
      App::VulkanInitializer::getDescriptorSetLayoutBinding(
          vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex,
          0);

  auto sceneBinding = App::VulkanInitializer::getDescriptorSetLayoutBinding(
      vk::DescriptorType::eUniformBufferDynamic,
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 1);

  std::array bindings{cameraBufferBinding, sceneBinding};

  vk::DescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.setBindings(bindings);

  m_descriptorSetlayout = m_device.createDescriptorSetLayout(layoutInfo);

  namespace init = App::VulkanInitializer;

  auto objectBinding = init::getDescriptorSetLayoutBinding(
      vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex, 0);

  vk::DescriptorSetLayoutCreateInfo objectSetInfo{};
  objectSetInfo.setBindings(objectBinding);

  m_objectSetLayout = m_device.createDescriptorSetLayout(objectSetInfo);

  auto poolSizes = std::to_array<vk::DescriptorPoolSize>(
      {{vk::DescriptorType::eUniformBuffer, 10},
       {vk::DescriptorType::eUniformBufferDynamic, 10},
       {vk::DescriptorType::eStorageBuffer, 10}});

  vk::DescriptorPoolCreateInfo poolInfo{};
  poolInfo.setPoolSizes(poolSizes);
  poolInfo.setMaxSets(10);
  poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
  m_descriptorPool = m_device.createDescriptorPool(poolInfo);

  auto sceneBufferSize = MAX_FRAMES_IN_FLIGHT * getPadUniformBufferOffsetSize(
                                                    sizeof(App::GPUSceneData));

  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
  bufferInfo.setSize(sceneBufferSize);
  VmaAllocationCreateInfo allocationInfo{};
  allocationInfo.usage = VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  allocationInfo.flags = VmaAllocationCreateFlagBits::
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  m_sceneParaBuffer = m_vulkanMemory->createBuffer(bufferInfo, allocationInfo);
}

void VulkanRender::initFrameDatas() { createFrameDatas(); }

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

  // 查看所有权之类删除
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

  auto queuePriority = 1.0F;

  for (uint32_t queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCreateInfo{.queueFamilyIndex = queueFamily,
                                              .queueCount = 1,
                                              .pQueuePriorities =
                                                  &queuePriority};
    queueCreateInfos.push_back(queueCreateInfo);
  }

  vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan12Features>
      deviceCreateChains{};
  // vk::PhysicalDeviceFeatures deviceFeatures{};
  auto &deviceFeatures = deviceCreateChains.get<vk::PhysicalDeviceFeatures2>();
  deviceFeatures.features.setSamplerAnisotropy(VK_TRUE);
  auto &device12Features =
      deviceCreateChains.get<vk::PhysicalDeviceVulkan12Features>();

  //不需要标量对齐方案
  //device12Features.setScalarBlockLayout(VK_TRUE);

  auto &createInfo = deviceCreateChains.get<vk::DeviceCreateInfo>();
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());

  // createInfo.pEnabledFeatures = &deviceFeatures;
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

  // 保持生命周期
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
  pipelineLayoutInfo.setSetLayouts(*m_descriptorSetlayout);

  // pipelineLayoutInfo.setSetLayouts(*m_descriptorSetLayout);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutInfo);

  pipelineFactory.m_pipelineLayout = *m_pipelineLayout;
  m_graphicsPipeline =
      pipelineFactory.buildPipeline(m_device, *(m_renderTarget->m_renderPass));
}

void VulkanRender::createFrameDatas() {
  m_frameDatas.reserve(MAX_FRAMES_IN_FLIGHT);
  App::QueueFamilyIndices queueFamilyIndices =
      App::VulkanInitializer::findQueueFamilies(*m_physicalDevice, *m_surface);

  vk::CommandPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    App::FrameData frameData{};

    frameData.commandPool = m_device.createCommandPool(poolInfo);

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = *frameData.commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;
    auto commandBuffers = m_device.allocateCommandBuffers(allocInfo);

    frameData.commandBuffer = std::move(commandBuffers.at(0));

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    frameData.availableSemaphore = m_device.createSemaphore(semaphoreInfo);
    frameData.renderFinishedSemaphore = m_device.createSemaphore(semaphoreInfo);
    frameData.inFlightFence = m_device.createFence(fenceInfo);

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
    bufferInfo.setSize(sizeof(App::GPUCameraData));
    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.flags = VmaAllocationCreateFlagBits::
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    frameData.cameraBuffer =
        m_vulkanMemory->createBuffer(bufferInfo, allocationInfo);

    vk::DescriptorSetAllocateInfo setAllocInfo{};
    setAllocInfo.setDescriptorPool(*m_descriptorPool);
    setAllocInfo.setDescriptorSetCount(1);
    setAllocInfo.setSetLayouts(*m_descriptorSetlayout);
    auto sets = m_device.allocateDescriptorSets(setAllocInfo);

    frameData.globalDescriptor = std::move(sets[0]);

    vk::DescriptorBufferInfo descriptorBufferInfo{};
    descriptorBufferInfo.setBuffer(frameData.cameraBuffer.get());
    descriptorBufferInfo.setOffset(0);
    descriptorBufferInfo.setRange(sizeof(App::GPUCameraData));

    namespace init = App::VulkanInitializer;

    std::array desCameraBufferInfos{descriptorBufferInfo};
    auto writeDescriptorSet = init::getWriteDescriptorSet(
        vk::DescriptorType::eUniformBuffer, *frameData.globalDescriptor,
        desCameraBufferInfos, 0);

    vk::DescriptorBufferInfo sceneBufferInfo{};

    sceneBufferInfo.setBuffer(m_sceneParaBuffer.get());
    sceneBufferInfo.setOffset(0);
    sceneBufferInfo.setRange(sizeof(App::GPUSceneData));

    std::array sceneBuffers{sceneBufferInfo};
    auto sceneWriteDes = init::getWriteDescriptorSet(
        vk::DescriptorType::eUniformBufferDynamic, *frameData.globalDescriptor,
        sceneBuffers, 1);

    constexpr int MAX_OBJECTS = 10000;
    vk::BufferCreateInfo objectBufferInfo{};
    objectBufferInfo.setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
    objectBufferInfo.setSize(sizeof(App::GPUObjectData) * MAX_OBJECTS);
    VmaAllocationCreateInfo objectAllocationInfo{};
    objectAllocationInfo.usage =
        VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    objectAllocationInfo.flags = VmaAllocationCreateFlagBits::
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    frameData.objectBuffer =
        m_vulkanMemory->createBuffer(objectBufferInfo, objectAllocationInfo);

    vk::DescriptorSetAllocateInfo objectSetAlloc{};
    objectSetAlloc.setDescriptorPool(*m_descriptorPool);
    objectSetAlloc.setSetLayouts(*m_objectSetLayout);

    auto objectSets = m_device.allocateDescriptorSets(objectSetAlloc);
    frameData.objectDescriptorSet = std::move(objectSets[0]);

    vk::DescriptorBufferInfo objectDesBufferInfo{};
    objectDesBufferInfo.setBuffer(frameData.objectBuffer.get());
    objectDesBufferInfo.setOffset(0);
    objectDesBufferInfo.setRange(sizeof(App::GPUObjectData) * MAX_OBJECTS);

    auto objectWrite = init::getWriteDescriptorSet(
        vk::DescriptorType::eStorageBuffer, *frameData.objectDescriptorSet,
        objectDesBufferInfo, 0);

    std::array writeDes{writeDescriptorSet, sceneWriteDes, objectWrite};
    m_device.updateDescriptorSets(writeDes, {});

    m_frameDatas.emplace_back(std::move(frameData));
  }
}

void VulkanRender::drawObjects(uint32_t frameIndex) {
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
  App::GPUCameraData camData{};
  camData.proj = projection;
  camData.view = lookAtView;
  camData.viewProj = projection * lookAtView;

  std::array camDatas{camData};

  m_vulkanMemory->upload(m_frameDatas[frameIndex].cameraBuffer,
                         std::span<App::GPUCameraData>(camDatas));

  m_sceneParameters.ambientColor = {std::sin(90.0 * frameIndex / 360), 0, 0, 1};

  std::array sceneParas{m_sceneParameters};

  m_vulkanMemory->upload(
      m_sceneParaBuffer, std::span<App::GPUSceneData>{sceneParas},
      getPadUniformBufferOffsetSize(sizeof(App::GPUSceneData)) * frameIndex);
}

void VulkanRender::drawFrame() {

  // auto startTime = chrono::high_resolution_clock().now();

  // updateUniformBuffer(m_currentFrame);

  using namespace std::literals::chrono_literals;
  auto onesecond = std::chrono::nanoseconds(1s);
  auto seconds = static_cast<uint64_t>(onesecond.count());

  // 可以和waitForFence做个异步
  drawObjects(m_currentFrame);

  auto result = m_device.waitForFences(
      *m_frameDatas[m_currentFrame].inFlightFence, VK_TRUE, seconds);
  if (result == vk::Result::eTimeout) {
    App::ThrowException(" wait fences time out");
  }
  m_device.resetFences(*m_frameDatas[m_currentFrame].inFlightFence);

  vk::AcquireNextImageInfoKHR acquireInfo{};

  // acquireInfo.swapchain = *m_swapChain;
  // acquireInfo.timeout = seconds;
  // acquireInfo.semaphore = *m_imageAvailableSemaphores[m_currentFrame];
  // acquireInfo.deviceMask = UINT32_MAX;
  // std::vector<std::future<uint32_t>> funtures;
  auto swapChain = *(m_renderTarget->m_swapChain);
  auto [acquireResult, imageIndex] = (*m_device).acquireNextImageKHR(
      swapChain, UINT64_MAX, *m_frameDatas[m_currentFrame].availableSemaphore);

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
  m_frameDatas[m_currentFrame].commandBuffer.reset();

  recordCommandBuffer(*m_frameDatas[m_currentFrame].commandBuffer, imageIndex);

  vk::SubmitInfo submitInfo{};

  std::array waitSemaphores = {
      *m_frameDatas[m_currentFrame].availableSemaphore};
  auto waitStages = std::to_array<vk::PipelineStageFlags>(
      {vk::PipelineStageFlagBits::eColorAttachmentOutput});
  std::array commandBuffers = {*m_frameDatas[m_currentFrame].commandBuffer};
  submitInfo.setWaitSemaphores(waitSemaphores);

  submitInfo.setWaitDstStageMask(waitStages);
  submitInfo.setCommandBuffers(commandBuffers);
  std::array signalSemaphores = {
      *m_frameDatas[m_currentFrame].renderFinishedSemaphore};
  submitInfo.setSignalSemaphores(signalSemaphores);
  m_graphicsQueue.submit(submitInfo,
                         *m_frameDatas[m_currentFrame].inFlightFence);

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
      // do nothing
    } else if (errorCode != vk::Result::eSuccess) {
      // do nothing
    }
  }

  m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

std::size_t
VulkanRender::getPadUniformBufferOffsetSize(std::size_t originSize) const {
  auto aligment = m_gpuProperties.limits.minUniformBufferOffsetAlignment;

  auto alignedSize = originSize;

  // 获取aligment的整数倍
  if (aligment > 0) {
    alignedSize = ((alignedSize + aligment - 1) / aligment) * aligment;
  }

  return alignedSize;
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

void VulkanRender::recordCommandBuffer(vk::CommandBuffer commandBuffer,
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

  commandBuffer.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, *m_pipelineLayout, 0,
      *m_frameDatas[m_currentFrame].globalDescriptor,
      m_currentFrame *
          getPadUniformBufferOffsetSize(sizeof(App::GPUSceneData)));

  // commandBuffer.drawIndexed(m_mesh.vertices.size(), 1, 0, 0, 0);

  // commandBuffer.drawIndexed(3, 1, 0, 0, 0);
  // glm::vec3 camPos = {0.f, 0.f, -2.f};
  // auto lookAtView =
  //     glm::lookAt(camPos, glm::vec3{0.f, 0.f, 0.f}, glm::vec3{0, -1, 0});

  // auto width = m_renderTarget->m_swapChainExtent.width;
  // auto height = m_renderTarget->m_swapChainExtent.height;
  // // // camera projection
  // glm::mat4 projection = glm::perspective(
  //     glm::radians(60.f), static_cast<float>(width) / height, 1.5f, 200.0f);
  // // // model rotation
  // glm::mat4 model =
  //     glm::rotate(glm::mat4{1.0f}, glm::radians(45.0f), glm::vec3(0, -1, 0));

  // auto big2 = projection * lookAtView ;
  // auto big2 = projection * lookAtView;

  App::MeshPushConstants constants = {};

  constants.renderMatrix = glm::mat4(1);

  commandBuffer.pushConstants<App::MeshPushConstants>(
      *m_pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, constants);

  // commandBuffer.draw(m_mesh.vertices.size(), 1, 0, 0);
  commandBuffer.draw(m_monkeyMesh.vertices.size(), 1, 0, 0);

  commandBuffer.endRenderPass();

  commandBuffer.end();
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

  // m_imageAvailableSemaphores.clear();
  // m_renderFinishedSemaphores.clear();
  // m_inFlightFences.clear();

  // m_commandBuffers.clear();

  // m_commandPools.clear();
  // m_swapChainFramebuffers.clear();
  m_frameDatas.clear();
  m_descriptorSetlayout.clear();
  m_objectSetLayout.clear();
  m_descriptorPool.clear();
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

  App::RenderTargetBuilder builder{m_device, *m_vulkanMemory};
  App::RenderTargetBuilder::CreateInfo info{};

  info.physicalDevice = *m_physicalDevice;
  info.renderExtent = vk::Extent2D{m_renderWidth, m_renderHeight};
  info.surface = *m_surface;

  m_renderTarget = builder.setCreateInfo(info).build();
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
