#include <render_target.hh>
namespace App {

void RenderTargetBuilder::buildSwapChainAndView() {

  auto physicalDevice = m_info.physicalDevice;
  auto surface = m_info.surface;
  auto renderExtent = m_info.renderExtent;

  vk::SwapchainCreateInfoKHR createInfo{};
  SwapChainSupportDetails swapChainSupport =
      VulkanInitializer::querySwapChainSupport(physicalDevice, surface);
  auto surfaceFormat =
      VulkanInitializer::chooseSwapSurfaceFormat(swapChainSupport.formats);
  auto presentMode =
      VulkanInitializer::chooseSwapPresentMode(swapChainSupport.presentModes);
  auto extent = VulkanInitializer::chooseSwapExtent(
      swapChainSupport.capabilities, renderExtent);
  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }
  createInfo.surface = surface;
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
  QueueFamilyIndices indices =
      VulkanInitializer::findQueueFamilies(physicalDevice, surface);
  std::array queueFamilyIndices = {indices.graphicsFamily.value(),
                                   indices.presentFamily.value()};

  if (indices.graphicsFamily != indices.presentFamily) {
    createInfo.imageSharingMode = vk::SharingMode::eConcurrent;

    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
  } else {
    createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    createInfo.queueFamilyIndexCount = 0;     // Optional
    createInfo.pQueueFamilyIndices = nullptr; // Optional
  }
  createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
  createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = *(m_product->m_swapChain);

  auto swapChain = m_device.createSwapchainKHR(createInfo);
  auto swapChainImages = (*m_device).getSwapchainImagesKHR(*swapChain);
  namespace init = App::VulkanInitializer;
  std::vector<raii::ImageView> swapChainViews;
  swapChainViews.reserve(swapChainImages.size());
  for (auto &image : swapChainImages) {
    auto imageViewInfo = init::getImageViewCreateInfo(
        createInfo.imageFormat, image, vk::ImageAspectFlagBits::eColor);
    auto imageView = m_device.createImageView(imageViewInfo);
    swapChainViews.emplace_back(std::move(imageView));
  }

  m_product->m_swapChain = std::move(swapChain);
  m_product->m_swapChainImages = std::move(swapChainImages);
  m_product->m_swapChainImageViews = std::move(swapChainViews);
  m_product->m_swapChainFormat = createInfo.imageFormat;
  m_product->m_swapChainExtent = createInfo.imageExtent;
}

void RenderTargetBuilder::buildDepthImageAndView() {

  vk::Extent3D depthImageExtent = {m_info.renderExtent.width,
                                   m_info.renderExtent.height, 1};
  auto format = vk::Format::eD32Sfloat;
  auto depthCreateInfo = App::VulkanInitializer::getImageCreateInfo(
      format, vk::ImageUsageFlagBits::eDepthStencilAttachment,
      depthImageExtent);

  VmaAllocationCreateInfo allocationInfo{};
  allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  allocationInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(
      vk::MemoryPropertyFlagBits::eDeviceLocal);
  auto depthImage = m_memory.createImage(depthCreateInfo, allocationInfo);
  auto depthImageViewInfo = App::VulkanInitializer::getImageViewCreateInfo(
      depthCreateInfo.format, depthImage.get(),
      vk::ImageAspectFlagBits::eDepth);
  auto depthImageView = m_device.createImageView(depthImageViewInfo);

  m_product->m_depthImage = std::move(depthImage);
  m_product->m_depthImageView = std::move(depthImageView);
  m_product->m_depthFormat = depthCreateInfo.format;
}

void RenderTargetBuilder::buildRenderPass() {
  vk::AttachmentDescription colorAttachment{};
  colorAttachment.format = m_product->m_swapChainFormat;
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

  vk::AttachmentDescription depthAttachment{};
  depthAttachment.setFormat(m_product->m_depthFormat);
  depthAttachment.setSamples(vk::SampleCountFlagBits::e1);
  depthAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
  depthAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
  depthAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
  depthAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
  depthAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
  depthAttachment.setFinalLayout(
      vk::ImageLayout::eDepthStencilAttachmentOptimal);

  vk::AttachmentReference depthAttachmentRef{};
  depthAttachmentRef.attachment = 1;
  depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

  vk::SubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

  vk::SubpassDependency depthDependency{};
  depthDependency.setSrcSubpass(VK_SUBPASS_EXTERNAL);
  depthDependency.setSrcStageMask(
      vk::PipelineStageFlagBits::eEarlyFragmentTests |
      vk::PipelineStageFlagBits::eLateFragmentTests);
  depthDependency.setDstStageMask(
      vk::PipelineStageFlagBits::eEarlyFragmentTests |
      vk::PipelineStageFlagBits::eLateFragmentTests);
  depthDependency.setDstAccessMask(
      vk::AccessFlagBits::eDepthStencilAttachmentWrite);

  std::array dependencies{dependency, depthDependency};

  vk::SubpassDescription subpass{};
  subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;
  subpass.setPDepthStencilAttachment(&depthAttachmentRef);

  std::array attachments{colorAttachment, depthAttachment};

  vk::RenderPassCreateInfo renderPassInfo{};
  renderPassInfo.setAttachments(attachments);
  renderPassInfo.setSubpasses(subpass);
  renderPassInfo.setDependencies(dependencies);

  m_product->m_renderPass = m_device.createRenderPass(renderPassInfo);
}

void RenderTargetBuilder::buildFrameBuffer() {
  std::vector<raii::Framebuffer> framebuffers;
  auto &swapChainViews = m_product->m_swapChainImageViews;
  auto &depthImageView = m_product->m_depthImageView;
  framebuffers.reserve(swapChainViews.size());
  for (auto &view : swapChainViews) {
    std::array attachments{*view, *depthImageView};
    vk::FramebufferCreateInfo info{};
    info.setRenderPass(*(m_product->m_renderPass));
    info.setLayers(1);
    info.setAttachments(attachments);
    info.setHeight(m_product->m_swapChainExtent.height);
    info.setWidth(m_product->m_swapChainExtent.width);
    framebuffers.emplace_back(m_device.createFramebuffer(info));
  }
  m_product->m_framebuffers = std::move(framebuffers);
}

vk::PresentModeKHR VulkanInitializer::chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  for (const auto &availablePresentMode : availablePresentModes) {
    if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
      return vk::PresentModeKHR::eMailbox;
    }
  }
  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D VulkanInitializer::chooseSwapExtent(
    const vk::SurfaceCapabilitiesKHR &capabilities, vk::Extent2D extent2d) {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {

    return capabilities.currentExtent;
  } else {
    uint32_t width = 0;
    uint32_t height = 0;

    height = extent2d.height;
    width = extent2d.width;

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
SwapChainSupportDetails
VulkanInitializer::querySwapChainSupport(const vk::PhysicalDevice &device,
                                         vk::SurfaceKHR surface) {

  SwapChainSupportDetails details;
  details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
  details.formats = device.getSurfaceFormatsKHR(surface);
  details.presentModes = device.getSurfacePresentModesKHR(surface);
  return details;
}

vk::SurfaceFormatKHR VulkanInitializer::chooseSwapSurfaceFormat(
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

struct QueueCountInFamily {
  uint32_t count = 0;
  std::vector<std::optional<uint32_t> *> countIndexInFamily;
};

QueueFamilyIndices
VulkanInitializer::findQueueFamilies(const vk::PhysicalDevice &device,
                                     vk::SurfaceKHR surface) {

  QueueFamilyIndices indices;

  auto queueFamilies = device.getQueueFamilyProperties();

  int index = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.graphicsFamily = index;
      indices.graphicsQueueIndex = 0;
    }

    if (queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) {
      indices.transferFamily = index;
      indices.transferQueueIndex = 0;
    }

    auto presentSupport = device.getSurfaceSupportKHR(index, surface);

    if (presentSupport == VK_TRUE) {
      indices.presentFamily = index;
      indices.presentQueueIndex = 0;
    }
    index++;
  }

  // //获得同一queueFamily下，需要的 queue 数量
  // using IndicesInFamily = std::vector<std::optional<uint32_t> *>;
  // std::vector<IndicesInFamily> queueCountInFamilyindices(queueFamilies.size());

  // queueCountInFamilyindices[indices.graphicsFamily.value()].push_back(
  //     &indices.graphicsQueueIndex);
  // queueCountInFamilyindices[indices.transferFamily.value()].push_back(
  //     &indices.transferQueueIndex);
  // queueCountInFamilyindices[indices.presentFamily.value()].push_back(
  //     &indices.presentQueueIndex);

  // //计算同一queueFamily下，如何求出queue index
  // for (uint i = 0; i < queueCountInFamilyindices.size(); ++i) {
  //   auto &queueIndices = queueCountInFamilyindices[i];
  //   auto const& queueFamily = queueFamilies[i];
  //   auto queueCount = queueFamily.queueCount;
  //   for (uint j = 0; j < queueIndices.size(); ++j) {
  //     *queueIndices[j] = j % queueCount;
  //   }
  // }

  // if (indices.transferFamily == indices.graphicsFamily) {
  //   if (queueFamilies[indices.graphicsFamily.value()].queueCount >= 2) {
  //     indices.transferQueueIndex = indices.graphicsQueueIndex.value() + 1;
  //   }

  //   if(indices.)
  // }

  return indices;
}

namespace VulkanInitializer {} // namespace VulkanInitializer

} // namespace App
