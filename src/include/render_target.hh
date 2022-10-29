#pragma once
#include "asset.hh"
#include <optional>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
namespace App {

namespace raii = vk::raii;

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

//多个参数最好传结构体，避免传多个参数，为了把各个抽象放在一起
class RenderTarget {
public:
  friend class RenderTargetBuilder;

private:
  RenderTarget() noexcept = default;

public:
  raii::SwapchainKHR m_swapChain{nullptr};
  std::vector<vk::Image> m_swapChainImages;
  std::vector<raii::ImageView> m_swapChainImageViews;
  vk::Format m_swapChainFormat{};
  vk::Extent2D m_swapChainExtent;
  vk::Format m_depthFormat{};
  App::VulkanImageHandle m_depthImage{};
  raii::ImageView m_depthImageView{nullptr};
  raii::RenderPass m_renderPass{nullptr};
  std::vector<raii::Framebuffer> m_framebuffers;
};

class RenderTargetBuilder {
public:
  explicit RenderTargetBuilder(const raii::Device &device,
                               App::VulkanMemory const &memory)
      : m_device(device), m_memory(memory) {}

  using Product = std::unique_ptr<RenderTarget>;
  struct CreateInfo {
    vk::PhysicalDevice physicalDevice;
    vk::SurfaceKHR surface;
    vk::Extent2D renderExtent;
  };
  RenderTargetBuilder &setCreateInfo(const CreateInfo &info) {
    m_info = info;
    return *this;
  }

  Product build() {
    buildSwapChainAndView();
    buildDepthImageAndView();
    buildRenderPass();
    buildFrameBuffer();
    return std::move(m_product);
  }

protected:
  void buildSwapChainAndView();
  void buildDepthImageAndView();
  void buildRenderPass();
  void buildFrameBuffer();

private:
  raii::Device const &m_device;
  App::VulkanMemory const &m_memory;
  Product m_product{new RenderTarget{}};
  CreateInfo m_info{};
};

namespace VulkanInitializer {

SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice &device,
                                              vk::SurfaceKHR surface);

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR> &availableFormats);

vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> &availablePresentModes);

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities,
                              vk::Extent2D extent2d);
QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice &device,
                                     vk::SurfaceKHR surface);


} // namespace VulkanInitializer

} // namespace App
