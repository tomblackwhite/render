#pragma once
#include "asset.hh"
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace App {

namespace raii = vk::raii;
// build pipeLine Factory
class PipelineFactory {
public:
  struct GraphicsPipelineCreateInfo {
    std::vector<vk::PipelineShaderStageCreateInfo> m_shaderStages;
    vk::PipelineVertexInputStateCreateInfo m_vertexInputInfo;
    vk::PipelineInputAssemblyStateCreateInfo m_inputAssembly;
    vk::PipelineDepthStencilStateCreateInfo m_depthStencilCreateInfo;
    vk::PipelineRasterizationStateCreateInfo m_rasterizer;
    vk::PipelineColorBlendAttachmentState m_colorBlendAttachment;
    vk::PipelineMultisampleStateCreateInfo m_multisampling;
    vk::PipelineLayout m_pipelineLayout;
  };
  raii::Pipeline createPipeline(GraphicsPipelineCreateInfo const &info);

  raii::ShaderModule createShaderModule(const std::span<unsigned char> code);

  explicit PipelineFactory(raii::Device *device, vk::RenderPass renderPass,
                           vk::Viewport const &viewPort,
                           vk::Rect2D const &m_scissor)
      : m_pDevice(device), m_renderPass(renderPass), m_viewPort(viewPort),
        m_scissor(m_scissor){};

private:
  raii::Device *m_pDevice{nullptr};
  vk::RenderPass m_renderPass;
  vk::Viewport m_viewPort;
  vk::Rect2D m_scissor;
};

namespace VulkanInitializer {
vk::PipelineShaderStageCreateInfo
getPipelineShaderStageCreateInfo(vk::ShaderStageFlagBits stage,
                                 vk::ShaderModule shaderModule);
vk::PipelineVertexInputStateCreateInfo
getPipelineVertexInputStateCreateInfo(App::VertexInputDescription const &);
vk::PipelineInputAssemblyStateCreateInfo
getPipelineInputAssemblyStateCreateInfo();
vk::PipelineRasterizationStateCreateInfo
getPipelineRasterizationStateCreateInfo();
vk::PipelineMultisampleStateCreateInfo getPipelineMultisampleStateCreateInfo();
vk::PipelineColorBlendAttachmentState getPipelineColorBlendAttachmentState();
vk::PipelineLayoutCreateInfo getPipelineLayoutCreateInfo();

vk::PipelineDepthStencilStateCreateInfo
getDepthStencilCreateInfo(bool depthTest, bool depthWrite,
                          vk::CompareOp compareOp);
vk::Viewport getViewPortInverseY(vk::Viewport const &viewPort);

} // namespace VulkanInitializer
} // namespace App
