#pragma once
#include <vulkan/vulkan.hpp>
#include "asset.hh"
#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace App {

namespace raii = vk::raii;
  // build pipeLine Factory
class PipelineFactory {
public:
  std::vector<vk::PipelineShaderStageCreateInfo> m_shaderStages;
  vk::PipelineVertexInputStateCreateInfo m_vertexInputInfo;
  vk::PipelineInputAssemblyStateCreateInfo m_inputAssembly;
  vk::PipelineDepthStencilStateCreateInfo m_depthStencilCreateInfo;
  vk::Viewport m_viewPort;
  vk::Rect2D m_scissor;
  vk::PipelineRasterizationStateCreateInfo m_rasterizer;
  vk::PipelineColorBlendAttachmentState m_colorBlendAttachment;
  vk::PipelineMultisampleStateCreateInfo m_multisampling;
  vk::PipelineLayout m_pipelineLayout;

  raii::Pipeline buildPipeline(raii::Device const &device, vk::RenderPass pass);
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
   vk::PipelineMultisampleStateCreateInfo
  getPipelineMultisampleStateCreateInfo();
   vk::PipelineColorBlendAttachmentState
  getPipelineColorBlendAttachmentState();
   vk::PipelineLayoutCreateInfo getPipelineLayoutCreateInfo();

   vk::PipelineDepthStencilStateCreateInfo
  getDepthStencilCreateInfo(bool depthTest, bool depthWrite,
                            vk::CompareOp compareOp);


}
}
