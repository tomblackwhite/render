#include <pipeline.hh>

namespace App {
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
  // 三角形列表模式
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
  rasterizer.cullMode = vk::CullModeFlagBits::eNone;
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

vk::PipelineDepthStencilStateCreateInfo
VulkanInitializer::getDepthStencilCreateInfo(bool depthTest, bool depthWrite,
                                             vk::CompareOp compareOp) {
  vk::PipelineDepthStencilStateCreateInfo info = {};
  info.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
  info.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
  info.depthCompareOp = depthTest ? compareOp : vk::CompareOp::eAlways;
  info.depthBoundsTestEnable = VK_FALSE;
  info.minDepthBounds = 0.0f; // Optional
  info.maxDepthBounds = 1.0f; // Optional
  info.stencilTestEnable = VK_FALSE;

  return info;
}

vk::Viewport
VulkanInitializer::getViewPortInverseY(vk::Viewport const &viewPort) {
  return {.x = viewPort.x,
          .y = viewPort.y + viewPort.height,
          .width = viewPort.width,
          .height = viewPort.height,
          .minDepth = viewPort.minDepth,
          .maxDepth = viewPort.maxDepth};
}

raii::ShaderModule
PipelineFactory::createShaderModule(const std::span<unsigned char> code) {
  vk::ShaderModuleCreateInfo createInfo{};
  createInfo.codeSize = code.size();

  createInfo.pCode =
      std::launder(reinterpret_cast<const uint32_t *>(code.data()));
  auto shaderModule = m_pDevice->createShaderModule(createInfo);
  return shaderModule;
}

raii::Pipeline PipelineFactory::createPipeline(const GraphicsPipelineCreateInfo & info) {
  vk::PipelineViewportStateCreateInfo viewportState = {};
  viewportState.setViewports(m_viewPort);
  viewportState.setScissors(m_scissor);

  vk::PipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = vk::LogicOp::eCopy; // Optional
  colorBlending.setAttachments(info.m_colorBlendAttachment);
  colorBlending.blendConstants[0] = 0.0f; // Optional
  colorBlending.blendConstants[1] = 0.0f; // Optional
  colorBlending.blendConstants[2] = 0.0f; // Optional
  colorBlending.blendConstants[3] = 0.0f; // Optional

  vk::GraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.setStages(info.m_shaderStages);
  pipelineInfo.pVertexInputState = &info.m_vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &info.m_inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &info.m_rasterizer;
  pipelineInfo.pMultisampleState = &info.m_multisampling;
  pipelineInfo.pDepthStencilState = &info.m_depthStencilCreateInfo; // Optional
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = nullptr; // Optional
  pipelineInfo.layout = info.m_pipelineLayout;

  pipelineInfo.renderPass = m_renderPass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
  pipelineInfo.basePipelineIndex = -1;

  auto graphicsPipeline =
      m_pDevice->createGraphicsPipeline(nullptr, pipelineInfo);

  return graphicsPipeline;
}

} // namespace App
