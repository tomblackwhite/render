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
  //三角形列表模式
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
  pipelineInfo.pDepthStencilState = &m_depthStencilCreateInfo; // Optional
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

} // namespace App
