#pragma once
#include "asset.hh"
#include "gltf_type.hh"
#include "pipeline.hh"
#include "script_implement.hh"
#include <concepts>
#include <utility>

#include <boost/type_index.hpp>
#include <node.hh>
#include <spdlog/spdlog.h>

namespace App {

namespace views = std::ranges::views;
namespace fs = std::filesystem;
using key = std::string;
using NodeShowMap = std::unordered_map<key, MeshInstance *>;


struct GPUCamera {
  glm::mat4 view{1};
  glm::mat4 proj{1};
  glm::mat4 viewProj{1};
};
struct MeshPushConstants {
  glm::uvec4 data;
  glm::mat4 renderMatrix;
};

// struct Object {
//   glm::mat4 model{1};
// };
// 场景数据
struct Scene {

  GPUCamera camera;

  glm::mat4 model{1};

  // 顶点数据
  NodeShowMap showMap;
  VertexBuffer vertexBuffer;

  std::vector<vk::DescriptorSetLayoutBinding> bindings = getBindings();
  VulkanBufferHandle cameraBuffer{nullptr};
  VulkanBufferHandle objectBuffer{nullptr};
  raii::DescriptorSetLayout sceneSetLayout{nullptr};
  raii::DescriptorSet sceneSet{nullptr};

  std::vector<vk::PushConstantRange> pushConstants = getPushConstantranges();
  raii::PipelineLayout pipelineLayout{nullptr};

  raii::Pipeline pipeline{nullptr};

  uint32_t cameraBinding = 0;
  uint32_t objectBinding = 1;

  std::filesystem::path vertShaderPath = "shader/object.vert.spv";
  std::filesystem::path fragShaderPath = "shader/object.frag.spv";

  vk::DescriptorSetLayoutCreateInfo getSceneSetLayoutInfo() {

    // std::array bindings;

    vk::DescriptorSetLayoutCreateInfo setInfo{};
    setInfo.setBindings(bindings);

    return setInfo;
  }

  vk::PipelineLayoutCreateInfo getPipelineLayoutInfo() {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};

    pipelineLayoutInfo.setPushConstantRanges(pushConstants);
    pipelineLayoutInfo.setSetLayouts(*sceneSetLayout);

    return pipelineLayoutInfo;
  }
  raii::Pipeline createScenePipeline(PipelineFactory &factory,
                                     fs::path const &homePath) const {

    auto vertShaderCode = readFile(homePath / vertShaderPath);
    auto fragShaderCode = readFile(homePath / fragShaderPath);
    auto vertShaderModule = factory.createShaderModule(vertShaderCode);
    auto fragShaderModule = factory.createShaderModule(fragShaderCode);

    using namespace App;

    PipelineFactory::GraphicsPipelineCreateInfo info{};

    // pipelineFactory
    info.m_shaderStages.push_back(
        VulkanInitializer::getPipelineShaderStageCreateInfo(
            vk::ShaderStageFlagBits::eVertex, *vertShaderModule));

    info.m_shaderStages.push_back(
        VulkanInitializer::getPipelineShaderStageCreateInfo(
            vk::ShaderStageFlagBits::eFragment, *fragShaderModule));

    // 保持生命周期
    auto inputDescriptor = App::Mesh::SubMesh::getVertexDescription();
    info.m_vertexInputInfo =
        VulkanInitializer::getPipelineVertexInputStateCreateInfo(
            inputDescriptor);

    info.m_inputAssembly =
        VulkanInitializer::getPipelineInputAssemblyStateCreateInfo();

    info.m_rasterizer =
        VulkanInitializer::getPipelineRasterizationStateCreateInfo();

    info.m_multisampling =
        VulkanInitializer::getPipelineMultisampleStateCreateInfo();

    info.m_colorBlendAttachment =
        VulkanInitializer::getPipelineColorBlendAttachmentState();
    info.m_depthStencilCreateInfo =
        VulkanInitializer::getDepthStencilCreateInfo(
            true, true, vk::CompareOp::eLessOrEqual);

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eLineWidth,
    };

    info.m_pipelineLayout = *pipelineLayout;

    return factory.createPipeline(info);
  }

  void recordCommand(vk::CommandBuffer commandBuffer,
                     NodeShowMap const &showMap) const {
    recordCommandDetail(commandBuffer, showMap);
    // recordCommandTest(commandBuffer, showMap);
  }
  void recordCommandDetail(vk::CommandBuffer commandBuffer,
                     NodeShowMap const &showMap) const {
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                     *pipelineLayout, 0, *sceneSet, {});

    // bind vertex
    auto vertexBufferView =
        views::all(vertexBuffer.buffers) |
        views::transform([](auto &elem) { return elem.get(); });

    std::vector<vk::Buffer> vertexBufferTemp(vertexBufferView.begin(),
                                             vertexBufferView.end());

    if (vertexBufferTemp.empty()) {
      return;
    }
    std::vector<vk::DeviceSize> vertexOffsets(vertexBufferTemp.size(), 0);
    commandBuffer.bindVertexBuffers(
        0,
        std::vector<vk::Buffer>(vertexBufferView.begin(),
                                vertexBufferView.end()),
        std::vector<vk::DeviceSize>(vertexBufferTemp.size(), 0));

    // bind vertex index
    commandBuffer.bindIndexBuffer(vertexBuffer.indexBuffer.get(), 0,
                                  vk::IndexTypeValue<App::IndexType>::value);

    int32_t currentVertexOffset = 0;
    uint32_t currentIndexOffset = 0;
    uint32_t currentMeshIndex = 0;
    for (auto const &[key, value] : showMap) {
      auto &currentMesh = value->mesh;

      App::MeshPushConstants constants = {};

      // first x is mesh index
      constants.data = glm::uvec4(currentMeshIndex, 0, 0, 0);
      auto matrix= value->modelMatrix;
      constants.renderMatrix = glm::mat4(1);

      // 上传object index
      commandBuffer.pushConstants<App::MeshPushConstants>(
          *pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, constants);

      // different subMesh
      for (auto &subMesh : currentMesh.subMeshs) {

        commandBuffer.drawIndexed(subMesh.indices.size(), 1, currentIndexOffset,
                                  currentVertexOffset, 0);
        currentIndexOffset += subMesh.indices.size();
        currentVertexOffset += static_cast<int32_t>(subMesh.positions.size());
      }
      currentMeshIndex += 1;
    }
  }

  raii::Pipeline createScenePipelineTest(PipelineFactory &factory,
                                     fs::path const &homePath) const {

    auto vertShaderCode = readFile(homePath / vertShaderPath);
    auto fragShaderCode = readFile(homePath / fragShaderPath);
    auto vertShaderModule = factory.createShaderModule(vertShaderCode);
    auto fragShaderModule = factory.createShaderModule(fragShaderCode);

    using namespace App;

    PipelineFactory::GraphicsPipelineCreateInfo info{};

    // pipelineFactory
    info.m_shaderStages.push_back(
        VulkanInitializer::getPipelineShaderStageCreateInfo(
            vk::ShaderStageFlagBits::eVertex, *vertShaderModule));

    info.m_shaderStages.push_back(
        VulkanInitializer::getPipelineShaderStageCreateInfo(
            vk::ShaderStageFlagBits::eFragment, *fragShaderModule));

    // 保持生命周期
    auto inputDescriptor = App::Mesh::SubMesh::getVertexDescription();
    info.m_vertexInputInfo =
        VulkanInitializer::getPipelineVertexInputStateCreateInfo(
            inputDescriptor);

    info.m_inputAssembly =
        VulkanInitializer::getPipelineInputAssemblyStateCreateInfo();

    info.m_rasterizer =
        VulkanInitializer::getPipelineRasterizationStateCreateInfo();

    info.m_multisampling =
        VulkanInitializer::getPipelineMultisampleStateCreateInfo();

    info.m_colorBlendAttachment =
        VulkanInitializer::getPipelineColorBlendAttachmentState();
    info.m_depthStencilCreateInfo =
        VulkanInitializer::getDepthStencilCreateInfo(
            true, true, vk::CompareOp::eLessOrEqual);

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eLineWidth,
    };

    info.m_pipelineLayout = *pipelineLayout;

    return factory.createPipeline(info);
  }
  void recordCommandTest(vk::CommandBuffer commandBuffer,
                     NodeShowMap const &showMap) const {
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                     *pipelineLayout, 0, *sceneSet, {});

    // bind vertex
    auto vertexBufferView =
        views::all(vertexBuffer.buffers) |
        views::transform([](auto &elem) { return elem.get(); });

    std::vector<vk::Buffer> vertexBufferTemp(vertexBufferView.begin(),
                                             vertexBufferView.end());

    if (vertexBufferTemp.empty()) {
      return;
    }
    std::vector<vk::DeviceSize> vertexOffsets(vertexBufferTemp.size(), 0);
    commandBuffer.bindVertexBuffers(
        0,
        std::vector<vk::Buffer>(vertexBufferView.begin(),
                                vertexBufferView.end()),
        std::vector<vk::DeviceSize>(vertexBufferTemp.size(), 0));

    // bind vertex index
    commandBuffer.bindIndexBuffer(vertexBuffer.indexBuffer.get(), 0,
                                  vk::IndexTypeValue<App::IndexType>::value);

    int32_t currentVertexOffset = 0;
    uint32_t currentIndexOffset = 0;
    uint32_t currentMeshIndex = 0;
    for (auto const &[key, value] : showMap) {
      auto &currentMesh = value->mesh;

      App::MeshPushConstants constants = {};

      // first x is mesh index
      constants.data = glm::uvec4(currentMeshIndex, 0, 0, 0);
      constants.renderMatrix = value->modelMatrix;

      // 上传object index
      commandBuffer.pushConstants<App::MeshPushConstants>(
          *pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, constants);

      // different subMesh
      // for (auto &subMesh : currentMesh.subMeshs) {

      //   commandBuffer.drawIndexed(subMesh.indices.size(), 1, currentIndexOffset,
      //                             currentVertexOffset, 0);
      //   currentIndexOffset += subMesh.indices.size();
      //   currentVertexOffset += static_cast<int32_t>(subMesh.positions.size());
      // }
      currentMeshIndex += 1;
    }

    commandBuffer.draw(3, 1, 0, 0);
  }
private:
  static std::vector<vk::DescriptorSetLayoutBinding> getBindings() {

    // 设置camera绑定关系
    vk::DescriptorSetLayoutBinding cameraBinding{};
    cameraBinding.setBinding(0);
    cameraBinding.setDescriptorCount(1);
    cameraBinding.setStageFlags(vk::ShaderStageFlagBits::eVertex);
    cameraBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);

    vk::DescriptorSetLayoutBinding objectBinding{};
    objectBinding.setBinding(1);
    objectBinding.setDescriptorCount(1);
    objectBinding.setStageFlags(vk::ShaderStageFlagBits::eVertex);
    objectBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
    return {cameraBinding, objectBinding};
  }

  static std::vector<vk::PushConstantRange> getPushConstantranges() {

    vk::PushConstantRange pushConstant = {};
    pushConstant.setOffset(0);
    pushConstant.setSize(sizeof(App::MeshPushConstants));
    pushConstant.setStageFlags(vk::ShaderStageFlagBits::eVertex);
    return {pushConstant};
  }
};

// 分离创建职责，区分素材管理和场景创建原则
class SceneFactory;

// 控制整个场景树。
class SceneManager {

public:
  explicit SceneManager(VulkanMemory *memory, PipelineFactory *factory,
                        const std::string &homePath);
  // 初始化游戏流程脚本之类的,绑定各种内容。
  void init();
  // 每帧更新
  void update();

  void physicalUpdate();

  // 设置主场景
  std::string mainScene() { return m_mainScene; }
  void setMainScene(const string &scene) { m_mainScene = scene; }

  // 只会被Scene修改
  // 供渲染使用.
  Scene m_scene;

private:
  // 加载场景
  void loadScene(const string &scene);

  // 绑定脚本
  void bindScript();

  // 显示场景 只用于第一次显示
  void showScene(const string &scene);

  // 节点遍历
  void visitNode(const string &key, std::function<void(Node *)> const &visitor);

  // 计算当前节点变换
  glm::mat4 getTransform(Node *node);

  std::string m_mainScene;
  // which data struct ?
  NodeContainer m_nodeContainer;
  NodeFactory m_nodeFactory{&m_nodeContainer};
  // NodeMap m_map{};
  // NodeTree m_tree{};

  AssetManager m_assetManager;

  VulkanMemory *m_vulkanMemory;
  PipelineFactory *m_pipelineFactory;
  std::filesystem::path m_homePath;

  std::unique_ptr<SceneFactory> m_factory;
};

class SceneFactory {
public:
  static void createScene(AssetManager &assetManager,
                          const std::string &sceneKey, NodeFactory *nodeFactory,NodeContainer * container);

private:
  static void
  createNode(tinygltf::Node const &, tinygltf::Model const &,
             std::vector<tinygltf::Buffer> &buffers,NodeFactory *nodeFactory);

  template <typename T, typename Param, std::size_t... Ints>
    requires IsAnyOf<T, glm::vec3, glm::quat, glm::mat4> &&
             requires(Param &param, std::size_t n) {
               { auto(param[n]) } -> std::floating_point;
             }
  static T castToGLMType(Param const &param,
                         std::index_sequence<Ints...> /*unused*/) {
    return T{static_cast<float>(param[Ints])...};
  }

  // 递归构建Node Tree
  static void createNodeTree(const tinygltf::Node &node,
                             const tinygltf::Model &model, NodeMap &map,
                             NodeTree &tree);

  // 创建Mesh
  static Mesh createMesh(int meshIndex, const tinygltf::Model &model,
                         std::vector<tinygltf::Buffer> &buffers);

  static Camera* createCamera(const tinygltf::Camera &camera, const std::string &name,
                                  NodeFactory *nodeFactory);

  // 根据acessor 获取span
  static GlTFSpanVariantType
  createSpanBuffer(const tinygltf::Accessor &acessor,
                   const tinygltf::Model &model,
                   std::vector<tinygltf::Buffer> &buffers);
};
} // namespace App
