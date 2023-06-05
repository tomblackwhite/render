#pragma once
#include "asset.hh"
#include "gltf_type.hh"
#include "pipeline.hh"
#include "script_implement.hh"
#include <concepts>
#include <map>
#include <utility>

#include <boost/type_index.hpp>
#include <node.hh>
#include <spdlog/spdlog.h>

namespace App {

using namespace std::literals::string_view_literals;
namespace views = std::ranges::views;
namespace fs = std::filesystem;
using key = std::string;
using NodeShowMap = std::unordered_map<key, MeshInstance *>;
using MeshShowMap = std::map<Mesh *, std::vector<glm::mat4>>;

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
struct TextureFilter {
  vk::Filter magFilter;
  vk::Filter minFilter;

  bool operator<(TextureFilter const &right) const {
    if (minFilter < right.minFilter) {
      return true;
    } else if (minFilter == right.minFilter) {
      return magFilter < right.magFilter;
    } else {
      return false;
    }
  }
};
// 场景数据
struct GPUScene {

  GPUCamera camera;

  glm::mat4 model{1};

  // 顶点数据
  MeshShowMap meshShowMap;
  NodeShowMap showMap;
  GPUMeshBlock vertexBuffer;

  std::vector<vk::DescriptorSetLayoutBinding> bindings = getBindings();
  VulkanBufferHandle cameraBuffer{nullptr};
  raii::DescriptorSetLayout sceneSetLayout{nullptr};
  raii::DescriptorSet sceneSet{nullptr};

  raii::DescriptorSetLayout objectSetLayout{nullptr};
  raii::DescriptorSet objectSet{nullptr};

  raii::DescriptorSetLayout textureSetLayout{nullptr};
  std::map<TextureFilter, raii::Sampler> samplerMap;
  // raii::DescriptorSet textureSet{nullptr};

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

  // vk::PipelineLayoutCreateInfo getPipelineLayoutInfo() {
  //   vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};

  //   pipelineLayoutInfo.setPushConstantRanges(pushConstants);
  //   pipelineLayoutInfo.setSetLayouts();

  //   return pipelineLayoutInfo;
  // }
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

  void recordCommand(vk::CommandBuffer commandBuffer) const {
    recordCommandDetail(commandBuffer);
    // recordCommandTest(commandBuffer, showMap);
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

  explicit GPUScene(VulkanMemory *memory, PipelineFactory *factory,
                    const std::string &homePath) {

    // 初始化memory信息
    // 创建pipeline
    this->sceneSetLayout =
        memory->createDescriptorSetLayout(this->getSceneSetLayoutInfo());

    vk::DescriptorSetLayoutCreateInfo textureSetLayoutInfo{};
    auto textureBindings = getTextureBindings();
    textureSetLayoutInfo.setBindings(textureBindings);
    this->textureSetLayout =
        memory->createDescriptorSetLayout(textureSetLayoutInfo);

    vk::DescriptorSetLayoutCreateInfo objectSetLayoutInfo{};
    auto objectSetBindings = getObjectBindings();
    objectSetLayoutInfo.setBindings(objectSetBindings);
    this->objectSetLayout =
        memory->createDescriptorSetLayout(objectSetLayoutInfo);

    std::array setLayouts{*sceneSetLayout, *objectSetLayout, *textureSetLayout};
    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.setPushConstantRanges(pushConstants);
    layoutInfo.setSetLayouts(setLayouts);

    this->pipelineLayout = factory->createPipelineLayout(layoutInfo);
    this->pipeline = this->createScenePipeline(*factory, homePath);

    std::array setLayoutForCreate{*sceneSetLayout, *objectSetLayout};
    auto sets = memory->createDescriptorSet(setLayoutForCreate);
    this->sceneSet = std::move(sets[0]);
    this->objectSet = std::move(sets[1]);

    //  初始化scene 成员。创建cameraBuffer;
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer |
                        vk::BufferUsageFlagBits::eTransferDst);
    bufferInfo.setSize(sizeof(GPUCamera));
    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    this->cameraBuffer = memory->createBuffer(bufferInfo, allocationInfo);

    vk::DescriptorBufferInfo descriptorBufferInfo{};
    descriptorBufferInfo.setBuffer(this->cameraBuffer.get());
    descriptorBufferInfo.setOffset(0);
    descriptorBufferInfo.setRange(sizeof(GPUCamera));

    // 绑定descriptorSet
    vk::WriteDescriptorSet writeSet{};
    writeSet.setDstSet(*this->sceneSet);
    writeSet.setDstBinding(0);
    writeSet.setDescriptorType(vk::DescriptorType::eUniformBuffer);
    writeSet.setBufferInfo(descriptorBufferInfo);

    memory->updateDescriptorSets(
        writeSet, std::initializer_list<vk::CopyDescriptorSet>{});
  }

  // void init(VulkanMemory* memory){

  // }
private:
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

      //   commandBuffer.drawIndexed(subMesh.indices.size(), 1,
      //   currentIndexOffset,
      //                             currentVertexOffset, 0);
      //   currentIndexOffset += subMesh.indices.size();
      //   currentVertexOffset +=
      //   static_cast<int32_t>(subMesh.positions.size());
      // }
      currentMeshIndex += 1;
    }

    commandBuffer.draw(3, 1, 0, 0);
  }
  void recordCommandDetail(vk::CommandBuffer commandBuffer) const {
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
    for (auto const &[key, value] : meshShowMap) {
      auto &currentMesh = *key;

      commandBuffer.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, *pipelineLayout, 1, *(objectSet),
          vertexBuffer.objectOffsets[currentMeshIndex]);
      App::MeshPushConstants constants = {};

      // first x is mesh index
      constants.data = glm::uvec4(currentMeshIndex, 0, 0, 0);
      constants.renderMatrix = glm::mat4(1);

      // 上传object index
      commandBuffer.pushConstants<App::MeshPushConstants>(
          *pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, constants);

      // different subMesh
      for (auto &subMesh : currentMesh.subMeshs) {

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                         *pipelineLayout, 2,
                                         *(subMesh.material->textureSet), {});
        commandBuffer.drawIndexed(subMesh.indices.size(), value.size(),
                                  currentIndexOffset, currentVertexOffset, 0);
        currentIndexOffset += subMesh.indices.size();
        currentVertexOffset += static_cast<int32_t>(subMesh.positions.size());
      }
      currentMeshIndex += 1;
    }
  }

  static std::vector<vk::DescriptorSetLayoutBinding> getBindings() {

    // 设置camera绑定关系
    vk::DescriptorSetLayoutBinding cameraBinding{};
    cameraBinding.setBinding(0);
    cameraBinding.setDescriptorCount(1);
    cameraBinding.setStageFlags(vk::ShaderStageFlagBits::eVertex);
    cameraBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);

    return {cameraBinding};
  }

  static std::vector<vk::DescriptorSetLayoutBinding> getTextureBindings() {

    // 设置texture绑定关系
    vk::DescriptorSetLayoutBinding textureBinding{};
    textureBinding.setBinding(0);
    textureBinding.setDescriptorCount(2);
    textureBinding.setStageFlags(vk::ShaderStageFlagBits::eFragment);
    textureBinding.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
    return {textureBinding};
  }

  static std::vector<vk::DescriptorSetLayoutBinding> getObjectBindings() {

    // 设置object绑定关系
    vk::DescriptorSetLayoutBinding objectBinding{};
    objectBinding.setBinding(0);
    objectBinding.setDescriptorCount(1);
    objectBinding.setStageFlags(vk::ShaderStageFlagBits::eVertex);
    objectBinding.setDescriptorType(vk::DescriptorType::eStorageBufferDynamic);
    return {objectBinding};
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

public:
  // 只会被Scene修改
  // 供渲染使用.
  GPUScene m_scene;
};

class SceneFactory {
public:
  static void createScene(AssetManager &assetManager,
                          const std::string &sceneKey, NodeFactory *nodeFactory,
                          NodeContainer *container);

private:
  static void createNode(tinygltf::Node const &, tinygltf::Model &,
                         std::vector<Mesh> *meshes, NodeFactory *nodeFactory);

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
  static void createNodeTree(const tinygltf::Node &node, tinygltf::Model &model,
                             NodeMap &map, NodeTree &tree);

  // 创建Mesh
  // [[deprecated]]
  // static Mesh createMesh(int meshIndex, tinygltf::Model &model,
  //                        std::vector<tinygltf::Buffer> &buffers);

  static Mesh createMesh(tinygltf::Mesh &mesh, tinygltf::Model &model,
                         std::vector<Material> *materials) {
    Mesh result;
    auto &buffers = model.buffers;
    // 构建mesh
    auto vec3Attributes = std::to_array<std::string>({"POSITION", "NORMAL"});
    for (auto const &primitive : mesh.primitives) {

      // 不存在indice 不会显示。

      if (primitive.indices != -1) {

        Mesh::SubMesh subMesh;
        auto const &accessor = model.accessors[primitive.indices];
        auto indexBuffer = createSpanBuffer(accessor, model, buffers);

        if (std::holds_alternative<Mesh::IndexSpanType>(indexBuffer)) {
          subMesh.indices = std::get<Mesh::IndexSpanType>(indexBuffer);

          std::map<int, std::span<glm::vec2>> texCoords;
          // 构建submesh中的attributes
          auto const &attributes = primitive.attributes;
          for (auto const &attri : attributes) {

            auto &attriName = attri.first;

            auto const &accessor = model.accessors[attri.second];
            // 构建属性span buffer
            auto spanBuffer = createSpanBuffer(accessor, model, buffers);

            if (attriName == "POSITION" &&
                holds_alternative<std::span<glm::vec3>>(spanBuffer)) {

              subMesh.positions = std::get<std::span<glm::vec3>>(spanBuffer);
            } else if (attriName == "NORMAL" &&
                       holds_alternative<std::span<glm::vec3>>(spanBuffer)) {
              subMesh.normals = std::get<std::span<glm::vec3>>(spanBuffer);
            } else {
              auto viewStrings = views::split(attriName, "_"sv);
              std::vector<std::string_view> viewStrs(viewStrings.begin(),
                                                     viewStrings.end());
              if (viewStrs.size() == 2) {
                std::string indexStr(viewStrs[1]);
                auto indexNumber = std::stoi(indexStr);

                if (viewStrs[0] == "TEXCOORD" &&
                    holds_alternative<std::span<glm::vec2>>(spanBuffer)) {
                  texCoords.insert_or_assign(
                      indexNumber, std::get<std::span<glm::vec2>>(spanBuffer));
                }

              } else {

                spdlog::warn("{}'s type in asset is not match  {}'s type' ",
                             mesh.name, attriName);
              }
            }
          }

          // 设置texCoords
          for (auto &texCoordElem : texCoords) {
            subMesh.texCoords.push_back(texCoordElem.second);
          }

          if (primitive.material != -1) {

            auto &currentMat = model.materials.at(primitive.material);
            auto &pbrCurrent = currentMat.pbrMetallicRoughness;
            auto materialItera = materials->begin() + primitive.material;
            subMesh.material = &*materialItera;
          }

          result.subMeshs.push_back(subMesh);
          // result.indexCount+=subMesh.indices.size();
          // result.vertexCount+=subMesh.positions.size();
        } else {
          // index type 不匹配时
          auto indexType = vk::IndexTypeValue<IndexType>::value;
          spdlog::warn(
              "{}'s index type {} in gltf asset is not match  indices's "
              "type {} ,so don't appear\n",
              mesh.name, accessor.componentType, std::to_underlying(indexType));
        }
      } else {
        spdlog::warn(
            "{} primitive don't have indices attribute, so don't appear \n",
            mesh.name);
      }
    }
    return result;
  }

  static Image createImage(tinygltf::Image &image) {
    vk::Format imageFormat = vk::Format::eR8G8B8Sint;

    if (image.bits == 8) {
      if (image.component == 3) {
        imageFormat = vk::Format::eR8G8B8Sint;
      } else if (image.component == 4) {
        imageFormat = vk::Format::eR8G8B8A8Sint;
      }
    } else if (image.bits == 16) {
      if (image.component == 3) {
        imageFormat = vk::Format::eR16G16B16Sint;
      } else if (image.component == 4) {
        imageFormat = vk::Format::eR16G16B16A16Sint;
      }
    }

    Image result;

    result.extent.width = image.width;
    result.extent.height = image.height;
    result.extent.setDepth(1);
    result.format = imageFormat;

    result.data =
        std::span<unsigned char>(image.image.data(), image.image.size());
    return result;
  }

  static Texture createTexture(tinygltf::Texture &texture,
                               std::vector<Image> *images,
                               tinygltf::Model &model) {

    auto getFilter = [](int filterNumber) {
      switch (filterNumber) {
      case 9728:
        return vk::Filter::eNearest;
      case 9729:
        return vk::Filter::eLinear;
      default:
        return vk::Filter::eLinear;
      }
    };
    Texture result;

    if (texture.sampler != -1) {
      auto &sampler = model.samplers[texture.sampler];
      result.magFilter = getFilter(sampler.magFilter);
      result.minFilter = getFilter(sampler.minFilter);
    } else {
      result.magFilter = vk::Filter::eLinear;
      result.minFilter = vk::Filter::eLinear;
    }
    result.imageIterator = images->begin() + texture.source;
    result.format = result.imageIterator->format;

    return result;
  }

  static Material createMaterial(tinygltf::Material &material,
                                 std::vector<Texture> *textures) {

    Material result;
    auto &pbrCurrent = material.pbrMetallicRoughness;

    auto &subMeshPbr = result.pbr;
    for (int i = 0; i < 3; ++i) {
      subMeshPbr.baseColorFactor[i] =
          static_cast<float>(pbrCurrent.baseColorFactor[i]);
    }
    subMeshPbr.baseColorTexture =
        textures->begin() + pbrCurrent.baseColorTexture.index;
    subMeshPbr.baseColorCoordIndex = pbrCurrent.baseColorTexture.texCoord;

    if (pbrCurrent.metallicRoughnessTexture.index != -1) {

      subMeshPbr.metallicRoughnessTexture =
          textures->begin() + pbrCurrent.metallicRoughnessTexture.index;
      subMeshPbr.metallicRoughnessCoordIndex =
          pbrCurrent.metallicRoughnessTexture.texCoord;
    }

    subMeshPbr.metallicFactor = static_cast<float>(pbrCurrent.metallicFactor);
    subMeshPbr.roughnessFactor = static_cast<float>(pbrCurrent.roughnessFactor);

    return result;
  }

  // 创建Texture
  // [[deprecated]]
  // static Texture createTexture(const tinygltf::TextureInfo &info,
  //                              tinygltf::Model &model);

  static Camera *createCamera(const tinygltf::Camera &camera,
                              const std::string &name,
                              NodeFactory *nodeFactory);

  // 根据acessor 获取span
  static GlTFSpanVariantType
  createSpanBuffer(const tinygltf::Accessor &acessor,
                   const tinygltf::Model &model,
                   std::vector<tinygltf::Buffer> &buffers);
};
} // namespace App
