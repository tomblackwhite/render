#include "scene.hh"
namespace App {
using glm::vec3;
using std::holds_alternative;
using std::index_sequence;
using std::initializer_list;

SceneManager::SceneManager(VulkanMemory *memory, PipelineFactory *factory,
                           const std::string &homePath)
    : m_vulkanMemory(memory), m_pipelineFactory(factory), m_homePath(homePath) {

  //创建pipeline
  m_scene.sceneSetLayout = m_vulkanMemory->createDescriptorSetLayout(
      m_scene.getSceneSetLayoutInfo());
  m_scene.pipelineLayout =
      m_pipelineFactory->createPipelineLayout(m_scene.getPipelineLayoutInfo());
  m_scene.pipeline=m_scene.createScenePipeline(*m_pipelineFactory, m_homePath);


  // 可以放在scene构造函数里，现在相当于二段初始化。
  //  初始化scene 成员。创建cameraBuffer;
  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer |
                      vk::BufferUsageFlagBits::eTransferDst);
  bufferInfo.setSize(sizeof(GPUCamera));
  VmaAllocationCreateInfo allocationInfo{};
  allocationInfo.usage = VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  allocationInfo.flags = VmaAllocationCreateFlagBits::
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  m_scene.cameraBuffer =
      m_vulkanMemory->createBuffer(bufferInfo, allocationInfo);


  auto sets = m_vulkanMemory->createDescriptorSet(*m_scene.sceneSetLayout);
  m_scene.sceneSet = std::move(sets[0]);

  vk::DescriptorBufferInfo descriptorBufferInfo{};
  descriptorBufferInfo.setBuffer(m_scene.cameraBuffer.get());
  descriptorBufferInfo.setOffset(0);
  descriptorBufferInfo.setRange(sizeof(GPUCamera));

  // 绑定descriptorSet
  vk::WriteDescriptorSet writeSet{};
  writeSet.setDstSet(*m_scene.sceneSet);
  writeSet.setDstBinding(0);
  writeSet.setDescriptorType(vk::DescriptorType::eUniformBuffer);
  writeSet.setBufferInfo(descriptorBufferInfo);

  m_vulkanMemory->updateDescriptorSets(
      writeSet, initializer_list<vk::CopyDescriptorSet>{});

}

void SceneManager::init() {

  loadScene(m_mainScene);

  // 初始化脚本
  ranges::for_each(m_map, [](auto const &keyValue) {
    auto &[key, node] = keyValue;

    auto *pscript = node->script();
    if (pscript) {
      pscript->init();
    }
  });

  // 执行脚本后根据node状态准备showScene
  showScene(m_mainScene);
}

void SceneManager::update() {

  ranges::for_each(m_map, [](auto const &keyValue) {
    auto &[key, node] = keyValue;
    auto *pscript = node->script();
    if (pscript) {
      pscript->update();
    }
  });
}

void SceneManager::physicalUpdate() {

  ranges::for_each(m_map, [](auto const &keyValue) {
    auto &[key, node] = keyValue;

    auto *pscript = node->script();
    if (pscript) {
      pscript->physicalUpdate();
    }
  });
}

void SceneManager::loadScene(const string &scene) {
  m_factory->createScene(m_assetManager, scene, m_map, m_tree);
}

void SceneManager::showScene(const string &scene) {

  auto visitor = [&showMap = this->m_scene.showMap,
                  &vulkanMemory = this->m_vulkanMemory, &scene = this->m_scene,
                  this](Node *pNode) {
    // 是否显示
    if ((pNode != nullptr) && pNode->visible()) {
      if (auto *mesh = dynamic_cast<MeshInstance *>(pNode); mesh) {

        showMap.insert_or_assign(mesh->name, mesh);
      }
      if (auto *camera = dynamic_cast<Camera *>(pNode); camera) {
        scene.camera.proj = camera->projection;
        scene.camera.view = this->getTransform(camera);
        scene.camera.viewProj = scene.camera.proj * scene.camera.view;
      }
    }
  };

  visitNode(m_mainScene, visitor);

  auto meshsView =
      views::all(m_scene.showMap) |
      views::transform([](auto &elem) -> Mesh & { return elem.second->mesh; });

#ifdef DEBUG
  // debug info
  auto meshCount = meshsView.size();
#endif
  // 上传到gpu
  m_scene.vertexBuffer = m_vulkanMemory->uploadMeshes(meshsView);

  // 上传camera
  m_vulkanMemory->upload(m_scene.cameraBuffer,
                         views::single(std::span(&m_scene.camera, 1)));
}

glm::mat4 SceneManager::getTransform(Node *node) {
  glm::mat4 trans{node->transform()};

  auto parentKey = node->parent();
  for (; !parentKey.empty();) {
    if (auto iter = m_map.find(parentKey); iter != m_map.end()) {
      trans = iter->second->transform() * trans;
      parentKey = iter->second->parent();
    } else {
      //
      spdlog::warn("some node lost so set transform to default");
      return glm::mat4{1};
    }
  }
  return trans;
}

void SceneManager::visitNode(const string &key,
                             std::function<void(Node *)> const &visitor) {
  auto children = m_tree[key];

  for (auto &elemKey : children) {
    auto &elem = m_map[elemKey];
    visitor(elem.get());
    visitNode(elemKey, visitor);
  }
}

void SceneFactory::createScene(AssetManager &assetManager,
                               const std::string &sceneKey, NodeMap &map,
                               NodeTree &tree) {

  auto &model = assetManager.getScene(sceneKey);
  auto &scene = model.scenes[model.defaultScene];

  for (auto &node : model.nodes) {
    map.insert({node.name, createNode(node, model, model.buffers)});
  }
  std::vector<std::string> sceneNodes;
  // 递归构建Node Tree
  for (auto index : scene.nodes) {
    auto &node = model.nodes[index];
    sceneNodes.push_back(node.name);
    createNodeTree(node, model, map, tree);
  }

  // sceneKey as root of scene nodes
  tree.insert({sceneKey, std::move(sceneNodes)});
}

std::unique_ptr<Node>
SceneFactory::createNode(tinygltf::Node const &node,
                         tinygltf::Model const &model,
                         std::vector<tinygltf::Buffer> &buffers) {

  auto result = std::unique_ptr<Node>(nullptr);

  // 创建mesh
  if (node.mesh != -1) {
    auto meshIns = std::make_unique<MeshInstance>();
    meshIns->mesh = createMesh(node.mesh, model, buffers);
    result = std::move(meshIns);
  } else if (node.camera != -1) {
    auto camera =
        std::make_unique<Camera>(createCamera(model.cameras[node.camera]));
    result = std::move(camera);

  } else {
    result = std::make_unique<Node>();
  }

  result->name = node.name;

  if (!node.scale.empty()) {
    auto scale =
        castToGLMType<glm::vec3>(node.scale, index_sequence<0, 1, 2>{});
    result->setScale(scale);
  }
  if (!node.rotation.empty()) {

    auto quat =
        castToGLMType<glm::quat>(node.rotation, index_sequence<3, 0, 1, 2>{});
    result->setRotation(quat);
  }

  if (!node.translation.empty()) {

    auto translation =
        castToGLMType<glm::vec3>(node.translation, index_sequence<0, 1, 2>{});
    result->setTranslation(translation);
  }
  if (!node.matrix.empty()) {
    auto matrix =
        castToGLMType<glm::mat4>(node.matrix, std::make_index_sequence<16>{});
    result->setTransform(matrix);
  }

  return result;
}

void SceneFactory::createNodeTree(const tinygltf::Node &node,
                                  const tinygltf::Model &model, NodeMap &map,
                                  NodeTree &tree) {
  if (node.children.empty()) {
    return;
  }
  NodeTree::value_type keyValue{node.name, NodeTree::mapped_type{}};
  tree.insert(keyValue);

  auto &value = keyValue.second;

  // 构建场景路径。
  auto &currentNode = map[node.name];

  for (auto index : node.children) {

    // 设置node 中路径信息。
    auto const &nodeKey = model.nodes[index].name;
    auto &childrenNode = map[nodeKey];
    childrenNode->parent() = node.name;

    currentNode->childern().push_back(nodeKey);
    value.push_back(model.nodes[index].name);
    createNodeTree(model.nodes[index], model, map, tree);
  }
}

Mesh SceneFactory::createMesh(int meshIndex, const tinygltf::Model &model,
                              std::vector<tinygltf::Buffer> &buffers) {

  Mesh result;
  auto &mesh = model.meshes[meshIndex];

  // 构建mesh
  auto vec3Attributes = std::to_array<std::string>({"POSITION", "NORMAL"});
  for (auto const &primitive : mesh.primitives) {

    // 不存在indice 不会显示。

    if (primitive.indices != -1) {

      Mesh::SubMesh subMesh;
      auto const &accessor = model.accessors[primitive.indices];
      auto spanBuffer = createSpanBuffer(accessor, model, buffers);

      if (std::holds_alternative<Mesh::IndexSpanType>(spanBuffer)) {
        subMesh.indices = std::get<Mesh::IndexSpanType>(spanBuffer);
        // 构建submesh中的attributes
        auto const &attributes = primitive.attributes;
        for (auto const &attri : attributes) {

          if (auto iter = ranges::find(vec3Attributes, attri.first);
              iter != ranges::end(vec3Attributes)) {

            auto const &accessor = model.accessors[attri.second];
            // 构建属性span buffer
            auto spanBuffer = createSpanBuffer(accessor, model, buffers);

            if (*iter == "POSITION" &&
                holds_alternative<std::span<glm::vec3>>(spanBuffer)) {

              subMesh.positions = std::get<std::span<glm::vec3>>(spanBuffer);
            } else if (*iter == "NORMAL" &&
                       holds_alternative<std::span<glm::vec3>>(spanBuffer)) {
              subMesh.normals = std::get<std::span<glm::vec3>>(spanBuffer);
            } // else if (*iter == "NORMAL" &&
            //            holds_alternative<std::span<glm::vec3>>(spanBuffer)) {

            // }
            else {
              spdlog::warn("{}'s type in asset is not match  {}'s type' ",
                           mesh.name, *iter);
            }
          }
        }

        result.subMeshs.push_back(subMesh);
        // result.indexCount+=subMesh.indices.size();
        // result.vertexCount+=subMesh.positions.size();
      } else {
        // index type 不匹配时
        auto indexType = vk::IndexTypeValue<IndexType>::value;
        spdlog::warn("{}'s index type {} in gltf asset is not match  indices's "
                     "type {} ,so don't appear\n",
                     mesh.name, accessor.componentType,
                     std::to_underlying(indexType));
      }
    } else {
      spdlog::warn(
          "{} primitive don't have indices attribute, so don't appear \n",
          mesh.name);
    }
  }
  return result;
}

Camera SceneFactory::createCamera(const tinygltf::Camera &camera) {
  Camera result;
  if (camera.type == "perspective") {
    auto const &perpective = camera.perspective;

    // finite
    if (perpective.zfar != 0.0) {

      result.projection =
          glm::perspective(perpective.yfov, perpective.aspectRatio,
                           perpective.znear, perpective.zfar);
    } else {
      result.projection = glm::infinitePerspective(
          perpective.yfov, perpective.aspectRatio, perpective.znear);
    }
  } else if (camera.type == "orthographic") {
    auto const &orth = camera.orthographic;
    result.projection = glm::ortho(0.0, 2 * orth.xmag, 0.0, 2 * orth.ymag,
                                   orth.znear, orth.zfar);
  } else {
    // do nothing
    spdlog::warn("camera don't have type");
  }
  return result;
}
GlTFSpanVariantType
SceneFactory::createSpanBuffer(const tinygltf::Accessor &acessor,
                               const tinygltf::Model &model,
                               std::vector<tinygltf::Buffer> &buffers) {

  auto const &bufferView = model.bufferViews[acessor.bufferView];
  auto &buffer = buffers[bufferView.buffer];

  auto *bufferStart =
      buffer.data.data() + bufferView.byteOffset + acessor.byteOffset;

  auto componentType = GLTFComponentType{acessor.componentType};
  auto type = GLTFType{acessor.type};

  GlTFSpanVariantType result;

  // 根据类型返回对应span
  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Scalar) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Scalar>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Vec2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Vec2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Vec3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Vec3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Vec4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Vec4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Mat2) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Mat2>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Mat3) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Mat3>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed8 && type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed8, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned8 && type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned8, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed16 && type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed16, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Unsigned16 &&
      type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Unsigned16, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::Signed32 && type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::Signed32, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  if (componentType == GLTFComponentType::SignedFloat32 &&
      type == GLTFType::Mat4) {
    using currentType =
        getGLTFType_t<GLTFComponentType::SignedFloat32, GLTFType::Mat4>;
    // 强转
    auto *positionBuffer =
        std::launder(reinterpret_cast<currentType *>(bufferStart));
    auto spanBuffer = std::span<currentType>(positionBuffer, acessor.count);

    result = spanBuffer;
  }

  return result;
}

} // namespace App
