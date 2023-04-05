#include "scene.hh"
namespace App {
using glm::vec3;
using std::holds_alternative;
using std::index_sequence;

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

void SceneFactory::createScene(AssetManager &assetManager,
                               const std::string &sceneKey,
                               SceneManager::NodeMap &map,
                               SceneManager::NodeTree &tree) {

  auto &model = assetManager.getScene(sceneKey);
  auto &scene = model.scenes[model.defaultScene];

  // // buffer Map
  // auto &bufferMap = assetManager.BufferMap();
  // std::vector<Buffer> buffers;
  // buffers.reserve(model.buffers.size());
  // // move buffer
  // for (auto &buffer : model.buffers) {
  //   buffers.emplace_back(std::move(buffer.data));
  // }
  // bufferMap.insert({sceneKey, std::move(buffers)});
  // auto &sceneBuffers = bufferMap[sceneKey];

  for (auto &node : model.nodes) {
    map.insert({node.name, createNode(node, model, model.buffers)});
  }

  // 递归构建Node Tree
  for (auto index : scene.nodes) {
    auto &node = model.nodes[index];
    createNodeTree(node, model, tree);
  }
}

std::unique_ptr<Node>
SceneFactory::createNode(tinygltf::Node const &node,
                         tinygltf::Model const &model,
                         std::vector<tinygltf::Buffer> const &buffers) {

  auto result = std::unique_ptr<Node>(nullptr);

  // 创建mesh
  if (node.mesh != -1) {

    result = std::make_unique<MeshInstance>();
  } else {
    result = std::make_unique<Node>();
  }

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
                                  const tinygltf::Model &model,
                                  SceneManager::NodeTree &tree) {
  if (node.children.empty()) {
    return;
  }
  SceneManager::NodeTree::value_type keyValue{
      node.name, SceneManager::NodeTree::mapped_type{}};
  tree.insert(keyValue);

  auto &value = keyValue.second;

  for (auto index : node.children) {
    value.push_back(model.nodes[index].name);
    createNodeTree(model.nodes[index], model, tree);
  }
}

std::unique_ptr<Mesh>
SceneFactory::createMesh(int meshIndex, const tinygltf::Model &model,
                         std::vector<tinygltf::Buffer> buffers) {

  auto result = std::make_unique<Mesh>();
  auto &mesh = model.meshes[meshIndex];

  // 构建mesh
  auto vec3Attributes = std::to_array<std::string>({"POSITION", "NORMAL"});
  for (auto const &primitive : mesh.primitives) {
    Mesh::SubMesh subMesh;

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

    // 构建index
    if (primitive.indices != -1) {

      auto const &accessor = model.accessors[primitive.indices];
      auto spanBuffer = createSpanBuffer(accessor, model, buffers);

      if (!VariantAssign(subMesh.indices, spanBuffer)) {
        // varinat 内部元素不匹配时。
        spdlog::warn("{}'s type in asset is not match  {}'s type' ", mesh.name,
                     "indices");
      }
    }

    result->subMeshs.push_back(subMesh);
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
