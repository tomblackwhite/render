#include "scene.hh"
namespace App {
using glm::vec3;
using std::index_sequence;

void SceneManager::init() {

  loadScene(m_mainScene);

  // 初始化脚本
  ranges::for_each(m_map, [](auto const &keyValue) {
    auto &[key, node] = keyValue;
    auto *pscript = node->script();
    if(pscript){
      pscript->init();
    }
  });
}

void SceneManager::update() {

  ranges::for_each(m_map, [](auto const &keyValue) {
    auto &[key, node] = keyValue;
    auto *pscript = node->script();
    if(pscript){
      pscript->update();
    }

  });
}

void SceneManager::physicalUpdate() {

  ranges::for_each(m_map, [](auto const &keyValue) {
    auto &[key, node] = keyValue;

    auto *pscript = node->script();
    if(pscript){
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

  auto model = assetManager.getScene(sceneKey);
  auto scene = model.scenes[model.defaultScene];

  for (auto &node : model.nodes) {
    map.insert({node.name, createNode(node)});
  }

  //递归构建Node Tree
  for (auto index : scene.nodes) {
    auto node = model.nodes[index];
    createNodeTree(node, model, tree);
  }
}

std::unique_ptr<Node> SceneFactory::createNode(tinygltf::Node const &node) {

  auto result = std::unique_ptr<Node>(nullptr);

  if(node.mesh!=-1){

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
} // namespace App
