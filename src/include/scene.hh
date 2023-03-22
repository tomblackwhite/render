#pragma once
#include <asset.hh>
#include <concepts>
#include <node.hh>

namespace App {

// 分离创建职责，区分素材管理和场景创建原则
class SceneFactory;

// 控制整个场景树。
class SceneManager {

public:
  using key = std::string;
  using NodeMap = std::unordered_map<key, std::unique_ptr<Node>>;

  using NodeTree = std::unordered_map<key, std::vector<key>>;

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

  std::string m_mainScene;
  // which data struct ?
  NodeMap m_map{};
  // std::vector<std::unique_ptr<Node>> m_nodes{};
  NodeTree m_tree{};

  AssetManager m_assetManager=AssetManager::instance();
  std::unique_ptr<SceneFactory> m_factory;
};

class SceneFactory {
public:
  static void createScene(AssetManager &assetManager, const std::string &sceneKey,
                   SceneManager::NodeMap &map, SceneManager::NodeTree &tree);

private:
  static std::unique_ptr<Node> createNode(tinygltf::Node const &);

  template <typename T, typename Param, std::size_t... Ints>
    requires IsAnyOf<T, glm::vec3, glm::quat, glm::mat4> &&
             requires(Param &param, std::size_t n) {
               { auto(param[n]) } -> std::floating_point;
             }
  static T castToGLMType(Param const &param, std::index_sequence<Ints...> ints) {
    return T{static_cast<float>(param[Ints])...};
  }

  //递归构建Node Tree
  static void createNodeTree(const tinygltf::Node &node,
                             const tinygltf::Model &model,
                             SceneManager::NodeTree &tree);
};
} // namespace App
