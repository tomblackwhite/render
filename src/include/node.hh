#pragma once
#include "asset.hh"
#include "script.hh"
#include <cstdint>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <memory>
#include <observer.hh>
#include <tiny_gltf.h>
#include <tool.hh>
#include <unordered_map>
#include <utility>
#include <vector>

namespace App {

class Node;
using key = std::string;
using NodeTree = std::unordered_map<key, std::vector<key>>;
using NodeMap = std::unordered_map<key, std::unique_ptr<Node>>;
struct NodeContainer {
  NodeTree tree;
  NodeMap map;
};
class NodeFactory;
// 节点接口
class Node {
public:
  Node(const Node &) = delete;
  Node &operator=(const Node &) = delete;
  Node(Node &&) = default;
  Node &operator=(Node &&) = default;
  [[nodiscard]] glm::mat4 const &transform() const { return m_transform; }
  void setTransform(glm::mat4 const &matrix) {
    m_transform = matrix;
    glm::vec4 per;
    glm::vec3 skew;
    glm::decompose(m_transform, m_scale, m_rotation, m_translation, skew, per);
  };

  [[nodiscard]] glm::vec3 const &translation() const { return m_translation; }
  void setTranslation(glm::vec3 const &translation) {
    m_translation = translation;
    setTransformInner();
  }

  [[nodiscard]] glm::quat const &rotation() const { return m_rotation; }
  void setRotation(glm::quat const &quat) {
    m_rotation = quat;
    setTransformInner();
  }

  [[nodiscard]] glm::vec3 const &scale() const { return m_scale; }
  void setScale(glm::vec3 const &scale) {
    m_scale = scale;
    setTransformInner();
  }

  [[nodiscard]] bool visible() const { return m_visible; }
  void setViible(bool visible) { m_visible = visible; }

  Script *script() { return m_script.get(); }

  void setScript(std::unique_ptr<Script> script) {
    m_script = std::move(script);
  }

  std::string &parent() { return m_parent; }
  std::vector<std::string> &childern() { return m_children; }

  void addChildren(key const &keyName) { m_children.push_back(keyName); }

  [[nodiscard]] std::string const &parent() const { return m_parent; }
  [[nodiscard]] std::vector<std::string> const &children() const {
    return m_children;
  }

  std::string name;

  // 相对于世界坐标的变换
  glm::mat4 model{1};

  virtual ~Node() noexcept = default;

  friend class NodeFactory;

protected:
  Node() = default;

private:
  void setTransformInner() {
    m_transform=glm::scale(glm::mat4(1), m_scale);
    m_transform *= glm::toMat4(m_rotation);
    m_transform=glm::translate(m_transform, m_translation);
  }

  glm::mat4 m_transform{1.0};
  glm::vec3 m_scale{1.0};
  glm::quat m_rotation{1.0, 0.0, 0.0, 0.0};
  glm::vec3 m_translation{0.0, 0.0, 0.0};
  bool m_visible = true;

  std::unique_ptr<Script> m_script{nullptr};

  NodeContainer *m_container{nullptr};
  std::string m_parent;
  std::vector<std::string> m_children;
};

class NodeFactory {
public:
  template <typename T>
    requires std::derived_from<T, Node>
  T *createNode() {
    auto nodeT = std::unique_ptr<T>(new T());
    auto result = nodeT.get();
    std::string idKey;

    do {
      idKey = "@" + std::to_string(getUniqueId());
    } while (m_container->map.contains(idKey));
    nodeT->name = idKey;

    std::unique_ptr<Node> node(nodeT.release());

    m_container->map.insert({idKey, std::move(node)});

    return result;
  }

  template <typename T>
    requires std::derived_from<T, Node>
  T *createNode(std::string key) {
    auto nodeT = std::unique_ptr<T>(new T());
    auto result = nodeT.get();
    std::unique_ptr<Node> node(nodeT.release());
    m_container->map.insert_or_assign(key, std::move(node));
    return result;
  }

  explicit NodeFactory(NodeContainer *container) : m_container(container){};

private:
  static std::size_t getUniqueId() {
    static std::size_t idNumber = 0;
    return idNumber++;
  }
  NodeContainer *m_container{nullptr};
};

class MeshInstance : public Node {

  friend class NodeFactory;

public: // std::string meshKey() { return m_meshKey; }
  // void setMeshKey(std::string key) {
  //   m_meshKey = std::move(key);
  //   notify(*this, "meshKey");
  // }

  // 相对于世界坐标系的变换。
  glm::mat4 modelMatrix{1.0};
  Mesh mesh;
};

class Camera : public Node {
  friend class NodeFactory;

public:
  glm::mat4 view{1};
  glm::mat4 projection{1};
};

} // namespace App
