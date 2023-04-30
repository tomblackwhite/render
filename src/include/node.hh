#pragma once
#include "asset.hh"
#include "script.hh"
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

// 节点接口
class Node : public Observable<Node> {
public:
  Node() = default;
  Node(const Node &) = delete;
  Node &operator=(const Node &) = delete;
  Node(Node &&) = default;
  Node &operator=(Node &&) = default;
  [[nodiscard]] glm::mat4 const &transform() const { return m_transform; }
  void setTransform(glm::mat4 const&matrix) {
    m_transform = matrix;
    glm::vec4 per;
    glm::vec3 skew;
    glm::decompose(m_transform, m_scale, m_rotation, m_translation, skew, per);

    notify(*this, "transform");
  };

  [[nodiscard]] glm::vec3 const& translation() const { return m_translation; }
  void setTranslation(glm::vec3 const& translation) {
    m_translation = translation;
    setTransformInner();
  }

  [[nodiscard]] glm::quat const& rotation() const { return m_rotation; }
  void setRotation(glm::quat const &quat) {
    m_rotation = quat;
    setTransformInner();
  }

  [[nodiscard]] glm::vec3 const& scale() const { return m_scale; }
  void setScale(glm::vec3 const&scale) {
    m_scale = scale;
    setTransformInner();
  }

  [[nodiscard]] bool visible() const { return m_visible; }
  void setViible(bool visible) {
    m_visible = visible;
    notify(*this, "visible");
  }

  Script *script() { return m_script.get(); }

  void setScript(Script *script) { m_script.reset(script); }

  std::string &parent() { return m_parent; }
  std::vector<std::string> &childern() { return m_children; }

  [[nodiscard]] std::string const &parent() const { return m_parent; }
  [[nodiscard]] std::vector<std::string> const &children() const {
    return m_children;
  }

  std::string name;

  //相对于世界坐标的变换
  glm::mat4 model{1};

  virtual ~Node() noexcept = default;

private:
  void setTransformInner() {
    glm::scale(m_transform, m_scale);
    m_transform *= glm::toMat4(m_rotation);
    glm::translate(m_transform, m_translation);
    notify(*this, "transform");
  }
  glm::mat4 m_transform{1.0};
  glm::vec3 m_scale{1.0};
  glm::quat m_rotation{1.0, 0.0, 0.0, 0.0};
  glm::vec3 m_translation{0.0, 0.0, 0.0};
  bool m_visible = true;

  std::unique_ptr<Script> m_script{nullptr};

  std::string m_parent;
  std::vector<std::string> m_children;
};

using key = std::string;

class MeshInstance : public Node, public Observable<MeshInstance> {

public:
  using Node::notify;
  using Node::subscrible;
  using Node::unsubscrible;
  using Observable<MeshInstance>::notify;
  using Observable<MeshInstance>::subscrible;
  using Observable<MeshInstance>::unsubscrible;

  // std::string meshKey() { return m_meshKey; }

  // void setMeshKey(std::string key) {
  //   m_meshKey = std::move(key);
  //   notify(*this, "meshKey");
  // }

  Mesh mesh;
};

class Camera : public Node {
public:
  // 默认相机 view matrix
  //+x is right , up is +y ,looks toward -z
  glm::mat4 view{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};
  glm::mat4 projection{1};
};

} // namespace App
