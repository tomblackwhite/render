#pragma once
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
  glm::mat4 transform() const { return m_transform; }
  void setTransform(glm::mat4 matrix) {
    m_transform = matrix;
    glm::vec4 per;
    glm::vec3 skew;
    glm::decompose(m_transform, m_scale, m_rotation, m_translation, skew, per);

    notify(*this, "transform");
  };

  glm::vec3 translation() const { return m_translation; }
  void setTranslation(glm::vec3 translation) {
    m_translation = translation;
    setTransformInner();
  }

  glm::quat rotation() const { return m_rotation; }
  void setRotation(glm::quat quat) {
    m_rotation = quat;
    setTransformInner();
  }

  glm::vec3 scale() const { return m_scale; }
  void setScale(glm::vec3 scale) {
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
};

using key = std::string;

namespace other {

class Mesh : public Node, public Observable<Mesh> {

  using Node::notify;
  using Node::subscrible;
  using Node::unsubscrible;
  using Observable<Mesh>::notify;
  using Observable<Mesh>::subscrible;
  using Observable<Mesh>::unsubscrible;

  std::string meshKey() { return m_meshKey; }

  void setMeshKey(std::string key) {
    m_meshKey = std::move(key);
    notify(*this, "meshKey");
  }

private:
  std::string m_meshKey;
};
} // namespace other

} // namespace App
