#pragma once
#include "script.hh"
#include <asset.hh>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <memory>
#include <tool.hh>
#include <unordered_map>
#include <vector>

namespace App {

/*观察者模式*/
template <typename T> class Observer {
public:
  virtual void fieldChanged(T &source, const string &fieldName) = 0;

  Observer(const Observer &) = delete;
  Observer(Observer &&) = delete;
  Observer &operator=(const Observer &) = delete;
  Observer &operator=(Observer &&) = delete;
  virtual ~Observer() noexcept = default;
};

template <typename T> class Observable {
public:
  void notify(T &source, const string &name) {

    for (auto observer : m_observers) {
      observer->fieldChanged(source, name);
    }
  }
  void subscrible(Observer<T> *observer) {
    m_observers.emplace_back(observer);
  };
  void unsubscrible(Observer<T> *observer){};

private:
  std::vector<Observer<T> *> m_observers;
};
/*观察者模式*/

// 节点接口
class Node : public Observable<Node> {
public:
  Node(const Node &) = delete;
  Node(Node &&) = default;
  Node &operator=(const Node &) = delete;
  Node &operator=(Node &&) = default;
  glm::mat4 transform() { return m_transform; }
  void setTransform(glm::mat4 matrix){
    m_transform = matrix;
    glm::vec4 per;
    glm::vec3 skew;
    glm::decompose(m_transform, m_scale, m_rotation, m_translation, skew, per);

    notify(*this, "transform");
  };

  glm::vec3 translation(){ return m_translation;}
  void setTranslation(glm::vec3 translation){
    m_translation = translation;
    setTransformInner();
  }

  glm::quat rotation(){return m_rotation;}
  void setRotation(glm::quat quat){
    m_rotation= quat;
    setTransformInner();
  }

  glm::vec3 scale(){return m_scale;}
  void setScale(glm::vec3 scale){
    m_scale=scale;
    setTransformInner();
  }

  [[nodiscard]] bool visible() const {return m_visible;}
  void setViible(bool visible){
    m_visible = visible;
    notify(*this, "visible");
  }

  Script &script() { return *m_script; }

  void setScript(Script *script) { m_script.reset(script); }

  virtual ~Node() noexcept = default;

private:
  void setTransformInner(){
    glm::scale(m_transform, m_scale);
    m_transform*=glm::toMat4(m_rotation);
    glm::translate(m_transform, m_translation);
    notify(*this, "transform");
  }
  glm::mat4 m_transform{1.0};
  glm::vec3 m_scale{1.0};
  glm::quat m_rotation{1.0, 0.0, 0.0, 0.0};
  glm::vec3 m_translation{0.0,0.0,0.0};
  bool m_visible = true;

  std::unique_ptr<Script> m_script{nullptr};
};

using key = std::string;

// 控制整个场景树。
class SceneManager {

public:
  // 初始化脚本之类的
  void init();
  // 每帧更新
  void update();

  void physicalUpdate();

private:
  // which data struct ?
  std::unordered_map<key, std::unique_ptr<Node>> m_nodes{};
  // std::vector<std::unique_ptr<Node>> m_nodes{};
  std::unordered_map<key, std::vector<key>> m_keys{};
};

namespace other {

class Mesh : public Node,public Observable<Mesh> {

private:

  AssetManager &m_assertManager = AssetManager::instance();
};
} // namespace other

} // namespace App
