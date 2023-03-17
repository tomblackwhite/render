#include "node.hh"

namespace App {
void SceneManager::init() {

  loadScene(m_mainScene);


  //初始化脚本
  ranges::for_each(m_nodes,[](auto const& keyValue){
    auto &[key,node] = keyValue;
    node->script().init();
  });
}

void SceneManager::update(){

  ranges::for_each(m_nodes,[](auto const& keyValue){
    auto &[key,node] = keyValue;
    node->script().update();
  });
}

void SceneManager::physicalUpdate(){

  ranges::for_each(m_nodes,[](auto const& keyValue){
    auto &[key,node] = keyValue;
    node->script().physicalUpdate();
  });
}

void SceneManager::loadScene(const string &scene) {


}
} // namespace App
