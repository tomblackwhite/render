#include "node.hh"

namespace App {
void SceneManager::init() {
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



} // namespace App
