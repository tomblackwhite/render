#include <script_implement.hh>

namespace App {

  void RootScript::init(){
    Script::init();
    auto *camera = m_nodeFactory->createNode<Camera>();
    // 设置相机
    // yfov aspecRatio znear zfar
    #ifdef GLM_FORCE_DEPTH_ZERO_TO_ONE
    std::cout << "depth zero to one\n";
    #endif
    auto orth =glm::ortho(0.0f,16.0f,0.0f,9.0f,0.1f,100.0f);
    auto per =glm::perspectiveZO(glm::radians(75.0f),
                                          16.0f/9.0f, 0.05f, 1000.0f);
    camera->projection = per;
    glm::vec3 location={0,1,1.5};
    camera->setTranslation(location);

    // camera->view = // glm::lookAt(location, {0.0f,0.0f,0.0f}, glm::vec3(0,1,0));

    m_node->addChildren(camera->name);
  }
      }
