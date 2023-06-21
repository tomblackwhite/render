#include <script_implement.hh>

namespace App {

void RootScript::init() {
  Script::init();
  auto *camera = m_nodeFactory->createNode<Camera>();

  m_camera = camera;
// 设置相机
// yfov aspecRatio znear zfar
#ifdef GLM_FORCE_DEPTH_ZERO_TO_ONE
  std::cout << "depth zero to one\n";
#endif
  auto orth = glm::ortho(0.0f, 16.0f, 0.0f, 9.0f, 0.1f, 100.0f);
  auto per =
      glm::perspectiveZO(glm::radians(75.0f), 16.0f / 9.0f, 0.05f, 1000.0f);
  camera->projection = per;
  glm::vec3 location = {0, 1, 1.5};
  camera->setTranslation(location);
  camera->rotationX = 0;
  camera->rotationY = 0;
  camera->rotationZ = 0;

  // camera->view = // glm::lookAt(location, {0.0f,0.0f,0.0f},
  // glm::vec3(0,1,0));

  m_node->addChildren(camera->name);

  auto *scene = m_nodeFactory->createNode<Scene>();

  auto vec = glm::vec4(1);
  scene->sceneData = {vec, vec, vec, vec, vec};
  auto identi = glm::mat4(1);
  scene->sceneData.sunlightColor = glm::vec4(1, 1, 1, 1);

  // I * Ry
  auto rotate = glm::rotate(identi, glm::radians(30.0f), {0.0f, 1.0f, 0.0f});
  // I*Rz * Ry
  rotate = glm::rotate(rotate, glm::radians(45.0f), {1.0f, 0.0f, 0.0f});

  scene->sceneData.sunlightDirection = -rotate * glm::vec4(0, -1, 0, 0);

  m_node->addChildren(scene->name);
}

void RootScript::update(Script::DeltaTime delta) {

  auto constexpr sensitive = 0.1f;
  auto constexpr pi = std::numbers::pi_v<float>;
  auto constexpr piDivide2 = std::numbers::pi_v<float> / 2;

  if (auto count = m_input->pressedCount(SDLK_F12); count != 0) {
    if (count % 2 == 1) {
      m_fpsMove = !m_fpsMove;
      SDL_SetRelativeMouseMode(m_fpsMove ? SDL_TRUE : SDL_FALSE);
    }
  }

  if (m_fpsMove) {

    if (m_input->isMouseMove()) {
      auto rotateByAxisX = glm::radians(-static_cast<float>(m_input->relY)*sensitive*delta.count()) ;
      auto rotateByAxisY = glm::radians(-static_cast<float>(m_input->relX)*sensitive*delta.count()) ;

      // Ry
      auto rotationY = m_camera->rotationY + rotateByAxisY;
      rotationY = std::fmod(rotationY,2*pi);
      auto rotation =
          glm::rotate(glm::quat(1, 0, 0, 0), rotationY, glm::vec3(0, 1, 0));

      auto rotationX = m_camera->rotationX + rotateByAxisX;
      rotationX = std::clamp(rotationX, -piDivide2, piDivide2);

      // glm::rotate()
      // Rx
      // 避免旋转超过 +-pi/2
      rotation = glm::rotate(rotation, rotationX, glm::vec3(1, 0, 0));

      // spdlog::info("origin add {},{}",glm::degrees(m_camera->rotationY),glm::degrees(rotateByAxisY));
      m_camera->rotationX = rotationX;
      m_camera->rotationY = rotationY;
      m_camera->setRotation(rotation);

    }
  }
}

} // namespace App
