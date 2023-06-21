#pragma once

#include "script.hh"
#include <glm/gtx/transform.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtx/projection.hpp>
#include <spdlog/spdlog.h>
#include <node.hh>
#include <memory>
#include <input.hh>
#include <cmath>

namespace App {
class RootScript : public Script {
public:
  using Script::Script;
  void init() final;
  void update(DeltaTime delta) final;

private:
  Camera * m_camera{nullptr};

  bool m_fpsMove=false;
};
} // namespace App
