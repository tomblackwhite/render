#pragma once

#include "script.hh"
#include <spdlog/spdlog.h>
#include <node.hh>
#include <memory>

namespace App {
class RootScript : public Script {
public:
  using Script::Script;
  void init() final;
  void update(DeltaTime delta) final;
};
} // namespace App
