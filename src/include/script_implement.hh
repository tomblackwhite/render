#pragma once

#include "script.hh"
#include <node.hh>
#include <memory>

namespace App {
class RootScript : public Script {
public:
  using Script::Script;
  void init() override;
};
} // namespace App
