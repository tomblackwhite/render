#pragma once
#include <chrono>
// #include "input.hh"

namespace App {

namespace chrono = std::chrono;

class Node;
class NodeFactory;
class Input;
class Script {
public:

  using DeltaTime = chrono::duration<float,chrono::milliseconds::period>;

  virtual void init(){};

  virtual void update(DeltaTime delta){};

  virtual void physicalUpdate(){};

  explicit Script(Node *node, NodeFactory *factory,Input *input)
      : m_node(node), m_nodeFactory(factory),m_input(input){};
  Script(const Script &) = default;
  Script(Script &&) = delete;
  Script &operator=(const Script &) = default;
  Script &operator=(Script &&) = delete;
  virtual ~Script(){}

  chrono::time_point<chrono::system_clock> preTimePoint;
protected:
  Node *m_node;
  NodeFactory *m_nodeFactory;
  Input *m_input;
};
} // namespace App
