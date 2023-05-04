#pragma once

namespace App {

class Node;
class NodeFactory;
class Script {
public:
  virtual void init(){};

  virtual void update(){};

  virtual void physicalUpdate(){};

  explicit Script(Node *node, NodeFactory *factory)
      : m_node(node), m_nodeFactory(factory){};
  Script(const Script &) = default;
  Script(Script &&) = delete;
  Script &operator=(const Script &) = default;
  Script &operator=(Script &&) = delete;
  virtual ~Script(){}

protected:
  Node *m_node;
  NodeFactory *m_nodeFactory;
};
} // namespace App
