#pragma once


class Script  {
public:

  virtual void init(){};

  virtual void update(){};

  virtual void physicalUpdate(){};

  Script();
  Script(const Script &) = default;
  Script(Script &&) = delete;
  Script &operator=(const Script &) = default;
  Script &operator=(Script &&) = delete;
  virtual ~Script();
};
