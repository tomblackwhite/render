#pragma once
#include <SDL.h>
#include <SDL_events.h>
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <unordered_map>
namespace App {

namespace ranges = std::ranges;
class Input {
  using KeyCodeMap =
      std::unordered_map<SDL_Keycode, std::vector<SDL_KeyboardEvent>>;

  constexpr static int bucketCount = 256;

  KeyCodeMap m_keyCodeMap = KeyCodeMap(bucketCount);
  KeyCodeMap m_tempkeyCodeMap = KeyCodeMap(bucketCount);
  std::vector<SDL_MouseMotionEvent> m_mouseEvents;
  bool m_mouseMove = false;

public:
  int32_t relX = 0;
  int32_t relY = 0;
  uint32_t timeDurtionX=0;
  uint32_t timeDurtionY=0;

  // 正在按
  bool isPressing(SDL_Keycode keyCode) {

    if (auto iter = m_keyCodeMap.find(keyCode); iter != m_keyCodeMap.end()) {
      if (!iter->second.empty()) {
        return iter->second.back().type == SDL_KEYDOWN;
      }
    }
    return false;
  }

  // 是否按过
  bool isPressed(SDL_Keycode keyCode) {
    if (auto iter = m_keyCodeMap.find(keyCode); iter != m_keyCodeMap.end()) {
      return ranges::any_of(iter->second, [](SDL_KeyboardEvent &event) {
        return event.type == SDL_KEYDOWN;
      });
    }
    return false;
  }

  uint32_t pressedCount(SDL_Keycode keyCode) {
    if (auto iter = m_keyCodeMap.find(keyCode); iter != m_keyCodeMap.end()) {
      return ranges::count_if(iter->second, [](SDL_KeyboardEvent &event) {
        return event.type == SDL_KEYDOWN;
      });
    }
    return 0;
  }

  bool isMouseMove() const { return m_mouseMove; }

  void inputEvent(SDL_Event *event) {
    switch (event->type) {
    case SDL_KEYUP:
    case SDL_KEYDOWN: {
      auto key = event->key.keysym.sym;
      if (auto iter = m_tempkeyCodeMap.find(key);
          iter != m_tempkeyCodeMap.end()) {
        iter->second.push_back(event->key);
      } else {
        m_tempkeyCodeMap.insert({key, {event->key}});
      }
      break;
    }
    case SDL_MOUSEMOTION: {

      auto mouse = event->motion;
      m_mouseEvents.push_back(mouse);
    }
    }
  };

  void processEvent() {

    m_keyCodeMap.swap(m_tempkeyCodeMap);
    m_tempkeyCodeMap.clear();

    if (!m_mouseEvents.empty()) {
      // spdlog::info("mouse motion event {}", m_mouseEvents.size());
      // for (auto &event : m_mouseEvents) {
      //   relX += event.xrel;
      //   relY += event.yrel;
      // }
      relX=m_mouseEvents.back().xrel;
      relY=m_mouseEvents.back().yrel;
      m_mouseMove = true;
    } else {
      relX = 0;
      relY = 0;
      m_mouseMove = false;
    }
    m_mouseEvents.clear();
  }
};
} // namespace App
