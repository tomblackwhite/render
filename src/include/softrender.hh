#pragma once
#include <cstdint>
#include <vector>
#include <new>
#include <spdlog/spdlog.h>

struct Pixel {
  unsigned char Red;
  unsigned char Green;
  unsigned char Blue;
  unsigned char Alpha;
};

class SoftRender {
public:
  SoftRender(){
    m_frameBuffer.reserve(m_width*m_height);

    spdlog::info("大小Pixel{}",sizeof(Pixel));

    auto size = m_width*m_height;
    for(std::size_t i=0;i < size;++i){
      m_frameBuffer.push_back({0xFF,0x00,0x00,0xFF});
    }
  }
  std::vector<Pixel> const& getFrameBuffer()  { return  m_frameBuffer; }

private:
  std::vector<Pixel> m_frameBuffer;
  unsigned long m_width=540;
  unsigned long m_height=960;
};
