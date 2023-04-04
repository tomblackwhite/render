#pragma once
#include <glm/glm.hpp>
#include <span>
#include <tiny_gltf.h>
#include <variant>

namespace App {

using Buffer = std::vector<unsigned char>;

/*

//生成VarintType
;;dash library
(let* ((a "glm::tvec2,glm::tvec3, glm::tvec4, glm::tmat2x2, glm::tmat3x3,
glm::tmat4x4") (b "glm::int8_t, glm::uint8_t, glm::int16_t, glm::uint16_t,
glm::int32_t, glm::float32_t") (split (-compose #'(lambda (a) (-map
#'string-trim a))
                        #'(lambda (a) (string-split a "," t))))
       (gettype (lambda (a b)(format "%s<%s>" a b)))
       (reslist (-table-flat gettype (funcall split a) (funcall split b)))
       result)
  (setq reslist (append reslist (funcall split b)))
  (setq result (string-join (mapcar #'(lambda (a) (format "std::span<%s>" a ))
                                    reslist) ",") )
  (format "using GlTFSpanVariantType = std::variant<%s>;" result))

 */

using GlTFSpanVariantType = std::variant<
    std::span<glm::tvec2<glm::int8_t>>, std::span<glm::tvec3<glm::int8_t>>,
    std::span<glm::tvec4<glm::int8_t>>, std::span<glm::tmat2x2<glm::int8_t>>,
    std::span<glm::tmat3x3<glm::int8_t>>, std::span<glm::tmat4x4<glm::int8_t>>,
    std::span<glm::tvec2<glm::uint8_t>>, std::span<glm::tvec3<glm::uint8_t>>,
    std::span<glm::tvec4<glm::uint8_t>>, std::span<glm::tmat2x2<glm::uint8_t>>,
    std::span<glm::tmat3x3<glm::uint8_t>>,
    std::span<glm::tmat4x4<glm::uint8_t>>, std::span<glm::tvec2<glm::int16_t>>,
    std::span<glm::tvec3<glm::int16_t>>, std::span<glm::tvec4<glm::int16_t>>,
    std::span<glm::tmat2x2<glm::int16_t>>,
    std::span<glm::tmat3x3<glm::int16_t>>,
    std::span<glm::tmat4x4<glm::int16_t>>, std::span<glm::tvec2<glm::uint16_t>>,
    std::span<glm::tvec3<glm::uint16_t>>, std::span<glm::tvec4<glm::uint16_t>>,
    std::span<glm::tmat2x2<glm::uint16_t>>,
    std::span<glm::tmat3x3<glm::uint16_t>>,
    std::span<glm::tmat4x4<glm::uint16_t>>, std::span<glm::tvec2<glm::int32_t>>,
    std::span<glm::tvec3<glm::int32_t>>, std::span<glm::tvec4<glm::int32_t>>,
    std::span<glm::tmat2x2<glm::int32_t>>,
    std::span<glm::tmat3x3<glm::int32_t>>,
    std::span<glm::tmat4x4<glm::int32_t>>,
    std::span<glm::tvec2<glm::float32_t>>,
    std::span<glm::tvec3<glm::float32_t>>,
    std::span<glm::tvec4<glm::float32_t>>,
    std::span<glm::tmat2x2<glm::float32_t>>,
    std::span<glm::tmat3x3<glm::float32_t>>,
    std::span<glm::tmat4x4<glm::float32_t>>, std::span<glm::int8_t>,
    std::span<glm::uint8_t>, std::span<glm::int16_t>, std::span<glm::uint16_t>,
    std::span<glm::int32_t>, std::span<glm::float32_t>>;

// gltf
enum class GLTFComponentType {

  Signed8 = 5120,
  Unsigned8 = 5121,
  Signed16 = 5122,
  Unsigned16 = 5123,
  Signed32 = 5125,
  SignedFloat32 = 5126
};

enum class GLTFType {
  Scalar = TINYGLTF_TYPE_SCALAR,
  Vec2 = TINYGLTF_TYPE_VEC2,
  Vec3 = TINYGLTF_TYPE_VEC3,
  Vec4 = TINYGLTF_TYPE_VEC4,
  Mat2 = TINYGLTF_TYPE_MAT2,
  Mat3 = TINYGLTF_TYPE_MAT3,
  Mat4 = TINYGLTF_TYPE_MAT4
};

template <GLTFComponentType ComponentTypeIndex, GLTFType TypeIndex>
constexpr auto getGLTFTypeIdentity() {
  using enum GLTFComponentType;
  using enum GLTFType;
  if constexpr (TypeIndex == Scalar) {
    if constexpr (ComponentTypeIndex == Signed8) {
      return std::type_identity<glm::int8_t>();
    }
    if constexpr (ComponentTypeIndex == Unsigned8) {
      return std::type_identity<glm::uint8_t>();
    }
    if constexpr (ComponentTypeIndex == Signed16) {
      return std::type_identity<glm::int16_t>();
    }
    if constexpr (ComponentTypeIndex == Unsigned16) {
      return std::type_identity<glm::uint16_t>();
    }
    if constexpr (ComponentTypeIndex == Signed32) {
      return std::type_identity<glm::int32_t>();
    }
    if constexpr (ComponentTypeIndex == SignedFloat32) {
      return std::type_identity<glm::float32_t>();
    }
  }

  using ComponentType =
      typename decltype(getGLTFTypeIdentity<ComponentTypeIndex,
                                            GLTFType::Scalar>())::type;
  if constexpr (TypeIndex == Vec2) {
    return std::type_identity<glm::tvec2<ComponentType>>();
  }
  if constexpr (TypeIndex == Vec3) {
    return std::type_identity<glm::tvec3<ComponentType>>();
  }
  if constexpr (TypeIndex == Vec4) {
    return std::type_identity<glm::tvec4<ComponentType>>();
  }
  if constexpr (TypeIndex == Mat2) {
    return std::type_identity<glm::tmat2x2<ComponentType>>();
  }
  if constexpr (TypeIndex == Mat3) {
    return std::type_identity<glm::tmat3x3<ComponentType>>();
  }
  if constexpr (TypeIndex == Mat4) {
    return std::type_identity<glm::tmat4x4<ComponentType>>();
  }
}

template <auto ComponentTypeIndex, auto TypeIndex>
using getGLTFType_t = typename decltype(getGLTFTypeIdentity<ComponentTypeIndex,
                                                            TypeIndex>())::type;
} // namespace App
