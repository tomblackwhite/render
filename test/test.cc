#define BOOST_TEST_MODULE main
#include <boost/test/unit_test.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// #include <glm/ext.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/matrix_decompose.hpp>
BOOST_AUTO_TEST_CASE( free_test_function )
{
  glm::mat4 transform{1.0};
  auto iden=glm::identity<glm::mat4>();
  auto translation = glm::translate(iden, {2,2,2});
  auto rotate = glm::rotate(iden,glm::half_pi<glm::float32_t>(),{1.0,0.,0.});
  auto scale = glm::scale(iden,{3,3,3});

  transform = translation * rotate * scale;

  glm::vec3 t;
  glm::quat r;
  glm::vec3 s;
  glm::vec3 skew;
  glm::vec4 p;

  glm::decompose(transform, s, r, t, skew, p);

  BOOST_TEST_MESSAGE("\n" << glm::to_string(s));
  BOOST_TEST_MESSAGE("\n" << glm::to_string(r));
  BOOST_TEST_MESSAGE("\n" << glm::to_string(t));
  BOOST_TEST_MESSAGE("\n" << glm::to_string(skew));
  BOOST_TEST_MESSAGE("\n" << glm::to_string(p));
}
