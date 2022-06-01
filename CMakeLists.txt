#set minimal cmake version

cmake_minimum_required(VERSION "3.20" )

project(render LANGUAGES CXX)

#export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")
set(CMAKE_BUILD_TYPE "Debug")


#set c++ standard
set(CMAKE_CXX_STANDARD 20)
#find boost for test
#BOOST_ROOT
#set(BOOST_ROOT "thirdparty/boost_1_75_0")
#set(BOOST_NO_SYSTEM_PATHS "NO")
#find_package(Boost 1.75.0 REQUIRED)
#find_package(Threads)
enable_testing()

add_compile_options(-g -fexceptions)
add_subdirectory("src")
