add_executable(yrestir yrestir.cpp)

set_target_properties(yrestir  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories(yrestir PUBLIC ${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(yrestir yocto yocto_gui)
