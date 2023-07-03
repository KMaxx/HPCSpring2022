#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.1.0 )

project ( RT
  VERSION 3.0.0
  LANGUAGES CXX
)

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

# Source
set ( COMMON_ALL
  src/rtweekend.h
  src/camera.h
  src/ray.h
  src/vec3.h
  src/utils.h
)

set ( SOURCE_RT
  ${COMMON_ALL}
  src/aabb.h
  src/external/stb_image.h
  src/perlin.h
  src/rtw_stb_image.h
  src/texture.h
  src/aarect.h
  src/box.h
  src/bvh.h
  src/constant_medium.h
  src/hittable.h
  src/hittable_list.h
  src/material.h
  src/moving_sphere.h
  src/sphere.h
  src/main.cc
)

# Executables
add_executable(rt       ${SOURCE_RT})

include_directories(src)
