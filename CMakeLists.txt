# This is an example of using common_utils and common_cmake_modules
#
# Required, and this line has to go atop. 3.15 is recommended, and 3.12 will
# probably work.
cmake_minimum_required(VERSION 3.15)

# Required. This set the project name
project(common_utils_boilerplate LANGUAGES CXX C)

# Required. This set the necessary global settings and load
# the helps for adding dependencies defined in common_cmake_modules.
#
# Adding common_utils or common_cmake_modules can be done in two ways:
# 1. Add the git repo as git submodule:
#    $ git submodule add -b kids_dev https://github.com/toltec-astro/common_utils.git
#    This will loosely bundle the source code in the repo and allow easy
#    updating of the source repo.
# 2. Add the git repo as a fetch content target.
#    This will allow cmake manage the updating of the repo, which works the best
#    for end users who do not modify the included repo.

# The below is needed for approch 1:
# include("${CMAKE_CURRENT_SOURCE_DIR}/common_utils/cmake/CommonSettings.cmake")
# add_library(common_utils INTERFACE)
# target_include_directories(common_utils
#     INTERFACE
#         <path/to/the/submodule>/src
#     )

# The below is needed for approach 2:
include(FetchContent)
FetchContent_Declare(
  common_utils
  GIT_REPOSITORY https://github.com/toltec-astro/common_utils.git
  GIT_TAG        kids_dev
)
FetchContent_GetProperties(common_utils)
if(NOT common_utils_POPULATED)
  FetchContent_Populate(common_utils)
  include("${common_utils_SOURCE_DIR}/cmake/CommonSettings.cmake")
  add_library(common_utils INTERFACE)
  target_include_directories(common_utils
      INTERFACE
          ${common_utils_SOURCE_DIR}/src
      )
endif()

# Actually add dependencies
# See common_cmake_modules/*.cmake for available items.
# The name here should match the filename stem of the CMake file there.
include(SpdlogAndFmt)
if (VERBOSE)  # VERBOSE is defined in CommonSettings.cmake
    # This is defined in CommonSettings and can be used to
    # display info of a target.
    print_target_properties(common_utils)
    print_target_properties(spdlog::spdlog)
    print_target_properties(fmt::fmt)
endif()

# The below is just normal cmake code.
# Notice the usage of imported targets, which are made available through
# the common_cmake_modules and the include(...) above.

add_executable(hello_world)
target_sources(hello_world
    PRIVATE
        hello_world.cpp
        )
target_link_libraries(hello_world
    PRIVATE
        common_utils
        spdlog::spdlog
        fmt::fmt
    )
