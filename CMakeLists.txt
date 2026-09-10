cmake_minimum_required(VERSION 3.20)
project(od_p3pf LANGUAGES CXX)

option(OD_P3PF_BUILD_EXPERIMENTS "Build the synthetic and TUM experiment drivers" ON)
option(OD_P3PF_BUILD_GB_BASELINE "Build the optional generated Groebner-basis baseline" OFF)

add_library(od_p3pf STATIC src/od_p3pf.cpp)
target_include_directories(od_p3pf PUBLIC include)
target_compile_features(od_p3pf PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(od_p3pf PRIVATE /O2 /W4 /permissive-)
else()
  target_compile_options(od_p3pf PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

if(OD_P3PF_BUILD_EXPERIMENTS)
  add_executable(od_p3pf_benchmark src/benchmark.cpp)
  target_link_libraries(od_p3pf_benchmark PRIVATE od_p3pf)
  target_compile_features(od_p3pf_benchmark PRIVATE cxx_std_17)

  add_executable(od_p3pf_tum src/tum_experiment.cpp)
  target_link_libraries(od_p3pf_tum PRIVATE od_p3pf)
  target_compile_features(od_p3pf_tum PRIVATE cxx_std_17)

  enable_testing()
  add_test(NAME od_p3pf_regression COMMAND od_p3pf_benchmark --quick)
endif()

if(OD_P3PF_BUILD_GB_BASELINE)
  find_package(Eigen3 3.3 QUIET NO_MODULE)
  if(NOT TARGET Eigen3::Eigen)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense PATH_SUFFIXES eigen3)
    if(NOT EIGEN3_INCLUDE_DIR)
      message(FATAL_ERROR
        "Eigen 3.3+ is required for OD_P3PF_BUILD_GB_BASELINE. "
        "Set EIGEN3_INCLUDE_DIR or Eigen3_DIR."
      )
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
    )
  endif()

  add_library(od_p3pf_gb STATIC src/od_p3pf_gb.cpp)
  target_include_directories(od_p3pf_gb PUBLIC include)
  target_link_libraries(od_p3pf_gb PUBLIC od_p3pf Eigen3::Eigen)
  target_compile_features(od_p3pf_gb PUBLIC cxx_std_17)

  if(MSVC)
    target_compile_options(od_p3pf_gb PRIVATE /O2 /W4 /permissive-)
  else()
    target_compile_options(od_p3pf_gb PRIVATE -O3 -Wall -Wextra -Wpedantic)
  endif()

  add_executable(od_p3pf_same_problem_benchmark src/same_problem_benchmark.cpp)
  target_link_libraries(
    od_p3pf_same_problem_benchmark PRIVATE od_p3pf od_p3pf_gb Eigen3::Eigen
  )
  target_compile_features(od_p3pf_same_problem_benchmark PRIVATE cxx_std_17)

  if(OD_P3PF_BUILD_EXPERIMENTS)
    add_test(
      NAME od_p3pf_same_problem_regression
      COMMAND od_p3pf_same_problem_benchmark --quick
    )
  endif()
endif()
