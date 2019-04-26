cmake_minimum_required (VERSION 3.1)
project (mhdsim C CXX)
set (CMAKE_CXX_STANDARD 11)

set(MHDSIM_SOURCES types.cpp operators.cpp util.cpp routines.cpp main.cpp)
INCLUDE_DIRECTORIES ("${PROJECT_SOURCE_DIR}")

#detect openmp
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DARMA_USE_OPENMP")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DARMA_USE_OPENMP")
endif()

#detect opencl
find_package(OpenCL REQUIRED)
INCLUDE_DIRECTORIES("${OpenCL_INCLUDE_DIRS}")
LINK_DIRECTORIES ("${OpenCL_LIBRARY}")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVIENNACL_WITH_OPENCL")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIENNACL_WITH_OPENCL")

#detect armadillo
find_package(Armadillo REQUIRED)
INCLUDE_DIRECTORIES ("${ARMADILLO_INCLUDE_DIRS}")
LINK_DIRECTORIES ("${ARMADILLO_LIBRARIES}")

# important defines
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DARMA_USE_CXX11 -DARMA_64BIT_WORD -g")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARMA_USE_CXX11 -DARMA_64BIT_WORD -g")

add_executable (mhdsim "${MHDSIM_SOURCES}")
set_target_properties(mhdsim PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries (mhdsim ${OpenCL_LIBRARY} ${ARMADILLO_LIBRARIES})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/steadymhd)
