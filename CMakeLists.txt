cmake_minimum_required (VERSION 3.1)
project (comfi C CXX)
set (CMAKE_CXX_STANDARD 11)

file(GLOB COMFI_SOURCES "src/*cpp")
include_directories("${PROJECT_SOURCE_DIR}")

#detect openmp
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DARMA_USE_OPENMP")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DARMA_USE_OPENMP")
endif()

#detect opencl
find_package(OpenCL REQUIRED)
include_directories("${OpenCL_INCLUDE_DIRS}")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVIENNACL_WITH_OPENCL")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIENNACL_WITH_OPENCL")

#detect HDF5
find_package(HDF5 REQUIRED)
include_directories("${HDF5_INCLUDE_DIRS}")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${HDF5_DEFINITIONS} -DARMA_USE_HDF5")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HDF5_DEFINITIONS} -DARMA_USE_HDF5")

#detect armadillo
find_package(Armadillo REQUIRED)
include_directories("${ARMADILLO_INCLUDE_DIRS}")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DARMA_USE_CXX11 -DARMA_64BIT_WORD -DVIENNACL_WITH_ARMADILLO")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARMA_USE_CXX11 -DARMA_64BIT_WORD -DVIENNACL_WITH_ARMADILLO")

# detect viennacl (file from Shogun devs)
set (ViennaCL_DIR "${PROJECT_SOURCE_DIR}")
find_package(ViennaCL REQUIRED)
include_directories("${VIENNACL_INCLUDE_DIRS}")

# important defines and flags
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -DARMA_NO_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -DARMA_NO_DEBUG")

add_executable (comfi "${COMFI_SOURCES}")
set_target_properties(comfi PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries (comfi ${OpenCL_LIBRARIES} ${ARMADILLO_LIBRARIES} ${HDF5_LIBRARIES})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/comfi)

# Make the directory structure in the binary directory
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/comfi/output)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/comfi/input)
file(COPY kernels_ocl DESTINATION ${PROJECT_BINARY_DIR}/comfi) 
