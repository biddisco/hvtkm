set(CMAKE_OSX_DEPLOYMENT_TARGET "")
project(hpx_vtkm)

#--------------------------------------------------
# cmake policies
# see http://www.cmake.org/cmake/help/v2.8.12/cmake.html#section_Policies
#--------------------------------------------------
foreach(p
    CMP0003 CMP0017 CMP0023 CMP0042
    )
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()

#--------------------------------------------------
# cmake old/new compatibility
#--------------------------------------------------
cmake_minimum_required(VERSION 2.8)

#--------------------------------------------------
# load script for checking out projects from git
# disable user fork before loading script
#--------------------------------------------------
set(GIT_EXTERNAL_USER_FORK "" CACHE STRING "")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GitExternal.cmake)

#------------------------------------------------------------------------------
# Boost 
#------------------------------------------------------------------------------
set(Boost_ADDITIONAL_VERSIONS
    ${Boost_ADDITIONAL_VERSIONS}
    "1.59.0" "1.59"
    "1.58.0" "1.58"
    "1.57.0" "1.57")

find_package(Boost 
  1.55  
  REQUIRED 
  COMPONENTS
    regex date_time serialization 
    program_options filesystem log log_setup system thread 
)

#--------------------------------------------------
# checkout projects we want to build together
#--------------------------------------------------
Git_External(
  ${CMAKE_CURRENT_SOURCE_DIR}/hpx
  https://github.com/STEllAR-GROUP/hpx.git
  master
  SHALLOW
  VERBOSE
)

Git_External(
  ${CMAKE_CURRENT_SOURCE_DIR}/vtkm
  git@github.com:biddisco/vtkm.git
  hpx-rebase
  SHALLOW
  VERBOSE
)

#--------------------------------------------------
# Testing : Include the standard testing module
#--------------------------------------------------
include(CTest)
include(Dart)
set(DART_TESTING_TIMEOUT "15" CACHE STRING "timeout" FORCE)

#--------------------------------------------------
# load add_subproject macro from common tools
#--------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SubProject.cmake)

#--------------------------------------------------
# Setup Output directories
#--------------------------------------------------
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables."
)
SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries"
)
SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries."
)

#--------------------------------------------------
# Add the subprojects to our project
# args are : project name in subproject, path from here
#--------------------------------------------------
#
# Some defaults we set for HPX
#
set(HPX_WITH_PARCELPORT_MPI ON CACHE BOOL "")
set(HPX_WITH_PARCELPORT_TCP ON CACHE BOOL "")
set(HPX_WITH_TESTS          OFF CACHE BOOL "")
set(HPX_WITH_EXAMPLES       OFF CACHE BOOL "")
#
# Some defaults we set for VTK-m
#
set(VTKm_ENABLE_CUDA        OFF CACHE BOOL "")
set(VTKm_ENABLE_HPX         ON CACHE BOOL "")

set(CMAKE_FIND_DEBUG_MODE 1)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY 1)
set(HPX_SKIP_FIND 1)
set(VTMm_SKIP_FIND 1)
set(vtkmtest_SKIP_FIND 1)

# add_subproject <PROJECT_NAME> <DIR>
add_subproject(HPX hpx)
add_subproject(VTKm vtkm)
add_subproject(vtkmtest vtkmtest)

#--------------------------------------------------
# Testing
#--------------------------------------------------
#add_subdirectory(testing)
