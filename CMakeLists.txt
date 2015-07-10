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

#--------------------------------------------------
# checkout projects we want to build together
#--------------------------------------------------
Git_External(
  ${CMAKE_CURRENT_SOURCE_DIR}/hpx
  https://github.com/STEllAR-GROUP/hpx.git
  master
  SHALLOW
  DISABLE_UPDATE
  VERBOSE
)

Git_External(
  ${CMAKE_CURRENT_SOURCE_DIR}/vtkm
  git@github.com:biddisco/vtkm.git
  hpx-rebase
  SHALLOW
  DISABLE_UPDATE
  VERBOSE
)

#--------------------------------------------------
# Testing : Include the standard testing module
#--------------------------------------------------
include(CTest)
include(Dart)

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
# DELETE THESE
# Custom setting for me to save some time, 
#--------------------------------------------------
set(HWLOC_ROOT "C:/hwloc-win64-build-1.9.1")

#--------------------------------------------------
# Add the subprojects to our project
# args are : project name in subproject, path from here
#--------------------------------------------------
set(HPX_WITH_PARCELPORT_MPI ON)
set(HPX_WITH_PARCELPORT_TCP ON)
set(HPX_WITH_TESTS          OFF)
set(HPX_WITH_EXAMPLES       OFF)
add_subproject(hpx hpx)

set(VTKm_ENABLE_CUDA ON)
set(VTKm_ENABLE_HPX  ON)
add_subproject(vtkm vtkm)

