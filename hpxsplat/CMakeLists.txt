project(hpxsplat)

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
    regex date_time serialization chrono atomic
    program_options filesystem log log_setup system thread 
)
if (WIN32)
  add_definitions(-DBOOST_ALL_DYN_LINK)
  link_directories(${Boost_LIBRARY_DIR})
endif()

#--------------------------------------------------
# Find HPX
#--------------------------------------------------
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY 1)

find_package(HPX)
include_directories(${HPX_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
  add_definitions(/bigobj)
endif()

#--------------------------------------------------
# VTK
#--------------------------------------------------
find_package(ParaView)
if (ParaView_FOUND)
  include(${ParaView_USE_FILE})
  add_definitions(-DHPX_HAVE_VTK)
  set(VTK_LIBS
		  vtkTestingRendering
		  vtkImagingCore
		  vtkImagingSources
		  vtkIOImage
		  vtkFiltersCore
		  vtkInteractionStyle
		  vtkParallelMPI
		  ${VTK_LIBRARIES}
		  vtksys
		  mpi
		  ${MPI_LIBRARIES}
		  )
endif()

#--------------------------------------------------
# hpx test
#--------------------------------------------------
aDD_EXECUTABLE(sort_by_key
        sort_by_key.cpp
        )

hpx_setup_target(sort_by_key)

#--------------------------------------------------
# HPX test
#--------------------------------------------------
ADD_EXECUTABLE(exclusive_scan_problem
        exclusive_scan_problem.cpp
        )

hpx_setup_target(exclusive_scan_problem)

#--------------------------------------------------
# HPX test
#--------------------------------------------------
ADD_EXECUTABLE(reduce_by_key
  reduce_by_key.cpp
)

hpx_setup_target(reduce_by_key)

#--------------------------------------------------
# HPX test
#--------------------------------------------------
ADD_EXECUTABLE(copy_if
        copy_if.cpp
)

hpx_setup_target(copy_if)

#--------------------------------------------------
# HPX test
#--------------------------------------------------
ADD_EXECUTABLE(sort_by_key_test
        sort_by_key_test.cpp
        sort_by_key_test.cpp)

hpx_setup_target(sort_by_key_test)

#--------------------------------------------------
# HPX test
#--------------------------------------------------
ADD_EXECUTABLE(TestHPXSplatter 
  TestHPXSplatter.cxx
#  ../testing/TestUtils.cxx
)

hpx_setup_target(TestHPXSplatter)

TARGET_LINK_LIBRARIES(TestHPXSplatter
  ${VTK_LIBS}
  iostreams_component
  ${Boost_LIBRARIES}
)

