# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.26.0)

project ("shape-regressor-cpp")
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmakes")
find_package(Eigen3 REQUIRED)


message(${Eigen3_FOUND})
# Include sub-projects

include(libigl)
igl_include(glfw)




set(prefix_mesh_shape ${CMAKE_CURRENT_SOURCE_DIR}/data/ICT-data/shapes_all)
#set(prefix_mesh_shape ${CMAKE_CURRENT_SOURCE_DIR}/data/ICT-data/shapes)
set(prefix_mesh_identity ${CMAKE_CURRENT_SOURCE_DIR}/data/ICT-data/identities)

function(GET_FILE_LIST prefix_mesh_shape result)
file(GLOB fff ${prefix_mesh_shape}/*.obj)
set(mesh_shapes_definition_string \")
foreach(f ${fff})
string(CONCAT res \"${f}\", )
string(APPEND mesh_shapes_definition_string ${res} )
endforeach()
message(${mesh_shapes_definition_string})
string(LENGTH ${mesh_shapes_definition_string} len )
math(EXPR t "${len}-2")
string(SUBSTRING ${mesh_shapes_definition_string} 0 ${t} test1)
string(SUBSTRING ${test1} 2 -1 comp_mesh_shape_definition_string)
set(${result} ${comp_mesh_shape_definition_string} PARENT_SCOPE)
endfunction()

message("GetFIELLIST:${prefix_mesh_shape}")
GET_FILE_LIST(${prefix_mesh_shape} comp_mesh_shape_definition_string)
message("test 2")
GET_FILE_LIST(${prefix_mesh_identity} mesh_identities_definition_string)



add_definitions(
	-DDLIB_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/dlib-data/"
	-DLANDMARK_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/dlib-data/shape_predictor_68_face_landmarks.dat"
	-DMESH_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/ICT-data"
	-DMESH_REF_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/ICT-data/generic_neutral_mesh.obj"
	-DMESH_SHAPES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/ICT-data/shapes"
	-DMESH_SHAPES_LIST="${comp_mesh_shape_definition_string}"
	-DMESH_IDENTITIES_LIST="${mesh_identities_definition_string}"
	-DFACE_68_LANDMARK_FILE="${CMAKE_CURRENT_SOURCE_DIR}/data/ICT-data/face_landmark.txt"
	-DEOS_DATA_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/data/eos-data/"
	-DWIN32_LEAN_AND_MEAN

)
remove_definitions(-DUNICODE)
find_package( KinectSDK2 REQUIRED )
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/lbfgs )


set(src 
src/main.cpp 
src/cameraPlugin.cpp
src/Fern.cpp src/cascade_regressor.cpp src/cnpy.cpp src/dataset.cpp src/weak_regressor.cpp
src/gaussian_mixture_model.cpp
)
set(viewer_src
src/viewer.cpp 
src/cnpy.cpp 

)
set(network_cpp
src/network_test.cpp 
src/cnpy.cpp 

)
set(main2_video_src
src/main2.cpp 
src/cameraPlugin.cpp
src/gaussian_mixture_model.cpp
src/Fern.cpp src/cascade_regressor.cpp src/cnpy.cpp src/dataset.cpp src/weak_regressor.cpp
)
set(header 
	#src/Fern.h

)
set(bshape_anim 
	#src/Fern.h
	src/view_blendshapes_anim_main.cpp  src/cnpy.cpp 
	src/gaussian_mixture_model.cpp
)
set(helper_src "src/cv_compatible_helper.cpp" "src/cv_compatible_helper.h")
add_library(cv_compatible ${helper_src})
target_link_libraries(cv_compatible ${OpenCV_LIBS} Eigen3::Eigen)
add_executable(facecapture ${src} ${header} )
add_executable(facecapture_video ${main2_video_src} ${header} )
add_executable(viewer ${viewer_src} ${header} )
add_executable(network_cpp ${network_cpp} ${header} )
add_executable(pre_anim_bshape ${bshape_anim} ${header} )



# Settings Kinect SDK v2
if(KinectSDK2_FOUND)
	include_directories( ${KinectSDK2_INCLUDE_DIRS} )
	link_directories( ${KinectSDK2_LIBRARY_DIRS} )
	target_link_libraries(facecapture PUBLIC dlib::dlib  cv_compatible igl::core igl::glfw ${OpenCV_LIBS}  ${KinectSDK2_LIBRARIES})
	add_custom_command( TARGET facecapture POST_BUILD ${KinectSDK2_COMMANDS} )
	
	target_link_libraries(viewer PUBLIC PUBLIC dlib::dlib  cv_compatible igl::core igl::glfw ${OpenCV_LIBS}  ${KinectSDK2_LIBRARIES})
	target_link_libraries(network_cpp PUBLIC PUBLIC dlib::dlib  cv_compatible igl::core igl::glfw ${OpenCV_LIBS}  ${KinectSDK2_LIBRARIES})
	target_link_libraries(pre_anim_bshape PUBLIC PUBLIC dlib::dlib  cv_compatible igl::core igl::glfw ${OpenCV_LIBS}  ${KinectSDK2_LIBRARIES})
	
	
	
	target_link_libraries(facecapture_video  PUBLIC dlib::dlib  cv_compatible igl::core igl::glfw ${OpenCV_LIBS}  ${KinectSDK2_LIBRARIES})
	add_custom_command( TARGET facecapture_video  POST_BUILD ${KinectSDK2_COMMANDS} )






else()
	#target_link_libraries(facecapture PUBLIC yaml-cpp dlib::dlib  igl::core igl::glfw ${OpenCV_LIBS}  )
	target_link_libraries(facecapture PUBLIC  dlib::dlib  cv_compatible igl::core igl::glfw ${OpenCV_LIBS}  )
endif()

