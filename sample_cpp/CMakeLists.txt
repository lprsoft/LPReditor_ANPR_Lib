
cmake_minimum_required(VERSION 3.5.1)

PROJECT(sample_cpp)
#************************
#PREPROCESSOR
#************************
add_definitions(
-DLPREDITOR_DEMO_NO_ARGS
)

#************************
#ROOT DIR
#************************
find_path(sample_cpp_ROOT sample_cpp.cpp ../sample_cpp/ )

#************************
#EXECUTABLE
#************************
FILE(GLOB sample_cppSources ${sample_cpp_ROOT}/*.cpp ${sample_cpp_ROOT}/*.h)
# Create a target for the library
ADD_EXECUTABLE( ${CMAKE_PROJECT_NAME} 
${sample_cppSources} ../include/Levenshtein.h ../src/Levenshtein.cpp ../include/utils_image_file.h ../src/utils_image_file.cpp
 )
 
set(CMAKE_CXX_STANDARD 17)
#************************
#OPENCV
#************************
if (MSVC)
#if necessary change the path ../opencv-master/build/ to the path where you build opencv
find_path(OPENCV4_ROOT OpenCV.sln ../../opencv-master/build/)
set(OpenCV_DIR OPENCV4_ROOT)
if(NOT OpenCV_FOUND)
find_path(OpenCV_DIR OpenCV.sln ../../opencv-master/build/)
endif(NOT OpenCV_FOUND)
else (MSVC)
endif (MSVC) 
#if necessary change the path ../opencv-master/build/ to the path where you build opencv
FIND_PACKAGE( OpenCV REQUIRED PATHS "../../opencv-master/build/")
set(OpenCV_DIR ../opencv-master/build)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
#************************
#LPReditor_ANPR_Lib
#************************
find_path(LPReditor_ANPR_Lib_ROOT Open_LPReditor_Lib.cpp ../ )
if (MSVC)
find_path(LPREDITOR_LIB_BUILD_ROOT LPReditor_ANPR_Lib.sln ${LPReditor_ANPR_Lib_ROOT}/build)

find_path(LPREDITOR_LIB_OUTPUT_ROOT LPReditor_ANPR_Lib.lib
 debug ${LPReditor_ANPR_Lib_ROOT}/build/Debug
 optimized ${LPReditor_ANPR_Lib_ROOT}/build/Release
  ) 
  
find_library(LPREDITOR_ANPR_LIB_TO_LINK_TO NAMES LPReditor_ANPR_Lib PATHS ${LPREDITOR_LIB_OUTPUT_ROOT})

else (MSVC)
find_path(LPREDITOR_LIB_BUILD_ROOT libLPReditor_ANPR_Lib.so ${LPReditor_ANPR_Lib_ROOT}/build) 
find_library(LPREDITOR_ANPR_LIB_TO_LINK_TO NAMES libLPReditor_ANPR_Lib.so PATHS ${LPREDITOR_LIB_BUILD_ROOT})
endif (MSVC) 

find_path(LPReditor_ANPR_Lib_INC Open_LPReditor_Lib.h ../ )

target_link_libraries(${CMAKE_PROJECT_NAME} ${LPREDITOR_ANPR_LIB_TO_LINK_TO})
MESSAGE( STATUS )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
MESSAGE( STATUS "LPREDITOR_ANPR_LIB_TO_LINK_TO =${LPREDITOR_ANPR_LIB_TO_LINK_TO}" )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
MESSAGE( STATUS )

if (MSVC)
    SET(EXECUTABLE_OUTPUT_PATH ${LPReditor_ANPR_Lib_ROOT}/build)
    SET(LIBRARY_OUTPUT_PATH ${LPReditor_ANPR_Lib_ROOT}/build)
endif (MSVC) 
    
#************************
#INC DIRS
#************************
include_directories(${LPReditor_ANPR_Lib_INC})


    
#************************
#LINKING
#************************
set_property(TARGET sample_cpp PROPERTY CXX_STANDARD 17)
set_property(TARGET sample_cpp PROPERTY CXX_STANDARD_REQUIRED ON)
