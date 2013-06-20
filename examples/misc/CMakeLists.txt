# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
cmake_minimum_required (VERSION 2.8)
PROJECT (ANNetMisc)

FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(Qt4)
FIND_PACKAGE(CUDA)

IF (CUDA_FOUND)
  SET(CUDA_SEPARABLE_COMPILATION ON)
  LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp)
  LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_20)
  LIST(APPEND CUDA_NVCC_FLAGS --pre-include gcc_cuda_compat.h)
  ADD_DEFINITIONS("-DCUDA") # needed for conditional compilation of some files
  IF (CUDATHRUST_FOUND)
    INCLUDE_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/C/common/inc/)
    INCLUDE_DIRECTORIES (${CUDATHRUST_INCLUDE_DIR})
  ENDIF (CUDATHRUST_FOUND)
ENDIF (CUDA_FOUND)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/gpgpu)
INCLUDE_DIRECTORIES (${ANNetMisc_SOURCE_DIR})

SET( ANGUIHeaderFiles 
  QSOMReader.h
)

SET( ANGUISourceFiles
  QSOMReader.cpp
)

if (QT4_FOUND)
  QT4_WRAP_CPP(ANGUIHeaderFiles_MOC ${ANGUIHeaderFiles} ${3rdPartyHeaderFiles})
  QT4_WRAP_UI(ANGUIHeaderFiles_MOC ${ANGUIFormularFiles})
  INCLUDE(${QT_USE_FILE})
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
  ADD_DEFINITIONS(${QT_DEFINITIONS})

  # Build Qt library
  add_library (SOMReader STATIC ${ANGUIHeaderFiles_MOC} ${ANGUISourceFiles})
  target_link_libraries (SOMReader ${QT_LIBRARIES})
endif (QT4_FOUND)

SET( ANSourceFiles 
  ${ANNetGPGPU_SOURCE_DIR}/2DArray.cpp
  ${ANNetGPGPU_SOURCE_DIR}/3DArray.cpp
  ${ANNetGPGPU_SOURCE_DIR}/AbsLayer.cpp
  ${ANNetGPGPU_SOURCE_DIR}/AbsNet.cpp
  ${ANNetGPGPU_SOURCE_DIR}/AbsNeuron.cpp
  ${ANNetGPGPU_SOURCE_DIR}/BPLayer.cpp
  ${ANNetGPGPU_SOURCE_DIR}/BPNet.cpp
  ${ANNetGPGPU_SOURCE_DIR}/BPNeuron.cpp
  ${ANNetGPGPU_SOURCE_DIR}/Edge.cpp
  ${ANNetGPGPU_SOURCE_DIR}/Functions.cpp
  ${ANNetGPGPU_SOURCE_DIR}/HFLayer.cpp
  ${ANNetGPGPU_SOURCE_DIR}/HFNet.cpp
  ${ANNetGPGPU_SOURCE_DIR}/HFNeuron.cpp
  ${ANNetGPGPU_SOURCE_DIR}/SOMLayer.cpp
  ${ANNetGPGPU_SOURCE_DIR}/SOMNet.cpp
  ${ANNetGPGPU_SOURCE_DIR}/SOMNeuron.cpp
  ${ANNetGPGPU_SOURCE_DIR}/TrainingSet.cpp
)

SET( ANCUDASourceFiles
  ${ANNetGPGPU_SOURCE_DIR}/2DArray.cu
  ${ANNetGPGPU_SOURCE_DIR}/SOMNetGPU.cpp
  ${ANNetGPGPU_SOURCE_DIR}/SOMNetGPU.cu
  ${ANNetGPGPU_SOURCE_DIR}/SOMKernel.cu
  ${ANNetGPGPU_SOURCE_DIR}/SOMExport.cu
  ${ANNetGPGPU_SOURCE_DIR}/BPNetGPU.cpp
  ${ANNetGPGPU_SOURCE_DIR}/BPKernel.cu
)

add_executable (BPNetCPU BPNetCPU.cpp)
target_link_libraries (BPNetCPU ANNet) 

# Build misc examples
if (CUDA_FOUND)
  if (QT4_FOUND)
    cuda_add_executable (SOMNetGPU SOMNetGPU.cpp)
    target_link_libraries (SOMNetGPU ANNet SOMReader) 
    
#     CUDA_ADD_LIBRARY (foo STATIC SetFcn.cu ${ANCUDASourceFiles} ${ANSourceFiles} ${BZIP_INCLUDE_DIRS})
#     cuda_add_executable (SimpFnExtGPU SimpFnExtGPU.cpp)
#     target_link_libraries (SimpFnExtGPU foo SOMReader ${BZIP2_LIBRARIES} -lgomp) 
  endif (QT4_FOUND)
  
  cuda_add_executable (BPNetGPU BPNetGPU.cpp)
  target_link_libraries (BPNetGPU ANNet) 
endif(CUDA_FOUND)

add_executable (LoadBPNet LoadBPNet.cpp)
target_link_libraries (LoadBPNet ANNet) 

add_executable (HFNet HFNet.cpp)
target_link_libraries (HFNet ANNet) 

if (QT4_FOUND)
  add_executable (SOMNetCPU SOMNetCPU.cpp)
  target_link_libraries (SOMNetCPU ANNet SOMReader) 
    
  add_executable (SimpFnExt SimpFnExt.cpp)
  target_link_libraries (SimpFnExt ANNet SOMReader) 

  add_executable (SOMNetClustering SOMNetClustering.cpp)
  target_link_libraries (SOMNetClustering ANNet SOMReader)  
endif(QT4_FOUND)

