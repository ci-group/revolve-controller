project(revolve-brain)

# C++11 ########################################################################
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR CMAKE_COMPILER_IS_GNUCXX)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(--std=c++11 SUPPORTS_STD_CXX11)
    if(SUPPORTS_STD_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --std=c++11")
    else()
        message(FATAL_ERROR "Compiler does not support --std=c++11")
    endif()
endif()


# Libraries ####################################################################
# build static libraries with position indipendent code, so they can be used in dynamic libraries
set (CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# boost libraries
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS system)

# add accneat
add_subdirectory("neat/accneat")
#TODO make include path for accneat nicer
include_directories("neat/accneat/src")

### RLPOWER DEPENDENCIES ###
# GSL
# Gnu mathematical library
find_package(GSL REQUIRED)

# end Libraries stuff
include_directories(${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${GSL_INCLUDE_DIRS})


file(GLOB_RECURSE BRAIN_SRCS
        brain/fakelightsensor.cpp
        brain/phototaxisevaluator.cpp
        brain/rlpower.cpp
        brain/neural_network.cpp
        brain/supgbrain.cpp
        brain/supgbrainphototaxis.cpp
        brain/supg/supgneuron.cpp
        brain/supg/supggenomemanager.cpp
        neat/asyncneat.cpp
        neat/neatevaluation.cpp 
        brain/ccpg/ExtendedNeuralNetwork.cpp
	brain/ccpg/Neuron.cpp
	brain/ccpg/DifferentialCPG.cpp
	brain/ccpg/BiasNeuron.cpp
	brain/ccpg/InputNeuron.cpp
	brain/ccpg/LeakyIntegrator.cpp
	brain/ccpg/LinearNeuron.cpp
	brain/ccpg/NeuralConnection.cpp
	brain/ccpg/OscillatorNeuron.cpp
	brain/ccpg/SigmoidNeuron.cpp
	brain/ccpg/VOscillator.cpp
	brain/ccpg/XOscillator.cpp)

# Compiling ####################################################################
add_library(revolve-brain-static STATIC ${BRAIN_SRCS})
add_library(revolve-brain SHARED)

# Linking ######################################################################
target_link_libraries(revolve-brain-static accneat ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${GSL_LIBRARIES} yaml-cpp)
target_link_libraries(revolve-brain revolve-brain-static)


# Install ######################################################################
install(TARGETS revolve-brain DESTINATION lib)


# Tests ########################################################################
#if tests are enabled
enable_testing()

add_executable(testAsyncNeat neat/test/testasyncneat.cpp)
add_executable(testCustomGenomeManager neat/test/testcustomgenomemanager.cpp)
add_executable(testMultiNNSpecies neat/test/testmultinnspeciesneat.cpp)
add_executable(testSUPGBrain test/testsupgbrain.cpp)
target_link_libraries(testAsyncNeat revolve-brain)
target_link_libraries(testCustomGenomeManager revolve-brain)
target_link_libraries(testMultiNNSpecies revolve-brain)
target_link_libraries(testSUPGBrain revolve-brain)
add_test(testAsyncNeat testAsyncNeat)
add_test(testCustomGenomeManager testCustomGenomeManager)
add_test(testMultiNNSpecies testMultiNNSpecies)
add_test(testSUPGBrain testSUPGBrain)


IF(WITH_PYTHON)
    message("BUILDING WITH PYTHON SUPPORT")
    add_subdirectory("brain/python/")
ELSE()
    message("BUILDING WITHOUT PYTHON SUPPORT. To enable it, run add `-DWITH_PYTHON=1` to your cmake parameters ")
ENDIF()
