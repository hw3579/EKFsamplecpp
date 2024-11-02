cmake_minimum_required(VERSION 3.10)
project(MyProject)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (MSVC)
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
else (LINUX)
    add_compile_options(-Wall -Wextra -Wpedantic -fpermissive)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

# 找到 Python 的头文件和库
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 获取 numpy 的头文件路径
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 将 Python 和 NumPy 路径添加为包含目录
include_directories(${Python3_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})

# 添加matplotlibcpp头文件
# find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
if (NOT MATPLOTLIB_CPP_INCLUDE_DIRS)
    message(INFO_ERROR "Could not find matplotlibcpp.h")
endif()

# 添加源文件
set(SOURCES
    src/standalone_localization_tester.cpp
)

# 添加头文件
include_directories(include)

# 添加可执行文件
add_executable(MyProject ${SOURCES})
target_include_directories(MyProject PRIVATE ${MATPLOTLIB_CPP_INCLUDE_DIRS})
target_link_libraries(MyProject ${Python3_LIBRARIES})