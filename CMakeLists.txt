cmake_minimum_required(VERSION 3.10)
project(MyProject)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加源文件

# 添加源文件
set(SOURCES
    src/standalone_localization_tester.cpp
)

# 添加头文件
include_directories(include)


# 添加可执行文件
add_executable(MyProject ${SOURCES})