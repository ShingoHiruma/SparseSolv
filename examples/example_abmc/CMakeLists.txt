project(MatTest CXX)
#出力先はベース
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Basedir}/build)

#コンパイル設定
add_executable(testABMC.out main.cpp)
#オプションとライブラリリンクの設定
if(CMAKE_CXX_COMPILER MATCHES "icpc")
 message("Intel Classical Compiler use in Example !") 
 target_compile_options(testABMC.out PUBLIC -std=c++14 -O3 -I${Basedir} -qopenmp -ipo)
 target_link_libraries(testABMC.out PUBLIC -qmkl -ipo SparseSolv)
#
elseif(CMAKE_CXX_COMPILER MATCHES "icpx")
 message("Intel New Compiler use in Example !") 
 target_compile_options(testABMC.out PUBLIC -std=c++14 -O3 -I${Basedir} -qopenmp -ipo)
 target_link_libraries(testABMC.out PUBLIC -qmkl -ipo SparseSolv)
#
else()
 target_compile_options(testABMC.out PUBLIC -std=c++14 -O3 -I${Basedir} -fopenmp)
 target_link_libraries(testABMC.out PUBLIC -lgomp SparseSolv)
endif()
