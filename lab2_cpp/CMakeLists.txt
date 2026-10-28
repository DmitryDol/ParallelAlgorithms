# Этап 2: последовательная программа на C++.
#
# Одни и те же исходники собираются в несколько вариантов, которые отличаются только ключами компиляции:
#   noopt         — без флага оптимизации («без флага»);
#   opt           — с флагом оптимизации и векторизацией под процессор («с флагом»);
#   opt_novec     — как opt, но без автовекторизации (вклад векторизации);
#   opt_fastmath  — как opt + разрешена перестановка операций с плавающей точкой.
# Для каждого варианта собираются программа замеров cholesky_<вариант> и тесты tests_<вариант>.
#
# Сборка (Linux, MinGW, MSVC):
#   cmake -S lab2_cpp -B lab2_cpp/build
#   cmake --build lab2_cpp/build --config Release
#   ctest --test-dir lab2_cpp/build -C Release
cmake_minimum_required(VERSION 3.16)
project(lab2_cholesky LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Уровень оптимизации задаёт вариант, а не тип сборки (Debug/Release): иначе, например, Release добавил бы -O3
# и к сборке «без флага». Поэтому ключи типов сборки очищаются.
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_CXX_FLAGS_${config} "")
endforeach()

if(MSVC)
    set(LAB2_FLAGS_noopt /Od)
    set(LAB2_FLAGS_opt /O2 /arch:AVX2)
    # Штатного ключа, отключающего автовекторизацию, у MSVC нет: вариант без AVX2 векторизует только SSE2
    # (2 числа double вместо 4).
    set(LAB2_FLAGS_opt_novec /O2)
    set(LAB2_FLAGS_opt_fastmath /O2 /arch:AVX2 /fp:fast)
    set(LAB2_COMMON_FLAGS /W4 /utf-8 /EHsc)
else()
    set(LAB2_FLAGS_noopt -O0)
    set(LAB2_FLAGS_opt -O3 -march=native)
    set(LAB2_FLAGS_opt_novec -O3 -march=native -fno-tree-vectorize)
    set(LAB2_FLAGS_opt_fastmath -O3 -march=native -ffast-math)
    set(LAB2_COMMON_FLAGS -Wall -Wextra -Wpedantic)
endif()

set(LAB2_CORE src/cholesky.cpp src/cli.cpp)

enable_testing()

function(lab2_variant name)
    set(flags ${LAB2_FLAGS_${name}})
    string(REPLACE ";" " " flags_text "${flags}")

    add_executable(cholesky_${name} src/main.cpp ${LAB2_CORE})
    add_executable(tests_${name} tests/tests.cpp ${LAB2_CORE})
    foreach(target cholesky_${name} tests_${name})
        target_include_directories(${target} PRIVATE src)
        target_compile_options(${target} PRIVATE ${LAB2_COMMON_FLAGS} ${flags})
        target_compile_definitions(${target} PRIVATE LAB2_VARIANT="${name}" LAB2_FLAGS="${flags_text}")
    endforeach()

    add_test(NAME unit_${name} COMMAND tests_${name})
    # Программа целиком: N не кратно b, N < b, N = 1; ненулевой код возврата, если погрешность велика.
    add_test(NAME run_${name} COMMAND cholesky_${name} -n 1 2 37 100 -b 8 -r 1)
    add_test(NAME rejects_bad_args_${name} COMMAND cholesky_${name} -n 0)
    set_tests_properties(rejects_bad_args_${name} PROPERTIES WILL_FAIL TRUE)
endfunction()

foreach(variant noopt opt opt_novec opt_fastmath)
    lab2_variant(${variant})
endforeach()
