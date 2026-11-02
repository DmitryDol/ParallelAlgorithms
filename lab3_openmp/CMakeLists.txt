# Этап 3: OpenMP. Одни и те же исходники собираются в несколько программ:
#   seq            — последовательная (директивы OpenMP игнорируются) — база для абсолютного ускорения;
#   omp            — вариант А: директивы OpenMP над циклами (число потоков задаётся при запуске);
#   autopar_p<P>   — вариант Б (GCC): автопараллелизация компилятором без директив, P потоков задаётся при сборке;
#   autopar        — вариант Б (MSVC): /Qpar, число потоков — во время выполнения.
# Ключи оптимизации у всех одинаковые — как у сборки «с флагом» этапа 2.
#
# Сборка (Linux, MinGW, MSVC):
#   cmake -S lab3_openmp -B lab3_openmp/build
#   cmake --build lab3_openmp/build --config Release
#   ctest --test-dir lab3_openmp/build -C Release
cmake_minimum_required(VERSION 3.16)
project(lab3_cholesky LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Уровень оптимизации задаёт вариант, а не тип сборки (Debug/Release) — как на этапе 2.
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_CXX_FLAGS_${config} "")
endforeach()

find_package(OpenMP COMPONENTS CXX)

set(LAB3_AUTOPAR_THREADS "2;4;6;8;12;16;24" CACHE STRING "Числа потоков для сборок с автопараллелизацией GCC")

if(MSVC)
    set(LAB3_OPT /O2 /arch:AVX2)
    set(LAB3_COMMON /W4 /utf-8 /EHsc)
    set(LAB3_IGNORE_OMP /wd4068)  # в сборках без OpenMP директивы #pragma omp намеренно не действуют
else()
    set(LAB3_OPT -O3 -march=native)
    set(LAB3_COMMON -Wall -Wextra -Wpedantic)
    set(LAB3_IGNORE_OMP -Wno-unknown-pragmas)
endif()

enable_testing()

# lab3_variant(<имя> [OPENMP] [THREADS <P>] [FLAGS <ключи>...] [REPORT <файл>] [TESTS])
function(lab3_variant name)
    cmake_parse_arguments(V "OPENMP;TESTS" "THREADS;REPORT" "FLAGS" ${ARGN})
    set(flags ${LAB3_OPT} ${V_FLAGS})
    if(V_OPENMP)
        set(shown ${flags} ${OpenMP_CXX_FLAGS})
    else()
        set(shown ${flags})
        list(APPEND flags ${LAB3_IGNORE_OMP})
    endif()
    string(REPLACE ";" " " flags_text "${shown}")

    # Вычислительное ядро — отдельной библиотекой, чтобы отчёт компилятора об автопараллелизации был только о нём.
    add_library(core_${name} STATIC src/cholesky.cpp)
    target_include_directories(core_${name} PUBLIC src)
    target_compile_options(core_${name} PRIVATE ${LAB3_COMMON} ${flags})
    if(V_REPORT)
        # Что распараллелено — в файл; почему остальные циклы не распараллелены — в дамп прохода parloops.
        target_compile_options(core_${name} PRIVATE -fopt-info-loop-all=${V_REPORT} -fdump-tree-parloops2-details)
    endif()
    if(V_OPENMP)
        target_link_libraries(core_${name} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    set(programs cholesky_${name})
    add_executable(cholesky_${name} src/main.cpp src/cli.cpp)
    if(V_TESTS)
        list(APPEND programs tests_${name})
        add_executable(tests_${name} tests/tests.cpp src/cli.cpp)
        add_test(NAME unit_${name} COMMAND tests_${name})
    endif()
    foreach(target ${programs})
        target_compile_options(${target} PRIVATE ${LAB3_COMMON} ${flags})
        target_compile_definitions(${target} PRIVATE LAB3_VARIANT="${name}" LAB3_FLAGS="${flags_text}")
        if(DEFINED V_THREADS)
            target_compile_definitions(${target} PRIVATE LAB3_AUTOPAR_THREADS=${V_THREADS})
        endif()
        target_link_libraries(${target} PRIVATE core_${name})
        if(V_FLAGS AND NOT MSVC)
            target_link_options(${target} PRIVATE ${V_FLAGS})  # -ftree-parallelize-loops подключает libgomp
        endif()
    endforeach()

    # Программа целиком: N не кратно b, N < b, N = 1; ненулевой код возврата, если погрешность велика.
    add_test(NAME run_${name} COMMAND cholesky_${name} -n 1 2 37 100 -b 8 -m all -r 1 --min-total 0)
    add_test(NAME rejects_bad_args_${name} COMMAND cholesky_${name} -t 0)
    set_tests_properties(rejects_bad_args_${name} PROPERTIES WILL_FAIL TRUE)
endfunction()

lab3_variant(seq TESTS)

if(OpenMP_CXX_FOUND)
    lab3_variant(omp OPENMP TESTS)
else()
    message(WARNING "OpenMP не найден: сборка omp (вариант А) пропущена")
endif()

if(MSVC)
    lab3_variant(autopar THREADS 0 FLAGS /Qpar /Qpar-report:2 TESTS)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(first TRUE)
    foreach(p ${LAB3_AUTOPAR_THREADS})
        if(first)
            lab3_variant(autopar_p${p} THREADS ${p} FLAGS -ftree-parallelize-loops=${p}
                         REPORT ${CMAKE_BINARY_DIR}/autopar_report.txt TESTS)
            set(first FALSE)
        else()
            lab3_variant(autopar_p${p} THREADS ${p} FLAGS -ftree-parallelize-loops=${p})
        endif()
    endforeach()
else()
    message(STATUS "Автопараллелизация (вариант Б) есть только у GCC и MSVC — сборки autopar пропущены")
endif()
