# Этап 4: MPI. Одни и те же исходники собираются в две программы:
#   seq — последовательная (без MPI): блочный алгоритм этапа 2 на целой матрице — база для абсолютного ускорения;
#   mpi — параллельная: циклическое распределение блочных строк, рассылки MPI_Bcast (запуск: mpiexec -n P …).
# Ключи оптимизации у обеих одинаковые — как у сборки «с флагом» этапа 2.
#
# Сборка (Linux: Open MPI или MPICH; Windows: MS-MPI + SDK):
#   cmake -S lab4_mpi -B lab4_mpi/build
#   cmake --build lab4_mpi/build --config Release
#   ctest --test-dir lab4_mpi/build -C Release
cmake_minimum_required(VERSION 3.16)
project(lab4_cholesky LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Уровень оптимизации задаёт вариант, а не тип сборки (Debug/Release) — как на этапах 2–3.
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_CXX_FLAGS_${config} "")
endforeach()

set(MPI_CXX_SKIP_MPICXX ON)  # нужен только интерфейс C (без устаревших привязок C++)
find_package(MPI COMPONENTS CXX)

if(MSVC)
    set(LAB4_OPT /O2 /arch:AVX2)
    set(LAB4_COMMON /W4 /utf-8 /EHsc)
else()
    set(LAB4_OPT -O3 -march=native)
    set(LAB4_COMMON -Wall -Wextra -Wpedantic)
endif()
string(REPLACE ";" " " LAB4_FLAGS_TEXT "${LAB4_OPT}")

enable_testing()

# Ядро без MPI: общее для обеих программ и тестов.
add_library(core STATIC src/cholesky.cpp src/cli.cpp)
target_include_directories(core PUBLIC src)
target_compile_options(core PRIVATE ${LAB4_COMMON} ${LAB4_OPT})

function(lab4_target target)
    target_compile_options(${target} PRIVATE ${LAB4_COMMON} ${LAB4_OPT})
    target_include_directories(${target} PRIVATE src tests)
    target_link_libraries(${target} PRIVATE core)
endfunction()

add_executable(cholesky_seq src/main.cpp)
lab4_target(cholesky_seq)
target_compile_definitions(cholesky_seq PRIVATE LAB4_VARIANT="seq" LAB4_FLAGS="${LAB4_FLAGS_TEXT}")

add_executable(tests tests/tests.cpp)
lab4_target(tests)
add_test(NAME unit COMMAND tests)
# Программа целиком: N не кратно b, N < b, N = 1; ненулевой код возврата, если погрешность велика.
add_test(NAME run_seq COMMAND cholesky_seq -n 1 2 37 100 -b 8 -r 1 --min-total 0)
add_test(NAME rejects_bad_args_seq COMMAND cholesky_seq -b 0)
add_test(NAME seq_rejects_phases COMMAND cholesky_seq -m phases)
set_tests_properties(rejects_bad_args_seq seq_rejects_phases PROPERTIES WILL_FAIL TRUE)

if(NOT MPI_CXX_FOUND)
    message(WARNING "MPI не найден: программа cholesky_mpi и её тесты не собираются "
                    "(Windows: установить MS-MPI и MS-MPI SDK; Linux: libopenmpi-dev или libmpich-dev)")
    return()
endif()

add_library(distributed STATIC src/distributed.cpp)
target_include_directories(distributed PUBLIC src)
target_compile_options(distributed PRIVATE ${LAB4_COMMON} ${LAB4_OPT})
target_link_libraries(distributed PUBLIC core MPI::MPI_CXX)

add_executable(cholesky_mpi src/main.cpp)
lab4_target(cholesky_mpi)
target_compile_definitions(cholesky_mpi PRIVATE LAB4_MPI LAB4_VARIANT="mpi" LAB4_FLAGS="${LAB4_FLAGS_TEXT}")
target_link_libraries(cholesky_mpi PRIVATE distributed)

add_executable(tests_mpi tests/tests_mpi.cpp)
lab4_target(tests_mpi)
target_link_libraries(tests_mpi PRIVATE distributed)

# Тесты под mpiexec: P = 1, 2, 3 и число ядер (не больше — без --oversubscribe Open MPI не запустит больше).
# P > p (у части процессов нет блочных строк) проверяется малыми N в самих тестах.
set(LAB4_TEST_PROCS 1 2 3)
if(MPIEXEC_MAX_NUMPROCS GREATER 3)
    list(APPEND LAB4_TEST_PROCS ${MPIEXEC_MAX_NUMPROCS})
endif()
foreach(p ${LAB4_TEST_PROCS})
    add_test(NAME mpi_unit_p${p}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${p} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:tests_mpi>
                     ${MPIEXEC_POSTFLAGS})
    add_test(NAME run_mpi_p${p}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${p} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cholesky_mpi>
                     ${MPIEXEC_POSTFLAGS} -n 1 2 37 100 -b 8 -m all -r 1 --min-total 0)
    set_tests_properties(mpi_unit_p${p} run_mpi_p${p} PROPERTIES TIMEOUT 120)
    if(NOT WIN32)  # Open MPI не запускает процессы от root без разрешения (облачный контейнер)
        set_tests_properties(mpi_unit_p${p} run_mpi_p${p} PROPERTIES
                             ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
    endif()
endforeach()
add_test(NAME rejects_bad_args_mpi
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cholesky_mpi>
                 ${MPIEXEC_POSTFLAGS} -b 0)
set_tests_properties(rejects_bad_args_mpi PROPERTIES WILL_FAIL TRUE TIMEOUT 60)
if(NOT WIN32)
    set_tests_properties(rejects_bad_args_mpi PROPERTIES
                         ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
endif()
