// Разбор аргументов командной строки программы замеров (проверка входных данных — только здесь).
// Число процессов задаёт mpiexec (-n P), а не программа.
#pragma once

#include <string>
#include <vector>

namespace lab4 {

// block — владелец строки j считает L_jj, затем рассылается вся строка j;
// phases — сначала рассылается внедиагональная часть строки j, модификации всех блоков столбца (включая
// диагональный) идут одновременно, затем рассылается только L_jj.
enum class Mode { block, phases };

struct Options {
    std::vector<int> sizes{128, 256, 512, 1024};  // размерности N (по возрастанию, без повторов)
    int block = 16;                               // размер блока b
    std::vector<Mode> modes{Mode::block, Mode::phases};
    int repeats = 3;                              // наименьшее число замеров на каждую точку
    double min_total = 1.0;                       // замеры повторяются, пока суммарно не займут столько, с
    double min_time = 0.05;                       // минимальная длительность одного замера, с
    bool csv = false;                             // вывод в CSV вместо таблицы
    bool info = false;                            // только напечатать сведения о сборке
    bool help = false;
    std::string dump;                             // файл для выгрузки L (одна N, один режим)
};

// Бросает std::invalid_argument с понятным сообщением, если аргументы некорректны.
Options parse_args(const std::vector<std::string>& args);

std::string usage();

}  // namespace lab4
