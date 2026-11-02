// Разбор аргументов командной строки программы замеров (проверка входных данных — только здесь).
#pragma once

#include <string>
#include <vector>

namespace lab3 {

// point — точечный алгоритм; block — блочный, параллельно поддиагональные блоки столбца;
// phases — блочный, параллельно модификации всех блоков столбца и затем треугольные решения.
enum class Mode { point, block, phases };

struct Options {
    std::vector<int> sizes{128, 256, 512, 1024};  // размерности N (по возрастанию, без повторов)
    int block = 16;                               // размер блока b (подобран на этапе 2)
    std::vector<Mode> modes{Mode::block, Mode::phases};
    int threads = 0;                              // число потоков; 0 — по умолчанию (OMP_NUM_THREADS или все)
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

}  // namespace lab3
