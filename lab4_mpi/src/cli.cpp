#include "cli.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace lab4 {

namespace {

// Параметр — это «-буква» или «--слово»; «-5» — значение (отрицательное число), а не параметр.
bool looks_like_option(const std::string& s) {
    return s.size() >= 2 && s[0] == '-' && (s[1] == '-' || std::isalpha(static_cast<unsigned char>(s[1])));
}

double parse_seconds(const std::string& text) {
    std::size_t pos = 0;
    double value = -1.0;
    try {
        value = std::stod(text, &pos);
    } catch (const std::exception&) {
        pos = 0;
    }
    if (pos == 0 || pos != text.size() || !(value >= 0.0 && value <= 3600.0))
        throw std::invalid_argument("время должно быть числом секунд от 0 до 3600, получено «" + text + "»");
    return value;
}

int parse_positive(const std::string& text, const std::string& what) {
    std::size_t pos = 0;
    int value = 0;
    try {
        value = std::stoi(text, &pos);
    } catch (const std::exception&) {
        pos = 0;
    }
    if (pos == 0 || pos != text.size() || value < 1)
        throw std::invalid_argument(what + " должно быть натуральным числом, получено «" + text + "»");
    return value;
}

}  // namespace

Options parse_args(const std::vector<std::string>& args) {
    Options o;
    for (std::size_t k = 0; k < args.size(); ++k) {
        const std::string& arg = args[k];
        auto next_value = [&]() -> const std::string& {
            if (k + 1 >= args.size() || looks_like_option(args[k + 1]))
                throw std::invalid_argument("после " + arg + " нужно значение");
            return args[++k];
        };

        if (arg == "-n" || arg == "--sizes") {
            o.sizes.clear();
            while (k + 1 < args.size() && !looks_like_option(args[k + 1]))
                o.sizes.push_back(parse_positive(args[++k], "N"));
            if (o.sizes.empty()) throw std::invalid_argument("после " + arg + " нужна хотя бы одна размерность N");
        } else if (arg == "-b" || arg == "--block") {
            o.block = parse_positive(next_value(), "b");
        } else if (arg == "-r" || arg == "--repeats") {
            o.repeats = parse_positive(next_value(), "число повторов");
        } else if (arg == "--min-total") {
            o.min_total = parse_seconds(next_value());
        } else if (arg == "-m" || arg == "--mode") {
            o.modes.clear();
            while (k + 1 < args.size() && !looks_like_option(args[k + 1])) {
                const std::string& m = args[++k];
                std::vector<Mode> chosen;
                if (m == "block")
                    chosen = {Mode::block};
                else if (m == "phases")
                    chosen = {Mode::phases};
                else if (m == "all")
                    chosen = {Mode::block, Mode::phases};
                else
                    throw std::invalid_argument("режим должен быть block, phases или all, получено «" + m + "»");
                for (Mode c : chosen)
                    if (std::find(o.modes.begin(), o.modes.end(), c) == o.modes.end()) o.modes.push_back(c);
            }
            if (o.modes.empty()) throw std::invalid_argument("после " + arg + " нужен хотя бы один режим");
        } else if (arg == "--dump") {
            o.dump = next_value();
        } else if (arg == "--csv") {
            o.csv = true;
        } else if (arg == "--info") {
            o.info = true;
        } else if (arg == "-h" || arg == "--help") {
            o.help = true;
        } else {
            throw std::invalid_argument("неизвестный параметр «" + arg + "»");
        }
    }

    std::sort(o.sizes.begin(), o.sizes.end());
    o.sizes.erase(std::unique(o.sizes.begin(), o.sizes.end()), o.sizes.end());
    if (!o.dump.empty() && (o.sizes.size() != 1 || o.modes.size() != 1))
        throw std::invalid_argument("для --dump нужны одна размерность (-n N) и один режим (-m block или phases)");
    return o;
}

std::string usage() {
    return "Блочный метод Холецкого через блочные скалярные произведения: замер времени и погрешности (этап 4, MPI).\n"
           "Запуск: mpiexec -n P cholesky_mpi [параметры]; число процессов P задаёт mpiexec.\n"
           "\n"
           "Параметры:\n"
           "  -n, --sizes N...    размерности матрицы (по умолчанию 128 256 512 1024)\n"
           "  -b, --block B       размер блока (по умолчанию 16)\n"
           "  -m, --mode M...     block — рассылка строки j после разложения L_jj;\n"
           "                      phases — сначала внедиагональная часть строки j, затем L_jj;\n"
           "                      all — обе (по умолчанию block phases)\n"
           "  -r, --repeats R     наименьшее число замеров на точку (по умолчанию 3)\n"
           "      --min-total S   повторять замеры, пока суммарно не займут S секунд (по умолчанию 1)\n"
           "      --csv           вывод в формате CSV\n"
           "      --dump FILE     записать найденную L в файл (одна N, один режим)\n"
           "      --info          сведения о сборке (вариант, ключи компиляции, компилятор, MPI)\n"
           "  -h, --help          эта справка\n";
}

}  // namespace lab4
