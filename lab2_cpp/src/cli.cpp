#include "cli.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace lab2 {

namespace {

// Параметр — это «-буква» или «--слово»; «-5» — значение (отрицательное число), а не параметр.
bool looks_like_option(const std::string& s) {
    return s.size() >= 2 && s[0] == '-' && (s[1] == '-' || std::isalpha(static_cast<unsigned char>(s[1])));
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
        } else if (arg == "-m" || arg == "--mode") {
            const std::string& m = next_value();
            if (m == "point")
                o.mode = Mode::point;
            else if (m == "block")
                o.mode = Mode::block;
            else if (m == "both")
                o.mode = Mode::both;
            else
                throw std::invalid_argument("режим должен быть point, block или both, получено «" + m + "»");
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
    if (!o.dump.empty() && (o.sizes.size() != 1 || o.mode == Mode::both))
        throw std::invalid_argument("для --dump нужны одна размерность (-n N) и один режим (-m point или -m block)");
    return o;
}

std::string usage() {
    return "Блочный метод Холецкого через блочные скалярные произведения: замер времени и погрешности.\n"
           "\n"
           "Параметры:\n"
           "  -n, --sizes N...    размерности матрицы (по умолчанию 128 256 512 1024)\n"
           "  -b, --block B       размер блока (по умолчанию 32)\n"
           "  -m, --mode M        point — точечный алгоритм, block — блочный, both — оба (по умолчанию)\n"
           "  -r, --repeats R     число замеров на точку (по умолчанию 3)\n"
           "      --csv           вывод в формате CSV\n"
           "      --dump FILE     записать найденную L в файл (одна N, один режим)\n"
           "      --info          сведения о сборке (вариант, ключи компиляции, компилятор)\n"
           "  -h, --help          эта справка\n";
}

}  // namespace lab2
