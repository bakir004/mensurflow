// Shared benchmark harness. See bench.hpp.
#include "bench.hpp"

#include "backends/cpu/parallel.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <ctime>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define BENCH_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(_MSC_VER)
#define BENCH_NOINLINE __declspec(noinline)
#else
#define BENCH_NOINLINE __attribute__((noinline))
#endif

namespace bench {

// ---------------------------------------------------------------- measuring

double process_cpu_seconds() {
#if defined(_WIN32)
    FILETIME creation, exit, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        return 0.0;
    }
    auto to_seconds = [](const FILETIME& t) {
        ULARGE_INTEGER v;
        v.LowPart = t.dwLowDateTime;
        v.HighPart = t.dwHighDateTime;
        return static_cast<double>(v.QuadPart) * 1e-7; // 100 ns units
    };
    return to_seconds(kernel) + to_seconds(user);
#else
    timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) {
        return 0.0;
    }
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
#endif
}

namespace {
std::atomic<std::uint64_t> sink{0};
}

BENCH_NOINLINE void consume(double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    sink.fetch_xor(bits, std::memory_order_relaxed);
}

BENCH_NOINLINE double checksum(const float* data, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(data[i]);
    }
    return sum;
}

// ----------------------------------------------------------------- checking

void check_bits(const float* expected, const float* actual, std::size_t n,
                const std::string& what) {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(expected + i, actual + i, sizeof(float)) != 0) {
            std::ostringstream message;
            message << what << ": result differs from the reference at index " << i << " (expected "
                    << expected[i] << ", got " << actual[i] << ")";
            throw std::runtime_error(message.str());
        }
    }
}

void check_close(const float* expected, const float* actual, std::size_t n, double rel_tol,
                 const std::string& what) {
    for (std::size_t i = 0; i < n; ++i) {
        const double e = expected[i], a = actual[i];
        const bool both_nan = std::isnan(e) && std::isnan(a);
        const bool close = std::abs(a - e) <= rel_tol * (1.0 + std::abs(e)) || (e == a);
        if (!both_nan && !close) {
            std::ostringstream message;
            message << what << ": result differs from the reference at index " << i << " (expected "
                    << e << ", got " << a << ")";
            throw std::runtime_error(message.str());
        }
    }
}

void check_value(double expected, double actual, double rel_tol, const std::string& what,
                 double scale) {
    if (!(std::abs(actual - expected) <= rel_tol * (1.0 + std::abs(expected) + scale))) {
        std::ostringstream message;
        message << what << ": result differs from the reference (expected " << expected << ", got "
                << actual << ")";
        throw std::runtime_error(message.str());
    }
}

// ------------------------------------------------------------------- inputs

void fill_a(float* data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i % 13) / 8 - 0.75f;
    }
}

void fill_b(float* data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i % 17) / 8 - 1.0f;
    }
}

Arrays::Arrays(std::size_t n, int count) {
    // Round n up to a whole 4 KB page (1024 floats) and add 1 KB, so each array starts 1 KB
    // further along within its page than the one before.
    stride_ = (n + 1023) / 1024 * 1024 + 256;
    storage_.assign(stride_ * static_cast<std::size_t>(count), 0.0f);
}

std::vector<int> thread_counts(int max_threads) {
    std::set<int> wanted{1, 2, 4, 8, 16, max_threads / 2, max_threads};
    std::vector<int> counts;
    for (int t : wanted) {
        if (t >= 1 && t <= max_threads) {
            counts.push_back(t);
        }
    }
    return counts;
}

// -------------------------------------------------------------- environment

std::string cpu_name() {
    std::string name;
#if defined(BENCH_X86)
    unsigned regs[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, static_cast<int>(0x80000000));
    const unsigned max_leaf = static_cast<unsigned>(info[0]);
    if (max_leaf >= 0x80000004u) {
        char brand[49] = {0};
        for (unsigned i = 0; i < 3; ++i) {
            __cpuid(info, static_cast<int>(0x80000002u + i));
            std::memcpy(brand + 16 * i, info, 16);
        }
        name = brand;
    }
#else
    if (__get_cpuid(0x80000000u, &regs[0], &regs[1], &regs[2], &regs[3]) &&
        regs[0] >= 0x80000004u) {
        char brand[49] = {0};
        for (unsigned i = 0; i < 3; ++i) {
            __get_cpuid(0x80000002u + i, &regs[0], &regs[1], &regs[2], &regs[3]);
            std::memcpy(brand + 16 * i, regs, 16);
        }
        name = brand;
    }
#endif
#elif defined(__APPLE__)
    char brand[256] = {0};
    std::size_t size = sizeof brand;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) {
        name = brand;
    }
#else
    std::ifstream cpuinfo("/proc/cpuinfo");
    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.rfind("model name", 0) == 0 || line.rfind("Model", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                name = line.substr(colon + 1);
                break;
            }
        }
    }
#endif
    const auto first = name.find_first_not_of(' ');
    const auto last = name.find_last_not_of(' ');
    return first == std::string::npos ? "unknown" : name.substr(first, last - first + 1);
}

bool avx2_supported() {
#if defined(BENCH_X86) && defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) {
        return false;
    }
    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) {
        return false;
    }
    if ((_xgetbv(0) & 0x6) != 0x6) { // the OS saves the YMM registers
        return false;
    }
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#elif defined(BENCH_X86) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}

std::string compiler_name() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

bool is_release_build() {
#ifdef NDEBUG
    return true;
#else
    return false;
#endif
}

namespace {

std::string openmp_name() {
#ifdef _OPENMP
    switch (_OPENMP) {
    case 200203:
        return "yes (2.0)";
    case 200505:
        return "yes (2.5)";
    case 200805:
        return "yes (3.0)";
    case 201107:
        return "yes (3.1)";
    case 201307:
        return "yes (4.0)";
    case 201511:
        return "yes (4.5)";
    case 201811:
        return "yes (5.0)";
    case 202011:
        return "yes (5.1)";
    case 202111:
        return "yes (5.2)";
    default:
        return "yes (" + std::to_string(_OPENMP) + ")";
    }
#else
    return "no";
#endif
}

} // namespace

void print_header(const std::string& title, const Config& config,
                  const std::vector<std::string>& notes) {
    std::cout << "=== " << title << " ===\n"
              << "cpu: " << cpu_name()
              << " | hardware threads: " << std::thread::hardware_concurrency()
              << " | OpenMP threads: " << mensurflow::cpu::max_threads() << '\n'
              << "compiler: " << compiler_name() << " | OpenMP: " << openmp_name()
              << " | AVX2: " << (avx2_supported() ? "yes" : "no")
              << " | build: " << (is_release_build() ? "Release" : "NOT Release") << '\n';
    if (!is_release_build()) {
        std::cout << "WARNING: this is not an optimized Release build; the numbers mean nothing.\n";
    }
    std::cout << "timing: " << config.warmups << " warm-up call(s), median of " << config.repeats
              << " samples (" << config.slow_repeats << " if a call takes over "
              << config.slow_call_s << " s), each sample at least " << config.min_sample_s * 1e3
              << " ms\n"
              << "for comparable numbers: plugged in, High performance power plan, other programs "
                 "closed, run twice\n";
    for (const std::string& note : notes) {
        std::cout << note << '\n';
    }
}

// ------------------------------------------------------------------- table

namespace {

std::string fixed(double value, int decimals) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

// Milliseconds with at least 4 decimals, and more for very fast calls so that a 50 ns call does not
// print as 0.0000: about 4 significant digits, up to 7 decimals.
std::string milliseconds(double seconds) {
    const double ms = seconds * 1e3;
    int decimals = 4;
    if (ms > 0.0) {
        decimals = std::max(4, std::min(7, 3 - static_cast<int>(std::floor(std::log10(ms)))));
    }
    return fixed(ms, decimals);
}

std::string speedup(const Sample* baseline, const Sample& sample) {
    if (baseline == nullptr || sample.median_s <= 0.0) {
        return "-";
    }
    return fixed(baseline->median_s / sample.median_s, 2) + "x";
}

} // namespace

Table::Table(std::string rate_unit, double rate_scale, std::string baseline_a,
             std::string baseline_b)
    : rate_unit_(std::move(rate_unit)), rate_scale_(rate_scale), baseline_a_(std::move(baseline_a)),
      baseline_b_(std::move(baseline_b)) {}

void Table::section(const std::string& title) {
    std::cout << "\n== " << title << " ==\n"
              << std::setw(10) << "n" << std::setw(13) << "impl" << std::setw(9) << "threads"
              << std::setw(12) << "median ms" << std::setw(12) << "min ms" << std::setw(12)
              << "max ms" << std::setw(11) << rate_unit_;
    if (!baseline_a_.empty()) {
        std::cout << std::setw(11) << baseline_a_;
    }
    if (!baseline_b_.empty()) {
        std::cout << std::setw(11) << baseline_b_;
    }
    std::cout << std::setw(10) << "cpu/wall" << '\n';
}

void Table::row(const std::string& n, const std::string& impl, const std::string& threads,
                const Sample& sample, double work, const Sample* baseline_a,
                const Sample* baseline_b) {
    const std::string rate =
        work > 0.0 && sample.median_s > 0.0 ? fixed(work / sample.median_s / rate_scale_, 3) : "-";
    const std::string cpu_over_wall =
        sample.cpu_over_wall > 0.0 ? fixed(sample.cpu_over_wall, 2) : "-";
    std::cout << std::setw(10) << n << std::setw(13) << impl << std::setw(9) << threads
              << std::setw(12) << milliseconds(sample.median_s) << std::setw(12)
              << milliseconds(sample.min_s) << std::setw(12) << milliseconds(sample.max_s)
              << std::setw(11) << rate;
    if (!baseline_a_.empty()) {
        std::cout << std::setw(11) << speedup(baseline_a, sample);
    }
    if (!baseline_b_.empty()) {
        std::cout << std::setw(11) << speedup(baseline_b, sample);
    }
    std::cout << std::setw(10) << cpu_over_wall << '\n';
}

void Table::blank() { std::cout << '\n'; }

} // namespace bench
