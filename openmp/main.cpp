#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#if defined(DOT_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(DOT_USE_CBLAS)
#include <cblas.h>
#endif

#if defined(_MSC_VER)
#define DOT_NOINLINE __declspec(noinline)
#else
#define DOT_NOINLINE __attribute__((noinline))
#endif

double dot_sequential(const double* a, const double* b, std::size_t n);

DOT_NOINLINE double dot_simd(const double* a, const double* b, std::size_t n) {
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

DOT_NOINLINE double dot_parallel_simd(const double* a, const double* b, std::size_t n) {
    double sum = 0.0;
#pragma omp parallel for simd schedule(static) reduction(+ : sum)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Extra independent accumulators can reduce dependency-chain latency. Compilers
// may already do this for dot_parallel_simd, so this is a candidate, not a promise
// of better performance. Handle the tail for arbitrary input sizes.
DOT_NOINLINE double dot_unrolled(const double* a, const double* b, std::size_t n) {
    const std::size_t groups = n / 4;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
#pragma omp parallel for simd schedule(static) reduction(+ : s0, s1, s2, s3)
    for (std::ptrdiff_t g = 0; g < static_cast<std::ptrdiff_t>(groups); ++g) {
        const std::size_t i = 4 * g;
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    double sum = (s0 + s1) + (s2 + s3);
    for (std::size_t i = groups * 4; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(DOT_USE_ACCELERATE) || defined(DOT_USE_CBLAS)
DOT_NOINLINE double dot_blas(const double* a, const double* b, std::size_t n) {
    return cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}
#endif

using Kernel = double (*)(const double*, const double*, std::size_t);

struct Method {
    const char* name;
    Kernel kernel;
    std::vector<double> times;
};

// Observable consumption plus non-inlined kernels prevents dead-result removal.
volatile double result_sink = 0.0;

std::size_t parse_positive(const char* text) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Arguments must be positive decimal integers");
    }
    const auto parsed = std::stoull(value);
    if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("Argument is zero or too large");
    }
    return static_cast<std::size_t>(parsed);
}

int main(int argc, char** argv) {
    try {
        if (argc > 4) {
            throw std::invalid_argument("Usage: dot_product [elements] [trials] [threads]");
        }
        const std::size_t n = argc > 1 ? parse_positive(argv[1]) : 8'000'000;
        const std::size_t trials = argc > 2 ? parse_positive(argv[2]) : 9;
        const std::size_t threads =
            argc > 3 ? parse_positive(argv[3]) : static_cast<std::size_t>(omp_get_max_threads());
        // Also keeps the optional BLAS call within the common 32-bit index API.
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            threads > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("Elements and threads must fit in a positive int");
        }
        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(threads));

        std::vector<double> a(n), b(n);
        std::mt19937 generator(42);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        long double reference = 0.0L;
        long double absolute_products = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = distribution(generator);
            b[i] = distribution(generator);
            const long double product = static_cast<long double>(a[i]) * b[i];
            reference += product;
            absolute_products += std::abs(product);
        }
        // Reduction orders differ. Scale tolerance by input magnitude so near-zero
        // dot products do not fail a relative-only comparison.
        const long double tolerance = 1e-10L * std::max(1.0L, absolute_products);
        auto validate = [&](double result) {
            if (!std::isfinite(result) || std::abs(result - reference) > tolerance) {
                throw std::runtime_error("Dot product correctness check failed");
            }
        };

        std::vector<Method> methods{
            {"Sequential scalar", dot_sequential, {}},
            {"SIMD, one thread", dot_simd, {}},
            {"SIMD + threads", dot_parallel_simd, {}},
            {"SIMD + threads, unrolled", dot_unrolled, {}},
        };
#if defined(DOT_USE_ACCELERATE) || defined(DOT_USE_CBLAS)
        methods.push_back({"Vendor BLAS ddot", dot_blas, {}});
#endif
        int actual_threads = 1;
#pragma omp parallel
        {
#pragma omp single
            actual_threads = omp_get_num_threads();
        }
        std::cout << "Dot product: double precision, " << n << " elements, " << trials
                  << " trials\nOpenMP _OPENMP=" << _OPENMP << ", requested threads=" << threads
                  << ", observed team size=" << actual_threads << '\n'
                  << "Input storage: " << (2.0 * n * sizeof(double) / (1024 * 1024)) << " MiB\n";
        std::cout << "BLAS threading, if enabled, is controlled by its own library/settings.\n";

        for (auto& method : methods) {
            method.times.reserve(trials);
            for (int warmup = 0; warmup < 2; ++warmup) {
                const double result = method.kernel(a.data(), b.data(), n);
                validate(result);
                result_sink = result;
            }
        }
        // Rotate method order to reduce consistent order/temperature bias.
        for (std::size_t trial = 0; trial < trials; ++trial) {
            for (std::size_t offset = 0; offset < methods.size(); ++offset) {
                auto& method = methods[(trial + offset) % methods.size()];
                const auto start = std::chrono::steady_clock::now();
                const double result = method.kernel(a.data(), b.data(), n);
                const auto end = std::chrono::steady_clock::now();
                result_sink = result;
                validate(result); // Outside timing.
                method.times.push_back(std::chrono::duration<double>(end - start).count());
            }
        }
        for (auto& method : methods) {
            std::sort(method.times.begin(), method.times.end());
        }
        auto median = [](const std::vector<double>& times) {
            const auto mid = times.size() / 2;
            return times.size() % 2 ? times[mid] : (times[mid - 1] + times[mid]) / 2;
        };
        const double baseline = median(methods.front().times);
        std::cout << std::fixed << std::setprecision(3) << '\n'
                  << std::left << std::setw(30) << "Method" << std::right << std::setw(13)
                  << "Median ms" << std::setw(13) << "Min ms" << std::setw(13) << "Max ms"
                  << std::setw(12) << "Speedup" << std::setw(12) << "GB/s*" << '\n';
        for (const auto& method : methods) {
            const double seconds = median(method.times);
            std::cout << std::left << std::setw(30) << method.name << std::right << std::setw(13)
                      << seconds * 1000 << std::setw(13) << method.times.front() * 1000
                      << std::setw(13) << method.times.back() * 1000 << std::setw(12)
                      << (seconds > 0 ? baseline / seconds : 0) << std::setw(12)
                      << (seconds > 0 ? 2.0 * n * sizeof(double) / seconds / 1e9 : 0) << '\n';
        }
        std::cout << "\nAll results checked. Reference dot = " << std::setprecision(12)
                  << static_cast<double>(reference) << '\n'
                  << "* Effective input bandwidth; cached data may not come from DRAM.\n";
#if !defined(DOT_USE_ACCELERATE) && !defined(DOT_USE_CBLAS)
        std::cout << "Vendor BLAS comparison skipped: no supported CBLAS library enabled.\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
