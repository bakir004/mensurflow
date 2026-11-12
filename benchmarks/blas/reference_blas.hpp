// Reference BLAS libraries loaded at run time, for benchmarks that compare against more than one.
//
// OpenBLAS, MKL and AOCL-BLIS all export the same BLAS names, so they cannot be linked into one
// executable. Here each library is opened from its own DLL (LoadLibrary / dlopen) and only the few
// functions the benchmark needs are looked up. A library that is not installed is skipped.
//
// Where a library is looked for: the environment variable first, then the path CMake found at
// configure time (-DMKL_ROOT=..., -DAOCL_BLIS_ROOT=..., see docs/BENCHMARKING.md).
//   MKL        MENSURFLOW_MKL_DLL        mkl_rt.3.dll / libmkl_rt.so (LP64 interface, Intel OpenMP)
//   AOCL-BLIS  MENSURFLOW_AOCL_BLIS_DLL  AOCL-LibBlis-Win-MT-dll.dll / libblis-mt.so
//
// MKL and AOCL-BLIS each bring their own copy of the LLVM/Intel OpenMP runtime (libiomp5md and
// libomp). Two copies in one process abort unless KMP_DUPLICATE_LIB_OK is set; it is set here. The
// two runtimes never run at the same time in a benchmark, but their idle workers spin for 200 ms
// (KMP_BLOCKTIME) after a call, so the benchmark must pause longer than that between libraries.
//
// Binding their threads to cores (OMP_PLACES=cores OMP_PROC_BIND=close in the environment) is the
// user's choice and applies to both runtimes. A runtime that binds also binds the thread that
// called it, and the next runtime to start would take that one core for the whole machine. So each
// library is started here, with one threaded product, from a thread that may run anywhere.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace reference_blas {

struct Library {
    std::string name;   // short column label
    std::string detail; // version and code path, for the benchmark header
    std::function<void(int)> set_threads;
    // Row-major C[m,n] = op(A) * op(B), alpha 1, beta 0. a_transposed: A is stored as [k, m];
    // b_transposed: B is stored as [n, k]. lda, ldb, ldc are the row lengths of the stored arrays.
    std::function<void(bool a_transposed, bool b_transposed, int m, int n, int k, const float* a,
                       int lda, const float* b, int ldb, float* c, int ldc)>
        sgemm;
};

namespace detail {

inline void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

inline std::string get_env(const char* name) {
#ifdef _WIN32
    char buffer[32768];
    const DWORD length = GetEnvironmentVariableA(name, buffer, sizeof(buffer));
    return length > 0 && length < sizeof(buffer) ? std::string(buffer, length) : std::string();
#else
    const char* value = std::getenv(name);
    return value ? value : "";
#endif
}

// Opens the library with its own folder first on the DLL search path, because MKL loads its
// threading and kernel DLLs by name on the first call.
inline void* open(const std::string& path) {
    if (path.empty()) {
        return nullptr;
    }
#ifdef _WIN32
    const auto slash = path.find_last_of("/\\");
    if (slash != std::string::npos) {
        set_env("PATH", path.substr(0, slash) + ";" + get_env("PATH"));
    }
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

template <class Function> Function symbol(void* library, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<Function>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return reinterpret_cast<Function>(dlsym(library, name));
#endif
}

// Lets the calling thread run on every processor of the process again.
inline void release_calling_thread() {
#ifdef _WIN32
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) && process_mask) {
        SetThreadAffinityMask(GetCurrentThread(), process_mask);
    }
#endif
}

// One product large enough for the library to start its threads.
template <class Sgemm> void start_threads(const Sgemm& sgemm) {
    const int n = 256;
    std::vector<float> a(static_cast<std::size_t>(n) * n, 1.0f);
    std::vector<float> c(static_cast<std::size_t>(n) * n, 0.0f);
    sgemm(false, false, n, n, n, a.data(), n, a.data(), n, c.data(), n);
    release_calling_thread();
}

inline bool module_loaded(const char* name) {
#ifdef _WIN32
    return GetModuleHandleA(name) != nullptr;
#else
    (void)name;
    return false;
#endif
}

inline std::string path_for(const char* variable, const char* configured) {
    const std::string from_env = get_env(variable);
    return from_env.empty() ? std::string(configured) : from_env;
}

} // namespace detail

#ifndef MENSURFLOW_MKL_DLL_DEFAULT
#define MENSURFLOW_MKL_DLL_DEFAULT ""
#endif
#ifndef MENSURFLOW_AOCL_BLIS_DLL_DEFAULT
#define MENSURFLOW_AOCL_BLIS_DLL_DEFAULT ""
#endif

// MKL through its single dynamic library (mkl_rt): cblas_sgemm, LP64 integers. MKL is left in its
// default "dynamic" mode, in which the thread count asked for is an upper limit (it does not use
// more threads than physical cores). MENSURFLOW_MKL_DYNAMIC=0 in the environment turns that off.
inline bool load_mkl(Library& out) {
    detail::set_env("KMP_DUPLICATE_LIB_OK", "TRUE");
    detail::release_calling_thread();
    void* library = detail::open(detail::path_for("MENSURFLOW_MKL_DLL", MENSURFLOW_MKL_DLL_DEFAULT));
    if (!library) {
        return false;
    }
    using Sgemm = void (*)(int, int, int, int, int, int, float, const float*, int, const float*,
                           int, float, float*, int);
    const auto sgemm = detail::symbol<Sgemm>(library, "cblas_sgemm");
    const auto set_threads = detail::symbol<void (*)(int)>(library, "MKL_Set_Num_Threads");
    const auto set_dynamic = detail::symbol<void (*)(int)>(library, "MKL_Set_Dynamic");
    const auto version = detail::symbol<void (*)(char*, int)>(library, "MKL_Get_Version_String");
    if (!sgemm || !set_threads) {
        return false;
    }
    const bool exact_threads = set_dynamic && detail::get_env("MENSURFLOW_MKL_DYNAMIC") == "0";
    if (exact_threads) {
        set_dynamic(0);
    }
    out.name = "mkl";
    out.set_threads = [set_threads](int threads) { set_threads(threads); };
    out.sgemm = [sgemm](bool ta, bool tb, int m, int n, int k, const float* a, int lda,
                        const float* b, int ldb, float* c, int ldc) {
        sgemm(101, ta ? 112 : 111, tb ? 112 : 111, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
    };
    // The first product also makes MKL load its kernel library; which one is the code path.
    detail::start_threads(out.sgemm);
    char text[256] = "MKL";
    if (version) {
        version(text, static_cast<int>(sizeof(text)));
        text[sizeof(text) - 1] = '\0';
    }
    std::string trimmed = text;
    while (!trimmed.empty() && trimmed.back() == ' ') {
        trimmed.pop_back();
    }
    std::string kernels;
    for (const char* name : {"avx10", "avx512", "avx2", "mc3", "def"}) {
        for (const char* major : {"3", "2"}) {
            const std::string file = std::string("mkl_") + name + "." + major + ".dll";
            if (detail::module_loaded(file.c_str())) {
                kernels += (kernels.empty() ? "" : " ") + file;
            }
        }
    }
    out.detail = "MKL: " + trimmed + (kernels.empty() ? "" : ", kernels " + kernels) +
                 (exact_threads ? ", MKL_DYNAMIC off" : "");
    return true;
}

// AOCL-BLIS (AMD's BLIS) through the Fortran BLAS name sgemm_, which is the entry point that has
// AMD's small-matrix and thread-count decisions. sgemm_ is column-major: a row-major product is
// the column-major product of the same arrays with the operands swapped (C^T = B^T * A^T). The
// integers are passed as 64-bit values by address, which a 32-bit (LP64) build reads correctly too.
inline bool load_aocl_blis(Library& out) {
    detail::set_env("KMP_DUPLICATE_LIB_OK", "TRUE");
    detail::release_calling_thread();
    void* library =
        detail::open(detail::path_for("MENSURFLOW_AOCL_BLIS_DLL", MENSURFLOW_AOCL_BLIS_DLL_DEFAULT));
    if (!library) {
        return false;
    }
    using Sgemm = void (*)(const char*, const char*, const std::int64_t*, const std::int64_t*,
                           const std::int64_t*, const float*, const float*, const std::int64_t*,
                           const float*, const std::int64_t*, const float*, float*,
                           const std::int64_t*);
    const auto sgemm = detail::symbol<Sgemm>(library, "sgemm_");
    const auto set_threads =
        detail::symbol<void (*)(std::int64_t)>(library, "bli_thread_set_num_threads");
    const auto version = detail::symbol<const char* (*)()>(library, "bli_info_get_version_str");
    const auto arch_id = detail::symbol<int (*)()>(library, "bli_arch_query_id");
    const auto arch_string = detail::symbol<const char* (*)(int)>(library, "bli_arch_string");
    if (!sgemm || !set_threads) {
        return false;
    }
    out.name = "aocl";
    out.set_threads = [set_threads](int threads) { set_threads(threads); };
    out.sgemm = [sgemm](bool ta, bool tb, int m, int n, int k, const float* a, int lda,
                        const float* b, int ldb, float* c, int ldc) {
        const std::int64_t rows = n, cols = m, depth = k, la = lda, lb = ldb, lc = ldc;
        const float one = 1.0f, zero = 0.0f;
        sgemm(tb ? "T" : "N", ta ? "T" : "N", &rows, &cols, &depth, &one, b, &lb, a, &la, &zero, c,
              &lc);
    };
    detail::start_threads(out.sgemm);
    out.detail = std::string("AOCL-BLIS: ") + (version ? version() : "unknown version");
    if (arch_id && arch_string) {
        out.detail += std::string(", configuration ") + arch_string(arch_id());
    }
    return true;
}

} // namespace reference_blas
