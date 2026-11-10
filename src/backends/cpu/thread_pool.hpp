// A small pool of persistent worker threads for kernels that want more control than OpenMP 2.0 (the
// version MSVC implements) gives: how long idle workers spin before they sleep, a cheap barrier, and
// threads that keep their place (worker i is always thread i of a team, so a thread pinned to a core
// stays there between calls).
//
//   pool_run(team, body)   runs body(tid) on `team` threads and returns when all are done. The calling
//                          thread is tid 0; pool workers are tids 1 .. team - 1.
//   pool_barrier()         inside body: waits until every thread of the team has reached it.
//   pool_thread_index()    inside body: this thread's tid; -1 on a thread that is not running a body.
//
// One team runs at a time. pool_run returns false, doing nothing, when the pool is already running a
// team (a concurrent caller, or a call from inside a body); the caller then uses another way to run.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// How long an idle worker keeps spinning on the job counter before it sleeps (microseconds). Spinning
// makes the next call start immediately but keeps a core busy; sleeping costs a wake-up of tens of
// microseconds. Default 2000 (2 ms).
void set_pool_spin_us(std::int64_t microseconds);
std::int64_t pool_spin_us();

// A type-erased reference to the body: a function pointer plus the object it is called on, so no
// allocation happens per call.
struct PoolBody {
    void (*call)(void* object, int tid);
    void* object;
};

// `skip` (1 .. team - 1, or 0 for nobody) leaves that worker out: the team is then the other
// team - 1 threads, with the tids 0 .. team - 1 except skip. For a caller that is running on the core
// worker `skip` is pinned to: the two would share the core.
bool pool_run(int team, PoolBody body, int skip = 0);

template <typename F>
bool pool_run(int team, F& body, int skip = 0) {
    return pool_run(team, PoolBody{[](void* object, int tid) { (*static_cast<F*>(object))(tid); },
                                   static_cast<void*>(&body)},
                    skip);
}

void pool_barrier();
int pool_thread_index();

// Number of worker threads currently started (they are started on first use, up to team - 1).
int pool_worker_count();

} // namespace mensurflow::cpu
