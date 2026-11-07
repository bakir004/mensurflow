// Where the threads of a parallel kernel run. On a CPU with SMT (two hardware threads per core) the
// operating system may put two of our threads on the two halves of one core and leave another core
// idle; compute-bound code then runs at half speed on the shared core and every barrier waits for it
// (a 12-thread matmul on 12 cores measured anywhere between 600 and 1050 GFLOP/s because of this).
// Pinning each thread to its own physical core avoids it. The OpenMP runtime of MSVC cannot do that
// (OpenMP 2.0), so the threads do it themselves.
#pragma once

namespace mensurflow::cpu {

// Number of physical cores, or 0 when the topology cannot be read on this platform.
int physical_core_count();

// How a thread is restricted:
//   Core:    to the hardware threads of one physical core (either SMT half). The OS may still move it
//            between the two halves, but never to another core.
//   Logical: to one hardware thread (the first SMT half of its core).
//   Ideal:   not restricted at all, but the scheduler is told which hardware thread to prefer (Windows:
//            the "ideal processor"). The thread can still move away, for example from a core that a
//            background program is using. On Linux this falls back to Core.
enum class PinMode { Core, Logical, Ideal };

// Restricts the calling thread to physical core (index modulo physical_core_count()), taking the
// cores in order from the fastest kind (performance cores before efficiency cores on hybrid CPUs).
// With more threads than cores, thread i and thread i + cores share a core. Returns false, and
// changes nothing, if the platform is not supported or the call failed.
bool pin_current_thread(int index, PinMode mode);

// The pin state of the calling thread, managed so that unpinning gives back what the thread had.
// Threads are placed compactly: thread i goes to core i, so a team fills one chip (one L3 cache) before
// it spills onto the next. Measured on a 12-core CPU made of two chiplets, that beat spreading the
// threads over both chiplets by 3-45% at 2-6 threads (the packed A and B panels are shared, and a
// cache line that has to cross between chiplets is slow). With more threads than cores, thread i and
// thread i + cores share a core. `wanted` is:
//   wanted 0: unpinned (if we pinned this thread before, its original restriction is restored)
//   wanted 1: pinned to a physical core, as PinMode::Core (index modulo physical_core_count())
//   wanted 2: pinned to one hardware thread of that core, as PinMode::Logical
//   wanted 3: only a preference for that core, as PinMode::Ideal
// Cheap when nothing changes (it remembers what it did on each thread), so a kernel can call it at
// the start of every parallel region.
void set_thread_pin(int index, int wanted);
} // namespace mensurflow::cpu
