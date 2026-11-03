// Execution: where an operation runs, chosen per call. Every op takes one as its last argument.
//
//   Execution::CPU   run on the CPU.
//   Execution::GPU   run on a GPU: the one the inputs already live on, otherwise the first
//                    available of CUDA and Metal. Throws runtime_error if there is none.
//   Execution::Auto  let the library choose. For now it always chooses the CPU.
//
// Where an op runs is independent of where its data lives. If they differ, the inputs are copied
// to the execution device and the result is copied back, so the returned tensor is always on the
// inputs' device. Those copies cost time; keep data on the device you run on to avoid them.
#pragma once

namespace mensurflow {

enum class Execution { Auto, CPU, GPU };

// "auto", "cpu", "gpu".
const char* execution_name(Execution execution);

} // namespace mensurflow
