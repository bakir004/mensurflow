# CPU/GPU BLAS and ML Kernel Library with Autotuning

This literature review covers four papers relevant to a small C++ library for
basic BLAS and machine learning operations on CPUs and GPUs. The CPU and
vendor GPU backends will call existing libraries, while the Vulkan backend
will use manually implemented compute kernels.

## 1. BLIS: A Framework for Rapidly Instantiating BLAS Functionality

**Authors:** Field G. Van Zee and Robert A. van de Geijn.

**Publication:** ACM Transactions on Mathematical Software, 41(3), 2015.

### Summary

BLAS defines common operations on vectors and matrices, such as matrix
multiplication. BLIS (*BLAS-like Library Instantiation Software*) is an
open-source CPU library and framework presented by the authors.

The main idea is to build larger operations from a small set of reusable
functions. For matrix multiplication, a small function called a
*micro-kernel* computes one block of the result. The library also arranges
data to make better use of fast CPU memory.

The paper explains this design and compares its performance with OpenBLAS,
ATLAS, and Intel MKL, which are other CPU numerical libraries. The results
show that BLIS can achieve similar performance on the tested operations.

[Paper](https://doi.org/10.1145/2764454) ·
[Author manuscript](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf).

## 2. CLBlast: A Tuned OpenCL BLAS Library

**Author:** Cedric Nugteren.

**Publication:** International Workshop on OpenCL, 2018.

### Summary

CLBlast is a BLAS library that uses OpenCL to run vector and matrix
operations on different devices, especially GPUs. Examples include adding
a scaled vector to another vector (AXPY) and multiplying matrices (GEMM).

Its kernels have adjustable settings, such as how many parallel workers
form a group and how much work each worker performs. The library can test
different settings to find a faster configuration for a device and input size.

The caller controls when data moves between CPU and GPU memory. This helps
avoid unnecessary transfers. The experiments show that settings suited to
the device and problem size can improve performance.

[Paper and full text](https://arxiv.org/abs/1705.05249) ·
[DOI](https://doi.org/10.1145/3204919.3204924).

## 3. vkpolybench: A Crossplatform Vulkan Compute Port of the PolyBench/GPU Benchmark Suite

**Authors:** Nicola Capodieci and Roberto Cavicchioli.

**Publication:** SoftwareX, 15, article 100793, 2021.

### Summary

vkpolybench is a collection of numerical benchmarks implemented using
Vulkan Compute. It adapts an existing benchmark collection so that its
calculations can run through Vulkan.

C++ code prepares the data and starts GPU work. Small GPU programs called
compute shaders perform the calculations. The shaders are written in GLSL
and compiled to SPIR-V. A helper module, `vkcomp`, handles tasks such as
creating buffers, transferring data, and launching shaders.

The paper includes an AXPY example, and the accompanying code also contains
matrix multiplication. The benchmarks measure performance and compare GPU
results with CPU results. This is a benchmark collection rather than a
complete BLAS library.

[Paper](https://doi.org/10.1016/j.softx.2021.100793) ·
[University record and PDF](https://iris.unimore.it/handle/11380/1256098) ·
[Accompanying code](https://github.com/ElsevierSoftwareX/SOFTX_2020_86).

## 4. CLTune: A Generic Auto-Tuner for OpenCL Kernels

**Authors:** Cedric Nugteren and Valeriu Codreanu.

**Publication:** IEEE International Symposium on Embedded Multicore/Many-core
Systems-on-Chip (MCSoC), 2015; arXiv version published in 2017.

### Summary

CLTune is a tool that automatically tests settings for OpenCL kernels.
This process is called *autotuning*. The user provides the settings to
change and their possible values. CLTune runs different combinations,
measures their execution time, and selects the fastest configuration it finds.

For example, it can change how many workers form a group or how large a
matrix block each group processes. It supports several search methods,
including testing every combination or testing a smaller selection.

The paper evaluates matrix multiplication and two-dimensional convolution.
The results show that the best settings can change with the device and
input dimensions.

[Paper and full text](https://arxiv.org/abs/1703.06503) ·
[DOI](https://doi.org/10.1109/MCSoC.2015.10).

## Overview

| Paper | Main topic |
| --- | --- |
| BLIS | A CPU library for vector and matrix operations. |
| CLBlast | A BLAS library using OpenCL, with adjustable kernel settings. |
| vkpolybench | Numerical benchmarks implemented using Vulkan Compute. |
| CLTune | A tool for automatically finding faster OpenCL kernel settings. |
