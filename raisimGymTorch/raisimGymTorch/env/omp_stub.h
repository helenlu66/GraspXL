#pragma once

// Minimal OpenMP stubs for macOS builds without libomp.
static inline void omp_set_num_threads(int) {}
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_max_threads() { return 1; }
