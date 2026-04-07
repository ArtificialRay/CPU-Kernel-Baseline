/*----------------------------------------------------------------------------
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
----------------------------------------------------------------------------*/

#include "helpers.h"
#include "loops.h"

#ifndef STANDALONE

// Loop information
#define LOOP(n, ...)  loop_function_t __ptr_loop_##n = NULL;
#include "loops.inc"
#undef LOOP

// Command line parse result
typedef struct Options {
  bool print;
  long iterations;
  int loop;
  const char *name;
} Options;

static Options parseOptions(int argc, char **argv) {
  Options opt = {0};

  int i = 1;
  while (i < argc) {
    if (strcmp("-p", argv[i]) == 0 || strcmp("--print", argv[i]) == 0) {
      opt.print = true;
    } else if (strcmp("-n", argv[i]) == 0 || strcmp("--iters", argv[i]) == 0) {
      i++;
      if (i == argc) {
        fprintf(stderr, "Expected number of iterations after '%s'\n",
                argv[i - 1]);
        exit(1);
      }

      char *end = NULL;
      long temp = strtol(argv[i], &end, 10);
      if (end != argv[i] + strlen(argv[i])) {
        fprintf(stderr, "Expected number of iterations after '%s'\n",
                argv[i - 1]);
        exit(1);
      }

      opt.iterations = temp;
    } else if (strcmp("-k", argv[i]) == 0 || strcmp("--loop", argv[i]) == 0) {
      i++;
      if (i == argc) {
        fprintf(stderr, "Expected loop number after '%s'\n", argv[i - 1]);
        exit(1);
      }

      char *end = NULL;
      long temp = strtol(argv[i], &end, 16);
      if (end != argv[i] + strlen(argv[i])) {
        fprintf(stderr, "Expected number of iterations after '%s'\n",
                argv[i - 1]);
        exit(1);
      }

      opt.loop = temp;
      opt.name = argv[i];
    } else {
      fprintf(stderr, "Unexpected '%s'\n", argv[i]);
      exit(1);
    }
    i++;
  }

  if (opt.iterations <= 0) {
    opt.iterations = 1;
  }

  return opt;
}

int main(int argc, char **argv) {
  Options opt = parseOptions(argc, argv);
  switch (opt.loop) {
#define LOOP(n, name, purpose, ...)             \
    case 0x##n:                                 \
      if (__ptr_loop_##n == NULL) break;        \
      printf("Loop " #n " - " name "\n");       \
      printf(" - Purpose: " purpose "\n");      \
      return __ptr_loop_##n (opt.iterations);
#include "loops.inc"
#undef LOOP
    default: break;
  }
  fprintf(stderr, "Unexpected loop number %s\n", opt.name);
  return 1;
}

#else  // STANDALONE

#ifndef STANDALONE_ITERS
#error "Expected STANDALONE_ITERS"
#endif

#define NAME2_HIDDEN(a, b) a##b
#define NAME2(a, b) NAME2_HIDDEN(a, b)
#define NAME(prefix) NAME2(prefix, STANDALONE)

loop_function_t NAME(__ptr_loop_) = NULL;
int NAME(loop_)(int);

int main() { return NAME(loop_)(STANDALONE_ITERS); }

#endif
