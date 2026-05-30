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

#include <signal.h>
#include <unistd.h>

#ifndef STANDALONE

// Loop information
#define LOOP(n, ...)  loop_function_t __ptr_loop_##n = NULL;
#include "loops.inc"
#undef LOOP

// ---------------------------------------------------------------------------
// Watchdog: kill the process if a candidate kernel runs for too long. Without
// this, an LLM-produced infinite loop or a pathologically slow implementation
// can hang the SSH-side eval harness up to its 300s socket timeout, wasting
// budget and forcing tear-down. Default 60s; tunable via -t/--timeout.
// Exit code 124 is the conventional `timeout(1)` exit so callers can detect it.
// ---------------------------------------------------------------------------
static int g_timeout_s = 60;

static void __watchdog_handler(int sig) {
    (void)sig;
    /* write(2) is async-signal-safe; printf is not. */
    static const char msg[] = "ABORT: kernel watchdog timeout reached\n";
    (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
    _exit(124);
}

static void __install_watchdog(void) {
    if (g_timeout_s <= 0) return;            /* 0 / negative disables */
    struct sigaction sa = {0};
    sa.sa_handler = __watchdog_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, NULL);
    alarm((unsigned int)g_timeout_s);
}

// Command line parse result
typedef struct Options {
  bool print;
  long iterations;
  int loop;
  const char *name;
} Options;

static long parseLong(const char *flag, const char *arg, int base) {
  char *end = NULL;
  long v = strtol(arg, &end, base);
  if (end != arg + strlen(arg)) {
    fprintf(stderr, "Expected integer after '%s'\n", flag);
    exit(1);
  }
  return v;
}

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
      opt.iterations = parseLong(argv[i - 1], argv[i], 10);
    } else if (strcmp("-k", argv[i]) == 0 || strcmp("--loop", argv[i]) == 0) {
      i++;
      if (i == argc) {
        fprintf(stderr, "Expected loop number after '%s'\n", argv[i - 1]);
        exit(1);
      }
      opt.loop = (int)parseLong(argv[i - 1], argv[i], 16);
      opt.name = argv[i];
    } else if (strcmp("-w", argv[i]) == 0 || strcmp("--warmup", argv[i]) == 0) {
      i++;
      if (i == argc) {
        fprintf(stderr, "Expected count after '%s'\n", argv[i - 1]);
        exit(1);
      }
      long v = parseLong(argv[i - 1], argv[i], 10);
      if (v < 0) v = 0;
      g_warmup_iters = (int)v;
    } else if (strcmp("-r", argv[i]) == 0 || strcmp("--reps", argv[i]) == 0) {
      i++;
      if (i == argc) {
        fprintf(stderr, "Expected count after '%s'\n", argv[i - 1]);
        exit(1);
      }
      long v = parseLong(argv[i - 1], argv[i], 10);
      if (v < 1) v = 1;
      g_reps = (int)v;
    } else if (strcmp("-t", argv[i]) == 0 || strcmp("--timeout", argv[i]) == 0) {
      i++;
      if (i == argc) {
        fprintf(stderr, "Expected seconds after '%s'\n", argv[i - 1]);
        exit(1);
      }
      long v = parseLong(argv[i - 1], argv[i], 10);
      g_timeout_s = (int)v;        /* 0 / negative disables the watchdog */
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
  __install_watchdog();
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
