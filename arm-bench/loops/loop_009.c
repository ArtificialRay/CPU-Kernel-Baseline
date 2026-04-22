/*----------------------------------------------------------------------------
#
#   Loop 009: Pointer chasing
#
#   Purpose:
#     Use of CTERM and BRK instructions.
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


typedef struct node {
  uint64_t payload;
  uint64_t payload2;
  struct node *next;
} node_t;

_Static_assert(sizeof(node_t) == 24, "node_t must be 192 bits");

struct loop_009_data {
  node_t *nodes;
  uint64_t res;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_009(struct loop_009_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_009(struct loop_009_data *restrict data) {
  node_t *nodes = data->nodes;

  uint64_t res = 0;
  for (node_t *p = nodes; p != NULL; p = p->next) {
    res ^= p->payload ^ p->payload2;
  }
  data->res = res;
}
#elif defined(HAVE_SVE_INTRINSICS)
static void inner_loop_009(struct loop_009_data *restrict data) {
  node_t *nodes = data->nodes;

  uint64_t res = 0;
  svuint64_t res_vec = svdup_u64(res);
  svuint64_t addr_vec = svdup_u64(res);
  node_t *p = nodes;
  uint64_t elems = svcntd();

  while (p != NULL) {
    svbool_t p1 = svpfalse_b();
    for (uint64_t i = 0; i < elems && p != NULL; i++) {
      p1 = svpnext_b64(svptrue_b8(), p1);
      addr_vec = svdup_u64_m(addr_vec, p1, (uint64_t)p);
      p = p->next;
    }
    svbool_t p2 = svbrka_z(svptrue_b8(), p1);
    svuint64_t payload_vec = svld1_gather_index_u64(p2, addr_vec, 0);
    svuint64_t payload2_vec = svld1_gather_index_u64(p2, addr_vec, 1);
    payload_vec = sveor_m(p2, payload_vec, payload2_vec);
    res_vec = sveor_m(p2, res_vec, payload_vec);
    res = sveorv(svptrue_b8(), res_vec);
  }
  data->res = res;
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_009(struct loop_009_data *restrict data) {
  node_t *nodes = data->nodes;

  uint64_t res = 0;
  node_t *p = nodes;

  asm volatile(
      // p0 = current partition mask
      "     ptrue   p0.b                            \n"  // p0 = all true
      "     dup     z0.d, #0                        \n"  // res' = 0
      // outer loop serialized sub-loop under p0
      "1:   pfalse  p1.b                            \n"  // first i
      // inner loop
      "2:   pnext   p1.d, p0, p1.d                  \n"  // next i in p0
      "     cpy     z1.d, p1/m, %[p]                \n"  // p'[i] = p
      "     ldr     %[p], [%[p], #16]               \n"  // p = p->next
      "     ctermeq %[p], xzr                       \n"  // p == NULL?
      "     b.tcont 2b                              \n"  // !(term | last)
      "     brka    p2.b, p0/z, p1.b                \n"  // p2[0..i] = T
      // vectorized main loop under p2
      "     ld1d    z2.d, p2/z, [z1.d, #0]          \n"  // p->payload
      "     ld1d    z3.d, p2/z, [z1.d, #8]          \n"  // p->payload2
      "     eor     z2.d, p2/m, z2.d, z3.d          \n"  // tmp = p->payload ^
                                                         // p->payload2
      "     eor     z0.d, p2/m, z0.d, z2.d          \n"  // res' ^= tmp
      "     cbnz    %[p], 1b                        \n"  // while p != NULL
      "     eorv    %d[res], p0, z0.d               \n"  // d0 = eor(res')
      // output operands, source operands, and clobber list
      : [res] "=&w"(res), [p] "+&r"(p)
      :
      : "v0", "v1", "v2", "x11", "p0", "p1", "cc", "memory");

  data->res = res;
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
// Scalar and Neon version (can't do better with Neon)
static void inner_loop_009(struct loop_009_data *restrict data) {
  node_t *nodes = data->nodes;

  uint64_t t0;
  uint64_t t1;
  uint64_t res = 0;
  node_t *p = nodes;

  if (p == NULL) {
    data->res = 0;
    return;
  }

  asm volatile(
      "1:   ldp   %[t0],  %[t1], [%[p]]       \n"
      "     ldr   %[p],   [%[p], #16]         \n"
      "     eor   %[t0],  %[t0], %[t1]        \n"
      "     eor   %[res], %[res], %[t0]       \n"
      "     cbnz  %[p],   1b                  \n"
      // output operands, source operands, and clobber list
      : [res] "+&r"(res), [p] "+&r"(p), [t0] "=&r"(t0), [t1] "=&r"(t1)
      :
      : "memory", "cc");

  data->res = res;
}
#else
static void inner_loop_009(struct loop_009_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

static void swap(node_t *p1, node_t *p2) {
  node_t *n1 = p1->next;
  node_t *n2 = p2->next;
  node_t *t = p2->next;

  p1->next = n2;
  p2->next = n1;
  t = n2->next;
  n2->next = n1->next;
  n1->next = t;
}

#ifndef SIZE
#define SIZE 1000
#endif

LOOP_DECL(009, NS_SVE_LOOP_ATTR)
{
  struct loop_009_data data = { .res = 0 };
  ALLOC_64B(data.nodes, SIZE, "node pool");
  fill_uint32((void *)data.nodes, sizeof(node_t) * SIZE / sizeof(uint32_t));

  for (int i = 0; i < SIZE - 1; i++) {
    data.nodes[i].next = &data.nodes[i + 1];
  }
  data.nodes[SIZE - 1].next = NULL;

  // shuffle list
  for (int i = 0; i < SIZE - 1; i++) {
    node_t *n1 = &data.nodes[i];
    if (!n1->next) {
      continue;
    }

    int lmt = 50;
    for (node_t *n2 = n1->next->next; n2 != NULL; n2 = n2->next) {
      if (!n2->next) {
        continue;
      }

      lmt--;
      if (lmt == 0) {
        break;  // limit amount of shuffling
      }

      if (n1->next->payload > n2->next->payload) {
        swap(n1, n2);
      }
    }
  }

  inner_loops_009(iters, &data);

  uint64_t res = data.res;
  uint64_t correct = 0xefdaee650e98f6e7UL;
  bool passed = res == correct;
#ifndef STANDALONE
  FINALISE_LOOP_I(9, passed, "0x%016"PRIx64, correct, res)
#endif
  return passed ? 0 : 1;
}
