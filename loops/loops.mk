# ----------------------------------------------------------------------------
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

LOOPS_IN_SRC += \
  loop_001  \
  loop_002  \
  loop_003  \
  loop_004  \
  loop_005  \
  loop_006  \
  loop_008  \
  loop_009  \
  loop_010  \
  loop_012  \
  loop_019  \
  loop_022  \
  loop_023  \
  loop_024  \
  loop_025  \
  loop_026  \
  loop_027  \
  loop_028  \
  loop_029  \
  loop_031  \
  loop_032  \
  loop_033  \
  loop_034  \
  loop_035  \
  loop_036  \
  loop_037  \
  loop_038  \
  loop_040  \
  loop_101  \
  loop_102  \
  loop_103  \
  loop_104  \
  loop_105  \
  loop_106  \
  loop_107  \
  loop_108  \
  loop_109  \
  loop_110  \
  loop_111  \
  loop_112  \
  loop_113  \
  loop_114  \
  loop_120  \
  loop_121  \
  loop_122  \
  loop_123  \
  loop_124  \
  loop_126  \
  loop_127  \
  loop_128  \
  loop_130  \
  loop_135  \
  loop_136  \
  loop_137  \
  loop_201  \
  loop_202  \
  loop_204  \
  loop_205  \
  loop_206  \
  loop_207  \
  loop_208  \
  loop_210  \
  loop_211  \
  loop_212  \
  loop_215  \
  loop_216  \
  loop_217  \
  loop_218  \
  loop_219  \
  loop_220  \
  loop_221  \
  loop_222  \
  loop_223  \
  loop_231  \
  loop_245

LOOPS += $(LOOPS_IN_SRC)
SRCFILES += $(addprefix loops/,$(addsuffix .c,$(LOOPS_IN_SRC)))

EXTRA_C_FILES := sample_json.c static_rand.c strops.c matmul_fp32.c
EXTRA_C_FILES := $(patsubst %, loops/%, $(EXTRA_C_FILES))
_EXTRA_OBJS += $(patsubst %.c, %.o, $(EXTRA_C_FILES))


