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

EXTRA_C_FILES := helpers.c sort.c
EXTRA_C_FILES := $(patsubst %, common/%, $(EXTRA_C_FILES))
_EXTRA_OBJS += $(patsubst %.c, %.o, $(EXTRA_C_FILES))


