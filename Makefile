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

ESC := $(shell printf '\033')
YELLOW := $(ESC)[33m
CYAN := $(ESC)[36m
RESET := $(ESC)[0m

LOOPS :=
SRCFILES :=
_EXTRA_OBJS :=
EXTRA_CFLAGS :=

# Number of loop iterations for standalone target
STANDALONE_ITERS ?= 20

# Limit for global allocator.
# 524288 KiB = 512 MB — enough for PERF_SIZE runs with working sets up to ~256 MB
# (e.g. fp32 inner product at SIZE=16M uses 2×16M×4B = 128 MB).
MAX_MEMORY_KIB ?= 524288

# Tuning problems sizes to default configuration (512 vector length and 64KiB buffer sizes)
TUNE_PBSIZE ?=

WANT_NATIVE ?= 1
WANT_SCALAR ?= 1
WANT_NEON ?= 1
WANT_SVE ?= 1
WANT_SVE2 ?= 1
WANT_SVE2P1 ?= 0
WANT_SME2 ?= 0
WANT_SME_SSVE ?= 0
WANT_SME2P1 ?= 0
WANT_SVE_INTRINSICS ?= 1
WANT_SME_INTRINSICS ?= 0
WANT_AUTOVEC_SVE ?= 0
WANT_AUTOVEC_SVE2 ?= 1

# Default toolchain paths
C_COMPILER ?= clang
C_LINKER ?= $(C_COMPILER)
OBJDUMP ?= llvm-objdump
FORMAT ?= clang-format

EXTRA_FLAGS ?=
EXTRA_LDFLAGS ?=
IFLAGS ?=
EXTRA_CFLAGS += -I./common -I./

# Full sanity check for debug
FULL_CHECK ?= 0
ifeq ($(FULL_CHECK), 1)
EXTRA_FLAGS += -DFULL_CHECK
endif

# Build static binaries
BUILD_STATIC ?= 1
ifeq ($(BUILD_STATIC), 1)
EXTRA_LDFLAGS += -static
endif

# Base architecture revision
ARCH_BASE ?= armv8.4-a

### End user options

# Feature extensions
MARCH_SCALAR := +bf16+fp16
MARCH_NEON   := +simd+bf16+fp16+fp16fml+dotprod
MARCH_SVE    := +sve+bf16
MARCH_SVE2   := +sve2+bf16
MARCH_SVE2P1 := +sve2p1+sve-b16b16
MARCH_SME2   := +sme2+sme-i16i64+sme-f64f64
MARCH_SME2P1 := +sme2p1+sme-f16f16+sme-b16b16

# Bit Perm optional selection flag
HAVE_BITPERM ?= 0
ifeq ($(HAVE_BITPERM), 1)
EXTRA_FLAGS  += -D__ARM_FEATURE_SVE2_BITPERM -D__ARM_FEATURE_SSVE_BITPERM
MARCH_SVE2   := $(MARCH_SVE2)+sve2-bitperm
MARCH_SME2   := $(MARCH_SVE2)$(MARCH_SME2)+ssve-bitperm
endif

# MMLA optional feature flags
HAVE_MMLA    ?= 0
ifeq ($(HAVE_MMLA),1)
EXTRA_FLAGS  += -D__ARM_FEATURE_SVE_MATMUL_FP32 -D__ARM_FEATURE_SVE_MATMUL_INT8
MARCH_NEON   := $(MARCH_NEON)+i8mm
MARCH_SVE2   := $(MARCH_SVE2)+f32mm+i8mm
MARCH_SME2   := $(MARCH_SVE2)$(MARCH_SME2)
endif

# LUT optional feature flags
HAVE_LUT    ?= 0
ifeq ($(HAVE_LUT),1)
EXTRA_FLAGS  += -D__ARM_FEATURE_LUT
MARCH_NEON   := $(MARCH_NEON)+lut
MARCH_SVE2   := $(MARCH_SVE2)+lut
MARCH_SME2   := $(MARCH_SVE2)$(MARCH_SME2)
endif

# Optional autovec using histcnt
ifeq ($(ENABLE_AUTOVEC_HISTCNT),1)
EXTRA_FLAGS  += -mllvm -enable-histogram-loop-vectorization
endif

# 16b float feature flags
ARCH_BASE := $(ARCH_BASE)+fp16+bf16

# Concatenate uArch flags
MARCH_SVE      := $(MARCH_NEON)$(MARCH_SVE)
MARCH_SVE2     := $(MARCH_SVE)$(MARCH_SVE2)
MARCH_SVE2P1   := $(MARCH_SVE2)$(MARCH_SVE2P1)
MARCH_SME2P1   := $(MARCH_SME2)$(MARCH_SME2P1)

CFLAGS_NO_AUTOVEC = -MMD -Wall -O2 -Werror -fno-tree-vectorize $(EXTRA_FLAGS)
CFLAGS_SCALAR = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_SCALAR) -U__ARM_NEON -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SVE2 -U__ARM_FEATURE_SVE2p1 -U__ARM_FEATURE_SME -U__ARM_FEATURE_SME2 -U__ARM_FEATURE_SME2p1
CFLAGS_NEON   = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_NEON) -D__ARM_NEON -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SVE2 -U__ARM_FEATURE_SVE2p1 -U__ARM_FEATURE_SME -U__ARM_FEATURE_SME2 -U__ARM_FEATURE_SME2p1
CFLAGS_SVE    = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_SVE) -D__ARM_FEATURE_SVE -U__ARM_FEATURE_SVE2 -U__ARM_FEATURE_SVE2p1 -U__ARM_FEATURE_SME -U__ARM_FEATURE_SME2 -U__ARM_FEATURE_SME2p1
CFLAGS_SVE2   = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_SVE2) -D__ARM_FEATURE_SVE2 -U__ARM_FEATURE_SVE2p1 -U__ARM_FEATURE_SME -U__ARM_FEATURE_SME2 -U__ARM_FEATURE_SME2p1
CFLAGS_SVE2P1 = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_SVE2P1) -D__ARM_FEATURE_SVE2p1 -U__ARM_FEATURE_SME -U__ARM_FEATURE_SME2 -U__ARM_FEATURE_SME2p1
CFLAGS_SME2   = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_SME2) -D__ARM_FEATURE_SME2 -U__ARM_FEATURE_SME2p1
CFLAGS_SME2P1 = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_SME2P1) -D__ARM_FEATURE_SME2p1

# Special target
MARCH_SME_SSVE := $(MARCH_SME2)
CFLAGS_SME_SSVE  = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_SME_SSVE) -D__ARM_FEATURE_SME -U__ARM_FEATURE_SME2 -U__ARM_FEATURE_SME2p1

# If clang enable the clang specific passes
ifeq ($(findstring clang,$(C_COMPILER)),clang)
CFLAGS_SME2     += -Rpass-analysis=sme
CFLAGS_SME_SSVE += -Rpass-analysis=sme
CFLAGS_SME2P1   += -Rpass-analysis=sme
endif

# Intrinsic targets
CFLAGS_SVE_INTRINSICS = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_SVE2) -DHAVE_SVE_INTRINSICS -D__ARM_FEATURE_SVE2 -U__ARM_FEATURE_SVE2p1 -U__ARM_FEATURE_SME -U__ARM_FEATURE_SME2 -U__ARM_FEATURE_SME2p1
CFLAGS_SME_INTRINSICS = $(CFLAGS_NO_AUTOVEC) -march=$(ARCH_BASE)$(MARCH_SME2) -DHAVE_SME_INTRINSICS -D__ARM_FEATURE_SME -U__ARM_FEATURE_SME2p1

# Auto-vectorised targets
CFLAGS_AUTOVEC_SVE  ?= -MMD -Wall -O3 -Werror -DHAVE_AUTOVEC $(EXTRA_FLAGS) -march=$(ARCH_BASE)+sve$(MARCH_SVE)
CFLAGS_AUTOVEC_SVE2 ?= -MMD -Wall -O3 -Werror -DHAVE_AUTOVEC $(EXTRA_FLAGS) -march=$(ARCH_BASE)+sve2$(MARCH_SVE2)

# Native target
CFLAGS_NATIVE ?= -march=$(ARCH_BASE) $(EXTRA_FLAGS) -MMD -Wall -O2 -Werror -fno-tree-vectorize -DHAVE_NATIVE -U__ARM_NEON -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SVE2 -U__ARM_FEATURE_SVE2p1 -U__ARM_FEATURE_SME -U__ARM_FEATURE_SME2 -U__ARM_FEATURE_SME2p1

INCLUDE_MK_FILES := $(wildcard **/*.mk)
ifneq ($(INCLUDE_MK_FILES),)
include $(INCLUDE_MK_FILES)
endif

ifeq ($(strip $(TUNE_PBSIZE)),)
# SME loops - define the max VL targetted
MAX_VL = 512
# Sets an upper bound on the memory footprint of the input data structures of Loop_2xx. It is not guaranteed that
# this limit will be met, as the ratios between and constraints on problem dimensions will dictate the final size.
# A value of `0` means each loop will use its own default problem size.
PROBLEM_SIZE_LIMIT_KIB = 64

else
# Problem sizes tuned.
# Checking MAX_VL
ifndef MAX_VL
$(warning The maximum targeted vector length (MAX_VL) is not specified.)
$(warning MAX_VL should be a multiple of 128 within the range of 128 and 2048.)
$(error MAX_VL is not set.)
endif

# Checking MAX_VL
ifndef PROBLEM_SIZE_LIMIT_KIB
$(warning The input buffer size limit (PROBLEM_SIZE_LIMIT_KIB) is not specified.)
$(warning PROBLEM_SIZE_LIMIT_KIB determines the maximum memory footprint of Loop_2xx input data structures.)
$(warning For default problem sizes 0 value should be used.)
$(error PROBLEM_SIZE_LIMIT_KIB is not set)
endif

endif

BUILDDIR = $(shell echo $$PWD)

_OBJS := $(patsubst %.c, %.o, $(SRCFILES)) common/main.o
ALL_OBJS :=

# PARAMETERS:
# $1: the C compiler to use
# $2: extra flags for the c compiler, next to the cflags for extensions already defined
# $3: the linker to use
# $4: flags to pass to the linker
# $5: libraries to link with
# $6: objdump bindary to use
# $7: name of the target (e.g. make sve2 bm_scalar)
# $8: the name of the architecture extension, used in previously defined variable names
# $9: extra prerequisites relative to the build directory for this target
# $10: extra prerequisites relative to the directory this Makefile is in
# $11: the directory where the build directory build/$7 will be placed in
define loops_for_kind
C_COMPILER_$7 := $1
EXTRA_CFLAGS_$7 := $2 $(EXTRA_CFLAGS)
C_LINKER_$7 := $3
LDFLAGS_$7 := $4 $(EXTRA_LDFLAGS)
LDLIBS_$7 := $5 $(EXTRA_LDLIBS)
OBJDUMP_$7 := $6
BUILDDIR_$7 := $(11)/build/$7

OBJS_$7 := $(patsubst %,$$(BUILDDIR_$7)/_obj/%,$(_OBJS))
EXTRA_OBJS_$7 := $(patsubst %,$$(BUILDDIR_$7)/_obj/%,$(_EXTRA_OBJS))
EXTRA2_OBJS_$7 := $(patsubst %,$$(BUILDDIR_$7)/_obj/%,$9)
EXTRA3_OBJS_$7 := $(10)
ALL_OBJS += $$(OBJS_$7) $$(EXTRA_OBJS_$7) $$(EXTRA2_OBJS_$7) $$(EXTRA3_OBJS_$7)

STANDALONE_$7 := $(patsubst %,$$(BUILDDIR_$7)/standalone/bin/%.elf,$(LOOPS))

ifeq (,$(findstring bm_,$7))
ifneq ($$(WANT_$8),0)
.PHONY: all
all: $7
endif
endif

# Rule for creating the build directory
$$(BUILDDIR_$7):
	mkdir -p $$@ $$@/_obj/_lnk $$@/bin $$@/standalone/bin $$@/dis

# Rules for building 'simd_loops' and its disassembly
$$(OBJS_$7) $$(EXTRA_OBJS_$7): $$(BUILDDIR_$7)/_obj/%.o: %.c | $$(BUILDDIR_$7)
	@mkdir -p $$(dir $$@)
	$$(C_COMPILER_$7) $$(IFLAGS) $$(CFLAGS_$8) $$(EXTRA_CFLAGS_$7) -DMAX_GLOBAL_MEMORY_KIB=$(MAX_MEMORY_KIB) -DPROBLEM_SIZE_LIMIT_KIB=$(PROBLEM_SIZE_LIMIT_KIB) -DMAX_VL=$(MAX_VL) -c $$< -o $$@
	ln -s $$@ $$(BUILDDIR_$7)/_obj/_lnk/$$(@F)

$$(BUILDDIR_$7)/bin/simd_loops: $$(EXTRA3_OBJS_$7) $$(OBJS_$7) $$(EXTRA_OBJS_$7) $$(EXTRA2_OBJS_$7)
	$$(C_LINKER_$7) -o $$@ $$(LDFLAGS_$7) $$^ $$(LDLIBS_$7)
	@echo "Compiled "$$@" successfully"

$$(BUILDDIR_$7)/dis/simd_loops.objdump: $$(BUILDDIR_$7)/bin/simd_loops
	$$(OBJDUMP_$7) -d $$< > $$@
	@echo "Generated "$$@" successfully"

# Rules for building standalone loops
$$(BUILDDIR_$7)/_obj/loop_%_main_standalone.o: common/main.c | $$(BUILDDIR_$7)
	$$(C_COMPILER_$7) $$(CFLAGS_$8) $$(EXTRA_CFLAGS_$7) -DSTANDALONE=$$* -DSTANDALONE_ITERS=$(STANDALONE_ITERS) -c $$< -o $$@

$$(STANDALONE_$7): $$(BUILDDIR_$7)/standalone/bin/%.elf : $$(EXTRA3_OBJS_$7) $$(BUILDDIR_$7)/_obj/_lnk/%.o $$(BUILDDIR_$7)/_obj/%_main_standalone.o $$(EXTRA_OBJS_$7) $$(EXTRA2_OBJS_$7)
	$$(C_LINKER_$7) -o $$@ $$(LDFLAGS_$7) $$^ $$(LDLIBS_$7)

# Special targets
$$(BUILDDIR_$7)/_obj/bytehist_sve2.o: loops/bytehist_sve2.S | $$(BUILDDIR_$7)
	$$(C_COMPILER_$7) -E -D__ARM_FEATURE_SVE2 -D__ARM_FEATURE_SVE $(EXTRA_FLAGS) $$(EXTRA_CFLAGS_$7) $$< > $$(BUILDDIR_$7)/_obj/bytehist_sve2.s
	$$(C_COMPILER_$7) -c -g -march=$(ARCH_BASE)+sve2+sve2-bitperm $(EXTRA_FLAGS) $$(EXTRA_CFLAGS_$7) -o $$@ $$(BUILDDIR_$7)/_obj/bytehist_sve2.s

# Top level target - build corresponding 'simd_loops', disassembly and standalone loops
.PHONY: $7
$7: $$(BUILDDIR_$7)/bin/simd_loops $$(BUILDDIR_$7)/dis/simd_loops.objdump $$(STANDALONE_$7)

.PHONY: clean-$7
clean-$7:
	rm -rf $(or $(BUILDDIR),.)/build/$7

endef

# Stamp out the subdirs for each of the kinds we care about.
# Note that the sve2 loops depend additionally on bytehist_sve2.o.
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),c-scalar,NATIVE,,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),scalar,SCALAR,,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),autovec-sve,AUTOVEC_SVE,,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),autovec-sve2,AUTOVEC_SVE2,,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),neon,NEON,,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),sve,SVE,,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),sve2,SVE2,bytehist_sve2.o,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),sme2,SME2,bytehist_sve2.o,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),sme-ssve,SME_SSVE,bytehist_sve2.o,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),sve2p1,SVE2P1,bytehist_sve2.o,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),sme2p1,SME2P1,bytehist_sve2.o,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),sve-intrinsics,SVE_INTRINSICS,bytehist_sve2.o,,$(BUILDDIR)))
$(eval $(call loops_for_kind,$(C_COMPILER),$(EXTRA_CFLAGS),$(C_LINKER),$(LDFLAGS),$(LDLIBS) -lm,$(OBJDUMP),sme-intrinsics,SME_INTRINSICS,bytehist_sve2.o,,$(BUILDDIR)))

ALL_TARGETS := c-scalar scalar autovec-sve autovec-sve2 neon sve sve2 sme2 sme-ssve sve2p1 sme2p1 sve-intrinsics sme-intrinsics
ALL_TARGETS_HELP := $(ALL_TARGETS)

INCLUDE_MK2_FILES := $(wildcard **/*.mk_)
ifneq ($(INCLUDE_MK2_FILES),)
include $(INCLUDE_MK2_FILES)
endif

# Include dependency tracking info generated by -MMD.
-include $(ALL_OBJS:.o=.d)

.DEFAULT_GOAL := _error_no_target

.PHONY: all
all: $(ALL_TARGETS)

.PHONY: fmt
fmt:
	$(FORMAT) --style="file:common/.clang-format" -i $(wildcard **/*.c) $(wildcard **/*.h)

.PHONY: clean
clean::
	rm -rf $(or $(BUILDDIR),.)/build

_error_no_target:
	@echo "\e[33mERROR\e[0m: Makefile target not specified. Please use one of the following targets:"
	@echo "\e[36mall fmt clean $(ALL_TARGETS_HELP)\e[0m"
