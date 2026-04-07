---
name: map-kernels-with-arm-baseline
description: Step-by-step tutorial to map partially optimized kernels with arm-bench baseline, then analyze the dependencies of a kernel implementation and create stubs to allow testing without the full framework build
---

# Tutorial: Map partially optimized kernels with arm-bench baseline

## Goal
Map partially optimized kernels with arm-bench baseline, analyze the dependencies of a kernel implementation and create stubs to allow testing without the full framework build

## Notice
Do not changes of the original kernel and dependency implementation

## Directory structure
```ncnn/
├── c-partially-optimized/
│   ├── attention/
│   ├── activation/
│   ├── convolution/
│   ├── gemm/
│   ├── ...
├── arm-heavy-optimized/
│   ├── attention/
│   ├── convolution/
│   ├── gemm/
│   ├── common/
│   ├── ...
```


## Step 1: generate kernel mapping
Fetch all kernel implementation directories (e.g. `attention/`, `activation/`, `convolution/`) in both `c-partially-optimized/` and `arm-heavy-optimized/`.Generates a 1-by-1 mapping for kernels in `c-partially-optimized/` and its arm baseline in `arm-heavy-optimized/` in `kernel_mapping.json`.
```
{
  "note": "Maps each kernel in c-partially-optimized to its ARM-optimized baseline in arm-heavy-optimized. 'arm_baseline' is null when no counterpart exists. Category rename: activation/ in c-partially-optimized was folded into tensor/ in arm-heavy-optimized.",
  "kernels": [
    {
      "name": "absval",
      "c_partially_optimized": "activation/absval",
      "arm_baseline": "tensor/absval_arm"
    },
    {
      "name": "bnll",
      "c_partially_optimized": "activation/bnll",
      "arm_baseline": null
    },
    {
      "name": "celu",
      "c_partially_optimized": "activation/celu",
      "arm_baseline": null
    },
    {
      "name": "clip",
      "c_partially_optimized": "activation/clip",
      "arm_baseline": "tensor/clip_arm"
    }
  ]
}
```

**key points:**
- For kernels that cannot find an arm baseline, set arm baseline to null.

## Step2: move kernel implementation based on mapping
Create two new directory `ncnn/mapped/` and `ncnn/unmapped/`.
Write a temporal script to conduct the task: 
For kernels that has an arm baseline, move the kernel implementation to directory `ncnn/mapped/`. For kernels that does not have an arm baseline, move the kernel implementation to directory `ncnn/unmapped/`. 

```python
import json
import os
import shutil
import glob

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_C   = os.path.join(BASE, "c-partially-optimized")
SRC_ARM = os.path.join(BASE, "arm-heavy-optimized")
MAPPED   = os.path.join(BASE, "mapped")
UNMAPPED = os.path.join(BASE, "unmapped")

with open(os.path.join(BASE, "kernel_mapping.json")) as f:
    data = json.load(f)

def copy_c_files(src_dir, stem, dest_dir):
    """Copy {stem}.h and {stem}.cpp exactly (no prefix matching)."""
    copied = []
    for ext in (".h", ".cpp"):
        path = os.path.join(src_dir, stem + ext)
        if os.path.isfile(path):
            shutil.copy2(path, dest_dir)
            copied.append(stem + ext)
    return copied

def copy_arm_files(src_dir, arm_stem, dest_dir):
    """Copy all arm files: {arm_stem}.h, {arm_stem}.cpp, and {arm_stem}_*.cpp variants."""
    copied = []
    for ext in (".h", ".cpp"):
        path = os.path.join(src_dir, arm_stem + ext)
        if os.path.isfile(path):
            shutil.copy2(path, dest_dir)
            copied.append(arm_stem + ext)
    # also pick up variant files: {arm_stem}_asimdhp.cpp, _vfpv4.cpp, etc.
    for path in sorted(glob.glob(os.path.join(src_dir, f"{arm_stem}_*.cpp"))):
        if os.path.isfile(path):
            shutil.copy2(path, dest_dir)
            copied.append(os.path.basename(path))
    return copied

for entry in data["kernels"]:
    name       = entry["name"]
    c_path     = entry["c_partially_optimized"]   # e.g. "activation/relu"
    arm_path   = entry["arm_baseline"]             # e.g. "tensor/relu_arm" or None

    c_category, c_stem = c_path.rsplit("/", 1)
    c_src_dir = os.path.join(SRC_C, c_category)

    if arm_path:
        # --- mapped: kernel + arm baseline go into mapped/{name}/ ---
        arm_category, arm_stem = arm_path.rsplit("/", 1)
        arm_src_dir = os.path.join(SRC_ARM, arm_category)

        dest = os.path.join(MAPPED, name)
        os.makedirs(dest, exist_ok=True)

        c_files   = copy_c_files(c_src_dir, c_stem, dest)
        arm_files = copy_arm_files(arm_src_dir, arm_stem, dest)

        print(f"[mapped]   {name:30s}  c={c_files}  arm={arm_files}")
    else:
        # --- unmapped: only c-partially-optimized files go into unmapped/{name}/ ---
        dest = os.path.join(UNMAPPED, name)
        os.makedirs(dest, exist_ok=True)

        c_files = copy_c_files(c_src_dir, c_stem, dest)

        print(f"[unmapped] {name:30s}  c={c_files}")

print("\nDone.")
print(f"  mapped/   : {len(os.listdir(MAPPED))} subdirs")
print(f"  unmapped/ : {len(os.listdir(UNMAPPED))} subdirs")
```

**key points:**
- if `ncnn/mapped` and `ncnn/unmapped` already exist, ignore this step

## Step 3: Analyze ncnn framework dependency
Dive in to the kernels in `ncnn/mapped/` and `ncnn/unmapped/`, perform dependency analysis to identify three categories:
- framework dependency: dependency inside the codebase directory ('/ncnn'); 
- cross-folder dependency: dependency inside the working directory (e.g. '/ncnn/c-partially-optimized'); 
- naive-C++ dependency: dependency supported by C/C++ (e,g. '<stdlib.h>') or hardware intrinsics (e.g. '<arm_neon.h>')

```cpp
#ifndef LAYER_DECONVOLUTION_H
#define LAYER_DECONVOLUTION_H

#include "layer.h"

namespace ncnn {

class Deconvolution : public Layer
{
public:
    Deconvolution();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
```
Dependency analysis: This implementation depends on `layer.h`, identify it is framework dependency

**key points:**
- Analyze #include directives at the begining of the file only
- #include in conditional compilation blocks (e.g. `#if NCNN_VULKAN`) should also be analyzed


## Step 4: Dive into the dependencies tree
For framework dependency and cross-folder dependency, looking into the exact file and to find new framework dependency and cross-folder dependency; For naive-C++ dependency, just leave it alone.
Analyze dependency until there are no more new framework dependency and cross-folder dependency.

```cpp
#ifndef NCNN_LAYER_H
#define NCNN_LAYER_H

#include "mat.h"
#include "modelbin.h"
#include "option.h"
#include "paramdict.h"
#include "platform.h"

#if NCNN_VULKAN
#include "command.h"
#include "pipeline.h"
#endif // NCNN_VULKAN

namespace ncnn {

class NCNN_EXPORT Layer
{
public:
    // empty
    Layer();
    // virtual destructor
    virtual ~Layer();

    // load layer specific parameter from parsed dict
    // return 0 if success
    virtual int load_param(const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    virtual int load_model(const ModelBin& mb);

    // layer implementation specific setup
    // return 0 if success
    virtual int create_pipeline(const Option& opt);

    // layer implementation specific clean
    // return 0 if success
    virtual int destroy_pipeline(const Option& opt);
```
Dependency analysis: This implementation depends on `mat.h`,`modelbin.h`,`option.h`,`paramdict.h`,`platform.h`,`command.h` , `pipeline.h`, identify they are all framework dependency, should dive in recursively until there are no more new dependency

**key points:**
- Analyze #include directives at the begining of the file only
- #include in conditional compilation blocks (e.g. `#if NCNN_VULKAN`) should also be analyzed

## Step 5: Fetch dependency header and impl
Fetch dependency header from all framework dependency and cross-folder dependency. For header file ends in `.h`, find the root-level `.cpp` implementation

```
layer.h -> layer.cpp;
mat.h -> mat.cpp;
pipeline.h -> pipeline.cpp;
option.h -> option.cpp;
paramdict.h -> paramdict.cpp;
...
```

**key points**
- only fetch the most relevant impl for header file

## Step 6: Move dependency to specific location
Move(not copy) all framework dependency, including header and impl, to `/ncnn/framework`; Move all cross-folder dependency, including header and impl, to `/ncnn/common`.

```
framework dependency: layer.h -> layer.cpp; mat.h -> mat.cpp; pipeline.h -> pipeline.cpp; option.h -> option.cpp;
paramdict.h -> paramdict.cpp;
cross folder dependency: fused_activation.h
```
then 
```bash
mv ncnn/layer.h ncnn/framework/layer.h
mv ncnn/mat.h ncnn/framework/mat.h
...
```

**For dependency that is already in `/ncnn/framework` or `/ncnn/common`**, no need to move.

## Step 7: Update include dependency path on kernel implementation
Since the last step has moved framework and cross-folder dependency, you should change the dependency path of kernel implementations to the new path

```cpp
#ifndef LAYER_DECONVOLUTION_H
#define LAYER_DECONVOLUTION_H

#include "../framework/layer.h" // <-- current path of framework dependency

namespace ncnn {

class Deconvolution : public Layer
{
public:
    Deconvolution();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

```

**For dependency that cannot fetch source file**, consider:
- Fetch files that includes dependency name
- Create dependency source file based on fetched files

→ **For testcase generation guide, see  [`.claude/skills/generate-testcase/SKILL.md`](../generate-testcase/SKILL.md)**

## Summary of Directory/Files Created/Modified
```
kernel_mapping.json # NEW: mapping result of kernel and arm baseline
ncnn/mapped/ # NEW: mapped kernel implementations
ncnn/unmapped/ # NEW: unmapped kernel implementations
ncnn/framework/ # MODIFIED: Framework dependency headers and implementations
ncnn/common/ #MODIFIED: Cross-folder dependency headers and implementations
```