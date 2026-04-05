---
name: analyze-ncnn-dependency
description: Step-by-step tutorial to analyze the dependencies of a kernel implementation and create stubs to allow testing without the full framework build
---

# Tutorial: Analyze dependency of a cpu kernel implementation

## Goal
Analyze the dependencies of a kernel implementation and create the necessary stubs to allow compiling and testing it without the full ncnn framework build.

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


## Step 1: Identify kernel implementation dependency
Fetch all kernel implementation directories (e.g. `attention/`, `activation/`, `convolution/`) in $ARGUMENTS and identify dependency of all kernel implementations (e.g. `attention/sdpa.cpp`, `conv/deconvolution.cpp`)

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
- Identify the dependency into three categories:
    - framework dependency: dependency inside the codebase directory ('/ncnn'); 
    - cross-folder dependency: dependency inside the working directory (e.g. '/ncnn/c-partially-optimized'); 
    - naive-C++ dependency: dependency supported by C/C++ (e,g. '<stdlib.h>') or hardware intrinsics (e.g. '<arm_neon.h>')
- Analyze #include directives at the begining of the file only
- #include in conditional compilation blocks (e.g. `#if NCNN_VULKAN`) should also be analyzed


## step2: Dive into the dependencies tree
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
- Identify the dependency into three categories:
    - framework dependency
    - cross-folder dependency 
    - naive-C++ dependency
- Analyze #include directives at the begining of the file only
- #include in conditional compilation blocks (e.g. `#if NCNN_VULKAN`) should also be analyzed

## step3: Fetch dependency header and impl
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

## step4: move dependecy to specific location
Move(not copy) all framework dependency, including header and impl, to `/ncnn/framework`; Move all cross-folder dependency, including header and impl, to `/ncnn/$ARGUMENTS/common`.

```
working on directory: /ncnn/c-partially-optimized
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

**For dependency that is already in `/ncnn/framework` or `/ncnn/$ARGUMENTS/common`**, no need to move.

## step5: update include dependency path on kernel implementation
since last step has moved framework and cross-folder dependency. You should change the dependency path of kernel implementations to new path

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
ncnn/framework # MODIFIED: Framework dependency headers and implementations
ncnn/$ARGUMENTS/common #MODIFIED: Cross-folder dependency headers and implementations
```