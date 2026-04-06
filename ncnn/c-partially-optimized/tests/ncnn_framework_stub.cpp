// ncnn_framework_stub.cpp
// Minimal stub implementations of ncnn framework symbols not provided by the
// framework source files compiled into ncnn_stub (mat.cpp, modelbin.cpp,
// expression.cpp). Provides: Allocator, Layer, ParamDict, Option,
// cpu helpers, and layer factory stubs.

// Include order matters: platform.h must resolve before everything else.
#include "../../framework/platform.h"
#include "../../framework/allocator.h"
#include "../../framework/mat.h"
#include "../../framework/option.h"
#include "../../framework/paramdict.h"
#include "../../framework/modelbin.h"
#include "../../framework/layer.h"
#include "../../framework/cpu.h"

#include <string.h>

namespace ncnn {

// ── Allocator ─────────────────────────────────────────────────────────────────

Allocator::~Allocator() {}

// ── Option ────────────────────────────────────────────────────────────────────

Option::Option()
{
    lightmode              = true;
    use_reserved_m0        = false;
    use_subgroup_ops       = false;
    use_reserved_0         = false;
    num_threads            = 1;
    blob_allocator         = 0;
    workspace_allocator    = 0;
    openmp_blocktime       = 0;
    use_winograd_convolution    = true;
    use_sgemm_convolution       = true;
    use_int8_inference          = true;
    use_vulkan_compute          = false;
    use_bf16_storage            = false;
    use_fp16_packed             = false;
    use_fp16_storage            = false;
    use_fp16_arithmetic         = false;
    use_int8_packed             = false;
    use_int8_storage            = false;
    use_int8_arithmetic         = false;
    use_packing_layout          = true;
    vulkan_device_index         = 0;
    use_bf16_packed             = false;
    use_tensor_storage          = false;
    use_reserved_1p             = false;
    use_weights_in_host_memory  = false;
    flush_denormals             = 3;
    use_reserved_2f             = false;
    use_reserved_3f             = false;
    use_mapped_model_loading    = false;
    use_local_pool_allocator    = true;
    use_shader_local_memory     = false;
    use_cooperative_matrix      = false;
    use_winograd23_convolution  = true;
    use_winograd43_convolution  = true;
    use_winograd63_convolution  = true;
    use_a53_a55_optimized_kernel = false;
    use_fp16_uniform            = false;
    use_int8_uniform            = false;
    use_reserved_9              = false;
    use_reserved_10             = false;
    use_reserved_11             = false;
}

// ── Layer ─────────────────────────────────────────────────────────────────────

Layer::Layer()
{
    one_blob_only           = false;
    support_inplace         = false;
    support_vulkan          = false;
    support_packing         = false;
    support_bf16_storage    = false;
    support_fp16_storage    = false;
    support_int8_storage    = false;
    support_tensor_storage  = false;
    support_vulkan_packing  = false;
    support_any_packing     = false;
    support_vulkan_any_packing = false;
    support_reserved_1 = false;
    support_reserved_2 = false;
    support_reserved_3 = false;
    support_reserved_4 = false;
    support_reserved_5 = false;
    support_reserved_6 = false;
    support_reserved_7 = false;
    support_reserved_8 = false;
    support_reserved_9 = false;
    featmask = 0;
    userdata  = 0;
    typeindex = -1;
}

Layer::~Layer() {}

int Layer::load_param(const ParamDict& /*pd*/) { return 0; }
int Layer::load_model(const ModelBin& /*mb*/) { return 0; }
int Layer::create_pipeline(const Option& /*opt*/) { return 0; }
int Layer::destroy_pipeline(const Option& /*opt*/) { return 0; }
int Layer::forward(const std::vector<Mat>& /*bottom*/, std::vector<Mat>& /*top*/, const Option& /*opt*/) const { return -1; }
int Layer::forward(const Mat& /*bottom*/, Mat& /*top*/, const Option& /*opt*/) const { return -1; }
int Layer::forward_inplace(std::vector<Mat>& /*blobs*/, const Option& /*opt*/) const { return -1; }
int Layer::forward_inplace(Mat& /*blob*/, const Option& /*opt*/) const { return -1; }

// ── ParamDict ─────────────────────────────────────────────────────────────────
// The real ParamDict uses a private-impl pattern (ParamDictPrivate* const d).
// Provide a minimal stub that always returns the caller's default value.

class ParamDictPrivate {};  // opaque stub

ParamDict::ParamDict() : d(new ParamDictPrivate()) {}

ParamDict::~ParamDict() { delete d; }

ParamDict::ParamDict(const ParamDict& rhs) : d(new ParamDictPrivate(*rhs.d)) {}

ParamDict& ParamDict::operator=(const ParamDict& rhs)
{
    if (this != &rhs) { delete d; const_cast<ParamDictPrivate*&>(d) = new ParamDictPrivate(*rhs.d); }
    return *this;
}

int   ParamDict::type(int /*id*/) const          { return 0; }
int   ParamDict::get(int /*id*/, int   def) const { return def; }
float ParamDict::get(int /*id*/, float def) const { return def; }
Mat   ParamDict::get(int /*id*/, const Mat& def) const { return def; }
std::string ParamDict::get(int /*id*/, const std::string& def) const { return def; }
void  ParamDict::set(int /*id*/, int   /*v*/) {}
void  ParamDict::set(int /*id*/, float /*v*/) {}
void  ParamDict::set(int /*id*/, const Mat& /*v*/) {}
void  ParamDict::set(int /*id*/, const std::string& /*v*/) {}
void  ParamDict::clear() {}
int   ParamDict::load_param(const DataReader& /*dr*/) { return 0; }
int   ParamDict::load_param_bin(const DataReader& /*dr*/) { return 0; }

// ── ModelBin ──────────────────────────────────────────────────────────────────

ModelBin::ModelBin() {}
ModelBin::~ModelBin() {}
Mat ModelBin::load(int /*w*/, int /*type*/) const { return Mat(); }
Mat ModelBin::load(int /*w*/, int /*h*/, int /*type*/) const { return Mat(); }
Mat ModelBin::load(int /*w*/, int /*h*/, int /*c*/, int /*type*/) const { return Mat(); }
Mat ModelBin::load(int /*w*/, int /*h*/, int /*d*/, int /*c*/, int /*type*/) const { return Mat(); }

class ModelBinFromMatArrayPrivate
{
public:
    ModelBinFromMatArrayPrivate(const Mat* _weights) : weights(_weights) {}
    mutable const Mat* weights;
};

ModelBinFromMatArray::ModelBinFromMatArray(const Mat* _weights)
    : ModelBin(), d(new ModelBinFromMatArrayPrivate(_weights)) {}

ModelBinFromMatArray::~ModelBinFromMatArray() { delete d; }

ModelBinFromMatArray::ModelBinFromMatArray(const ModelBinFromMatArray&) : d(0) {}

ModelBinFromMatArray& ModelBinFromMatArray::operator=(const ModelBinFromMatArray&) { return *this; }

Mat ModelBinFromMatArray::load(int /*w*/, int /*type*/) const
{
    if (!d->weights) return Mat();
    Mat m = d->weights[0];
    d->weights++;
    return m;
}

// ── Layer factory stubs ───────────────────────────────────────────────────────
// Returns a heap-allocated base Layer (no-op forward) for any type index.
// Sufficient for tests that only call load_param / load_model on the sub-layer.

Layer* create_layer(const char* /*type*/)        { return new Layer(); }
Layer* create_layer_naive(const char* /*type*/)  { return new Layer(); }
Layer* create_layer_cpu(const char* /*type*/)    { return new Layer(); }
Layer* create_layer(int /*index*/)               { return new Layer(); }
Layer* create_layer_naive(int /*index*/)         { return new Layer(); }
Layer* create_layer_cpu(int /*index*/)           { return new Layer(); }

// ── CPU helpers ───────────────────────────────────────────────────────────────

int get_omp_thread_num()           { return 0; }
int get_big_cpu_count()            { return 1; }
int get_little_cpu_count()         { return 0; }
int get_cpu_count()                { return 1; }
int get_physical_big_cpu_count()   { return 1; }
int get_physical_little_cpu_count(){ return 0; }
int get_physical_cpu_count()       { return 1; }
int get_omp_num_threads()          { return 1; }
void set_omp_num_threads(int /*n*/) {}
int get_omp_dynamic()              { return 0; }
void set_omp_dynamic(int /*d*/)    {}
int get_kmp_blocktime()            { return 0; }
void set_kmp_blocktime(int /*t*/)  {}
int get_flush_denormals()          { return 0; }
int set_flush_denormals(int /*v*/) { return 0; }

} // namespace ncnn
