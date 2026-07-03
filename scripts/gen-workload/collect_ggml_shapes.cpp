// Capture GGML_OP_MUL_MAT shapes from a llama.cpp model forward pass.
//
// Compile as a llama.cpp example: place in examples/collect-ggml-shapes/
// and add add_subdirectory(collect-ggml-shapes) to examples/CMakeLists.txt.
//
// Usage:
//   collect-ggml-shapes -m model.gguf --prompts-file prompts.txt --output shapes.json
//
// prompts.txt: one prompt per line (whitespace-normalized, no embedded newlines).
// Output JSON: array of {M, K, N, name, prompt_len} records.
//   M = number of tokens (batch / sequence length)
//   K = contraction dimension (input features)
//   N = output dimension

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct ShapeRecord {
    int64_t M, K, N;
    std::string name;
    int prompt_len;
};

struct CollectState {
    std::vector<ShapeRecord> records;
    int current_prompt_len = 0;
};

// cb_eval fires twice per tensor: ask=true (should we compute?) then ask=false (done).
static bool shape_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;
    if (t->op != GGML_OP_MUL_MAT) return true;

    auto * state = static_cast<CollectState *>(user_data);
    // GGML convention: ggml_mul_mat(A, B) → A^T * B
    //   src[0] = A (weight): ne[0]=K, ne[1]=N
    //   src[1] = B (input):  ne[0]=K, ne[1]=M
    //   output:              ne[0]=N, ne[1]=M
    const int64_t K = t->src[0]->ne[0];
    const int64_t N = t->src[0]->ne[1];
    const int64_t M = t->src[1]->ne[1];
    state->records.push_back({M, K, N, std::string(t->name), state->current_prompt_len});
    return true;
}

static std::string escape_json(const std::string & s) {
    std::string r;
    r.reserve(s.size() + 4);
    for (unsigned char c : s) {
        if      (c == '"')  r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else if (c == '\r') r += "\\r";
        else if (c == '\t') r += "\\t";
        else                r += c;
    }
    return r;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string prompts_file;
    std::string output_file = "shapes.json";

    // Strip our custom flags before passing to common_params_parse.
    std::vector<char *> fwd;
    fwd.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--prompts-file" && i + 1 < argc) {
            prompts_file = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else {
            fwd.push_back(argv[i]);
        }
    }

    common_init();
    common_params params;
    int fwd_argc = static_cast<int>(fwd.size());
    if (!common_params_parse(fwd_argc, fwd.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // CPU-only inference: no token generation, just prefill.
    params.warmup   = false;
    params.n_predict = 0;

    CollectState state;
    params.cb_eval           = shape_cb;
    params.cb_eval_user_data = &state;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init    = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();

    if (!model || !ctx) {
        fprintf(stderr, "[collect] Failed to init model/context\n");
        return 1;
    }

    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const bool          add_bos = llama_vocab_get_add_bos(vocab);
    const int           n_ctx  = static_cast<int>(llama_n_ctx(ctx));

    // Load prompts: one per line from --prompts-file, or single -p prompt.
    std::vector<std::string> prompts;
    if (!prompts_file.empty()) {
        std::ifstream pf(prompts_file);
        if (!pf) {
            fprintf(stderr, "[collect] Cannot open prompts file: %s\n", prompts_file.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(pf, line)) {
            if (!line.empty()) prompts.push_back(line);
        }
    } else if (!params.prompt.empty()) {
        prompts.push_back(params.prompt);
    } else {
        fprintf(stderr, "[collect] Provide --prompts-file <file> or -p <prompt>\n");
        return 1;
    }

    fprintf(stderr, "[collect] %zu prompt(s), context=%d\n", prompts.size(), n_ctx);

    for (size_t pi = 0; pi < prompts.size(); pi++) {
        std::vector<llama_token> tokens = common_tokenize(ctx, prompts[pi], add_bos, true);
        if (tokens.empty()) continue;

        // Truncate to fit context (leave 4 slots of headroom).
        if (static_cast<int>(tokens.size()) > n_ctx - 4) {
            tokens.resize(static_cast<size_t>(n_ctx - 4));
        }

        const int prompt_len = static_cast<int>(tokens.size());
        state.current_prompt_len = prompt_len;
        const size_t before = state.records.size();

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[collect] Decode failed for prompt %zu, skipping\n", pi);
            continue;
        }

        fprintf(stderr, "[collect] %zu/%zu len=%d new_records=%zu\n",
                pi + 1, prompts.size(), prompt_len, state.records.size() - before);
    }

    // Write JSON output.
    std::ofstream out(output_file);
    if (!out) {
        fprintf(stderr, "[collect] Cannot write output: %s\n", output_file.c_str());
        return 1;
    }

    out << "[\n";
    for (size_t i = 0; i < state.records.size(); i++) {
        const auto & r = state.records[i];
        out << "  {\"M\":" << r.M
            << ",\"K\":"   << r.K
            << ",\"N\":"   << r.N
            << ",\"name\":\"" << escape_json(r.name) << "\""
            << ",\"prompt_len\":" << r.prompt_len << "}";
        if (i + 1 < state.records.size()) out << ",";
        out << "\n";
    }
    out << "]\n";

    fprintf(stderr, "[collect] Wrote %zu records to %s\n",
            state.records.size(), output_file.c_str());

    llama_backend_free();
    return 0;
}
