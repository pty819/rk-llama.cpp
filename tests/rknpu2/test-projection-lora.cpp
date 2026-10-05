#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3) return 1;
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    auto cp = llama_context_default_params();
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_LAST;
    cp.n_ctx = 8192;
    cp.n_batch = cp.n_ubatch = 4096;
    cp.n_threads = cp.n_threads_batch = 4;
    auto * ctx = llama_init_from_model(model, cp);
    auto * adapter = llama_adapter_lora_init(model, argv[2]);
    if (!ctx || !adapter) return 3;
    const char * text = "Document: Climate change is reshaping coastal cities through rising seas, stronger storms, and hotter summers. Municipal planners must invest in resilient infrastructure, restore wetlands, and update building codes. Scientists warn that delaying action increases adaptation costs and risks irreversible ecosystem loss. Community engagement, open data, and equitable funding are essential.";
    std::vector<llama_token> tokens(256);
    int n = llama_tokenize(llama_model_get_vocab(model), text, strlen(text), tokens.data(), tokens.size(), true, true);
    if (n < 53) return 4;
    auto batch = llama_batch_init(53, 0, 1);
    batch.n_tokens = 53;
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    std::vector<std::vector<float>> vectors;
    for (float scale : {1.f, 0.f, 1.f}) {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0) return 5;
        if (llama_decode(ctx, batch) != 0) return 6;
        const auto * result = llama_get_embeddings_seq(ctx, 0);
        if (!result) return 7;
        vectors.emplace_back(result, result + llama_model_n_embd(model));
        for (float value : vectors.back()) if (!std::isfinite(value)) return 8;
    }
    float error = 0.f, effect = 0.f;
    for (size_t i = 0; i < vectors[0].size(); ++i) {
        error = std::max(error, std::abs(vectors[0][i] - vectors[2][i]));
        effect = std::max(effect, std::abs(vectors[0][i] - vectors[1][i]));
    }
    printf("LoRA clear-memory roundtrip max_error=%g effect=%g\n", error, effect);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_adapter_lora_free(adapter);
    llama_model_free(model);
    llama_backend_free();
    return error <= 1e-5f && effect > 1e-5f ? 0 : 9;
}
