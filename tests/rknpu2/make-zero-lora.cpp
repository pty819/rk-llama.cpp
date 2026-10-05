#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <initializer_list>

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) return 1;
    ggml_init_params params{1024*1024, nullptr, false};
    auto * ctx = ggml_init(params);
    auto * file = gguf_init_empty();
    gguf_set_val_str(file, "general.type", "adapter");
    gguf_set_val_str(file, "general.architecture", "qwen3");
    gguf_set_val_str(file, "adapter.type", "lora");
    gguf_set_val_f32(file, "adapter.lora.alpha", 4.f);
    for (const auto * name : {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"}) {
        const int N = strstr(name, "attn_q") ? 2048 : strstr(name, "attn_") ? 1024 : 3072;
        auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 4);
        auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, N);
        ggml_format_name(a, "%s.lora_a", name);
        ggml_format_name(b, "%s.lora_b", name);
        memset(a->data, 0, ggml_nbytes(a));
        memset(b->data, 0, ggml_nbytes(b));
        if (argc == 3) {
            for (size_t i = 0; i < ggml_nbytes(a) / sizeof(float); ++i) ((float *)a->data)[i] = (int(i % 17) - 8) * .005f;
            for (size_t i = 0; i < ggml_nbytes(b) / sizeof(float); ++i) ((float *)b->data)[i] = (int(i % 13) - 6) * .005f;
        }
        gguf_add_tensor(file, a);
        gguf_add_tensor(file, b);
    }
    bool ok = gguf_write_to_file(file, argv[1], false);
    gguf_free(file);
    ggml_free(ctx);
    return ok ? 0 : 2;
}
