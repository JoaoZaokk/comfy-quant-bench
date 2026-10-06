// SwiGLU prologue fused into comfy-kitchen's ConvRot int4 activation quantizer (COMFY_PORTABLE, 2026-10-03).
//
// An MLP down-projection in W4A4 ConvRot computes out = W4A4(silu(gate) * up). Without fusion torch writes the
// intermediate silu(gate) * up in bf16 (two kernels, ~470 GB/s) and the quantizer reads it back. This kernel is
// comfy-kitchen's own `quantize_int4_rowwise_convrot64_kernel` (ops/convrot_w4a4.cu, identical in v0.2.35 and v0.2.37)
// with one change: each element is loaded as silu(gate) * up, computed in fp32 and rounded to the input dtype once,
// straight from h = gate | up. Same 256-point FHT, same absmax, same int4 packing, same scales.
// Measured on the RTX 3090 (h = 4096 x 24576 bf16, Qwen-Image-2.1 img_mlp): 0.982 ms -> 0.378 ms per call;
// nibbles 100 % equal to an fp32-intermediate reference, where the bf16-intermediate path gets 99.18 %.
// Built by build.py against a comfy-kitchen source checkout (the ops/ include path); see README in this folder.
#include "ops/convrot_w4a4.cu"

// The included file references these launchers on GEMM paths this module does not use.
extern "C" void launch_cublas_gemm_int8_kernel(const void*, const void*, void*, int64_t, int64_t, int64_t, void*, int64_t, cudaStream_t) {}
extern "C" bool launch_cutlass_int8_dequant_strided(const void*, const void*, const void*, const void*, const void*, void*, int64_t, int64_t, int64_t, int64_t, int, cudaStream_t) { return false; }

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

template<typename InType, int BLOCK_THREADS>
__global__ void quantize_int4_rowwise_convrot64_swiglu_kernel(
    const InType* __restrict__ h,      // (M, 2K): gate | up
    int8_t* __restrict__ q,            // (M, K/2)
    float* __restrict__ scales,        // (M,)
    int K)
{
    constexpr int kGroupThreads = 64;
    constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    InType* row_buf = reinterpret_cast<InType*>(smem_raw);
    InType* tmp = row_buf + K;

    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int sub = tid / kGroupThreads;
    const int lane = tid % kGroupThreads;
    const int64_t gate_offset = static_cast<int64_t>(row) * (2 * K);
    const int64_t up_offset = gate_offset + K;
    const int n_groups = K / kConvRotGroup;

    float abs_max = 0.0f;
    InType* buf0 = tmp + sub * (2 * kConvRotGroup);
    InType* buf1 = buf0 + kConvRotGroup;
    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;
    for (int it = 0; it < iters; ++it) {
        const int group = it * kGroupsInFlight + sub;
        const bool active = group < n_groups;
        const int base = lane * 4;
        const int col = group * kConvRotGroup + base;

        float a[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (active) {
                const float g = to_float<InType>(h[gate_offset + col + i]);
                const float u = to_float<InType>(h[up_offset + col + i]);
                a[i] = g / (1.0f + __expf(-g)) * u;       // silu(g) * u in fp32, one rounding below
            } else {
                a[i] = 0.0f;
            }
        }
        convrot_h4_store_typed(from_float<InType>(a[0]), from_float<InType>(a[1]), from_float<InType>(a[2]), from_float<InType>(a[3]), buf1, base);
        __syncwarp();

        convrot_fht_stage64_vec2_typed(buf1, buf0, lane, 4);
        __syncwarp();
        convrot_fht_stage64_vec2_typed(buf0, buf1, lane, 16);
        __syncthreads();

        if (active) {
            abs_max = fmaxf(abs_max, convrot_fht_stage64_store_absmax_typed(buf1, row_buf + group * kConvRotGroup, lane, 64));
        }
        __syncthreads();
    }

    abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(finite_absmax_for_int4_scale<InType>(abs_max) * (1.0f / static_cast<float>(kInt4Max)), 1.0e-10f);
    if (tid == 0) {
        scales[row] = scale;
    }
    const float inv_scale = 1.0f / scale;
    const int K_half = K / 2;
    int8_t* q_row = q + static_cast<int64_t>(row) * K_half;
    uint32_t* q_words = reinterpret_cast<uint32_t*>(q_row);
    const int word_count = K / 8;
    for (int word = tid; word < word_count; word += BLOCK_THREADS) {
        q_words[word] = quantize_int4_pack4_word<InType, false>(row_buf, word * 8, inv_scale, 0ull, static_cast<int64_t>(row) * K);
    }
}

template<typename InType, int BLOCK_THREADS>
void launch_swiglu(const InType* h, int8_t* q, float* scales, int64_t M, int64_t K, cudaStream_t stream) {
    const int groups_in_flight = BLOCK_THREADS / 64;
    const size_t smem_bytes = (static_cast<size_t>(K) + groups_in_flight * 2 * kConvRotGroup) * sizeof(InType);
    auto kernel = quantize_int4_rowwise_convrot64_swiglu_kernel<InType, BLOCK_THREADS>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    kernel<<<static_cast<unsigned>(M), BLOCK_THREADS, smem_bytes, stream>>>(h, q, scales, static_cast<int>(K));
}

}  // namespace

// h: (M, 2K) bf16/fp16, contiguous, CUDA. Returns (q (M, K/2) int8, scales (M,) f32); K a multiple of 256, <= 16384.
std::vector<torch::Tensor> swiglu_quantize_int4_rowwise_convrot64(torch::Tensor h) {
    TORCH_CHECK(h.is_cuda() && h.dim() == 2 && h.is_contiguous(), "h must be a contiguous (M, 2K) CUDA tensor");
    TORCH_CHECK(h.size(1) % 2 == 0, "h must hold gate | up halves");
    const int64_t M = h.size(0);
    const int64_t K = h.size(1) / 2;
    TORCH_CHECK(K % kConvRotGroup == 0 && K <= 16384, "K must be a multiple of 256 and <= 16384");
    const at::cuda::OptionalCUDAGuard guard(h.device());
    auto q = torch::empty({M, K / 2}, h.options().dtype(torch::kInt8));
    auto scales = torch::empty({M}, h.options().dtype(torch::kFloat32));
    if (M == 0) {
        return {q, scales};
    }
    auto stream = at::cuda::getCurrentCUDAStream(h.device().index());
    if (h.scalar_type() == torch::kBFloat16) {
        auto ptr = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr());
        if (K <= 4096) launch_swiglu<__nv_bfloat16, 256>(ptr, q.data_ptr<int8_t>(), scales.data_ptr<float>(), M, K, stream);
        else launch_swiglu<__nv_bfloat16, 1024>(ptr, q.data_ptr<int8_t>(), scales.data_ptr<float>(), M, K, stream);
    } else if (h.scalar_type() == torch::kHalf) {
        auto ptr = reinterpret_cast<const __half*>(h.data_ptr());
        if (K <= 4096) launch_swiglu<__half, 256>(ptr, q.data_ptr<int8_t>(), scales.data_ptr<float>(), M, K, stream);
        else launch_swiglu<__half, 1024>(ptr, q.data_ptr<int8_t>(), scales.data_ptr<float>(), M, K, stream);
    } else {
        TORCH_CHECK(false, "h must be bf16 or fp16");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_quantize_int4_rowwise_convrot64", &swiglu_quantize_int4_rowwise_convrot64,
          "silu(gate) * up + ConvRot-256 rotation + rowwise int4 quantization, fused");
}
