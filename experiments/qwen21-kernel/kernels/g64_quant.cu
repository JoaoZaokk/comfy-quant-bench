// Fase 7, etapa 2: quantizador de ativacao ConvRot com escala por grupo (64 ou 128), int4 simetrico.
// x (M, K) bf16 -> rotacao pela Hadamard regular de bloco 256 (a `_build_hadamard(256)` do ck: kron de H4 regular,
// normalizada) -> escala por grupo = amax/7 arredondada PARA CIMA ate um valor bf16 (exata no primeiro fma do GEMM e
// sem saturar) -> q = rn(v / s) em [-7, 7], empacotado como o ck (nibble baixo = coluna par).
// Saidas: q (M_pad, K/2) int8, s (K/G, M_pad) fp32. Linhas M..M_pad-1 saem com q = 0 e s = 0.
// Um warp por bloco de 256 (8 valores por lane); a FWHT radix-4 (4 estagios, passos 1/4/16/64) roda na smem do warp em
// fp32. H4 regular: y_i = (x0 + x1 + x2 + x3)/2 - x_{3-i}.
// SWIGLU: x (M, 2K) = [gate | up], a entrada da linear e silu(gate) * up (em fp32), como a mlp.out do Qwen-Image-2.1.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int WARPS = 8;

__device__ __forceinline__ float bf16_up(float v) {
    // menor valor bf16 >= v (v > 0): arredonda a mantissa para cima
    uint32_t u = __float_as_uint(v);
    const uint32_t low = u & 0xFFFFu;
    u &= 0xFFFF0000u;
    if (low) u += 0x10000u;
    return __uint_as_float(u);
}

template<int G, bool SWIGLU>
__global__ void __launch_bounds__(WARPS * 32)
quant_kernel(const __nv_bfloat16* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ s, int M, int M_pad, int K) {
    __shared__ float buf[WARPS][256];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nblk = K / 256;
    const int64_t task = static_cast<int64_t>(blockIdx.x) * WARPS + warp;   // (linha, bloco de 256)
    if (task >= static_cast<int64_t>(M_pad) * nblk) return;
    const int row = static_cast<int>(task / nblk), blk = static_cast<int>(task % nblk);
    float* b = buf[warp];

    float v[8];
    if (row < M) {
        if constexpr (SWIGLU) {
            const __nv_bfloat16* gp = x + static_cast<int64_t>(row) * (2 * K) + blk * 256 + lane * 8;
            const uint4 gr = *reinterpret_cast<const uint4*>(gp);
            const uint4 ur = *reinterpret_cast<const uint4*>(gp + K);
            const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gr);
            const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&ur);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 gf = __bfloat1622float2(g2[j]), uf = __bfloat1622float2(u2[j]);
                v[2 * j] = gf.x / (1.0f + __expf(-gf.x)) * uf.x;
                v[2 * j + 1] = gf.y / (1.0f + __expf(-gf.y)) * uf.y;
            }
        } else {
            const uint4 r = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(row) * K + blk * 256 + lane * 8);
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&r);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(h2[j]);
                v[2 * j] = f.x; v[2 * j + 1] = f.y;
            }
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = 0.0f;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) b[lane * 8 + j] = v[j];
    __syncwarp();
    // FWHT radix-4 regular: 64 borboletas por estagio, 2 por lane
    #pragma unroll
    for (int st = 1; st <= 64; st *= 4) {
        float y[2][4];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int bf = lane + 32 * h, base = (bf % st) + (bf / st) * 4 * st;
            const float x0 = b[base], x1 = b[base + st], x2 = b[base + 2 * st], x3 = b[base + 3 * st];
            const float hs = 0.5f * (x0 + x1 + x2 + x3);             // (H4/2) x: y_i = soma/2 - x_{3-i}
            y[h][0] = hs - x3; y[h][1] = hs - x2; y[h][2] = hs - x1; y[h][3] = hs - x0;
        }
        __syncwarp();
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int bf = lane + 32 * h, base = (bf % st) + (bf / st) * 4 * st;
            b[base] = y[h][0]; b[base + st] = y[h][1]; b[base + 2 * st] = y[h][2]; b[base + 3 * st] = y[h][3];
        }
        __syncwarp();
    }
    // quatro estagios de H4/2 (ortonormal) = kron^4(H4) / 16 = _build_hadamard(256)
    float am = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) { v[j] = b[lane * 8 + j]; am = fmaxf(am, fabsf(v[j])); }
    constexpr int LANES = G / 8;                                     // lanes por grupo (8 para g64, 16 para g128)
    #pragma unroll
    for (int o = 1; o < LANES; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am > 0.0f ? bf16_up(am / 7.0f) : 0.0f;
    const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    uint32_t w = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int qi = __float2int_rn(v[j] * inv);
        qi = max(-7, min(7, qi));
        w |= (static_cast<uint32_t>(qi) & 0xFu) << (4 * j);
    }
    *reinterpret_cast<uint32_t*>(q + static_cast<int64_t>(row) * (K / 2) + blk * 128 + lane * 4) = w;
    if ((lane % LANES) == 0) {
        const int g = blk * (256 / G) + lane / LANES;
        s[static_cast<int64_t>(g) * M_pad + row] = sc;
    }
}

}  // namespace

// group = 64 ou 128; swiglu: x tem 2K colunas ([gate | up]); pad: M_pad multiplo de `pad`
std::vector<torch::Tensor> quant(torch::Tensor x, int64_t group, bool swiglu, int64_t pad) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x bf16 2D contiguo");
    const int64_t M = x.size(0), K = swiglu ? x.size(1) / 2 : x.size(1);
    TORCH_CHECK(K % 256 == 0, "K % 256");
    const int64_t M_pad = (M + pad - 1) / pad * pad;
    auto q = torch::empty({M_pad, K / 2}, x.options().dtype(torch::kInt8));
    auto s = torch::empty({K / group, M_pad}, x.options().dtype(torch::kFloat32));
    const int64_t tasks = M_pad * (K / 256);
    const int blocks = static_cast<int>((tasks + WARPS - 1) / WARPS);
    auto st = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    if (group == 64 && !swiglu) quant_kernel<64, false><<<blocks, WARPS * 32, 0, st>>>(xp, q.data_ptr<int8_t>(), s.data_ptr<float>(), M, M_pad, K);
    else if (group == 64 && swiglu) quant_kernel<64, true><<<blocks, WARPS * 32, 0, st>>>(xp, q.data_ptr<int8_t>(), s.data_ptr<float>(), M, M_pad, K);
    else if (group == 128 && !swiglu) quant_kernel<128, false><<<blocks, WARPS * 32, 0, st>>>(xp, q.data_ptr<int8_t>(), s.data_ptr<float>(), M, M_pad, K);
    else if (group == 128 && swiglu) quant_kernel<128, true><<<blocks, WARPS * 32, 0, st>>>(xp, q.data_ptr<int8_t>(), s.data_ptr<float>(), M, M_pad, K);
    else TORCH_CHECK(false, "group 64 ou 128");
    return {q, s};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant", &quant, "ConvRot (H256 regular) + int4 simetrico por grupo de 64/128, escala bf16 para cima");
}
