// Fase 7: microteste de capacidade no GA102 -- quantas FFMA cabem por IMMA m16n8k64 s4 sem desacelerar o tensor core.
// Cada warp repete L vezes: NM IMMA independentes (acumuladores proprios, entradas fixas) e F FFMA por IMMA.
//   modo 0: FFMA independentes (cadeias proprias, sem depender do IMMA)
//   modo 1: FFMA dependem do resultado do IMMA da iteracao anterior (como a escala por grupo: D -> t -> acc)
// Tudo em asm volatile, na ordem escrita: IMMA(i), F FFMA, IMMA(i+1), ...
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

namespace {

template<int F, int MODE>
__global__ void ub_kernel(int L, float* out, int* outi) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u};
    uint32_t b[2] = {threadIdx.x * 11u, threadIdx.x * 13u};
    constexpr int NM = 8;
    int d[NM][4];
    float acc[8];
    #pragma unroll
    for (int i = 0; i < NM; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) d[i][j] = 0x4B400000;
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = static_cast<float>(threadIdx.x + j);
    const float s = 1.0001f, c = -0.5f;
    for (int it = 0; it < L; ++it) {
        a[0] = static_cast<uint32_t>(it) * 2654435761u;               // entrada muda a cada volta: o IMMA nao sai do laco
        #pragma unroll
        for (int i = 0; i < NM; ++i) {
            if constexpr (MODE == 0) {                                 // acumula no proprio d (cadeia por i)
                asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+r"(d[i][0]), "+r"(d[i][1]), "+r"(d[i][2]), "+r"(d[i][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            } else {
                asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                    : "=r"(d[i][0]), "=r"(d[i][1]), "=r"(d[i][2]), "=r"(d[i][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0x4B400000));
            }
            #pragma unroll
            for (int f = 0; f < F; ++f) {
                float x;
                if constexpr (MODE == 0) x = acc[f & 7];
                else x = __int_as_float(d[(i + NM - 2) % NM][f & 3]);   // resultado de dois IMMA atras
                asm volatile("fma.rn.f32 %0, %1, %2, %3;\n" : "=f"(acc[f & 7]) : "f"(x), "f"(s), "f"(acc[f & 7]));
                (void)c;
            }
        }
    }
    float t = 0.0f;
    int ti = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) t += acc[j];
    #pragma unroll
    for (int i = 0; i < NM; ++i) ti += d[i][0] ^ d[i][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = t;
    outi[blockIdx.x * blockDim.x + threadIdx.x] = ti;
}

template<int F, int MODE>
double run(int blocks, int threads, int L) {
    auto o = torch::empty({blocks * threads}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto oi = torch::empty({blocks * threads}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto st = at::cuda::getCurrentCUDAStream();
    ub_kernel<F, MODE><<<blocks, threads, 0, st>>>(L, o.data_ptr<float>(), oi.data_ptr<int>());   // aquecimento
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, st);
    for (int r = 0; r < 5; ++r) ub_kernel<F, MODE><<<blocks, threads, 0, st>>>(L, o.data_ptr<float>(), oi.data_ptr<int>());
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / 5.0;
}

}  // namespace

// devolve ms por lancamento para F em {0, 2, 4, 6, 8, 10, 12, 16, 24, 32}
std::vector<double> bench(int64_t blocks, int64_t threads, int64_t L, int64_t mode) {
    std::vector<double> r;
    if (mode == 0) {
        r.push_back(run<0, 0>(blocks, threads, L)); r.push_back(run<2, 0>(blocks, threads, L));
        r.push_back(run<4, 0>(blocks, threads, L)); r.push_back(run<6, 0>(blocks, threads, L));
        r.push_back(run<8, 0>(blocks, threads, L)); r.push_back(run<10, 0>(blocks, threads, L));
        r.push_back(run<12, 0>(blocks, threads, L)); r.push_back(run<16, 0>(blocks, threads, L));
        r.push_back(run<24, 0>(blocks, threads, L)); r.push_back(run<32, 0>(blocks, threads, L));
    } else {
        r.push_back(run<0, 1>(blocks, threads, L)); r.push_back(run<2, 1>(blocks, threads, L));
        r.push_back(run<4, 1>(blocks, threads, L)); r.push_back(run<6, 1>(blocks, threads, L));
        r.push_back(run<8, 1>(blocks, threads, L)); r.push_back(run<10, 1>(blocks, threads, L));
        r.push_back(run<12, 1>(blocks, threads, L)); r.push_back(run<16, 1>(blocks, threads, L));
        r.push_back(run<24, 1>(blocks, threads, L)); r.push_back(run<32, 1>(blocks, threads, L));
    }
    return r;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bench", &bench, "IMMA m16n8k64 s4 com F FFMA por IMMA");
}
