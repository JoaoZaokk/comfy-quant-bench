// Fase 7, v3 do GEMM ConvRot W4A4 s4 x s4 (mma.sync m16n8k64, sm_80/86). Contrato de dados igual a v1/v2:
// A (M, K/2), B (N, K/2) int8 empacotados (nibble baixo = coluna par); bias bf16 (N,) opcional; saida bf16 (M, N).
// Politicas (escalas da ativacao `sa`, do peso `sb`):
//   0 row/row          sa (M,) fp32, sb (N,) fp32; acumulador int32, escalas no epilogo (= ConvRot do ck)
//   1 peso g64 FP      sa (M,), sb (K/64, N) fp32 bf16-exata; acc += fma(D, w, -M*w)
//   2 ativacao g64 FP  sa (K/64, M) fp32 bf16-exata, sb (N,); acc += fma(D, a, -M*a)
//   3 g64/g64 FP       sa (K/64, M) fp32, sb (K/64, N) bf16-exata; acc = fma(fma(D, w, -M*w), a, acc)
//   4 g128/g128 FP     sa (K/128, M), sb (K/128, N) bf16-exata; os dois IMMA k64 do k-tile encadeiam em int32
//   5 peso g64 inteiro sa (M,) fp32, sb = (sw (N,) fp32, cw (K/64, N) int32 em [1, 255]); acc_i += P * cw (IMAD)
//   6 ativ. g64 inteira sa = (sa (M,), ca (K/64, M) int32 em [1, 255]), sb (N,); acc_i += P * ca (IMAD)
// Nas FP, o IMMA recebe C = 0x4B400000 (1,5 * 2^23): D lido como float = 1,5*2^23 + P (|P| <= 3136), e fma(D, s, -M*s)
// = P*s com um arredondamento (exato com s de ate 8 bits de mantissa, bf16). Nas inteiras, |acc_i| <= 3136*255*K/64.
// IMMA e a parte por grupo em asm volatile, em ordem fixa (IMMA(i), parte por grupo de IMMA(i - PD)); constantes por
// grupo (escalas e -M*s) lidas uma vez por passo k64. Bloco 128 x BN x 128, 8 warps 2 x 4, STAGES estagios cp.async,
// fragmentos em buffer duplo, escalas num anel de STAGES + 1 posicoes, ordem dos blocos agrupada (GROUP_M), epilogo pela
// smem com escrita de 16 B. Requer M % 128, N % BN, K % 128.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int BM = 128, BKB = 64;
constexpr int THREADS = 256, GROUP_M = 8;
constexpr float MAGIC_F = 12582912.0f;
constexpr int MAGIC_I = 0x4B400000;

template<int BN, int STAGES, int POL>
struct Cfg {
    static constexpr int WN = BN / 4, NI = WN / 8;
    static constexpr int A_BYTES = BM * BKB, B_BYTES = BN * BKB, AB_BYTES = A_BYTES + B_BYTES;
    static constexpr bool WG = POL == 1 || POL == 3 || POL == 4 || POL == 5;   // escala do peso vem por grupo
    static constexpr bool AG = POL == 2 || POL == 3 || POL == 4 || POL == 6;   // escala da ativacao vem por grupo
    static constexpr bool INT_ACC = POL == 0 || POL == 5 || POL == 6;
    static constexpr int GPK = POL == 4 ? 1 : 2;                                // grupos por k-tile
    static constexpr int SB_ELEMS = WG ? GPK * BN : 0, SA_ELEMS = AG ? GPK * BM : 0;
    static constexpr int S_BYTES = (SB_ELEMS + SA_ELEMS) * 4, S_SLOTS = STAGES + 1;
    static constexpr int LDC = BN + 8;
    static constexpr int PIPE_BYTES = STAGES * AB_BYTES + S_SLOTS * S_BYTES, EPI_BYTES = BM * LDC * 2;
    static constexpr int SMEM = PIPE_BYTES > EPI_BYTES ? PIPE_BYTES : EPI_BYTES;
    static constexpr int A_CP = BM * 4 / THREADS, B_CP = BN * 4 / THREADS;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_acc(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_c(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int c) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}
__device__ __forceinline__ void mma_chain(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ffma_v(float a, float b, float c) {
    float d;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;\n" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}
__device__ __forceinline__ float fadd_v(float a, float b) {
    float d;
    asm volatile("add.rn.f32 %0, %1, %2;\n" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ int imad_v(int a, int b, int c) {
    int d;
    asm volatile("mad.lo.s32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
__device__ __forceinline__ int swz(int row, int chunk) { return row * BKB + ((chunk ^ ((row >> 1) & 3)) << 4); }

template<int BN, int STAGES, int POL, int PD>
__global__ void __launch_bounds__(THREADS, 1)
gemm_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B, const float* __restrict__ sa,
            const float* __restrict__ sb, const int* __restrict__ ca, const int* __restrict__ cw,
            const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Cout, int M, int N, int K) {
    using C = Cfg<BN, STAGES, POL>;
    constexpr int NI = C::NI, WN = C::WN;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;

    const int pid = blockIdx.x, npm = M / BM, npn = N / BN;
    const int per_group = GROUP_M * npn, gid = pid / per_group, first_m = gid * GROUP_M;
    const int gsz = min(npm - first_m, GROUP_M);
    const int pm = first_m + (pid % per_group) % gsz, pn = (pid % per_group) / gsz;
    const int m0 = pm * BM, n0 = pn * BN;
    const int Kb = K / 2, KT = K / 128;

    // ---- enderecos fixos ----
    const uint32_t sbase = smem_u32(smem);
    const uint32_t sScale = sbase + STAGES * C::AB_BYTES;
    uint32_t a_dst[C::A_CP], b_dst[C::B_CP];
    const int8_t* a_src[C::A_CP];
    const int8_t* b_src[C::B_CP];
    #pragma unroll
    for (int j = 0; j < C::A_CP; ++j) {
        const int i = tid + j * THREADS, row = i >> 2, ch = i & 3;
        a_dst[j] = swz(row, ch);
        a_src[j] = A + static_cast<int64_t>(m0 + row) * Kb + ch * 16;
    }
    #pragma unroll
    for (int j = 0; j < C::B_CP; ++j) {
        const int i = tid + j * THREADS, row = i >> 2, ch = i & 3;
        b_dst[j] = C::A_BYTES + swz(row, ch);
        b_src[j] = B + static_cast<int64_t>(n0 + row) * Kb + ch * 16;
    }
    // escalas por grupo: peso = GPK x BN elementos de 4 B (GPK*BN/4 blocos de 16 B), ativacao = GPK x BM
    constexpr int SB_CH = C::SB_ELEMS / 4, SA_CH = C::SA_ELEMS / 4;
    const bool sb_on = tid < SB_CH, sa_on = tid >= SB_CH && tid < SB_CH + SA_CH;
    uint32_t s_dst = 0;
    const char* s_src = nullptr;
    int64_t s_step = 0;
    if (sb_on) {
        const int g = tid / (BN / 4), c = tid % (BN / 4);
        const char* base = (POL == 5) ? reinterpret_cast<const char*>(cw) : reinterpret_cast<const char*>(sb);
        s_dst = (g * BN + c * 4) * 4;
        s_src = base + (static_cast<int64_t>(g) * N + n0 + c * 4) * 4;
        s_step = static_cast<int64_t>(C::GPK) * N * 4;
    } else if (sa_on) {
        const int t = tid - SB_CH, g = t / (BM / 4), c = t % (BM / 4);
        const char* base = (POL == 6) ? reinterpret_cast<const char*>(ca) : reinterpret_cast<const char*>(sa);
        s_dst = (C::SB_ELEMS + g * BM + c * 4) * 4;
        s_src = base + (static_cast<int64_t>(g) * M + m0 + c * 4) * 4;
        s_step = static_cast<int64_t>(C::GPK) * M * 4;
    }
    auto load = [&](int kt) {
        const uint32_t sb_ = sbase + (kt % STAGES) * C::AB_BYTES;
        #pragma unroll
        for (int j = 0; j < C::A_CP; ++j) cp_async16(sb_ + a_dst[j], a_src[j] + kt * BKB);
        #pragma unroll
        for (int j = 0; j < C::B_CP; ++j) cp_async16(sb_ + b_dst[j], b_src[j] + kt * BKB);
        if (sb_on || sa_on) cp_async16(sScale + (kt % C::S_SLOTS) * C::S_BYTES + s_dst, s_src + kt * s_step);
    };

    const int q = lane >> 3, r8 = lane & 7, gq = lane >> 2, tq = lane & 3;
    const int a_row = wm * 64 + (q & 1) * 8 + r8, b_row = wn * WN + (q >> 1) * 8 + r8;
    const uint32_t a_off0 = swz(a_row, q >> 1), b_off0 = C::A_BYTES + swz(b_row, q & 1);
    auto ld_frags = [&](uint32_t (&af)[4][4], uint32_t (&bf)[NI / 2][4], int kt, int kk) {
        const uint32_t sb_ = sbase + (kt % STAGES) * C::AB_BYTES;
        const uint32_t ao = (a_off0 ^ (kk << 5)), bo = (b_off0 ^ (kk << 5));
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) ldsm_x4(af[mi], sb_ + ao + mi * 16 * BKB);
        #pragma unroll
        for (int np = 0; np < NI / 2; ++np) ldsm_x4(bf[np], sb_ + bo + np * 16 * BKB);
    };

    int acc_i[C::INT_ACC ? 4 : 1][C::INT_ACC ? NI : 1][4];
    float acc_f[C::INT_ACC ? 1 : 4][C::INT_ACC ? 1 : NI][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < NI; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if constexpr (C::INT_ACC) acc_i[mi][ni][j] = 0; else acc_f[mi][ni][j] = 0.0f;
            }

    // constantes do grupo `gk` (0/1 dentro do k-tile) lidas uma vez: peso por coluna (w, -M*w), ativacao por linha
    struct GrpConst { float w[NI][2], kw[NI][2], a[4][2], ka[4][2]; int cwi[NI][2], cai[4][2]; };
    auto grp_const = [&](int kt, int gk, GrpConst& g) {
        const char* sS = reinterpret_cast<const char*>(smem) + STAGES * C::AB_BYTES + (kt % C::S_SLOTS) * C::S_BYTES;
        if constexpr (C::WG) {
            const int off = gk * BN + wn * WN + tq * 2;
            #pragma unroll
            for (int ni = 0; ni < NI; ++ni) {
                if constexpr (POL == 5) {
                    const int2 v = *reinterpret_cast<const int2*>(sS + (off + ni * 8) * 4);
                    g.cwi[ni][0] = v.x; g.cwi[ni][1] = v.y;
                } else {
                    const float2 v = *reinterpret_cast<const float2*>(sS + (off + ni * 8) * 4);
                    g.w[ni][0] = v.x; g.w[ni][1] = v.y;
                    g.kw[ni][0] = -MAGIC_F * v.x; g.kw[ni][1] = -MAGIC_F * v.y;
                }
            }
        }
        if constexpr (C::AG) {
            const int off = C::SB_ELEMS + gk * BM + wm * 64 + gq;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if constexpr (POL == 6) {
                        g.cai[mi][h] = *reinterpret_cast<const int*>(sS + (off + mi * 16 + h * 8) * 4);
                    } else {
                        const float v = *reinterpret_cast<const float*>(sS + (off + mi * 16 + h * 8) * 4);
                        g.a[mi][h] = v; g.ka[mi][h] = -MAGIC_F * v;
                    }
                }
            }
        }
    };
    auto epi = [&](const int (&dd)[4], int mi, int ni, const GrpConst& g) {
        if constexpr (POL == 1) {
            float* acc = acc_f[mi][ni];
            acc[0] = fadd_v(acc[0], ffma_v(__int_as_float(dd[0]), g.w[ni][0], g.kw[ni][0]));
            acc[1] = fadd_v(acc[1], ffma_v(__int_as_float(dd[1]), g.w[ni][1], g.kw[ni][1]));
            acc[2] = fadd_v(acc[2], ffma_v(__int_as_float(dd[2]), g.w[ni][0], g.kw[ni][0]));
            acc[3] = fadd_v(acc[3], ffma_v(__int_as_float(dd[3]), g.w[ni][1], g.kw[ni][1]));
        } else if constexpr (POL == 2) {
            float* acc = acc_f[mi][ni];
            acc[0] = fadd_v(acc[0], ffma_v(__int_as_float(dd[0]), g.a[mi][0], g.ka[mi][0]));
            acc[1] = fadd_v(acc[1], ffma_v(__int_as_float(dd[1]), g.a[mi][0], g.ka[mi][0]));
            acc[2] = fadd_v(acc[2], ffma_v(__int_as_float(dd[2]), g.a[mi][1], g.ka[mi][1]));
            acc[3] = fadd_v(acc[3], ffma_v(__int_as_float(dd[3]), g.a[mi][1], g.ka[mi][1]));
        } else if constexpr (POL == 3 || POL == 4) {
            float* acc = acc_f[mi][ni];
            acc[0] = ffma_v(ffma_v(__int_as_float(dd[0]), g.w[ni][0], g.kw[ni][0]), g.a[mi][0], acc[0]);
            acc[1] = ffma_v(ffma_v(__int_as_float(dd[1]), g.w[ni][1], g.kw[ni][1]), g.a[mi][0], acc[1]);
            acc[2] = ffma_v(ffma_v(__int_as_float(dd[2]), g.w[ni][0], g.kw[ni][0]), g.a[mi][1], acc[2]);
            acc[3] = ffma_v(ffma_v(__int_as_float(dd[3]), g.w[ni][1], g.kw[ni][1]), g.a[mi][1], acc[3]);
        } else if constexpr (POL == 5) {
            int* acc = acc_i[mi][ni];
            acc[0] = imad_v(dd[0], g.cwi[ni][0], acc[0]); acc[1] = imad_v(dd[1], g.cwi[ni][1], acc[1]);
            acc[2] = imad_v(dd[2], g.cwi[ni][0], acc[2]); acc[3] = imad_v(dd[3], g.cwi[ni][1], acc[3]);
        } else if constexpr (POL == 6) {
            int* acc = acc_i[mi][ni];
            acc[0] = imad_v(dd[0], g.cai[mi][0], acc[0]); acc[1] = imad_v(dd[1], g.cai[mi][0], acc[1]);
            acc[2] = imad_v(dd[2], g.cai[mi][1], acc[2]); acc[3] = imad_v(dd[3], g.cai[mi][1], acc[3]);
        }
    };
    constexpr int CINIT = (POL == 5 || POL == 6) ? 0 : MAGIC_I;
    // um passo k64 com escala por grupo (pol 1, 2, 3, 5, 6): IMMA(i) e a parte por grupo de IMMA(i - PD)
    auto step_g64 = [&](const uint32_t (&af)[4][4], const uint32_t (&bf)[NI / 2][4], int kt, int kk) {
        GrpConst g;
        grp_const(kt, kk, g);
        constexpr int NM = 4 * NI;
        int d[PD + 1][4];
        #pragma unroll
        for (int i = 0; i < NM; ++i) {
            const int ni = i / 4, mi = i % 4;
            mma_c(d[i % (PD + 1)], af[mi], bf[ni / 2][(ni & 1) * 2], bf[ni / 2][(ni & 1) * 2 + 1], CINIT);
            if (i >= PD) epi(d[(i - PD) % (PD + 1)], (i - PD) % 4, (i - PD) / 4, g);
        }
        #pragma unroll
        for (int i = NM - PD; i < NM; ++i) epi(d[i % (PD + 1)], i % 4, i / 4, g);
    };
    // g128 (pol 4): os dois passos k64 do k-tile, IMMA(kk=0) com C magico e IMMA(kk=1) encadeado
    auto step_g128 = [&](const uint32_t (&af0)[4][4], const uint32_t (&bf0)[NI / 2][4],
                         const uint32_t (&af1)[4][4], const uint32_t (&bf1)[NI / 2][4], int kt) {
        GrpConst g;
        grp_const(kt, 0, g);
        constexpr int NM = 4 * NI;
        int d[PD + 1][4];
        #pragma unroll
        for (int i = 0; i < NM; ++i) {
            const int ni = i / 4, mi = i % 4;
            int (&di)[4] = d[i % (PD + 1)];
            mma_c(di, af0[mi], bf0[ni / 2][(ni & 1) * 2], bf0[ni / 2][(ni & 1) * 2 + 1], MAGIC_I);
            mma_chain(di, af1[mi], bf1[ni / 2][(ni & 1) * 2], bf1[ni / 2][(ni & 1) * 2 + 1]);
            if (i >= PD) epi(d[(i - PD) % (PD + 1)], (i - PD) % 4, (i - PD) / 4, g);
        }
        #pragma unroll
        for (int i = NM - PD; i < NM; ++i) epi(d[i % (PD + 1)], i % 4, i / 4, g);
    };
    auto step_row = [&](const uint32_t (&af)[4][4], const uint32_t (&bf)[NI / 2][4]) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi)
            #pragma unroll
            for (int ni = 0; ni < NI; ++ni)
                mma_acc(acc_i[mi][ni], af[mi], bf[ni / 2][(ni & 1) * 2], bf[ni / 2][(ni & 1) * 2 + 1]);
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s);
        cp_async_commit();
    }
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    uint32_t fa[2][4][4], fb[2][NI / 2][4];
    ld_frags(fa[0], fb[0], 0, 0);
    for (int kt = 0; kt < KT; ++kt) {
        ld_frags(fa[1], fb[1], kt, 1);
        if (kt + STAGES - 1 < KT) load(kt + STAGES - 1);
        cp_async_commit();
        if constexpr (POL == 4) {
            step_g128(fa[0], fb[0], fa[1], fb[1], kt);                 // usa os dois passos antes da barreira
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            if (kt + 1 < KT) ld_frags(fa[0], fb[0], kt + 1, 0);
        } else {
            if constexpr (POL == 0) step_row(fa[0], fb[0]); else step_g64(fa[0], fb[0], kt, 0);
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            if (kt + 1 < KT) ld_frags(fa[0], fb[0], kt + 1, 0);
            if constexpr (POL == 0) step_row(fa[1], fb[1]); else step_g64(fa[1], fb[1], kt, 1);
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // epilogo: escalas por linha que sobraram (pol 0/1/5: ativacao; pol 0/2/6: peso), bias, bf16 pela smem
    __nv_bfloat16* sC = reinterpret_cast<__nv_bfloat16*>(smem);
    constexpr bool ROW_A = POL == 0 || POL == 1 || POL == 5 || POL == 6;
    constexpr bool ROW_B = POL == 0 || POL == 2 || POL == 5 || POL == 6;
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int rl = wm * 64 + mi * 16 + gq;
        float s_r0 = 1.0f, s_r1 = 1.0f;
        if constexpr (ROW_A) { s_r0 = sa[m0 + rl]; s_r1 = sa[m0 + rl + 8]; }
        #pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
            const int cl = wn * WN + ni * 8 + tq * 2;
            float2 s_c = make_float2(1.0f, 1.0f);
            if constexpr (ROW_B) s_c = *reinterpret_cast<const float2*>(sb + n0 + cl);
            float2 bv = make_float2(0.0f, 0.0f);
            if (bias != nullptr) bv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + n0 + cl));
            float v[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if constexpr (C::INT_ACC) v[j] = static_cast<float>(acc_i[mi][ni][j]); else v[j] = acc_f[mi][ni][j];
            }
            v[0] = v[0] * s_r0 * s_c.x + bv.x;
            v[1] = v[1] * s_r0 * s_c.y + bv.y;
            v[2] = v[2] * s_r1 * s_c.x + bv.x;
            v[3] = v[3] * s_r1 * s_c.y + bv.y;
            *reinterpret_cast<__nv_bfloat162*>(sC + rl * C::LDC + cl) = __floats2bfloat162_rn(v[0], v[1]);
            *reinterpret_cast<__nv_bfloat162*>(sC + (rl + 8) * C::LDC + cl) = __floats2bfloat162_rn(v[2], v[3]);
        }
    }
    __syncthreads();
    constexpr int CH_ROW = BN * 2 / 16;
    #pragma unroll 4
    for (int i = tid; i < BM * CH_ROW; i += THREADS) {
        const int r = i / CH_ROW, c = i % CH_ROW;
        const uint4 v = *reinterpret_cast<const uint4*>(sC + r * C::LDC + c * 8);
        *reinterpret_cast<uint4*>(Cout + static_cast<int64_t>(m0 + r) * N + n0 + c * 8) = v;
    }
}

template<int BN, int STAGES, int POL, int PD>
void launch(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa, const torch::Tensor& sb,
            const torch::Tensor& ca, const torch::Tensor& cw, const torch::Tensor& bias, torch::Tensor& c) {
    using C = Cfg<BN, STAGES, POL>;
    const int M = a.size(0), N = b.size(0), K = a.size(1) * 2;
    TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % 128 == 0, "M%128, N%BN, K%128");
    auto kernel = gemm_kernel<BN, STAGES, POL, PD>;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        attr = true;
    }
    const __nv_bfloat16* bp = bias.numel() ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()) : nullptr;
    kernel<<<(M / BM) * (N / BN), THREADS, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), sa.data_ptr<float>(), sb.data_ptr<float>(),
        ca.numel() ? ca.data_ptr<int>() : nullptr, cw.numel() ? cw.data_ptr<int>() : nullptr, bp,
        reinterpret_cast<__nv_bfloat16*>(c.data_ptr()), M, N, K);
}

template<int BN, int STAGES, int PD>
void by_pol(int pol, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa, const torch::Tensor& sb,
            const torch::Tensor& ca, const torch::Tensor& cw, const torch::Tensor& bias, torch::Tensor& c) {
    switch (pol) {
        case 0: launch<BN, STAGES, 0, PD>(a, b, sa, sb, ca, cw, bias, c); break;
        case 1: launch<BN, STAGES, 1, PD>(a, b, sa, sb, ca, cw, bias, c); break;
        case 2: launch<BN, STAGES, 2, PD>(a, b, sa, sb, ca, cw, bias, c); break;
        case 3: launch<BN, STAGES, 3, PD>(a, b, sa, sb, ca, cw, bias, c); break;
        case 4: launch<BN, STAGES, 4, PD>(a, b, sa, sb, ca, cw, bias, c); break;
        case 5: launch<BN, STAGES, 5, PD>(a, b, sa, sb, ca, cw, bias, c); break;
        case 6: launch<BN, STAGES, 6, PD>(a, b, sa, sb, ca, cw, bias, c); break;
        default: TORCH_CHECK(false, "pol 0..6");
    }
}

}  // namespace

// cfg: 0 = BN 256 / 3 estagios / PD 2, 1 = BN 128 / 4 estagios / PD 2, 2 = BN 128 / 4 estagios / PD 1
torch::Tensor gemm(torch::Tensor a, torch::Tensor b, torch::Tensor sa, torch::Tensor sb, torch::Tensor ca,
                   torch::Tensor cw, torch::Tensor bias, int64_t pol, int64_t cfg) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "A/B int8 CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && sa.is_contiguous() && sb.is_contiguous(), "contiguos");
    TORCH_CHECK(sa.dtype() == torch::kFloat32 && sb.dtype() == torch::kFloat32, "escalas fp32");
    TORCH_CHECK(bias.numel() == 0 || (bias.dtype() == torch::kBFloat16 && bias.numel() == b.size(0)), "bias bf16 (N,)");
    const int64_t M = a.size(0), N = b.size(0), K = a.size(1) * 2;
    TORCH_CHECK(b.size(1) * 2 == K, "K");
    const int64_t ga = pol == 4 ? K / 128 : K / 64;
    const bool a_grp_f = pol == 2 || pol == 3 || pol == 4, b_grp_f = pol == 1 || pol == 3 || pol == 4;
    TORCH_CHECK(a_grp_f ? (sa.dim() == 2 && sa.size(0) == ga && sa.size(1) == M) : sa.numel() == M, "escala da ativacao");
    TORCH_CHECK(b_grp_f ? (sb.dim() == 2 && sb.size(0) == ga && sb.size(1) == N) : sb.numel() == N, "escala do peso");
    if (pol == 5) TORCH_CHECK(cw.dtype() == torch::kInt32 && cw.is_contiguous() && cw.size(0) == K / 64 && cw.size(1) == N, "cw int32 (K/64, N)");
    if (pol == 6) TORCH_CHECK(ca.dtype() == torch::kInt32 && ca.is_contiguous() && ca.size(0) == K / 64 && ca.size(1) == M, "ca int32 (K/64, M)");
    auto c = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));
    if (cfg == 0) by_pol<256, 3, 2>(static_cast<int>(pol), a, b, sa, sb, ca, cw, bias, c);
    else if (cfg == 1) by_pol<128, 4, 2>(static_cast<int>(pol), a, b, sa, sb, ca, cw, bias, c);
    else if (cfg == 2) by_pol<128, 4, 1>(static_cast<int>(pol), a, b, sa, sb, ca, cw, bias, c);
    else TORCH_CHECK(false, "cfg 0..2");
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "ConvRot W4A4 s4 x s4, politicas de escala 0..6 (v3)");
}
