// Fase 7, v6 do GEMM ConvRot W4A4 s4 x s4 (mma.sync m16n8k64, sm_80/86): as candidatas de menor custo depois das
// baterias fqF/fqG/fqH (peso g64 sozinho ~77 % do ganho; g128/g128 ~86 %, ambos a ~4 instrucoes por IMMA).
// Contrato de dados igual a v1..v5: A (M, K/2), B (N, K/2) int8 empacotados (nibble baixo = coluna par); sa (M,) fp32;
// sb (N,) fp32; bias bf16 (N,) opcional; saida bf16 (M, N).
//   0 row/row      acumulador int32, escalas no epilogo (controle; = ConvRot do ck)
//   4 g128/g128    sa (K/128, M), sb (K/128, N) fp32 (sb bf16-exata); dois IMMA encadeados com C = 0x4B400000 e
//                  acc = fma(fma(D, w, -M*w), a, acc) em fp32 (como no v3)
//   5 peso g64     cw (K/64, N) int32 em [1, 255], sb = escala da linha / 255: acc += P_g * cw (4 IMAD por IMMA)
//   7 peso g128    cw (K/128, N): os dois IMMA k64 do k-tile encadeiam em int32, depois acc += P * cw (2 IMAD por IMMA)
//   15, 17         = 5, 7 com cw em uint8 (VRAM das razoes / 4); na smem continuam uint8, lidas com LDS.U8 por coluna
//                  (asm: o compilador juntava as duas em LDS.U16 + 2 PRMT)
// |acc| <= 3136 * 255 * K/64 < 2^31 para K < 171840 (pol 7: 6272 * 255 * K/128, o mesmo limite).
// Esquema de fragmentos do v3: os fragmentos A e B de um passo k64 sao lidos da smem antes da barreira que libera o
// estagio para o proximo cp.async (o v4/v5 le B do passo 1 depois da barreira: corrida de escrita sobre o estagio).
// Mudancas sobre o v3: (a) anel de razoes com 4/8 posicoes (potencia de 2 >= STAGES + 1: sem divisao por 5);
// (b) destino dos cp.async = um registrador + imediatos, passado por asm vazio para nao ser recalculado a cada k-tile;
// (c) ponteiro das razoes incremental; (d) warps por bloco e blocos por SM no template (como o v5).
// Bloco 128 x BN x 128, warps 2 x NW/2, STAGES estagios, ordem dos blocos agrupada (GROUP_M), epilogo pela smem.
// Requer M % 128, N % BN, K % 128.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int BM = 128, BKB = 64;
constexpr int GROUP_M = 8;
constexpr float MAGIC_F = 12582912.0f;
constexpr int MAGIC_I = 0x4B400000;

template<int NW, int BN, int STAGES, int POL>
struct Cfg {
    static constexpr int THREADS = NW * 32, WARPS_N = NW / 2;
    static constexpr int WN = BN / WARPS_N, NI = WN / 8;
    static constexpr int A_BYTES = BM * BKB, B_BYTES = BN * BKB, AB_BYTES = A_BYTES + B_BYTES;
    static constexpr int BASE = POL % 10, CW_BYTES = POL >= 10 ? 1 : 4;          // 15/17 = 5/7 com cw uint8
    static constexpr bool FP = BASE == 4;                                       // g128/g128 com escalas fp32
    static constexpr int GPK = BASE == 5 ? 2 : ((BASE == 7 || FP) ? 1 : 0);    // grupos do peso por k-tile
    static constexpr int S_BYTES = FP ? (BN + BM) * 4 : GPK * BN * CW_BYTES, S_SLOTS = STAGES + 1 <= 4 ? 4 : 8;
    static constexpr int LDC = BN + 8;
    static constexpr int PIPE_BYTES = STAGES * AB_BYTES + S_SLOTS * S_BYTES, EPI_BYTES = BM * LDC * 2;
    static constexpr int SMEM = PIPE_BYTES > EPI_BYTES ? PIPE_BYTES : EPI_BYTES;
    static constexpr int RC = THREADS / 4;                                      // linhas por rodada de cp.async
    static constexpr int A_CP = BM / RC, B_CP = BN / RC;
    static constexpr int SB_CH = S_BYTES / 16;                                  // blocos de 16 B de razoes por k-tile
    static_assert(BM % RC == 0 && BN % RC == 0 && RC % 8 == 0, "copias inteiras; linhas j > 0 na fase do swizzle");
    static_assert(SB_CH <= THREADS, "um bloco de razoes por thread");
    static_assert(NI % 2 == 0, "pares de colunas n8");
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
__device__ __forceinline__ void mma_zero(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ void mma_c(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int c) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}
__device__ __forceinline__ float ffma_v(float a, float b, float c) {
    float d;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;\n" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}
__device__ __forceinline__ void mma_chain(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int imad_v(int a, int b, int c) {
    int d;
    asm volatile("mad.lo.s32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
__device__ __forceinline__ int swz(int row, int chunk) { return row * BKB + ((chunk ^ ((row >> 1) & 3)) << 4); }

template<int NW, int BN, int STAGES, int POL, int PD, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB)
gemm_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B, const float* __restrict__ sa,
            const float* __restrict__ sb, const void* __restrict__ cw, const __nv_bfloat16* __restrict__ bias,
            __nv_bfloat16* __restrict__ Cout, int M, int N, int K) {
    using C = Cfg<NW, BN, STAGES, POL>;
    constexpr int NI = C::NI, WN = C::WN, THREADS = C::THREADS, RC = C::RC;
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / C::WARPS_N, wn = warp % C::WARPS_N;

    const int pid = blockIdx.x, npm = M / BM, npn = N / BN;
    const int per_group = GROUP_M * npn, gid = pid / per_group, first_m = gid * GROUP_M;
    const int gsz = min(npm - first_m, GROUP_M);
    const int pm = first_m + (pid % per_group) % gsz, pn = (pid % per_group) / gsz;
    const int m0 = pm * BM, n0 = pn * BN;
    const int Kb = K / 2, KT = K / 128;

    // ---- cp.async: a thread copia o bloco de 16 B tid%4 das linhas tid/4 + j*RC; destino = dst0 + imediato ----
    const uint32_t sbase = smem_u32(smem);
    uint32_t dst0 = swz(tid >> 2, tid & 3);
    asm volatile("" : "+r"(dst0));                                              // nao recalcular por k-tile
    const int8_t* a_src[C::A_CP];
    const int8_t* b_src[C::B_CP];
    #pragma unroll
    for (int j = 0; j < C::A_CP; ++j) a_src[j] = A + static_cast<int64_t>(m0 + (tid >> 2) + j * RC) * Kb + (tid & 3) * 16;
    #pragma unroll
    for (int j = 0; j < C::B_CP; ++j) b_src[j] = B + static_cast<int64_t>(n0 + (tid >> 2) + j * RC) * Kb + (tid & 3) * 16;
    // razoes do peso: GPK x BN por k-tile (int32 ou uint8); a thread tid < SB_CH copia o bloco de 16 B tid (tid*16 no anel)
    constexpr int PER16 = 16 / C::CW_BYTES;                                     // colunas por bloco de 16 B
    // (pol 4: escalas fp32 do peso, BN, e da ativacao, BM, do grupo g128 do k-tile; cada uma com o proprio passo)
    const bool s_on = tid < C::SB_CH;
    const char* s_ptr = static_cast<const char*>(cw);
    int64_t s_step = static_cast<int64_t>(C::GPK) * N * C::CW_BYTES;
    if constexpr (C::FP) {
        if (tid < BN / 4) {
            s_ptr = reinterpret_cast<const char*>(sb + n0 + tid * 4);
        } else {
            s_ptr = reinterpret_cast<const char*>(sa + m0 + (tid - BN / 4) * 4);
            s_step = static_cast<int64_t>(M) * 4;
        }
    } else if constexpr (C::GPK > 0) {
        if (s_on) s_ptr += (static_cast<int64_t>(tid / (BN / PER16)) * N + n0 + (tid % (BN / PER16)) * PER16) * C::CW_BYTES;
    }
    auto load = [&](int kt) {                                                   // kt crescente, uma vez cada
        const uint32_t d = sbase + (kt % STAGES) * C::AB_BYTES + dst0;
        #pragma unroll
        for (int j = 0; j < C::A_CP; ++j) cp_async16(d + j * RC * BKB, a_src[j] + kt * BKB);
        #pragma unroll
        for (int j = 0; j < C::B_CP; ++j) cp_async16(d + C::A_BYTES + j * RC * BKB, b_src[j] + kt * BKB);
        if constexpr (C::GPK > 0) {
            if (s_on) {
                cp_async16(sbase + STAGES * C::AB_BYTES + (kt % C::S_SLOTS) * C::S_BYTES + tid * 16, s_ptr);
                s_ptr += s_step;
            }
        }
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

    int acc[C::FP ? 1 : 4][C::FP ? 1 : NI][4];
    float accf[C::FP ? 4 : 1][C::FP ? NI : 1][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < NI; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if constexpr (C::FP) accf[mi][ni][j] = 0.0f; else acc[mi][ni][j] = 0;
            }

    // razoes do grupo gk (posicao no k-tile) para as colunas 2tq, 2tq+1 de cada n8 do warp
    auto w_ratio = [&](int kt, int gk, int (&c)[NI][2]) {
        const char* sS = reinterpret_cast<const char*>(smem) + STAGES * C::AB_BYTES + (kt % C::S_SLOTS) * C::S_BYTES;
        const int off = gk * BN + wn * WN + tq * 2;
        #pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
            if constexpr (C::CW_BYTES == 4) {
                const int2 v = *reinterpret_cast<const int2*>(sS + (off + ni * 8) * 4);
                c[ni][0] = v.x; c[ni][1] = v.y;
            } else {                                                            // dois LDS.U8 (sem PRMT)
                const uint32_t u = smem_u32(sS) + off + ni * 8;
                asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(c[ni][0]) : "r"(u));
                asm volatile("ld.shared.u8 %0, [%1+1];\n" : "=r"(c[ni][1]) : "r"(u));
            }
        }
    };
    auto epi = [&](const int (&dd)[4], int mi, int ni, const int (&c)[NI][2]) {
        int* a = acc[mi][ni];
        a[0] = imad_v(dd[0], c[ni][0], a[0]); a[1] = imad_v(dd[1], c[ni][1], a[1]);
        a[2] = imad_v(dd[2], c[ni][0], a[2]); a[3] = imad_v(dd[3], c[ni][1], a[3]);
    };
    // passo k64 da politica 5: IMMA(i) com C = 0 e o IMAD do IMMA(i - PD)
    auto step_g64 = [&](const uint32_t (&af)[4][4], const uint32_t (&bf)[NI / 2][4], int kt, int kk) {
        int c[NI][2];
        w_ratio(kt, kk, c);
        constexpr int NM = 4 * NI;
        int d[PD + 1][4];
        #pragma unroll
        for (int i = 0; i < NM; ++i) {
            const int ni = i / 4, mi = i % 4;
            mma_zero(d[i % (PD + 1)], af[mi], bf[ni / 2][(ni & 1) * 2], bf[ni / 2][(ni & 1) * 2 + 1]);
            if (i >= PD) epi(d[(i - PD) % (PD + 1)], (i - PD) % 4, (i - PD) / 4, c);
        }
        #pragma unroll
        for (int i = NM - PD; i < NM; ++i) epi(d[i % (PD + 1)], i % 4, i / 4, c);
    };
    // politica 7 (g128): IMMA(kk=0) com C = 0, IMMA(kk=1) encadeado no mesmo int32, depois o IMAD (atrasado PD)
    auto step_g128 = [&](const uint32_t (&af0)[4][4], const uint32_t (&bf0)[NI / 2][4],
                         const uint32_t (&af1)[4][4], const uint32_t (&bf1)[NI / 2][4], int kt) {
        int c[NI][2];
        w_ratio(kt, 0, c);
        constexpr int NM = 4 * NI;
        int d[PD + 1][4];
        #pragma unroll
        for (int i = 0; i < NM; ++i) {
            const int ni = i / 4, mi = i % 4;
            int (&di)[4] = d[i % (PD + 1)];
            mma_zero(di, af0[mi], bf0[ni / 2][(ni & 1) * 2], bf0[ni / 2][(ni & 1) * 2 + 1]);
            mma_chain(di, af1[mi], bf1[ni / 2][(ni & 1) * 2], bf1[ni / 2][(ni & 1) * 2 + 1]);
            if (i >= PD) epi(d[(i - PD) % (PD + 1)], (i - PD) % 4, (i - PD) / 4, c);
        }
        #pragma unroll
        for (int i = NM - PD; i < NM; ++i) epi(d[i % (PD + 1)], i % 4, i / 4, c);
    };
    // politica 4 (g128/g128 FP): IMMA(kk=0) com C = 0x4B400000, IMMA(kk=1) encadeado; D lido como float = 1,5*2^23 + P
    // e acc = fma(fma(D, w, -M*w), a, acc) (w bf16-exata: o primeiro fma da P*w exato antes de um arredondamento)
    struct FConst { float w[NI][2], kw[NI][2], a[4][2]; };
    auto f_const = [&](int kt, FConst& g) {
        const char* sS = reinterpret_cast<const char*>(smem) + STAGES * C::AB_BYTES + (kt % C::S_SLOTS) * C::S_BYTES;
        const int offw = wn * WN + tq * 2, offa = BN + wm * 64 + gq;
        #pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
            const float2 v = *reinterpret_cast<const float2*>(sS + (offw + ni * 8) * 4);
            g.w[ni][0] = v.x; g.w[ni][1] = v.y;
            g.kw[ni][0] = -MAGIC_F * v.x; g.kw[ni][1] = -MAGIC_F * v.y;
        }
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi)
            #pragma unroll
            for (int h = 0; h < 2; ++h) g.a[mi][h] = *reinterpret_cast<const float*>(sS + (offa + mi * 16 + h * 8) * 4);
    };
    auto epi_f = [&](const int (&dd)[4], int mi, int ni, const FConst& g) {
        float* a = accf[mi][ni];
        a[0] = ffma_v(ffma_v(__int_as_float(dd[0]), g.w[ni][0], g.kw[ni][0]), g.a[mi][0], a[0]);
        a[1] = ffma_v(ffma_v(__int_as_float(dd[1]), g.w[ni][1], g.kw[ni][1]), g.a[mi][0], a[1]);
        a[2] = ffma_v(ffma_v(__int_as_float(dd[2]), g.w[ni][0], g.kw[ni][0]), g.a[mi][1], a[2]);
        a[3] = ffma_v(ffma_v(__int_as_float(dd[3]), g.w[ni][1], g.kw[ni][1]), g.a[mi][1], a[3]);
    };
    auto step_g128f = [&](const uint32_t (&af0)[4][4], const uint32_t (&bf0)[NI / 2][4],
                          const uint32_t (&af1)[4][4], const uint32_t (&bf1)[NI / 2][4], int kt) {
        FConst g;
        f_const(kt, g);
        constexpr int NM = 4 * NI;
        int d[PD + 1][4];
        #pragma unroll
        for (int i = 0; i < NM; ++i) {
            const int ni = i / 4, mi = i % 4;
            int (&di)[4] = d[i % (PD + 1)];
            mma_c(di, af0[mi], bf0[ni / 2][(ni & 1) * 2], bf0[ni / 2][(ni & 1) * 2 + 1], MAGIC_I);
            mma_chain(di, af1[mi], bf1[ni / 2][(ni & 1) * 2], bf1[ni / 2][(ni & 1) * 2 + 1]);
            if (i >= PD) epi_f(d[(i - PD) % (PD + 1)], (i - PD) % 4, (i - PD) / 4, g);
        }
        #pragma unroll
        for (int i = NM - PD; i < NM; ++i) epi_f(d[i % (PD + 1)], i % 4, i / 4, g);
    };
    auto step_row = [&](const uint32_t (&af)[4][4], const uint32_t (&bf)[NI / 2][4]) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi)
            #pragma unroll
            for (int ni = 0; ni < NI; ++ni)
                mma_acc(acc[mi][ni], af[mi], bf[ni / 2][(ni & 1) * 2], bf[ni / 2][(ni & 1) * 2 + 1]);
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
        ld_frags(fa[1], fb[1], kt, 1);                                          // o passo 1 inteiro antes da barreira
        if (kt + STAGES - 1 < KT) load(kt + STAGES - 1);
        cp_async_commit();
        if constexpr (C::BASE == 7 || C::FP) {
            if constexpr (C::FP) step_g128f(fa[0], fb[0], fa[1], fb[1], kt); else step_g128(fa[0], fb[0], fa[1], fb[1], kt);
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

    // epilogo: escala da linha da ativacao x escala do peso (pol 5/7/15/17: escala da linha / 255; pol 4: ja aplicadas),
    // bias, bf16 pela smem
    __nv_bfloat16* sC = reinterpret_cast<__nv_bfloat16*>(smem);
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int rl = wm * 64 + mi * 16 + gq;
        float s_r0 = 1.0f, s_r1 = 1.0f;
        if constexpr (!C::FP) { s_r0 = sa[m0 + rl]; s_r1 = sa[m0 + rl + 8]; }
        #pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
            const int cl = wn * WN + ni * 8 + tq * 2;
            float2 bv = make_float2(0.0f, 0.0f);
            if (bias != nullptr) bv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + n0 + cl));
            float v0, v1, v2, v3;
            if constexpr (C::FP) {
                const float* a = accf[mi][ni];
                v0 = a[0] + bv.x; v1 = a[1] + bv.y; v2 = a[2] + bv.x; v3 = a[3] + bv.y;
            } else {
                const float2 s_c = *reinterpret_cast<const float2*>(sb + n0 + cl);
                const int* a = acc[mi][ni];
                v0 = static_cast<float>(a[0]) * s_r0 * s_c.x + bv.x;
                v1 = static_cast<float>(a[1]) * s_r0 * s_c.y + bv.y;
                v2 = static_cast<float>(a[2]) * s_r1 * s_c.x + bv.x;
                v3 = static_cast<float>(a[3]) * s_r1 * s_c.y + bv.y;
            }
            *reinterpret_cast<__nv_bfloat162*>(sC + rl * C::LDC + cl) = __floats2bfloat162_rn(v0, v1);
            *reinterpret_cast<__nv_bfloat162*>(sC + (rl + 8) * C::LDC + cl) = __floats2bfloat162_rn(v2, v3);
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

template<int NW, int BN, int STAGES, int POL, int PD, int MINB>
void launch(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa, const torch::Tensor& sb,
            const torch::Tensor& cw, const torch::Tensor& bias, torch::Tensor& c) {
    using C = Cfg<NW, BN, STAGES, POL>;
    const int M = a.size(0), N = b.size(0), K = a.size(1) * 2;
    TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % 128 == 0, "M%128, N%BN, K%128");
    auto kernel = gemm_kernel<NW, BN, STAGES, POL, PD, MINB>;
    static bool attr = false;
    if (!attr) {
        TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) == cudaSuccess,
                    "smem dinamica");
        attr = true;
    }
    const __nv_bfloat16* bp = bias.numel() ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()) : nullptr;
    kernel<<<(M / BM) * (N / BN), C::THREADS, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), sa.data_ptr<float>(), sb.data_ptr<float>(),
        cw.numel() ? cw.data_ptr() : nullptr, bp, reinterpret_cast<__nv_bfloat16*>(c.data_ptr()), M, N, K);
}

template<int NW, int BN, int STAGES, int PD, int MINB>
void by_pol(int pol, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa, const torch::Tensor& sb,
            const torch::Tensor& cw, const torch::Tensor& bias, torch::Tensor& c) {
    switch (pol) {
        case 0: launch<NW, BN, STAGES, 0, PD, MINB>(a, b, sa, sb, cw, bias, c); break;
        case 5: launch<NW, BN, STAGES, 5, PD, MINB>(a, b, sa, sb, cw, bias, c); break;
        case 4: launch<NW, BN, STAGES, 4, PD, MINB>(a, b, sa, sb, cw, bias, c); break;
        case 7: launch<NW, BN, STAGES, 7, PD, MINB>(a, b, sa, sb, cw, bias, c); break;
        case 15: launch<NW, BN, STAGES, 15, PD, MINB>(a, b, sa, sb, cw, bias, c); break;
        case 17: launch<NW, BN, STAGES, 17, PD, MINB>(a, b, sa, sb, cw, bias, c); break;
        default: TORCH_CHECK(false, "pol 0, 4, 5, 7, 15, 17");
    }
}

}  // namespace

// cfg: 0 = 8 warps BN 256 3 est. | 1 = 8 warps BN 128 4 est. | 2 = 8 warps BN 128 3 est.
//      3 = 4 warps BN 128 2 est. 2/SM | 4 = 4 warps BN 64 3 est. 2/SM
// Assinatura igual a v3..v5 (ca ignorado) para os mesmos bancos de teste e o mesmo no.
torch::Tensor gemm(torch::Tensor a, torch::Tensor b, torch::Tensor sa, torch::Tensor sb, torch::Tensor ca,
                   torch::Tensor cw, torch::Tensor bias, int64_t pol, int64_t cfg) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "A/B int8 CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && sa.is_contiguous() && sb.is_contiguous(), "contiguos");
    TORCH_CHECK(sa.dtype() == torch::kFloat32 && sb.dtype() == torch::kFloat32, "escalas fp32");
    TORCH_CHECK(bias.numel() == 0 || (bias.dtype() == torch::kBFloat16 && bias.numel() == b.size(0)), "bias bf16 (N,)");
    const int64_t M = a.size(0), N = b.size(0), K = a.size(1) * 2;
    TORCH_CHECK(b.size(1) * 2 == K && K < 171840, "K (< 171840: acumulador int32)");
    if (pol == 4) {
        TORCH_CHECK(sa.dim() == 2 && sa.size(0) == K / 128 && sa.size(1) == M && sb.dim() == 2 && sb.size(0) == K / 128 &&
                    sb.size(1) == N, "pol 4: sa (K/128, M), sb (K/128, N) fp32");
    } else {
        TORCH_CHECK(sa.numel() == M && sb.numel() == N, "escalas por linha (M,), (N,)");
    }
    if (pol % 10 == 5 || pol % 10 == 7) {
        const int64_t gw = pol % 10 == 5 ? 64 : 128;
        TORCH_CHECK(cw.dtype() == (pol >= 10 ? torch::kUInt8 : torch::kInt32) && cw.is_contiguous() && cw.dim() == 2 &&
                    cw.size(0) == K / gw && cw.size(1) == N, "cw (K/G, N): int32 (pol 5/7) ou uint8 (15/17)");
    }
    auto c = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));
    const int p = static_cast<int>(pol);
    if (cfg == 0) by_pol<8, 256, 3, 2, 1>(p, a, b, sa, sb, cw, bias, c);
    else if (cfg == 1) by_pol<8, 128, 4, 2, 1>(p, a, b, sa, sb, cw, bias, c);
    else if (cfg == 2) by_pol<8, 128, 3, 2, 1>(p, a, b, sa, sb, cw, bias, c);
    else if (cfg == 3) by_pol<4, 128, 2, 2, 2>(p, a, b, sa, sb, cw, bias, c);
    else if (cfg == 4) by_pol<4, 64, 3, 2, 2>(p, a, b, sa, sb, cw, bias, c);
    else TORCH_CHECK(false, "cfg 0..4");
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "ConvRot W4A4 s4 x s4, politicas 0, 4, 5, 7, 15, 17 (v6)");
}
