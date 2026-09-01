// SparseGemm 2:4 do CUTLASS em INT8 e INT4, para medir a linha de 3 bits/peso.
//
// POR QUE
// -------
// `sp24_gemm.cu` fechou o 2:4 em bf16: 1,7x-1,95x sobre o denso, medido. Aquilo custa 9,0
// bits/peso e o W4A4 que esta bancada ja entrega custa 4,0. A linha que ganharia das duas e a
// composicao **2:4 + 4 bits = 3,0 bits/peso**, e o registro dela dizia "SEM KERNEL, nao medido".
// Este arquivo e a tentativa de tirar aquela celula do vazio.
//
//     formato          bits/peso   como se conta
//     2:4 + bf16            9,0    (2 x 16 + 4 de indice) / 4
//     2:4 + int8            5,0    (2 x  8 + 4 de indice) / 4
//     2:4 + int4            3,0    (2 x  4 + 4 de indice) / 4
//
// O INT8 nao esta aqui so de escada: ele e o braco C de `probe_esparso_promocao.py` (promover a 8
// bits so as camadas que aceitam) e nunca foi cronometrado. E, se o INT4 falhar, o INT8 diz se a
// falha e do caminho inteiro ou so daquela largura -- sem ele um INT4 quebrado nao se distingue de
// "esparso inteiro nao funciona".
//
// O QUE `params()` EXISTE PARA EVITAR
// -----------------------------------
// Empacotar 2:4 exige tres numeros que dependem do tipo do elemento: `kSparse`,
// `kElementsPerElementE` e o tamanho de `ElementE`. Eu podia deduzi-los da documentacao; ja errei
// hoje deduzindo `sizeof(SharedStorage)` de um nome desmangled. `params()` pergunta ao COMPILADOR,
// e o empacotador em Python le a resposta em vez de assumir.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>

namespace {

template <typename Elem, typename ElemAcc, typename TBShape, typename WarpShape,
          typename InstShape, int kStages>
struct Cfg {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElemAcc, 128 / cutlass::sizeof_bits<ElemAcc>::value, ElemAcc, float>;
  using Gemm = cutlass::gemm::device::SparseGemm<
      Elem, cutlass::layout::RowMajor,
      Elem, cutlass::layout::ColumnMajor,   // B column-major: e o layout que os kernels
      ElemAcc, cutlass::layout::RowMajor,   // inteiros do CUTLASS aceitam na sm_80
      ElemAcc,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TBShape, WarpShape, InstShape,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<3>,
      kStages>;

  static int64_t smem() { return (int64_t)sizeof(typename Gemm::GemmKernel::SharedStorage); }
  // (kSparse, kElementsPerElementE, sizeof(ElementE), bits do elemento de dado)
  static std::vector<int64_t> params() {
    return {(int64_t)Gemm::kSparse, (int64_t)Gemm::kElementsPerElementE,
            (int64_t)sizeof(typename Gemm::ElementE),
            (int64_t)cutlass::sizeof_bits<Elem>::value};
  }

  static torch::Tensor run(const torch::Tensor& a, const torch::Tensor& b,
                           const torch::Tensor& meta, int64_t m, int64_t n, int64_t k) {
    using G = Gemm;
    using ElementE = typename G::ElementE;
    using LayoutE = typename G::LayoutE;
    const int meta_ncols = (int)(k / G::kSparse / G::kElementsPerElementE);

    auto d = torch::empty({m, n}, a.options().dtype(
        std::is_same<ElemAcc, int32_t>::value ? torch::kInt32 : torch::kFloat32));
    cutlass::gemm::GemmCoord problem((int)m, (int)n, (int)k);

    // As formas logicas nao vem do `size()` dos tensores: em int4 o container e int8 e um
    // `size()` mediria bytes, nao elementos. Por isso m/n/k chegam por argumento.
    auto ref_a = cutlass::TensorRef<Elem, cutlass::layout::RowMajor>(
        (Elem*)a.data_ptr(), cutlass::layout::RowMajor((int)(k / G::kSparse)));
    auto ref_b = cutlass::TensorRef<Elem, cutlass::layout::ColumnMajor>(
        (Elem*)b.data_ptr(), cutlass::layout::ColumnMajor((int)k));
    auto ref_d = cutlass::TensorRef<ElemAcc, cutlass::layout::RowMajor>(
        (ElemAcc*)d.data_ptr(), cutlass::layout::RowMajor((int)n));
    auto ref_e = cutlass::TensorRef<ElementE, LayoutE>(
        (ElementE*)meta.data_ptr(), LayoutE::packed({(int)m, meta_ncols}));

    typename G::Arguments args{problem, ref_a, ref_b, ref_d, ref_d, ref_e, {1, 0}, 1};
    G op;
    auto st = op.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "can_implement recusou: ", cutlassGetStatusString(st));
    auto ws = torch::empty({(int64_t)G::get_workspace_size(args)},
                           a.options().dtype(torch::kUInt8));
    st = op.initialize(args, ws.data_ptr(), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(st == cutlass::Status::kSuccess, "initialize recusou: ",
                cutlassGetStatusString(st), " (smem = ", smem(), " bytes)");
    st = op.run(at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(st == cutlass::Status::kSuccess, "run recusou: ", cutlassGetStatusString(st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return d;
  }
};

using S8_a = Cfg<int8_t, int32_t, cutlass::gemm::GemmShape<128, 128, 128>,
                 cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 64>, 3>;
using S8_b = Cfg<int8_t, int32_t, cutlass::gemm::GemmShape<128, 128, 128>,
                 cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 64>, 2>;
using S4_a = Cfg<cutlass::int4b_t, int32_t, cutlass::gemm::GemmShape<128, 128, 256>,
                 cutlass::gemm::GemmShape<64, 64, 256>, cutlass::gemm::GemmShape<16, 8, 128>, 3>;
using S4_b = Cfg<cutlass::int4b_t, int32_t, cutlass::gemm::GemmShape<128, 128, 256>,
                 cutlass::gemm::GemmShape<64, 64, 256>, cutlass::gemm::GemmShape<16, 8, 128>, 2>;

}  // namespace

std::vector<int64_t> smems() { return {S8_a::smem(), S8_b::smem(), S4_a::smem(), S4_b::smem()}; }

std::vector<std::vector<int64_t>> params_de_cada() {
  return {S8_a::params(), S8_b::params(), S4_a::params(), S4_b::params()};
}

torch::Tensor mm(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& meta,
                 int64_t cfg, int64_t m, int64_t n, int64_t k) {
  switch (cfg) {
    case 0: return S8_a::run(a, b, meta, m, n, k);
    case 1: return S8_b::run(a, b, meta, m, n, k);
    case 2: return S4_a::run(a, b, meta, m, n, k);
    case 3: return S4_b::run(a, b, meta, m, n, k);
  }
  TORCH_CHECK(false, "config desconhecida: ", cfg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("smems", &smems, "sizeof(GemmKernel::SharedStorage) de cada config");
  m.def("params_de_cada", &params_de_cada,
        "(kSparse, kElementsPerElementE, sizeof(ElementE), bits do dado) por config");
  m.def("mm", &mm, "GEMM 2:4 esparso inteiro; m/n/k logicos vao por argumento");
}
