// GEMM 2:4 esparso do CUTLASS, com o tile escolhido para CABER na sm_86.
//
// POR QUE ESTE ARQUIVO EXISTE
// ---------------------------
// O kernel 2:4 que o xformers distribui recusa esta placa com `Got CUTLASS error: Error Internal`
// em `sparse24/gemm.cu:190`. Medido nesta bancada em 2026-09-01, a causa nao e a placa:
//
//   * `can_implement()` PASSA -- shapes e layouts sao aceitaveis (a falha e na linha 190, que e o
//     check depois de `initialize()`, nao depois de `can_implement()` na 180);
//   * falha identica em 12 de 12 combinacoes de forma (64 a 3840) e nos dois dtypes, ou seja e
//     limite de recurso fixo e nao propriedade da entrada;
//   * `cutlass/gemm/device/gemm_sparse.h:438-446` chama `cudaFuncSetAttribute` com
//     `sizeof(GemmKernel::SharedStorage)` e devolve `kErrorInternal` se ele recusar;
//   * MEDIDO nesta 3090: `cudaFuncSetAttribute` aceita ate 101.376 bytes e recusa 100 KB. A A100
//     (sm_80) aceita 163.840. A config que o xformers compila e `GemmShape<256,128,64>` com
//     `NumStages = 4`, lida no nome desmangled dentro do proprio `_C.pyd` e confirmada no fonte
//     deles (`gemm.cu:66` e `:280`).
//
// Ou seja: a 3090 TEM tensor core esparso, e o kernel nao cabe na memoria compartilhada dela
// porque o tile foi dimensionado para a A100. O conserto e a config, nao o hardware.
//
// O QUE ESTE ARQUIVO ACRESCENTA E QUE NENHUM DOS PASSOS ACIMA DAVA
// ----------------------------------------------------------------
// `smem_bytes(cfg)` devolve `sizeof(typename Gemm::GemmKernel::SharedStorage)` -- o numero que o
// COMPILADOR calculou, nao a aritmetica de guardanapo de quem leu o nome do tipo. Sem isto, dizer
// "a config do xformers pede ~139 KB" seria deducao; com isto e leitura.
//
// NAO COBERTO
// -----------
// bf16 apenas, A e B row-major apenas, sem split-k, sem bias. Quatro configs, escolhidas para
// isolar UM eixo de cada vez (mesmo tile com menos estagios; tile menor com os mesmos estagios) --
// nao e uma varredura do espaco de tuning, e a config mais rapida que cabe nao foi procurada.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>

namespace {

// Uma instanciacao do SparseGemm do CUTLASS. Os parametros que variam entre as configs estao no
// template; todo o resto e fixo e igual ao que o xformers usa, de proposito -- se algo mais mudasse
// junto, a comparacao entre configs mediria duas coisas ao mesmo tempo.
template <typename TBShape, typename WarpShape, int kStages>
struct Config {
  using Element = cutlass::bfloat16_t;
  using ElementAcc = float;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      Element, 128 / cutlass::sizeof_bits<Element>::value, ElementAcc, ElementAcc>;
  using Gemm = cutlass::gemm::device::SparseGemm<
      Element, cutlass::layout::RowMajor,
      Element, cutlass::layout::RowMajor,
      Element, cutlass::layout::RowMajor,
      ElementAcc,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,          // igual ao xformers: a sm_86 e "CC 8.x" e e aceita
      TBShape, WarpShape, InstructionShape,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<3>,
      kStages>;

  static int64_t smem() {
    return (int64_t)sizeof(typename Gemm::GemmKernel::SharedStorage);
  }

  // a: [m, k/2] comprimido, row-major.  b: [k, n] row-major.  meta: reordenado, uint16.
  static torch::Tensor run(const torch::Tensor& a, const torch::Tensor& b,
                           const torch::Tensor& meta) {
    using G = Gemm;
    using ElementE = typename G::ElementE;
    using LayoutE = typename G::LayoutE;

    const int m = a.size(0);
    const int k = b.size(0);
    const int n = b.size(1);
    const int meta_ncols = k / G::kSparse / G::kElementsPerElementE;

    auto d = torch::empty({m, n}, a.options());
    cutlass::gemm::GemmCoord problem(m, n, k);

    auto ref_a = cutlass::TensorRef<Element, cutlass::layout::RowMajor>(
        (Element*)a.data_ptr(), cutlass::layout::RowMajor(a.stride(0)));
    auto ref_b = cutlass::TensorRef<Element, cutlass::layout::RowMajor>(
        (Element*)b.data_ptr(), cutlass::layout::RowMajor(b.stride(0)));
    auto ref_d = cutlass::TensorRef<Element, cutlass::layout::RowMajor>(
        (Element*)d.data_ptr(), cutlass::layout::RowMajor(d.stride(0)));
    auto ref_e = cutlass::TensorRef<ElementE, LayoutE>(
        (ElementE*)meta.data_ptr(), LayoutE::packed({m, meta_ncols}));

    typename G::Arguments args{problem, ref_a, ref_b, ref_d, ref_d, ref_e,
                               {ElementAcc(1), ElementAcc(0)}, 1};
    G op;
    // Cada status vai separado: "can_implement recusou" e "initialize recusou" sao diagnosticos
    // diferentes, e foi exatamente distinguir os dois que localizou o problema no xformers.
    TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess,
                "can_implement recusou: ", cutlassGetStatusString(op.can_implement(args)));
    auto ws = torch::empty({(int64_t)G::get_workspace_size(args)},
                           a.options().dtype(torch::kUInt8));
    auto st = op.initialize(args, ws.data_ptr(), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "initialize recusou: ", cutlassGetStatusString(st),
                " (smem pedida = ", smem(), " bytes)");
    st = op.run(at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "run recusou: ", cutlassGetStatusString(st));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return d;
  }
};

using G256x128s4 = Config<cutlass::gemm::GemmShape<256, 128, 64>,
                          cutlass::gemm::GemmShape<64, 64, 64>, 4>;  // <- a do xformers
using G256x128s2 = Config<cutlass::gemm::GemmShape<256, 128, 64>,
                          cutlass::gemm::GemmShape<64, 64, 64>, 2>;  // so os estagios mudam
using G128x128s3 = Config<cutlass::gemm::GemmShape<128, 128, 64>,
                          cutlass::gemm::GemmShape<64, 64, 64>, 3>;
using G128x128s2 = Config<cutlass::gemm::GemmShape<128, 128, 64>,
                          cutlass::gemm::GemmShape<64, 64, 64>, 2>;

}  // namespace

std::vector<int64_t> smem_de_cada_config() {
  return {G256x128s4::smem(), G256x128s2::smem(), G128x128s3::smem(), G128x128s2::smem()};
}

torch::Tensor mm(const torch::Tensor& a, const torch::Tensor& b,
                 const torch::Tensor& meta, int64_t cfg) {
  switch (cfg) {
    case 0: return G256x128s4::run(a, b, meta);
    case 1: return G256x128s2::run(a, b, meta);
    case 2: return G128x128s3::run(a, b, meta);
    case 3: return G128x128s2::run(a, b, meta);
  }
  TORCH_CHECK(false, "config desconhecida: ", cfg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("smem_de_cada_config", &smem_de_cada_config,
        "sizeof(GemmKernel::SharedStorage) de cada config, calculado pelo compilador");
  m.def("mm", &mm, "GEMM 2:4 esparso: a[m,k/2] comprimido x b[k,n] -> [m,n]");
}
