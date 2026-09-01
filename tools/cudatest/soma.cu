// Extensao CUDA minima: prova que nvcc 13.2 + MSVC 14.50 compilam contra torch 2.13.0+cu130.
//
// O ponto NAO e a soma. E responder se um toolkit 13.2 consegue construir uma extensao para um
// torch construido com 13.0 -- a duvida que eu levantei e nao tinha testado.
//
// O kernel escreve no tensor de saida a partir de dois de entrada, o que exercita o caminho todo:
// cabecalhos do torch, ABI do MSVC, ligacao com torch_cuda, dispatch de dtype e lancamento real.
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void soma_kernel(const float* a, const float* b, float* saida, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) saida[i] = a[i] + b[i] * 2.0f;
}

torch::Tensor soma(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "os dois tensores precisam estar na GPU");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "so float32 neste teste");
    auto saida = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocos = (n + threads - 1) / threads;
    soma_kernel<<<blocos, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                     saida.data_ptr<float>(), n);
    // Erro de lancamento nao aparece sozinho: sem esta checagem um kernel que nem rodou devolveria
    // um tensor de lixo e o teste passaria comparando lixo com lixo.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "falha no lancamento: ", cudaGetErrorString(err));
    return saida;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soma", &soma, "a + 2b em CUDA");
}
