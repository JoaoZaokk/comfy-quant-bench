# R6: Como se mede dano de quantização em LLM vs Difusão

## LLM: KL Divergence é o padrão estabelecido

**Metrica**: KL Divergence contra FP16 baseline
- ikawrakow [PR #5076 llama.cpp](https://github.com/ggml-org/llama.cpp/pull/5076) estabeleceu KLD como gold standard (2024)
- Gerada via `./perplexity --kl-divergence-base` em calibração
- Unsloth Dynamic (v2.0): usa KLD com 150+ benchmarks em 121 configurações para atribuir bits **por layer**
- Resultado: modelo custom-tailored per-layer, média ~3-bit com qualidade superior

## Difusão: Timestep-wise + Sensibilidade por Camada

**Metrica**: Não é epsilon simples; três abordagens competem
- **PTQ4DM, PTQD, Q-DiT**: Quantização timestep-wise + sensitivity scoring
- **MixDQ**: [SQNR](https://arxiv.org/pdf/2405.17873) (signal-to-quantization-noise ratio) + SSIM por layer
- **DiffPro**: [Manifold-aware sensitivity](https://arxiv.org/pdf/2511.11446) para alocar bits jointly em timesteps + layers
- FID score mede dano final; MSE inter-layer vs resultado ponta-a-ponta = desconexão aberta

## Epsilon passo-a-passo: SIM, já foi feito (2024)

[**Timestep-Aware Correction** (ECCV 2024, arxiv 2407.03917)](https://arxiv.org/pdf/2407.03917): 
- Modela erro epsilon em cada timestep como N(0, λ̄_t² I)
- [**Error Propagation Mechanisms** (arxiv 2508.12094)](https://arxiv.org/pdf/2508.12094): rastreia erro cumulativo através dos passos
- **Mas**: essas medem/corrigem erro; nenhuma compara "erro-por-layer vs epsilon-por-passo como critério de escolha"

## Gap: Ninguém compara critérios de atribuição de bits

- LLM: usa KLD **global** (não por-layer) → evita correlação fraca tipo "crest factor"
- Difusão: usa **por-layer** (SQNR, Fisher, SSIM) → foi provado desconectado do resultado final aqui
- **Aberto**: qual critério prediz melhor: erro de epsilon passo-a-passo, ou KLD análogo, ou loss de treinamento?

**Fontes**:
- [llama.cpp KL divergence PR #5076](https://github.com/ggml-org/llama.cpp/pull/5076)
- [Unsloth Dynamic 2.0](https://unsloth.ai/blog/dynamic-v2)
- [MixDQ arxiv 2405.17873](https://arxiv.org/pdf/2405.17873)
- [DiffPro arxiv 2511.11446](https://arxiv.org/pdf/2511.11446)
- [Timestep-Aware Correction arxiv 2407.03917](https://arxiv.org/pdf/2407.03917)
