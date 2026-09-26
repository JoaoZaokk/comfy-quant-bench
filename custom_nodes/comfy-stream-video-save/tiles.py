"""Conta do decode em tiles por janelas temporais (sem dependencia do ComfyUI, para o teste de CPU)."""
import itertools

import torch


def _escalas(upscale_amount, index_formulas, dims):
    if not isinstance(upscale_amount, (tuple, list)):
        upscale_amount = [upscale_amount] * dims
    if index_formulas is None:
        index_formulas = upscale_amount
    if not isinstance(index_formulas, (tuple, list)):
        index_formulas = [index_formulas] * dims

    def escala(d, v):
        u = upscale_amount[d]
        return u(v) if callable(u) else u * v

    def posicao(d, v):
        u = index_formulas[d]
        return u(v) if callable(u) else u * v

    return escala, posicao


def quadros_em_tiles(samples, function, tile, overlap, upscale_amount, out_channels, output_device,
                     index_formulas=None, finalize=None):
    """Gera tensores [C, T_trecho, H, W] ja finalizados, em ordem, somando exatamente como
    `comfy.utils.tiled_scale_multidim` (upscale, dims = 3, tempo na dimensao 0 do tile).

    `finalize` recebe o trecho [1, C, t, H, W] depois da divisao pelos pesos (no lugar do `process_output` que o
    VAE aplica ao video inteiro; como e elemento a elemento, aplicar por trecho da o mesmo resultado)."""
    dims = len(tile)
    assert dims == 3 and samples.shape[0] == 1, "so video com lote 1"
    if not isinstance(overlap, (tuple, list)):
        overlap = [overlap] * dims
    escala, posicao = _escalas(upscale_amount, index_formulas, dims)
    s = samples
    saida = [round(escala(d, s.shape[d + 2])) for d in range(dims)]

    if all(s.shape[d + 2] <= tile[d] for d in range(dims)):  # cabe num tile so
        ps = function(s).to(output_device)
        yield finalize(ps) if finalize else ps
        return

    posicoes = [range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d]) if s.shape[d + 2] > tile[d] else [0]
                for d in range(dims)]
    ini = 0  # quadro absoluto do indice 0 do acumulador
    acc = torch.zeros([1, out_channels, 0] + saida[1:], device=output_device)
    div = torch.zeros([1, 1, 0] + saida[1:], device=output_device)
    pos_t = [max(0, min(s.shape[2] - overlap[0], p)) for p in posicoes[0]]

    for i, it_t in enumerate(posicoes[0]):
        for it_xy in itertools.product(*posicoes[1:]):
            it = (it_t,) + it_xy
            s_in = s
            up = []
            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                n = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, n)
                up.append(round(posicao(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones([1, 1] + list(ps.shape[2:]), device=output_device)
            for d in range(2, dims + 2):
                feather = round(escala(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            fim_t = up[0] + min(ps.shape[2], saida[0] - up[0])
            falta = fim_t - (ini + acc.shape[2])
            if falta > 0:  # o acumulador cresce com zeros, como o `out.zero_()` do original
                acc = torch.cat([acc, torch.zeros([1, out_channels, falta] + saida[1:], device=output_device)], 2)
                div = torch.cat([div, torch.zeros([1, 1, falta] + saida[1:], device=output_device)], 2)

            o, o_d, ps_v, m_v = acc, div, ps, mask
            for d in range(dims):
                base = up[d] - ini if d == 0 else up[d]
                n = min(ps_v.shape[d + 2], saida[d] - up[d])
                o = o.narrow(d + 2, base, n)
                o_d = o_d.narrow(d + 2, base, n)
                if n < ps_v.shape[d + 2]:
                    ps_v = ps_v.narrow(d + 2, 0, n)
                    m_v = m_v.narrow(d + 2, 0, n)
            o.add_(ps_v * m_v)
            o_d.add_(m_v)

        # nenhum tile das proximas posicoes temporais comeca antes de `pronto`
        pronto = round(posicao(0, pos_t[i + 1])) if i + 1 < len(pos_t) else saida[0]
        n = pronto - ini
        if n > 0:
            trecho = acc[:, :, :n].div_(div[:, :, :n])
            yield finalize(trecho) if finalize else trecho
            acc, div = acc[:, :, n:].clone(), div[:, :, n:].clone()
            ini = pronto
