"""Monta tools/bench_server.html com as quatro linguas embutidas.

    python_embeded/python.exe -s tools/build_bench_html.py

As strings em ingles moram AQUI (sao a fonte); as tres traducoes em tools/bench_i18n.json.
Nao edite bench_server.html a mao -- ele e gerado, e a proxima geracao apaga a edicao.

As tres traducoes vieram de um tradutor + um revisor nativo por lingua e estao em i18n.json; o
ingles e a fonte e mora aqui. Este script existe para que as strings NAO sejam digitadas a mao
dentro do HTML -- uma chave faltando numa lingua e um buraco visivel na tela, e o jeito de nunca
ter isso e gerar e conferir.

NAO COBERTO: nao valida a qualidade da traducao, so a completude. E nao renderiza a pagina.
"""
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ALVO = AQUI / "bench_server.html"
I18N = AQUI / "bench_i18n.json"

EN = {
    "titulo": "Quantization bench",
    "subtitulo": "Convert checkpoints, and see what has already been measured",
    "aviso_somente_leitura": "Read-only. A dry run works; a real conversion needs the owner to "
                             "restart the server with --permitir-escrita.",
    "aviso_escrita_liberada": "Writing enabled. A real conversion will use the GPU for hours and "
                              "write gigabytes.",
    "aba_converter": "Convert",
    "aba_checkpoints": "Checkpoints",
    "aba_achados": "Findings",
    "aba_como_medir": "How this bench measures",
    "gpu_livre": "GPU lock free",
    "gpu_ocupada": "GPU lock held by",
    "conv_titulo": "Run a converter",
    "conv_qual": "Which converter",
    "conv_entrada": "Input file",
    "conv_saida": "Output file (optional)",
    "conv_ensaio": "Dry run (writes nothing)",
    "conv_rodar": "Run",
    "conv_ver_flags": "Show every flag for this converter",
    "conv_linha": "Command line",
    "conv_recusado": "Refused",
    "conv_escolha_entrada": "Pick an input file",
    "trab_titulo": "Jobs",
    "trab_nenhum": "Nothing has been run yet",
    "trab_rodando": "running",
    "trab_terminou": "finished",
    "trab_falhou": "failed",
    "trab_codigo": "exit code",
    "trab_saida": "Output",
    "ckpt_titulo": "Checkpoints on disk",
    "ckpt_arquivo": "File",
    "ckpt_pasta": "Folder",
    "ckpt_tamanho": "Size",
    "ckpt_quantizado": "Quantized",
    "ckpt_camadas": "Layers",
    "ckpt_formatos": "Formats",
    "ckpt_sim": "yes",
    "ckpt_nao": "no",
    "ach_titulo": "Search what has already been measured",
    "ach_dica": "Search before you measure. Most of it has been measured once, and the caveat "
                "matters as much as the number.",
    "ach_campo": "e.g. text encoder lock, int8 branch, reserve-vram",
    "ach_buscar": "Search",
    "ach_vazio": "Nothing found",
    "ach_documento": "Document",
    "ach_linha": "Line",
    "erro_requisicao": "The request failed",
    "carregando": "Loading",
    "rodape": "This page runs the same command line you would type in a terminal. It verifies "
              "nothing on its own: a clean conversion proves the kernel dispatches, never that "
              "the output is good. Only a render answers that.",
}

CSS = """
:root{
  --tinta:#e8e6e3; --fraca:#9a978f; --fundo:#141416; --painel:#1b1b1f; --borda:#2c2c33;
  --acento:#c9a227; --bom:#7bc47f; --ruim:#e06c6c; --mono:ui-monospace,"Cascadia Mono",Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--fundo);color:var(--tinta);
     font:15px/1.55 ui-sans-serif,"Segoe UI",system-ui,sans-serif}
a{color:var(--acento)}
header{padding:22px 26px 14px;border-bottom:1px solid var(--borda);
       display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{margin:0;font-size:20px;font-weight:600;letter-spacing:-.01em}
.sub{color:var(--fraca);font-size:14px}
.linguas{margin-left:auto;display:flex;gap:4px}
.linguas button{background:none;border:1px solid var(--borda);color:var(--fraca);
                border-radius:5px;padding:3px 9px;cursor:pointer;font-size:12px}
.linguas button[aria-pressed=true]{color:var(--fundo);background:var(--acento);border-color:var(--acento)}
.faixa{padding:9px 26px;font-size:13px;border-bottom:1px solid var(--borda)}
.faixa.leitura{background:#20201a;color:#d8c98a}
.faixa.escrita{background:#2a1a1a;color:#e8a0a0}
nav{display:flex;gap:2px;padding:0 20px;border-bottom:1px solid var(--borda);flex-wrap:wrap}
nav button{background:none;border:0;border-bottom:2px solid transparent;color:var(--fraca);
           padding:11px 14px;cursor:pointer;font-size:14px}
nav button[aria-selected=true]{color:var(--tinta);border-bottom-color:var(--acento)}
main{padding:22px 26px;max-width:1100px}
section[hidden]{display:none}
h2{font-size:16px;font-weight:600;margin:0 0 4px}
.dica{color:var(--fraca);font-size:13px;margin:0 0 16px;max-width:66ch}
label{display:block;font-size:13px;color:var(--fraca);margin:12px 0 4px}
input[type=text],select{width:100%;max-width:660px;background:var(--painel);color:var(--tinta);
  border:1px solid var(--borda);border-radius:6px;padding:8px 10px;font-size:14px}
input[type=text]{font-family:var(--mono);font-size:13px}
.caixa{display:flex;gap:8px;align-items:center;margin:14px 0}
.caixa label{margin:0}
button.acao{background:var(--acento);color:#1a1a1a;border:0;border-radius:6px;
            padding:9px 20px;font-size:14px;font-weight:600;cursor:pointer}
button.acao:disabled{opacity:.45;cursor:not-allowed}
button.leve{background:none;border:1px solid var(--borda);color:var(--fraca);
            border-radius:6px;padding:7px 13px;cursor:pointer;font-size:13px}
pre{background:var(--painel);border:1px solid var(--borda);border-radius:6px;padding:12px;
    overflow-x:auto;font-family:var(--mono);font-size:12.5px;line-height:1.5;margin:10px 0;
    white-space:pre-wrap;word-break:break-word}
pre.log{max-height:420px;overflow-y:auto;white-space:pre}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--borda);vertical-align:top}
th{color:var(--fraca);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
.rolagem{overflow-x:auto}
.pilula{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11.5px;
        border:1px solid var(--borda);color:var(--fraca)}
.pilula.sim{color:var(--bom);border-color:#2f4a31}
.pilula.rodando{color:var(--acento);border-color:#4a3f1a}
.pilula.falhou{color:var(--ruim);border-color:#4a2626}
.erro{color:var(--ruim)}
.achado{border-left:2px solid var(--borda);padding:2px 0 2px 12px;margin:10px 0}
.achado .onde{color:var(--fraca);font-size:12px;font-family:var(--mono)}
.achado .txt{font-size:13.5px}
footer{padding:20px 26px 40px;color:var(--fraca);font-size:12.5px;max-width:78ch;
       border-top:1px solid var(--borda);margin-top:30px}
@media (prefers-color-scheme:light){
 :root{--tinta:#1b1b1e;--fraca:#6b6862;--fundo:#faf9f7;--painel:#f0eeea;--borda:#dcd8d1;
       --acento:#8a6d0f;--bom:#2f7a36;--ruim:#a83232}
 .faixa.leitura{background:#fdf6dd;color:#6b5510}
 .faixa.escrita{background:#fbe6e6;color:#8f2a2a}
 button.acao{color:#fff}
}
"""

JS = r"""
const I18N = __I18N__;
const ORDEM = ["en","pt","es","zh"];
const NOMES = {en:"EN", pt:"PT", es:"ES", zh:"\u4e2d\u6587"};
let L = localStorage.getItem("bancada_lingua") || (navigator.language||"en").slice(0,2);
if (!I18N[L]) L = "en";
const t = k => (I18N[L] && I18N[L][k]) || I18N.en[k] || k;
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

async function pede(nome, args, metodo){
  try{
    const r = metodo === "POST"
      ? await fetch("/api/"+nome, {method:"POST", headers:{"Content-Type":"application/json"},
                                   body: JSON.stringify(args||{})})
      : await fetch("/api/"+nome+"?"+new URLSearchParams(args||{}));
    const j = await r.json();
    if (!j.ok) throw new Error(j.erro || t("erro_requisicao"));
    return j.valor;
  }catch(e){ throw new Error(e.message || t("erro_requisicao")); }
}

function pinta(){
  document.documentElement.lang = L === "zh" ? "zh-Hans" : L;
  document.title = t("titulo");
  document.querySelectorAll("[data-t]").forEach(e => e.textContent = t(e.dataset.t));
  document.querySelectorAll("[data-tp]").forEach(e => e.placeholder = t(e.dataset.tp));
  $("#linguas").innerHTML = ORDEM.map(c =>
    `<button aria-pressed="${c===L}" onclick="trocaLingua('${c}')">${NOMES[c]}</button>`).join("");
  // Tudo que o JS escreveu tem de ser reescrito: a primeira versao disto so redesenhava o que
  // tinha sido carregado uma vez, entao "GPU lock free" e o estado vazio de tarefas ficavam em
  // ingles depois de trocar a lingua. O que o data-t alcanca e so o HTML estatico.
  if (window.__ultimosCkpt) desenhaCkpt(window.__ultimosCkpt);
  desenhaTrabalhos();
  pintaGpu();
  pintaFaixa();
}
function trocaLingua(c){ L = c; localStorage.setItem("bancada_lingua", c); pinta(); }

function aba(nome){
  document.querySelectorAll("nav button").forEach(b =>
    b.setAttribute("aria-selected", b.dataset.aba === nome));
  document.querySelectorAll("main section").forEach(s => s.hidden = s.id !== "aba-"+nome);
  if (nome === "checkpoints" && !window.__ultimosCkpt) carregaCkpt();
  if (nome === "medir" && !$("#texto-medir").textContent.trim()) carregaMedir();
}

function pintaGpu(){
  const g = window.__gpu;
  if (!g){ $("#gpu").innerHTML = ""; return; }
  $("#gpu").innerHTML = g.erro ? `<span class="erro">${esc(g.erro)}</span>`
    : g.livre ? `<span class="pilula sim">${esc(t("gpu_livre"))}</span>`
              : `<span class="pilula falhou">${esc(t("gpu_ocupada"))} ${esc(g.dono)}</span>`;
}
async function carregaGpu(){
  try{ window.__gpu = await pede("estado_da_gpu"); }
  catch(e){ window.__gpu = {erro: e.message}; }
  pintaGpu();
}
function pintaFaixa(){
  if (window.__escrita === undefined) return;
  const f = $("#faixa");
  f.className = "faixa " + (window.__escrita ? "escrita" : "leitura");
  f.textContent = t(window.__escrita ? "aviso_escrita_liberada" : "aviso_somente_leitura");
}

async function carregaConversores(){
  const cs = await pede("listar_conversores");
  window.__conv = cs;
  $("#subcomando").innerHTML = cs.map(c =>
    `<option value="${esc(c.subcomando)}">${esc(c.subcomando)}</option>`).join("");
  mostraDescricao();
}
function mostraDescricao(){
  // A descricao vem do proprio conversor e NAO e traduzida -- fica marcada como texto da
  // ferramenta em vez de fingir que faz parte da interface.
  const c = (window.__conv||[]).find(x => x.subcomando === $("#subcomando").value);
  $("#descricao").textContent = c ? `${c.modulo} · ${c.o_que_faz}` : "";
}

async function verFlags(){
  const s = $("#subcomando").value;
  $("#flags").textContent = t("carregando");
  try{ $("#flags").textContent = await pede("flags_do_conversor", {subcomando:s}); }
  catch(e){ $("#flags").textContent = e.message; }
}

async function roda(){
  const entrada = $("#entrada").value.trim();
  if (!entrada){ $("#resultado").innerHTML = `<span class="erro">${esc(t("conv_escolha_entrada"))}</span>`; return; }
  const flags = {"--input": entrada};
  if ($("#saida").value.trim()) flags["--output"] = $("#saida").value.trim();
  if ($("#ensaio").checked) flags["--dry-run"] = true;
  $("#rodar").disabled = true;
  try{
    const v = await pede("converter", {subcomando:$("#subcomando").value, flags}, "POST");
    if (v.recusado){
      $("#resultado").innerHTML =
        `<p class="erro"><b>${esc(t("conv_recusado"))}</b> \u2014 ${esc(v.porque)}</p>`+
        `<p class="dica">${esc(t("conv_linha"))}</p><pre>${esc(v.linha_de_comando)}</pre>`;
    } else {
      $("#resultado").innerHTML =
        `<p class="dica">${esc(t("conv_linha"))}</p><pre>${esc(v.linha_de_comando)}</pre>`;
      acompanha(v.id);
    }
  }catch(e){ $("#resultado").innerHTML = `<p class="erro">${esc(e.message)}</p>`; }
  finally{ $("#rodar").disabled = false; }
}

const TRAB = {};
async function acompanha(id){
  const passo = async () => {
    try{
      const j = await pede("estado_do_trabalho", {id, linhas: 300});
      TRAB[id] = j; window.__ultimosTrab = 1; desenhaTrabalhos();
      if (j.estado === "rodando") setTimeout(passo, 1200);
    }catch(e){ /* servidor caiu; para de pedir */ }
  };
  passo();
}
function desenhaTrabalhos(){
  const ids = Object.keys(TRAB);
  if (!ids.length){ $("#trabalhos").innerHTML = `<p class="dica">${esc(t("trab_nenhum"))}</p>`; return; }
  $("#trabalhos").innerHTML = ids.reverse().map(id => {
    const j = TRAB[id];
    const cls = j.estado === "rodando" ? "rodando" : j.estado === "falhou" ? "falhou" : "sim";
    const rot = j.estado === "rodando" ? t("trab_rodando")
              : j.estado === "falhou" ? t("trab_falhou") : t("trab_terminou");
    const cod = j.codigo === null ? "" : ` \u00b7 ${esc(t("trab_codigo"))} ${j.codigo}`;
    return `<div class="achado"><div class="onde">${esc(j.subcomando)} \u00b7 ${esc(id)}${cod}
      <span class="pilula ${cls}">${esc(rot)}</span></div>
      <pre class="log">${esc((j.saida||[]).join("\n"))}</pre></div>`;
  }).join("");
}

async function carregaCkpt(){
  $("#tabela-ckpt").innerHTML = `<p class="dica">${esc(t("carregando"))}</p>`;
  try{ window.__ultimosCkpt = await pede("listar_checkpoints"); desenhaCkpt(window.__ultimosCkpt); }
  catch(e){ $("#tabela-ckpt").innerHTML = `<p class="erro">${esc(e.message)}</p>`; }
}
function desenhaCkpt(cs){
  $("#tabela-ckpt").innerHTML = `<div class="rolagem"><table><thead><tr>
    <th>${esc(t("ckpt_arquivo"))}</th><th>${esc(t("ckpt_pasta"))}</th>
    <th style="text-align:right">${esc(t("ckpt_tamanho"))}</th>
    <th>${esc(t("ckpt_quantizado"))}</th>
    <th style="text-align:right">${esc(t("ckpt_camadas"))}</th>
    <th>${esc(t("ckpt_formatos"))}</th></tr></thead><tbody>` +
    cs.map(c => `<tr><td style="font-family:var(--mono);font-size:12px">${esc(c.arquivo)}</td>
      <td>${esc(c.pasta)}</td><td class="num">${c.gib.toFixed(2)}</td>
      <td>${c.quantizado ? `<span class="pilula sim">${esc(t("ckpt_sim"))}</span>`
                         : `<span class="pilula">${esc(t("ckpt_nao"))}</span>`}</td>
      <td class="num">${c.camadas_quantizadas || ""}</td>
      <td style="font-size:12px">${esc(Object.entries(c.formatos||{})
          .map(([k,v]) => `${k} \u00d7${v}`).join(", "))}</td></tr>`).join("") +
    "</tbody></table></div>";
}

async function busca(){
  const q = $("#consulta").value.trim();
  if (!q) return;
  $("#achados").innerHTML = `<p class="dica">${esc(t("carregando"))}</p>`;
  try{
    const r = await pede("buscar_achados", {consulta:q, limite:14});
    $("#achados").innerHTML = r.length ? r.map(a =>
      `<div class="achado"><div class="onde">${esc(a.documento)} \u00b7 ${esc(t("ach_linha"))} ${a.linha}</div>
       <div class="txt">${esc(a.texto)}</div></div>`).join("")
      : `<p class="dica">${esc(t("ach_vazio"))}</p>`;
  }catch(e){ $("#achados").innerHTML = `<p class="erro">${esc(e.message)}</p>`; }
}

async function carregaMedir(){
  try{ $("#texto-medir").textContent = await pede("como_medir"); }
  catch(e){ $("#texto-medir").textContent = e.message; }
}

async function carregaServidor(){
  try{
    const s = await pede("estado_do_servidor");
    window.__escrita = !!s.permitir_escrita;
    pintaFaixa();
  }catch(e){ $("#faixa").textContent = e.message; }
}

pinta(); aba("converter"); carregaServidor(); carregaGpu(); carregaConversores(); desenhaTrabalhos();
setInterval(carregaGpu, 15000);
$("#consulta").addEventListener("keydown", e => { if (e.key === "Enter") busca(); });
"""

CORPO = """<header>
  <h1 data-t="titulo"></h1>
  <span class="sub" data-t="subtitulo"></span>
  <span id="gpu"></span>
  <span class="linguas" id="linguas"></span>
</header>
<div class="faixa leitura" id="faixa"></div>
<nav>
  <button data-aba="converter"   data-t="aba_converter"   onclick="aba('converter')"></button>
  <button data-aba="checkpoints" data-t="aba_checkpoints" onclick="aba('checkpoints')"></button>
  <button data-aba="achados"     data-t="aba_achados"     onclick="aba('achados')"></button>
  <button data-aba="medir"       data-t="aba_como_medir"  onclick="aba('medir')"></button>
</nav>
<main>
  <section id="aba-converter">
    <h2 data-t="conv_titulo"></h2>
    <label data-t="conv_qual"></label>
    <select id="subcomando" onchange="mostraDescricao()"></select>
    <p class="dica" id="descricao" style="margin:6px 0 0;font-family:var(--mono);font-size:12px"></p>
    <label data-t="conv_entrada"></label>
    <input type="text" id="entrada" data-tp="conv_escolha_entrada">
    <label data-t="conv_saida"></label>
    <input type="text" id="saida">
    <div class="caixa">
      <input type="checkbox" id="ensaio" checked>
      <label for="ensaio" data-t="conv_ensaio"></label>
    </div>
    <div class="caixa">
      <button class="acao" id="rodar" data-t="conv_rodar" onclick="roda()"></button>
      <button class="leve" data-t="conv_ver_flags" onclick="verFlags()"></button>
    </div>
    <pre id="flags" style="display:block"></pre>
    <div id="resultado"></div>
    <h2 data-t="trab_titulo" style="margin-top:26px"></h2>
    <div id="trabalhos"></div>
  </section>

  <section id="aba-checkpoints" hidden>
    <h2 data-t="ckpt_titulo"></h2>
    <div id="tabela-ckpt"></div>
  </section>

  <section id="aba-achados" hidden>
    <h2 data-t="ach_titulo"></h2>
    <p class="dica" data-t="ach_dica"></p>
    <input type="text" id="consulta" data-tp="ach_campo">
    <div class="caixa"><button class="acao" data-t="ach_buscar" onclick="busca()"></button></div>
    <div id="achados"></div>
  </section>

  <section id="aba-medir" hidden>
    <h2 data-t="aba_como_medir"></h2>
    <pre id="texto-medir"></pre>
  </section>
</main>
<footer data-t="rodape"></footer>
"""


def main() -> int:
    trad = json.loads((I18N).read_text(encoding="utf-8"))
    i18n = {"en": EN, **trad}

    faltando = {c: sorted(set(EN) - set(s)) for c, s in i18n.items() if set(EN) - set(s)}
    sobrando = {c: sorted(set(s) - set(EN)) for c, s in i18n.items() if set(s) - set(EN)}
    if faltando or sobrando:
        raise SystemExit(f"chaves fora de sincronia\nfaltando: {faltando}\nsobrando: {sobrando}")

    html = ("<!-- gerado por scratchpad/monta_html.py -- nao edite a mao, edite o gerador -->\n"
            f"<style>{CSS}</style>\n{CORPO}\n"
            f"<script>{JS.replace('__I18N__', json.dumps(i18n, ensure_ascii=False))}</script>\n")
    ALVO.write_text(html, encoding="utf-8", newline="\n")

    print(f"{ALVO}  {len(html) / 1024:.1f} KiB")
    for c, s in i18n.items():
        print(f"  {c}: {len(s)} chaves")
    print("NAO COBERTO: confere completude das chaves, nunca a qualidade da traducao. E nao "
          "renderiza a pagina -- um erro de JS so aparece no navegador.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
