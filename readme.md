# Unsupervised Cosegmentation Pipeline

Este repositório contém um protótipo de **cosegmentação não supervisionada** baseado em extração de features com **DINOv2**, correspondência entre imagens, **co-saliência**, refinamento com **SAM 2.1** e refinamento iterativo.

> Este README assume que o pipeline antigo (`level1_dino_kmeans.py`, `level2_coseg.py`, `level3_sam_refine.py`, `level4_refine.py`, `benchmark.py`, `run_eval_only.sh`) foi removido e que o script principal do projeto agora é **`run_pipeline.sh`**. Pastas e arquivos órfãos de execuções antigas (`resultados_pipeline/`, `pastas.txt`) também foram removidos — ver [2.1](#21-itens-removidos-do-repositório).

---

## Sumário

- [1. Visão geral](#1-visão-geral)
- [2. Estrutura do projeto](#2-estrutura-do-projeto)
  - [2.1. Itens removidos do repositório](#21-itens-removidos-do-repositório)
- [3. Requisitos](#3-requisitos)
- [4. Dataset esperado](#4-dataset-esperado)
- [5. Scripts principais](#5-scripts-principais)
- [6. Pipeline principal (`run_pipeline.sh`)](#6-pipeline-principal-run_pipelinesh)
- [7. Modos de execução: subset vs dataset completo](#7-modos-de-execução-subset-vs-dataset-completo)
- [8. Etapas do pipeline](#8-etapas-do-pipeline)
- [9. Avaliação](#9-avaliação)
- [10. Checkpoints, retomada e estado do pipeline](#10-checkpoints-retomada-e-estado-do-pipeline)
- [11. Execução periódica e SSH](#11-execução-periódica-e-ssh)
- [12. Scripts de análise/auditoria](#12-scripts-de-análiseauditoria)
- [13. Scripts de sweep / ablação](#13-scripts-de-sweep--ablação)
- [14. Fluxos recomendados de uso](#14-fluxos-recomendados-de-uso)
- [15. Troubleshooting](#15-troubleshooting)

---

## 1. Visão geral

O pipeline segue, em alto nível, as etapas abaixo:

1. **Extração de features** com DINOv2.
2. **Correspondência entre imagens** para encontrar regiões consistentes entre exemplos da mesma coleção.
3. **Co-saliência** para estimar regiões de foreground compartilhadas.
4. **Refinamento com SAM 2.1** para obter máscaras mais precisas.
5. **Refinamento iterativo** para melhorar consistência e estabilidade.
6. **Avaliação** contra ground truth, quando disponível.

O projeto foi pensado principalmente para experimentos com o **iCoseg**, mas pode ser adaptado para outras coleções organizadas de forma compatível.

---

## 2. Estrutura do projeto

Estrutura resumida esperada:

```text
cosegmentation/
├── analyze_step1_features.py
├── analyze_step2_correspondence.py
├── analyze_step3_cosaliency.py
├── analyze_step4_sam_refinement.py
├── evaluate.py
├── gt_index.json
├── icoseg_index.json
├── run_pipeline.sh                      # SCRIPT PRINCIPAL
├── run_step1_feature_sweep.sh
├── run_step2_correspondence_sweep.sh
├── run_step3_cosaliency_sweep.sh
├── run_step4_sam_sweep.sh
├── step1_feature_extraction.py
├── step2_correspondence.py
├── step3_cosaliency.py
├── step4_sam_refinement.py
├── step5_iterative.py
├── icoseg/
│   ├── images/
│   └── ground_truth/
├── features/
├── results/
└── logs/
```

Papel de cada pasta raiz:

| Pasta | Papel | Referenciada em |
|---|---|---|
| `icoseg/` | Dataset bruto (input). Nunca é escrita pelo pipeline. | `ICOSEG_ROOT`, `GT_ROOT` |
| `features/` | Saída das Etapas 1-5 (`.npz` por classe + `.state/`). | `FEATURES_ROOT` (default `"features"`) |
| `results/` | Saída da avaliação (`metrics.json`, `resumo_agregado.*`). | `RESULTS_ROOT` (default `"results"`) |
| `logs/` | Logs de execução por classe/etapa. | `LOG_ROOT` (default `"logs"`) |

Essas quatro pastas são as únicas que o `run_pipeline.sh` cria e espera encontrar. Qualquer outra pasta na raiz do projeto (nomes em português, sufixos de versões antigas, etc.) provavelmente é resíduo de execuções anteriores e pode ser conferida contra este README antes de ser removida.

### 2.1. Itens removidos do repositório

Durante uma limpeza do projeto, os seguintes itens foram removidos por serem de uma geração anterior do pipeline (baseada em GrabCut/Graph Cuts, sem checkpoint, sem integração com `run_pipeline.sh`) ou por serem saídas/anotações órfãs:

**Scripts (geração antiga "level1-4"):**
```text
level1_dino_kmeans.py     # DINOv2 + K-Means, sem correspondência cross-image
level2_coseg.py           # PCA conjunto + K-Means K=2 (protótipo das Etapas 1-3 atuais)
level3_sam_refine.py      # refinamento SAM com points_prompt (substituído por step4)
level4_refine.py          # GrabCut/Graph Cuts com perfil global (substituído por step4+step5)
benchmark.py              # benchmark de variações do level2_coseg.py
run_eval_only.sh          # workaround para um bug de run_pipeline.sh já corrigido
```

**Pastas e arquivos órfãos:**
```text
resultados_pipeline/      # saída de uma versão anterior a RESULTS_ROOT="results"
pastas.txt                # anotação/rascunho manual, não referenciado por nenhum script
```

Se precisar recuperar algo dessa geração antiga (por exemplo, para citar em uma seção de "abordagens exploradas" de um paper), recorra ao histórico do Git em vez de mantê-los no diretório de trabalho.

### Diretórios gerados pelo pipeline

Depois da execução, espera-se algo como:

```text
features/
├── icoseg16/
│   ├── Classe_A/
│   ├── Classe_B/
│   └── .state/
└── icoseg_full/
    ├── Classe_A/
    ├── Classe_B/
    └── .state/

results/
├── icoseg16/
│   ├── Classe_A/
│   └── Classe_B/
└── icoseg_full/
    ├── Classe_A/
    └── Classe_B/

logs/
├── icoseg16/
└── icoseg_full/
```

Os diretórios de resultados são separados por set:

```text
features/icoseg16/       # subset atual
features/icoseg_full/    # dataset completo
results/icoseg16/
results/icoseg_full/
logs/icoseg16/
logs/icoseg_full/
```

---

## 3. Requisitos

### 3.1. Sistema

Recomendado:

- Linux ou WSL2
- Python 3.9+
- GPU compatível com PyTorch/CUDA, fortemente recomendada para DINOv2 e SAM

### 3.2. Bibliotecas Python

Você provavelmente precisará de bibliotecas como:

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `scikit-learn`
- `matplotlib`
- `opencv-python`
- `scipy`
- dependências do SAM 2.1 usadas no projeto

Se ainda não houver um `requirements.txt`, vale a pena criá-lo depois.

---

## 4. Dataset esperado

O projeto assume a existência de uma pasta `icoseg/` com esta organização:

```text
icoseg/
├── images/
│   ├── Classe_1/
│   ├── Classe_2/
│   └── ...
└── ground_truth/
    ├── Classe_1/
    ├── Classe_2/
    └── ...
```

Além disso, dois índices podem ser usados:

- `icoseg_index.json`: índice das coleções/classes e suas imagens.
- `gt_index.json`: índice de mapeamento entre imagens e respectivas máscaras de ground truth.

### 4.1. Conferindo as classes disponíveis

Antes de tudo, rode:

```bash
python step1_feature_extraction.py --list
```

Esse comando é importante para confirmar os **nomes exatos** das classes disponíveis no índice.

### 4.2. Se o índice do iCoseg não existir

Se `icoseg_index.json` ainda não existir, o script principal tenta gerar o índice automaticamente via varredura do diretório de imagens.

---

## 5. Scripts principais

### Núcleo do pipeline

- `step1_feature_extraction.py` — extrai features DINOv2.
- `step2_correspondence.py` — computa correspondência cross-image.
- `step3_cosaliency.py` — estima mapas de co-saliência.
- `step4_sam_refinement.py` — refina máscaras com SAM 2.1.
- `step5_iterative.py` — refinamento iterativo adicional.
- `evaluate.py` — avaliação contra ground truth.

### Script principal de orquestração

- `run_pipeline.sh` — executa o pipeline completo ou parcial, com validação de saídas, checkpoints e retomada.

### Análise / auditoria

- `analyze_step1_features.py`
- `analyze_step2_correspondence.py`
- `analyze_step3_cosaliency.py`
- `analyze_step4_sam_refinement.py`

### Sweeps / ablações

- `run_step1_feature_sweep.sh`
- `run_step2_correspondence_sweep.sh`
- `run_step3_cosaliency_sweep.sh`
- `run_step4_sam_sweep.sh`

---

## 6. Pipeline principal (`run_pipeline.sh`)

Este é o script principal do projeto.

### 6.1. Ideia geral

Ele permite:

- rodar o **subset atual do iCoseg**;
- rodar o **dataset completo**;
- rodar **classes específicas** manualmente;
- controlar quais etapas executar;
- retomar o processo a partir de uma etapa intermediária;
- validar se saídas `.npz` e `.json` estão íntegras antes de pular etapas;
- registrar estado por classe/etapa em `.running`, `.done` e `.failed`;
- impedir execuções simultâneas no mesmo set por meio de lock automático;
- pular SAM;
- exportar segmentações;
- fazer `dry-run`;
- avaliar os resultados.

### 6.2. Uso básico

```bash
bash run_pipeline.sh --set icoseg16 --export
bash run_pipeline.sh --set full --export
```

### 6.3. Ajuda

```bash
bash run_pipeline.sh --help
```

### 6.4. Opções principais

```text
--set NOME          Set experimental: icoseg16 ou full
--steps N,M,...     Etapas a executar. Ex.: 1,2,3 ou 4,5,eval
--from-step N       Executa da etapa N até o final
--force             Reprocessa mesmo se saída válida já existir
--dry-run           Mostra os comandos sem executar
--visualize         Ativa visualizações nas etapas que suportam isso
--export            Exporta máscaras PNG em E4/E5
--no-sam            Pula etapas 4 e 5
--no-eval           Pula avaliação contra ground truth
--no-lock           Desativa o lock automático
--max-images N      Limita o número de imagens por coleção
```

---

## 7. Modos de execução: subset vs dataset completo

O script suporta dois modos principais.

### `--set icoseg16`

Usa o subconjunto selecionado de classes definido no próprio script:

```bash
bash run_pipeline.sh --set icoseg16 --export
```

Esse modo é recomendado para:

- debug;
- testes rápidos;
- desenvolvimento;
- ablações iniciais.

As saídas ficam em:

```text
features/icoseg16/
results/icoseg16/
logs/icoseg16/
```

### `--set full`

Usa todas as classes encontradas em `icoseg_index.json`:

```bash
bash run_pipeline.sh --set full --export
```

Esse modo é recomendado para:

- benchmark completo;
- geração de resultados finais;
- comparação global.

As saídas ficam em:

```text
features/icoseg_full/
results/icoseg_full/
logs/icoseg_full/
```

### Compatibilidade com aliases antigos

Também funcionam aliases como:

```bash
bash run_pipeline.sh selected
bash run_pipeline.sh all
```

Mas a forma recomendada é usar explicitamente:

```bash
--set icoseg16
--set full
```

### Classes específicas

Também é possível ignorar o set e passar classes manualmente:

```bash
bash run_pipeline.sh Panda Kite Ferrari
```

ou nomes exatos do índice:

```bash
bash run_pipeline.sh "Alaskan Brown Bear" "Stonehenge"
```

> Use sempre os nomes retornados por `python step1_feature_extraction.py --list`.

---

## 8. Etapas do pipeline

### Etapa 1 — `step1_feature_extraction.py`

Extrai features DINOv2 para todas as imagens de uma coleção.

Saída típica:

```text
features/<set>/<classe>/features.npz
```

### Etapa 2 — `step2_correspondence.py`

Usa as features para estimar correspondência entre imagens.

Saída típica:

```text
features/<set>/<classe>/correspondence.npz
```

### Etapa 3 — `step3_cosaliency.py`

Computa co-saliência/foreground compartilhado.

Saída típica:

```text
features/<set>/<classe>/cosaliency.npz
```

### Etapa 4 — `step4_sam_refinement.py`

Refina as máscaras usando SAM 2.1.

Saídas típicas:

```text
features/<set>/<classe>/refined_masks.npz
features/<set>/<classe>/segmented_e4/       # quando --export é usado
```

### Etapa 5 — `step5_iterative.py`

Faz refinamento iterativo adicional, normalmente partindo da saída refinada do SAM.

Saídas típicas:

```text
features/<set>/<classe>/iterative_masks.npz
features/<set>/<classe>/segmented_e5/       # quando --export é usado
```

### Avaliação — `evaluate.py`

Compara as máscaras produzidas com o ground truth.

Saídas típicas:

```text
results/<set>/<classe>/metrics.json
results/<set>/<classe>/metrics_summary.txt
```

A avaliação procura máscaras nesta ordem:

```text
segmented_e5/    # preferencial
segmented_e4/    # fallback
segmented/       # compatibilidade com versões antigas
```

---

## 9. Avaliação

### 9.1. Gerando `gt_index.json`

Se `gt_index.json` não existir, o pipeline tenta gerá-lo automaticamente quando a avaliação estiver habilitada. Se precisar fazer manualmente:

```bash
python evaluate.py --scan_gt icoseg/ground_truth --images_root icoseg/images
```

### 9.2. Avaliando ao final do pipeline

Para que a avaliação tenha máscaras PNG prontas para comparação, use `--export`:

```bash
bash run_pipeline.sh --set icoseg16 --export
bash run_pipeline.sh --set full --export
```

### 9.3. Pular avaliação

```bash
bash run_pipeline.sh --set icoseg16 --no-eval
```

### 9.4. Onde encontrar os resultados numéricos

Subset atual:

```text
results/icoseg16/<classe>/metrics.json
results/icoseg16/<classe>/metrics_summary.txt
```

Dataset completo:

```text
results/icoseg_full/<classe>/metrics.json
results/icoseg_full/<classe>/metrics_summary.txt
```

Para localizar todos os resultados:

```bash
find results -name "metrics.json"
find results -name "metrics_summary.txt"
```

---

## 10. Checkpoints, retomada e estado do pipeline

A versão atual do `run_pipeline.sh` implementa uma retomada mais segura.

Antes, o pipeline pulava uma etapa apenas porque o arquivo de saída existia. Agora ele só pula quando a saída existe **e passa em validação básica**.

### 10.1. O que é validado

- `.npz`: precisa existir, não estar vazio e ser carregável pelo `numpy.load`.
- `metrics.json`: precisa existir, não estar vazio e ser um JSON válido.
- E4 com `--export`: além de `refined_masks.npz`, precisa existir `segmented_e4/` com arquivos `*_mask.png`.
- E5 com `--export`: além de `iterative_masks.npz`, precisa existir `segmented_e5/` com arquivos `*_mask.png`.

### 10.2. Arquivos de estado

O pipeline registra o estado por classe e por etapa em:

```text
features/<set>/.state/<classe>/
```

Exemplo:

```text
features/icoseg_full/.state/Panda/E1.done
features/icoseg_full/.state/Panda/E2.done
features/icoseg_full/.state/Panda/E3.done
features/icoseg_full/.state/Panda/E4.done
features/icoseg_full/.state/Panda/E5.running
features/icoseg_full/.state/Panda/EVAL.failed
```

Estados possíveis:

```text
.running   # etapa em execução
.done      # etapa concluída ou saída válida reaproveitada
.failed    # etapa falhou ou avaliação ficou incompleta
```

### 10.3. Como retomar após interrupção

Rode o mesmo comando novamente, sem `--force`:

```bash
bash run_pipeline.sh --set full --export
```

O script vai:

1. validar as saídas já existentes;
2. pular as etapas válidas;
3. refazer as etapas ausentes, incompletas ou corrompidas;
4. limpar `.running` antigo quando a etapa for concluída ou marcada como falha.

### 10.4. Quando usar `--force`

Use `--force` somente quando quiser reprocessar mesmo que a saída atual esteja válida:

```bash
bash run_pipeline.sh --set icoseg16 --export --force
```

Para retomar execução interrompida, normalmente **não use `--force`**.

### 10.5. Como descobrir onde parou

Para ver etapas em execução ou deixadas como interrompidas:

```bash
find features/icoseg_full/.state -name "*.running"
```

Para ver falhas:

```bash
find features/icoseg_full/.state -name "*.failed"
```

Para ver etapas concluídas:

```bash
find features/icoseg_full/.state -name "*.done"
```

---

## 11. Execução periódica e SSH

### 11.1. Rodando periodicamente

A forma recomendada é rodar sempre o mesmo comando, sem `--force`:

```bash
bash run_pipeline.sh --set full --export
```

Como o script valida checkpoints, ele deve continuar do ponto em que parou.

### 11.2. Lock automático

O script usa lock automático por set em:

```text
features/<set>/.run_pipeline.lock
```

Isso evita duas execuções simultâneas do mesmo set, por exemplo duas execuções de `--set full` ao mesmo tempo.

Se precisar desativar explicitamente:

```bash
bash run_pipeline.sh --set full --export --no-lock
```

Use `--no-lock` com cuidado.

### 11.3. Rodando via `cron`

Exemplo para tentar retomar a cada 30 minutos:

```cron
*/30 * * * * cd /caminho/para/cosegmentation && bash run_pipeline.sh --set full --export >> logs/cron_pipeline_full.log 2>&1
```

Como o script já tem lock interno, uma nova execução não deve iniciar se outra ainda estiver rodando para o mesmo set.

### 11.4. Rodando via SSH

Para execuções longas, prefira `tmux` ou `screen`.

Exemplo com `tmux`:

```bash
tmux new -s coseg
bash run_pipeline.sh --set full --export
```

Para desconectar do `tmux`, use `Ctrl+B`, depois `D`.

Para voltar:

```bash
tmux attach -t coseg
```

---

## 12. Scripts de análise/auditoria

Esses scripts ajudam a diagnosticar gargalos do pipeline por etapa.

### 12.1. Auditoria da Etapa 1

```bash
python analyze_step1_features.py \
    --features features/icoseg16/Panda/features.npz \
    --gt_index gt_index.json \
    --output features/icoseg16/Panda/feature_quality
```

Esse script responde perguntas como:

- as features globais estão coesas?
- as features locais geram correspondências consistentes?
- há outliers?
- com GT, as features separam foreground/background?
- existe sinal suficiente antes das etapas seguintes?

### 12.2. Auditoria da Etapa 2

```bash
python analyze_step2_correspondence.py \
    --correspondence features/icoseg16/Panda/correspondence.npz \
    --gt_index gt_index.json \
    --output features/icoseg16/Panda/step2_quality
```

Perguntas típicas:

- `fg_scores` estão polarizados ou achatados?
- os pares concordam entre si?
- o resultado depende demais de poucos pares?
- com GT, qual threshold funciona melhor?

### 12.3. Auditoria da Etapa 3

```bash
python analyze_step3_cosaliency.py \
    --cosaliency features/icoseg16/Panda/cosaliency.npz \
    --gt_index gt_index.json \
    --output features/icoseg16/Panda/step3_quality
```

### 12.4. Auditoria da Etapa 4

```bash
python analyze_step4_sam_refinement.py \
    --refined features/icoseg16/Panda/refined_masks.npz \
    --gt_index gt_index.json \
    --output features/icoseg16/Panda/step4_quality
```

---

## 13. Scripts de sweep / ablação

O projeto inclui scripts para exploração de hiperparâmetros.

### Sweep da Etapa 1

```bash
bash run_step1_feature_sweep.sh
```

### Sweep da Etapa 2

```bash
bash run_step2_correspondence_sweep.sh
```

### Sweep da Etapa 3

```bash
bash run_step3_cosaliency_sweep.sh
```

### Sweep da Etapa 4

```bash
bash run_step4_sam_sweep.sh
```

---

## 14. Fluxos recomendados de uso

### Fluxo A — verificação inicial do dataset

1. Conferir classes disponíveis:

```bash
python step1_feature_extraction.py --list
```

2. Verificar se os nomes do subset no `run_pipeline.sh` estão corretos.

3. Testar um `dry-run`:

```bash
bash run_pipeline.sh --set icoseg16 --dry-run
```

### Fluxo B — primeiro experimento rápido

```bash
bash run_pipeline.sh --set icoseg16 --export
```

### Fluxo C — benchmark completo

```bash
bash run_pipeline.sh --set full --export
```

### Fluxo D — continuar de uma etapa intermediária

Exemplo: recomeçar da etapa 3:

```bash
bash run_pipeline.sh --from-step 3 --set icoseg16 --export
```

### Fluxo E — rodar apenas certas etapas

Exemplo: somente etapas 1 e 2:

```bash
bash run_pipeline.sh --steps 1,2 --set icoseg16
```

### Fluxo F — rodar sem SAM

Útil para isolar o desempenho até a co-saliência:

```bash
bash run_pipeline.sh --set icoseg16 --no-sam
```

### Fluxo G — forçar reprocessamento

```bash
bash run_pipeline.sh --set icoseg16 --export --force
```

### Fluxo H — limitar número de imagens

```bash
bash run_pipeline.sh --set icoseg16 --max-images 10 --export
```

### Fluxo I — retomar execução interrompida

```bash
bash run_pipeline.sh --set full --export
```

Não use `--force` nesse caso.

### Fluxo J — refazer só E4/E5 e avaliação

Útil depois de mudar parâmetros do SAM ou corrigir exportação:

```bash
bash run_pipeline.sh --set icoseg16 --from-step 4 --export --force
```

---

## 15. Troubleshooting

### 15.1. “Nenhuma classe encontrada”

Possíveis causas:

- `icoseg_index.json` inexistente ou inconsistente;
- nomes de classes errados no subset;
- dataset em local diferente do esperado.

Soluções:

- rode `python step1_feature_extraction.py --list`;
- confira `icoseg/images`;
- ajuste `ICOSEG_SELECTED_CLASSES` no script.

### 15.2. O subset não roda certas classes

Provavelmente os nomes no array `ICOSEG_SELECTED_CLASSES` não batem exatamente com as chaves do índice.

Exemplo clássico:

- `Panda` vs `Pandas`
- `Elephant` vs `Elephants`
- `Ferrari` vs `Ferrari 599 GTB Fiorano`

Use sempre os nomes retornados por `--list`.

### 15.3. A avaliação não encontra ground truth

Verifique:

- se `icoseg/ground_truth` existe;
- se `gt_index.json` foi gerado;
- se os caminhos das imagens e máscaras correspondem.

Se necessário:

```bash
python evaluate.py --scan_gt icoseg/ground_truth --images_root icoseg/images
```

### 15.4. A avaliação não gera `metrics.json`

Verifique se foram exportadas máscaras PNG:

```bash
find features/icoseg16 -path "*/segmented_e5/*_mask.png"
find features/icoseg16 -path "*/segmented_e4/*_mask.png"
```

Se não houver máscaras, rode novamente com `--export`:

```bash
bash run_pipeline.sh --set icoseg16 --from-step 4 --export --force
```

### 15.5. A etapa 4/5 falha por causa do SAM

Possíveis causas:

- dependências do SAM não instaladas;
- modelo não baixado/configurado;
- VRAM insuficiente.

Teste primeiro sem SAM:

```bash
bash run_pipeline.sh --set icoseg16 --no-sam
```

### 15.6. Há `.running` antigo depois de uma queda

Isso indica que o processo provavelmente foi interrompido durante aquela etapa.

Veja os estados:

```bash
find features/icoseg_full/.state -name "*.running" -o -name "*.failed"
```

Depois rode novamente o mesmo comando, sem `--force`:

```bash
bash run_pipeline.sh --set full --export
```

Se a saída correspondente estiver válida, o script marca a etapa como `.done`. Se estiver ausente ou corrompida, ele refaz a etapa.

### 15.7. Quero saber onde está o gargalo

Use os scripts de auditoria, principalmente nesta ordem:

1. `analyze_step1_features.py`
2. `analyze_step2_correspondence.py`
3. `analyze_step3_cosaliency.py`
4. `analyze_step4_sam_refinement.py`

Isso ajuda a evitar ajustar etapas posteriores quando o problema real está na extração de features.

---

## Observações finais

- O pipeline principal do projeto é `run_pipeline.sh`.
- O modo `icoseg16` é o melhor para desenvolvimento rápido.
- O modo `full` é o mais apropriado para benchmark final.
- Use `--export` quando quiser avaliação automática.
- Para retomar execução interrompida, rode novamente o mesmo comando sem `--force`.
- Se algo der estranho, a primeira coisa a conferir é a consistência entre os nomes das classes e o que existe em `icoseg_index.json`.
