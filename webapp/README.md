# Cosegmentation Web

Interface Flask para executar o pipeline de cosegmentação sobre uma coleção enviada manualmente pelo navegador e para auditar isoladamente as Etapas 1–4.

## Princípio de execução do webapp

O webapp separa explicitamente duas responsabilidades:

- **coleção**: é fornecida pelo usuário e salva em `webapp/runs/<run_id>/input/`;
- **configuração algorítmica**: no modo **Pipeline completo**, é fixada pelo preset `PRODUCTION` de `pipeline_runner.py`.

Assim, o webapp corresponde ao trecho do pipeline em que uma coleção já foi resolvida e está pronta para passar por E1 → E2 → E2B → E3 → E4 → E5. Ele **não** replica a lógica de seleção de benchmarks/índices de `run_pipeline.sh`.

Para coleções enviadas manualmente, E1 usa `protocol=semantic`. Protocolos específicos de benchmark, como `icoseg38`, permanecem restritos ao orquestrador experimental em shell.

> `pipeline/run_pipeline.sh` não é modificado nem chamado pelo webapp. Ele permanece independente para preservar o contexto experimental usado em outros trabalhos.

## Pipeline completo

O modo **Pipeline completo** usa o seguinte preset no webapp:

1. **E1 — DINOv2**
   - `protocol=semantic`
   - `dinov2_vits14`
   - `img_size=518`
   - `resize=keep_ar`
   - `feature_mode=l2`
2. **E2 — Cross-image correspondence**
   - `top_k=250`
   - `temperature=0.05`
   - `orientation=spatial`
   - `degree_alpha=0.10`
3. **E2B — Adaptive Recurring-Background Suppression**
   - `mode=recurrence`
   - `lambda=0.50`
   - `gamma=0.00`
   - `border_width=2`
   - `top_fg_fraction=0.10`
4. **E3 — Co-saliency**
   - usa `correspondence_bg.npz`
   - `n_clusters=8`
   - `k_neighbors=15`
   - `n_spectral=8`
   - `saliency_alpha=0.5`
   - `fg_decision_mode=saliency`
   - `empty_fg_bypass=e2_topk`
   - `empty_fg_top_fraction=0.20`
5. **E4 — SAM 2**
   - `facebook/sam2.1-hiera-large`
   - `prompt_mode=box_points_mask`
   - `n_iter=2`
   - exporta `segmented_e4/`
6. **E5 — Iterative refinement**
   - mesmo SAM 2.1 Hiera Large
   - `max_iter=20`
   - `converge_thresh=0.99`
   - `n_fg_pts=8`
   - `n_bg_pts=8`
   - `--no_mask_history`
   - exporta `segmented_e5/`

O dicionário `PRODUCTION` é a fonte de verdade **do webapp** para o modo completo e para a correção global. O `run_pipeline.sh` continua com sua própria configuração e não é alterado por esta aplicação.

## Snapshot de configuração por execução

Ao criar um run, `app.py` grava em `metadata.json`:

- `collection_source=uploaded_folder`;
- `e1_protocol=semantic`;
- `configuration_source=production_preset` no pipeline completo;
- o snapshot completo de parâmetros em `options`.

`pipeline_runner.py` reutiliza esse snapshot durante a execução e também na correção global. Isso evita que uma alteração futura em `PRODUCTION` mude silenciosamente a configuração associada a um run já criado.

## Validação das saídas

No pipeline completo, o webapp não considera uma etapa concluída apenas porque o processo retornou código zero. Após cada estágio ele valida os principais artefatos:

```text
E1   -> features.npz
E2   -> correspondence.npz
E2B  -> correspondence_bg.npz
E3   -> cosaliency.npz
E4   -> refined_masks.npz + segmented_e4/*_mask.png
E5   -> iterative_masks.npz + segmented_e5/*_mask.png
```

Os arquivos NPZ são verificados quanto à existência, tamanho e abertura válida pelo NumPy. Na E3, o webapp também confere o metadata de `empty_fg_bypass`/`empty_fg_top_fraction`, acompanhando a proteção usada no shell contra uma saída incompatível. Como cada execução do webapp recebe um diretório novo, não há a mesma necessidade de checkpoint/reuso de saídas antigas existente no shell experimental.

## Avaliação com ground truth

Se ground truth for enviado, `pipeline/evaluate.py` roda depois da E5 sobre `segmented_e5`.

A avaliação é tratada como uma operação posterior à cosegmentação. Portanto:

- se E1–E5 terminarem corretamente e a avaliação também funcionar, o run termina como concluído com avaliação;
- se E1–E5 terminarem corretamente, mas `evaluate.py` falhar, o run continua marcado como **concluído**, com `evaluation_state=failed` e um aviso no `status.json`;
- uma falha em E1–E5 continua marcando o run como **failed**.

A mesma separação é aplicada à avaliação posterior à correção global.

## Auditorias

### Etapa 1

Executa E1 e `ablation/analyze_step1_features.py`. A interface permite variar backbone DINOv2, resolução, resize e modo de features. O protocolo permanece `semantic` porque a coleção é fornecida manualmente.

### Etapa 2

Executa E1 → E2 → `ablation/analyze_step2_correspondence.py`. Além de `top_k` e `temperature`, a interface expõe `orientation` e `degree_alpha`. Os valores iniciais são os de produção.

### Etapa 3

Executa E1 → E2 → **E2B** → E3 → `ablation/analyze_step3_cosaliency.py`.

A E2B pode ser desativada para comparação. Quando ativa, E3 recebe `correspondence_bg.npz`; quando desativada, recebe `correspondence.npz`. Também podem ser ajustados os parâmetros principais de E3 e o mecanismo `empty_fg_bypass`.

### Etapa 4

Executa a mesma cadeia da auditoria de E3, acrescenta SAM 2 e então chama `ablation/analyze_step4_sam_refinement.py`. O modelo inicial é `facebook/sam2.1-hiera-large`.

## Correção global por scribble

A correção reutiliza E1/E3 e altera `cosaliency.npz` antes de reexecutar E4 e E5. O rerun pós-correção usa o snapshot do preset salvo no `metadata.json` da execução original.

Os resultados corrigidos ficam isolados em `output/correction/`. As saídas corrigidas de E4/E5 também são validadas antes de a correção ser considerada concluída.

## Ground truth

O envio de GT é opcional. Os arquivos são pareados pelo nome; são aceitos stems iguais ou sufixos como `_mask`, `_gt`, `-mask`, `-gt`, `_ground_truth` e `_segmentation`.

Exemplo:

```text
urso_01.jpg  <->  urso_01_mask.png
```

As máscaras devem representar foreground em branco e background em preto.

## Estrutura de saída

Cada execução fica em:

```text
webapp/runs/<run_id>/
```

Principais artefatos do pipeline completo:

```text
input/
ground_truth/       # quando houver GT
output/
  features.npz
  correspondence.npz
  correspondence_bg.npz
  cosaliency.npz
  refined_masks.npz
  iterative_masks.npz
  segmented_e4/
  segmented_e5/
  evaluation/        # quando houver GT e a avaliação for produzida
metadata.json
status.json
pipeline.log
```

## Execução

Partindo da raiz do projeto:

```bash
cd webapp
python app.py
```

Abra `http://127.0.0.1:5000`.

As dependências do pipeline e das rotinas de auditoria precisam estar instaladas no mesmo ambiente Python usado para iniciar o Flask.
