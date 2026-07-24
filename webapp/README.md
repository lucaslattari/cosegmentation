# Auditorias das Etapas 1, 2, 3 e 4 no webapp

Esta versão mantém o pipeline completo e acrescenta uma execução isolada da
Etapa 1. Na página inicial, selecione **Auditar Etapa 1**, arraste pelo menos
duas imagens, seus ground truths opcionais e escolha a configuração do DINOv2.

Também é possível selecionar **Auditar Etapa 2**. Nesse modo, o webapp executa
a Etapa 1 como pré-requisito, calcula as correspondências e roda
`ablation/analyze_step2_correspondence.py`. É possível configurar `top_k` e
`temperature`, além dos parâmetros usados na extração DINOv2.

Em **Auditar Etapa 3**, o app executa E1 e E2 como pré-requisitos, produz
`cosaliency.npz` e chama `ablation/analyze_step3_cosaliency.py`. A interface
expõe `k_neighbors`, `n_spectral`, `n_clusters`, `fg_threshold`,
`saliency_alpha`, refinamento intra-imagem e clusters adaptativos ao padding.

Em **Auditar Etapa 4**, o app também executa o refinamento SAM 2 e chama
`ablation/analyze_step4_sam_refinement.py`. É possível escolher o modelo SAM,
o modo de prompting (`box_only`, `box_points` ou `box_points_mask`) e o número
de iterações.

O servidor executa, em sequência:

1. `pipeline/step1_feature_extraction.py`, que cria `features.npz`;
2. `ablation/analyze_step1_features.py`, que recebe o índice de ground truths
   quando disponível e cria o JSON, o relatório Markdown e as visualizações.

Os resultados de cada execução ficam em `webapp/runs/<id>/`. O navegador exibe
o diagnóstico, os vereditos por métrica, os testes sugeridos e os gráficos do
analisador.

Na auditoria da Etapa 1, o webapp também apresenta dois heatmaps específicos
para as métricas de ciclo:

- `cycle_top1_heatmap.png`: fração de patches que completam A→B→A e retornam
  exatamente ao patch de origem;
- `cycle_reconstruction_heatmap.png`: similaridade entre o patch original e o
  patch recuperado depois do ciclo, mesmo quando a posição exata não é
  recuperada.

Essa separação evita interpretar `local_similarity_heatmap.png` como se ele
representasse diretamente a consistência cíclica. O primeiro heatmap mede
reciprocidade exata; o segundo mede preservação semântica da reconstrução.

Na Etapa 2, as métricas sem GT cobrem polarização, bimodalidade, fração de
foreground, concordância entre pares, estabilidade leave-one-pair-out e
correlação com a saliência CLS. Com GT, também são calculados o melhor IoU por
threshold, AUC ROC, acurácia da orientação FG/BG e uma comparação post-hoc das
heurísticas `cls_saliency`, `size_based`, `spatial_based` e `combined`.

Na Etapa 3, os vereditos avaliam fração FG, coerência espacial, margem dos
clusters, concordância entre E2 e o prior espacial, sensibilidade a α,
centralidade e compacidade. Com GT, também avaliam IoU patch-level, oracle dos
clusters e acurácia do mapeamento cluster→foreground/background.

Na Etapa 4, a auditoria distingue a confiança interna do SAM do IoU real. Ela
avalia fração FG, coesão espacial e semântica e, com GT, mede o IoU final, o
ganho E4−E3 e quantas imagens foram degradadas. Também gera comparações E3/E4,
gráficos por imagem, overlays e mapas de TP/FP/FN.

No modo **Pipeline completo**, o envio de ground truth também é opcional. Se
as máscaras forem fornecidas, depois da Etapa 5 o app executa
`pipeline/evaluate.py` sobre `segmented_e5` e mostra uma tabela com as médias
de accuracy, precision, recall, F1 e IoU, além do desvio-padrão do IoU
(`iou_std`). A avaliação usa a orientação produzida pelo pipeline, sem inverter
automaticamente foreground e background. Sem GT, o fluxo permanece com cinco
etapas e exibe apenas as máscaras finais.

## Execução

Partindo da raiz do projeto:

```bash
cd webapp
python app.py
```

Abra `http://127.0.0.1:5000`.

As dependências do pipeline e do analisador precisam estar instaladas no mesmo
ambiente Python usado para iniciar o Flask. A primeira execução de um modelo
DINOv2 pode baixar pesos e, por isso, tende a demorar mais.

## Interpretação

Sem ground truth, o analisador usa coesão por CLS, outliers, vizinhos locais,
consistência cíclica e recorrência de clusters. Com ground truth, também calcula:

- margem de similaridade foreground/background;
- AUC leave-one-image-out baseada em protótipos FG/BG;
- AUC, F1 e balanced accuracy de um linear probe com grupos por imagem;
- IoU oracle do melhor subconjunto global de clusters;
- IoU oracle do melhor subconjunto por imagem.

As máscaras são pareadas pelo nome. Podem ter o mesmo stem da imagem ou os
sufixos `_mask`, `_gt`, `-mask`, `-gt`, `_ground_truth` e `_segmentation`.
Exemplo: `urso_01.jpg` ↔ `urso_01_mask.png`. O servidor exige pareamento
completo quando algum GT é enviado e gera `webapp/runs/<id>/gt_index.json`.

As máscaras devem usar branco para foreground e preto para background. O
analisador as alinha ao padding de `keep_ar` e reduz para a grade de patches.
Os limiares dos vereditos continuam heurísticos: compare configurações
diferentes sobre a mesma coleção antes de concluir que uma feature é melhor.
