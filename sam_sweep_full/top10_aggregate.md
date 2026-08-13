# Top 10 agregado — configurações da Etapa 4 (SAM 2)

> Ranking por `mean_iou_gain` (ganho real de IoU E3->E4 contra GT) quando há ground truth disponível; `mean_proxy` (confiança SAM + coesão) é usado apenas como fallback para configs sem GT. Ver coluna `rank_basis`.

Classes avaliadas: **14**. Mínimo para ranking: **14**.

**Baseline (E3, sem refinamento SAM2)**: IoU médio contra GT = **0.6135** (±0.1395, n=14 classes com GT). Use esta linha como referência para o ganho absoluto (`mean_gain` abaixo é o delta E4−E3; IoU absoluto da E4 é `mean_e4_iou_gt`).

| # | prompt_mode | n_iter | model | classes (GT) | cov | rank_basis | mean_proxy | mean_sam_iou | mean_fg_coh | mean_spatial | mean_gt_iou±std | mean_e4_iou | mean_gain±std |
|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | box_points_mask | 2 | sam2.1-hiera-base-plus | 14/14 | 100.0% | gt_iou_gain | 0.9469 | 0.9215 | 0.7711 | 0.9146 | 0.7263±0.1488 | 0.7263 | 0.1128±0.0582 |
| 2 | box_points_mask | 2 | sam2.1-hiera-small | 14/14 | 100.0% | gt_iou_gain | 0.9487 | 0.9141 | 0.7774 | 0.9349 | 0.7263±0.1426 | 0.7263 | 0.1127±0.0637 |
| 3 | box_only | 2 | sam2.1-hiera-tiny | 14/14 | 100.0% | gt_iou_gain | 0.9406 | 0.9120 | 0.6836 | 0.9263 | 0.7142±0.1870 | 0.7142 | 0.1006±0.1576 |
| 4 | box_only | 1 | sam2.1-hiera-tiny | 14/14 | 100.0% | gt_iou_gain | 0.9295 | 0.8899 | 0.6876 | 0.9115 | 0.7083±0.1828 | 0.7083 | 0.0948±0.1489 |
| 5 | box_points_mask | 2 | sam2.1-hiera-tiny | 14/14 | 100.0% | gt_iou_gain | 0.9569 | 0.9243 | 0.7820 | 0.9487 | 0.7007±0.1516 | 0.7007 | 0.0872±0.0545 |
| 6 | box_only | 2 | sam2.1-hiera-small | 14/14 | 100.0% | gt_iou_gain | 0.9256 | 0.8914 | 0.6662 | 0.9264 | 0.6975±0.1889 | 0.6975 | 0.0839±0.1450 |
| 7 | box_points_mask | 1 | sam2.1-hiera-small | 14/14 | 100.0% | gt_iou_gain | 0.9796 | 0.9825 | 0.7846 | 0.9465 | 0.6953±0.1377 | 0.6953 | 0.0817±0.0419 |
| 8 | box_points_mask | 1 | sam2.1-hiera-base-plus | 14/14 | 100.0% | gt_iou_gain | 0.9744 | 0.9768 | 0.7841 | 0.9347 | 0.6937±0.1409 | 0.6937 | 0.0801±0.0348 |
| 9 | box_only | 1 | sam2.1-hiera-small | 14/14 | 100.0% | gt_iou_gain | 0.8991 | 0.8548 | 0.6639 | 0.8910 | 0.6894±0.1870 | 0.6894 | 0.0759±0.1450 |
| 10 | box_points_mask | 1 | sam2.1-hiera-tiny | 14/14 | 100.0% | gt_iou_gain | 0.9817 | 0.9862 | 0.7861 | 0.9491 | 0.6802±0.1458 | 0.6802 | 0.0667±0.0333 |
