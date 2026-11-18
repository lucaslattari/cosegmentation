#!/usr/bin/env python3
"""Varrer todas as colecoes iCoseg e encontrar pares S2B bons para slides.

Usa apenas NPZ existentes e GT para ranqueamento pos-hoc (nunca para inferencia).
Se correspondence_bg.npz nao existir, reconstroi S2B temporariamente em RAM
usando a implementacao original do step2b_background_suppression.py.

Instalar este arquivo em pipeline/, ao lado de analyze_s2b_numeric.py e
step2b_background_suppression.py.

Exemplo (executar da raiz do projeto):
  python pipeline/scan_s2b_dataset.py --features-root features/icoseg38_topk250

Dependencias: numpy, pillow, matplotlib. Nao requer GPU, DINOv2, SAM2 ou OpenCV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Funcoes ja usadas na auditoria individual, para garantir a mesma convencao.
try:
    import analyze_s2b_numeric as audit
except ImportError as exc:
    raise SystemExit("Coloque scan_s2b_dataset.py em pipeline/, ao lado de "
                     "analyze_s2b_numeric.py") from exc


def load_correspondence(npz: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(npz, allow_pickle=True) as data:
        if 'fg_scores_2d' in data.files:
            scores = data['fg_scores_2d'].astype(np.float32)
        else:
            grid = tuple(int(v) for v in data['patch_grid'])
            scores = data['fg_scores'].astype(np.float32).reshape((-1, *grid))
        paths = [str(x) for x in data['image_paths']]
    return scores, paths


def compute_s2b_in_memory(features_path: Path, before_path: Path, lam: float,
                          gamma: float, border_width: int, top_fg_fraction: float):
    """Replica E2B original sem salvar/substituir NPZ e sem inferencia com redes."""
    try:
        import step2b_background_suppression as s2b
    except ImportError as exc:
        raise RuntimeError("step2b_background_suppression.py ausente em pipeline/") from exc
    # O calculo segue as mesmas funcoes da versao publicada do script E2B.
    with np.load(features_path, allow_pickle=True) as feat, \
         np.load(before_path, allow_pickle=True) as corr:
        pf = s2b.l2_normalize(feat['patch_features'].astype(np.float32), axis=2).astype(np.float32)
        fg = corr['fg_scores'].astype(np.float32)
        grid = tuple(int(v) for v in corr['patch_grid'])
        resize = feat['resize_info'] if 'resize_info' in feat.files else None
        content = s2b.content_mask_from_resize_info(resize, grid)
        border = s2b.border_mask(grid, border_width)
        centroids = s2b.border_centroids(pf, border, content)
        bg_feats, weights = s2b.border_recurrence_weights(pf, centroids, border, content)
        proto = s2b.build_recurring_background_prototype(bg_feats, weights)
        corrected, diagnostics = s2b.suppress_background(
            patch_features=pf, fg_scores=fg, prototype=proto, border=border,
            content_mask=content, lam=lam, gamma=gamma,
            top_fg_fraction=top_fg_fraction)
        corrected = corrected.reshape((-1, *grid))
    del pf
    return corrected, diagnostics


def find_after_path(before_dir: Path, root_before: Path, root_after: Path | None) -> Path | None:
    # Se after-root especificado, usa apenas os scores daquela variante.
    candidates = []
    if root_after is not None:
        candidates.append(root_after / before_dir.relative_to(root_before) / 'correspondence_bg.npz')
    candidates.append(before_dir / 'correspondence_bg.npz')
    for p in candidates:
        if p.is_file():
            return p
    return None


def save_pair(item: dict, a: np.ndarray, b: np.ndarray, resize_info, out: Path,
              alpha: float = .53):
    """Exporta originais e overlays comparaveis: RdYlGn, escala global [0,1]."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    i = item['index']
    with Image.open(item['image_path']) as im:
        orig = im.convert('RGB')
    wh = orig.size
    row = resize_info[i] if resize_info is not None else None
    prefix = f"rank{item['rank']:02d}_{item['collection'][:35]}_img{i:03d}"
    prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)
    original_out = out / (prefix + '_original.png')
    orig.save(original_out)
    generated = {'original': str(original_out)}
    for label, patch in [('before',a[i]), ('after', b[i])]:
        scores = audit.pixel_scores(patch, wh, row)
        rgb = (plt.get_cmap('RdYlGn')(scores)[..., :3]*255).astype('uint8')
        color = Image.fromarray(rgb, 'RGB')
        over = Image.blend(orig, color, alpha)
        path = out / (prefix+f'_{label}_s2b_overlay.png')
        over.save(path)
        generated[label] = str(path)
    return generated


def make_gallery(selected: list[dict], out: Path, rows_per_page: int, alpha: float):
    if not selected:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for pg, start in enumerate(range(0, len(selected), rows_per_page), 1):
        subset = selected[start:start + rows_per_page]
        fig, axes = plt.subplots(len(subset), 4, figsize=(16, 3.25*len(subset)), squeeze=False)
        for j, item in enumerate(subset):
            with Image.open(item['image_path']) as im:
                rgb = np.asarray(im.convert('RGB'))
            # Pares ja exportados, usar as mesmas cores/alpha do slide.
            overlays = item['pngs']
            with Image.open(overlays['before']) as im:
                b = np.asarray(im.convert('RGB'))
            with Image.open(overlays['after']) as im:
                a = np.asarray(im.convert('RGB'))
            with Image.open(item['gt_path']) as im:
                gt = np.asarray(im.convert('L').resize(
                    (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)) > 128
            for k, (im, title) in enumerate(zip(
                (rgb, b, a, gt), ('Original', 'Before S2B', 'After S2B', 'Ground truth'))):
                ax = axes[j,k]
                ax.imshow(im, cmap='gray' if k == 3 else None, vmin=0 if k==3 else None,
                          vmax=1 if k==3 else None)
                ax.set_title(title, fontsize=10)
                ax.set_axis_off()
            axes[j,0].set_title(
                f"#{item['rank']:02d} {item['collection'][:23]}\n"
                f"BG FPR {item['fpr_before']:.0%}→{item['fpr_after']:.0%} | "
                f"FG TPR {item['tpr_before']:.0%}→{item['tpr_after']:.0%} | "
                f"IoU {item['iou_before']:.0%}→{item['iou_after']:.0%}\nOriginal",
                fontsize=8)
        fig.suptitle('Melhores pares S2B — GT somente para selecao (verde = maior FG score)', fontsize=13)
        fig.tight_layout(rect=(0.03, .01, 1, .975))
        path = out / f'gallery_{pg:02d}.png'
        fig.savefig(path, dpi=115, bbox_inches='tight')
        plt.close(fig)


def evaluate_collection(directory: Path, root: Path, features_root: Path,
                        after_root: Path | None, gt_candidates, mapping, args,
                        problems: list[str]):
    feat_path = directory/'features.npz'
    before_path = directory/'correspondence.npz'
    if not feat_path.is_file():
        problems.append(f'{directory}: features.npz ausente')
        return [], None

    before, paths = load_correspondence(before_path)
    after_path = find_after_path(directory, features_root, after_root)
    diag = None
    if args.recompute_all:
        after_path = None
    if after_path is not None:
        after, paths_after = load_correspondence(after_path)
        if before.shape != after.shape or paths != paths_after:
            problems.append(f'{directory}: after NPZ com shape ou ordem diferentes; ignorado')
            return [], None
        source = 'npz:' + str(after_path)
        diagnostic_path = after_path.parent/'background_suppression.json'
        if diagnostic_path.is_file():
            try:
                diag = json.loads(diagnostic_path.read_text(encoding='utf8'))
            except Exception:
                pass
    elif args.compute_missing or args.recompute_all:
        after, diag = compute_s2b_in_memory(feat_path, before_path,
                                            args.bg_lambda, args.bg_gamma,
                                            args.border_width, args.top_fg_fraction)
        if after.shape != before.shape:
            raise ValueError('Resultado calculado nao tem mesmo shape do before')
        source = 'recomputed_in_memory'
    else:
        problems.append(f'{directory}: correspondence_bg.npz nao encontrado; use --compute-missing')
        return [], None

    with np.load(feat_path, allow_pickle=True) as feat:
        resize = feat['resize_info'] if 'resize_info' in feat.files else None
        if 'image_paths' in feat.files and [str(x) for x in feat['image_paths']] != paths:
            problems.append(f'{directory}: E1 image_paths em ordem diferente de E2; ignorado')
            return [], None
        if resize is not None and len(resize) != len(paths):
            problems.append(f'{directory}: resize_info incompatível; ignorado')
            return [], None

    rows=[]
    for i, path_raw in enumerate(paths):
        img_path=audit.choose_path(path_raw, root)
        if img_path is None:
            problems.append(f'{directory.name} #{i}: imagem original nao localizada')
            continue
        gt_path, reason = audit.find_gt(img_path, path_raw, gt_candidates, mapping)
        if gt_path is None:
            problems.append(f'{directory.name} #{i}: {reason}')
            continue
        try:
            with Image.open(img_path) as im:
                wh=im.size
            with Image.open(gt_path) as im:
                gt=np.asarray(im.convert('L').resize(wh, Image.Resampling.NEAREST))>128
            if not np.any(gt) or not np.any(~gt):
                problems.append(f'{directory.name} #{i}: GT vazio ou completamente FG')
                continue
            rr=resize[i] if resize is not None else None
            bmap=audit.pixel_scores(before[i], wh, rr)
            amap=audit.pixel_scores(after[i], wh, rr)
            m0=audit.calc_metrics(bmap, gt, args.threshold)
            m1=audit.calc_metrics(amap, gt, args.threshold)
        except Exception as ex:
            problems.append(f'{directory.name} #{i}: {ex}')
            continue
        d_fpr=m0['fpr']-m1['fpr']  # positivo = BG menos classificado como FG
        d_iou=m1['iou']-m0['iou']
        d_tpr=m1['tpr']-m0['tpr']
        sep0=m0['mean_fg']-m0['mean_bg']
        sep1=m1['mean_fg']-m1['mean_bg']
        d_sep=sep1-sep0
        fg_fraction=float(gt.mean())
        # Filtros para uma visualizacao fiel: excluir melhora aparente que
        # aconteceu sacrificando muito FG ou sem reduzir FP no BG.
        acceptable=(
            d_fpr >= args.min_bg_reduction and
            d_iou >= args.min_iou_gain and
            d_tpr >= -args.max_recall_loss and
            m1['tpr'] >= args.min_tpr_after and
            args.min_fg_fraction <= fg_fraction <= args.max_fg_fraction
        )
        strong=(acceptable and m1['iou'] >= args.min_after_iou and
                d_fpr >= 0.05 and d_iou >= 0.02)
        # Escore para escolher uma BOA IMAGEM DE SLIDE, nao para medir o metodo:
        # reducao de FPR e aumento de IoU dominam; bonus pela separacao e IoU final;
        # punicao explicita se FG ficou pior. Valores em pontos percentuais.
        rank_score=100.0*(
            0.45*d_fpr + 0.30*d_iou + 0.15*d_sep + 0.10*m1['iou']
            - 0.6*max(0.0,-d_tpr)
        )
        rows.append({
            'collection':directory.name,'index':i,'image':img_path.name,
            'image_path':str(img_path),'gt_path':str(gt_path),
            'directory':str(directory),'source_after':source,
            'fg_fraction':fg_fraction, 'fpr_before':m0['fpr'],
            'fpr_after':m1['fpr'],'fpr_reduction':d_fpr,
            'tpr_before':m0['tpr'],'tpr_after':m1['tpr'],'tpr_change':d_tpr,
            'iou_before':m0['iou'],'iou_after':m1['iou'],'iou_gain':d_iou,
            'sep_before':sep0,'sep_after':sep1,'sep_gain':d_sep,
            'bg_score_before':m0['mean_bg'],'bg_score_after':m1['mean_bg'],
            'fg_score_before':m0['mean_fg'],'fg_score_after':m1['mean_fg'],
            'score':rank_score, 'strong':int(strong),'acceptable':int(acceptable),
            'lambda_effective':diag.get('lambda_effective') if diag else '',
        })
    # Nao deixar mapas de 38 grupos inteiros residentes: somente scores para os melhores depois.
    del before, after
    return rows, len(paths)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features-root',type=Path,default=Path('features/icoseg38_topk250'),
                   help='Pasta com subpastas das colecoes e NPZ antigos')
    p.add_argument('--after-root',type=Path,default=None,
                   help='Pasta correspondente de outra variante contendo correspondence_bg.npz')
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p.add_argument('--gt-root',type=Path,default=None)
    p.add_argument('--gt-index',type=Path,default=None)
    p.add_argument('--output',type=Path,default=Path('s2b_best_examples'))
    p.add_argument('--top',type=int,default=12)
    p.add_argument('--rows-per-page',type=int,default=3)
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--min-bg-reduction',type=float,default=.02,
                   help='Reducao minima da FPR do BG, ex. .02 = 2 pontos percentuais')
    p.add_argument('--min-iou-gain',type=float,default=.005)
    p.add_argument('--min-after-iou',type=float,default=.10)
    p.add_argument('--max-recall-loss',type=float,default=.05)
    p.add_argument('--min-tpr-after',type=float,default=.60)
    p.add_argument('--min-fg-fraction',type=float,default=.005)
    p.add_argument('--max-fg-fraction',type=float,default=.65)
    p.add_argument('--alpha',type=float,default=.53,help='Opacidade do heatmap sobre original')
    # Explicit --no-compute-missing permite auditar somente resultados ja salvos.
    p.add_argument('--compute-missing',action=argparse.BooleanOptionalAction,default=True,
                   help='Recalcular S2B em RAM se NPZ after nao existe (padrao: sim)')
    p.add_argument('--recompute-all',action='store_true',
                   help='Ignorar todos NPZ after e aplicar mesma configuracao S2B em RAM a todos')
    p.add_argument('--bg-lambda',type=float,default=.5)
    p.add_argument('--bg-gamma',type=float,default=0.)
    p.add_argument('--border-width',type=int,default=2)
    p.add_argument('--top-fg-fraction',type=float,default=.1)
    args=p.parse_args()
    if not 0 < args.threshold < 1 or not 0 <= args.alpha <= 1:
        p.error('--threshold deve estar (0,1); --alpha deve estar [0,1]')
    if args.top<1 or args.rows_per_page<1:
        p.error('--top e --rows-per-page devem ser >= 1')
    root=args.project_root.resolve()
    features_root=(root/args.features_root).resolve()
    after_root=(root/args.after_root).resolve() if args.after_root else None
    out=(root/args.output).resolve()
    gt_root=(root/args.gt_root).resolve() if args.gt_root else root/'datasets/icoseg/ground_truth'
    gt_index=(root/args.gt_index).resolve() if args.gt_index else root/'gt_index.json'
    if not features_root.is_dir():
        p.error(f'Pasta de features nao encontrada: {features_root}')
    if not gt_root.is_dir() and not gt_index.is_file():
        p.error('Nao localizei GT em datasets/icoseg/ground_truth nem gt_index.json')
    groups=sorted(q.parent for q in features_root.rglob('correspondence.npz') if q.parent != features_root)
    if not groups:
        p.error(f'Nenhuma colecao com correspondence.npz encontrada em {features_root}')
    gt_candidates, specific=audit.load_gt_candidates(root, gt_root, gt_index)
    print(f'Encontradas {len(groups)} colecoes com NPZ em {features_root}')
    print(f'S2B faltante sera calculado em RAM: {args.compute_missing}; '
          f'recalcular todos: {args.recompute_all} (sem modificar NPZ)')
    all_rows=[]
    problems=[]
    totals={'collections_seen':len(groups),'collections_scored':0,'images_npz':0}
    for n, directory in enumerate(groups, 1):
        print(f'[{n:02d}/{len(groups):02d}] {directory.name}',flush=True)
        try:
            rows, nimages = evaluate_collection(directory,root,features_root,after_root,
                                                gt_candidates,specific,args,problems)
            if nimages is not None:
                totals['collections_scored']+=1
                totals['images_npz']+=nimages
            all_rows.extend(rows)
            print(f'    imagens avaliadas: {len(rows)} | candidatas: {sum(x["acceptable"] for x in rows)}',flush=True)
        except Exception as ex:
            problems.append(f'{directory}: {type(ex).__name__}: {ex}')
            print(f'    AVISO: {ex}',file=sys.stderr,flush=True)
    if not all_rows:
        raise SystemExit('Nenhuma imagem avaliada. Consulte mensagens de erro e caminhos dos GT.')
    all_rows.sort(key=lambda x:(-x['strong'],-x['acceptable'],-x['score'],-x['iou_after'],x['collection'],x['index']))
    # Pode nao existir imagem forte. Nesse caso nao inventar sucesso:
    # ordenar tambem os fallbacks e marcar explicitamente.
    selected=all_rows[:args.top]
    for i, item in enumerate(selected,1):
        item['rank']=i
    out.mkdir(parents=True,exist_ok=True)
    export_dir=out/'selected_pairs'
    export_dir.mkdir(exist_ok=True)
    # Carregamos somente as colecoes selecionadas, agrupadas para evitar leituras repetidas.
    cache={}
    for item in selected:
        dr=Path(item['directory'])
        if dr not in cache:
            before,paths=load_correspondence(dr/'correspondence.npz')
            ap=None if args.recompute_all else find_after_path(dr,features_root,after_root)
            if ap:
                after,apaths=load_correspondence(ap)
                if paths!=apaths:
                    raise ValueError(f'After com outra ordem em {dr}')
            else:
                after,_=compute_s2b_in_memory(dr/'features.npz',dr/'correspondence.npz',
                    args.bg_lambda,args.bg_gamma,args.border_width,args.top_fg_fraction)
            with np.load(dr/'features.npz', allow_pickle=True) as ft:
                resize=ft['resize_info'] if 'resize_info' in ft.files else None
            cache[dr]=(before,after,resize)
        b,a,ri=cache[dr]
        item['pngs']=save_pair(item,b,a,ri,export_dir,args.alpha)
    make_gallery(selected,out,args.rows_per_page,args.alpha)
    public_fields=[k for k in all_rows[0] if k not in ('image_path','gt_path','directory')]
    with (out/'ranking.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=public_fields,extrasaction='ignore');w.writeheader();w.writerows(all_rows)
    with (out/'selected_manifest.json').open('w',encoding='utf-8') as f:
        json.dump(selected,f,ensure_ascii=False,indent=2)
    summary={**totals, 'images_evaluated':len(all_rows),
             'strong_examples':sum(x['strong'] for x in all_rows),
             'acceptable_examples':sum(x['acceptable'] for x in all_rows),
             'ranking_definition':'Pos-hoc GT; FPR reducao + IoU ganho + FG-BG separacao + IoU final; recall FG protegido',
             'color_scale':'RdYlGn [0,1]; verde = alto foreground score',
             'threshold':args.threshold,
             'note':'Selecao de figuras, NAO avaliacao do metodo final SAM2. S2B renormaliza independentemente cada imagem.',
             'warnings':problems}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('\nMELHORES EXEMPLOS (FPR menor = BG menos destacado; TPR maior = FG preservado)')
    for item in selected:
        level='FORTE' if item['strong'] else 'BOM' if item['acceptable'] else 'NAO SATISFAZ FILTROS'
        print(f"#{item['rank']:02d} {item['collection'][:36]:36} img {item['index']:03d} "
              f"FPR {item['fpr_before']:.1%}->{item['fpr_after']:.1%}, "
              f"IoU {item['iou_before']:.1%}->{item['iou_after']:.1%}, "
              f"TPR {item['tpr_before']:.1%}->{item['tpr_after']:.1%}  [{level}]")
    print(f'\nPares para slide: {export_dir}')
    print(f'Galerias: {out}/gallery_01.png (e seguintes)')
    print(f'Ranking completo: {out}/ranking.csv')
    print(f'Warnings: {len(problems)} (em summary.json)')
    if not any(x['acceptable'] for x in all_rows):
        print('ATENCAO: nenhuma imagem cumpre os requisitos de reducao BG sem perda FG. '
              'As imagens selecionadas sao alternativas, nao bons exemplos confirmados.')

if __name__=='__main__':
    main()
