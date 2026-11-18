#!/usr/bin/env python3
"""Analise POS-HOC de foreground scores antes e depois do S2B, com GT iCoseg.

Uso (na raiz do repositorio):
  python pipeline/analyze_s2b_numeric.py \
      'features/icoseg38_topk250_s2b_slide/<classe>'

Requer: numpy, Pillow, matplotlib. Nao modifica NPZ nem utiliza GT no pipeline.

Os scores de S2B foram renormalizados por imagem; suas diferencas nao
representam diretamente a quantidade subtraida antes da renormalizacao.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def pixel_scores(patch_map: np.ndarray, wh: tuple[int,int], resize_row) -> np.ndarray:
    """Mesmo mapeamento patch -> original do export_s2b_heatmaps.py."""
    im = Image.fromarray(np.asarray(patch_map, dtype=np.float32), mode="F")
    if resize_row is not None:
        rh, rw, pt, pl, ch, cw = map(int, resize_row[:6])
        im = im.resize((cw, ch), Image.Resampling.BILINEAR)
        im = im.crop((pl, pt, pl + rw, pt + rh))
    im = im.resize(wh, Image.Resampling.BILINEAR)
    return np.clip(np.asarray(im, dtype=np.float32), 0, 1)


def choose_path(raw: str, root: Path) -> Path | None:
    """Trata paths absolutos, relativos e salvos em outro checkout WSL/Windows."""
    p = Path(str(raw))
    if p.is_file():
        return p.resolve()
    if (root / p).is_file():
        return (root / p).resolve()
    norm = str(raw).replace('\\', '/').replace('E:/', '/mnt/e/')
    q = Path(norm)
    if q.is_file():
        return q.resolve()
    for marker in ('datasets/icoseg/', 'icoseg/'):
        if marker in norm:
            suffix = norm.split(marker, 1)[1]
            for prefix in (root/'datasets'/'icoseg', root/'icoseg'):
                test = prefix / suffix
                if test.is_file():
                    return test.resolve()
    return None


def load_gt_candidates(root: Path, gt_root: Path, gt_index_path: Path):
    """Indexa GTs por stem; nao escolhe candidatos ambiguos silenciosamente."""
    exts = {'.png','.bmp','.tif','.tiff'}
    candidates: dict[str,set[Path]] = {}
    def register(stem: str, path: Path):
        if path.is_file():
            candidates.setdefault(stem, set()).add(path.resolve())
    if gt_root.is_dir():
        for path in gt_root.rglob('*'):
            if path.is_file() and path.suffix.lower() in exts:
                name = path.stem
                register(name, path)
                for suffix in ('_gt','_mask','_seg','_ground_truth','_label'):
                    if name.lower().endswith(suffix):
                        register(name[:-len(suffix)], path)
    specific_mapping = {}
    if gt_index_path.is_file():
        data = json.loads(gt_index_path.read_text(encoding='utf-8'))
        for stem, raw in data.get('gt_index',{}).items():
            resolved = choose_path(raw, root)
            if resolved is not None:
                register(stem, resolved)
        for raw_img, raw_gt in data.get('mappings',{}).items():
            resolved = choose_path(raw_gt, root)
            if resolved is not None:
                specific_mapping[str(raw_img).replace('\\','/')] = resolved
    return candidates, specific_mapping


def find_gt(image_path: Path, raw_image_path: str, candidates, specific_mapping):
    raw_normal = str(raw_image_path).replace('\\','/')
    if raw_normal in specific_mapping:
        return specific_mapping[raw_normal], ''
    matching_mappings = [v for k,v in specific_mapping.items() if Path(k).name == image_path.name]
    if len(set(matching_mappings)) == 1:
        return matching_mappings[0], ''
    options = list(candidates.get(image_path.stem, set()))
    if not options:
        return None, 'GT nao encontrado'
    if len(options) == 1:
        return options[0], ''
    # Preferir pasta de mesmo nome dentro de images/ e ground_truth/.
    same_album = [x for x in options if x.parent.name == image_path.parent.name]
    if len(same_album) == 1:
        return same_album[0], ''
    return None, f'GT ambiguo ({len(options)} arquivos para {image_path.stem})'


def calc_metrics(s: np.ndarray, gt: np.ndarray, threshold: float) -> dict:
    pred = s >= threshold
    fg = gt
    bg = ~fg
    tp = int(np.count_nonzero(pred & fg))
    fp = int(np.count_nonzero(pred & bg))
    fn = int(np.count_nonzero(~pred & fg))
    tn = int(np.count_nonzero(~pred & bg))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,
                iou=tp/(tp+fp+fn) if (tp+fp+fn) else float('nan'),
                tpr=tp/(tp+fn) if (tp+fn) else float('nan'),
                fpr=fp/(fp+tn) if (fp+tn) else float('nan'),
                mean_fg=float(s[fg].mean()) if np.any(fg) else float('nan'),
                mean_bg=float(s[bg].mean()) if np.any(bg) else float('nan'),
                precision=tp/(tp+fp) if (tp+fp) else 0.0)


def draw_contact_sheet(items, scores_before, scores_after, resize_info, output, threshold):
    if not items:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count = len(items)
    fig, axes = plt.subplots(count, 4, figsize=(15, 3.5*count), squeeze=False)
    for row, item in enumerate(items):
        path = Path(item['image_path'])
        gt_path = Path(item['gt_path'])
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB'))
            wh = im.size
        with Image.open(gt_path) as im:
            gt = np.asarray(im.convert('L').resize(wh, Image.Resampling.NEAREST)) > 128
        i = item['index']
        rinfo = resize_info[i] if resize_info is not None else None
        a = pixel_scores(scores_before[i], wh, rinfo)
        b = pixel_scores(scores_after[i], wh, rinfo)
        for col, title in enumerate(('Original','Before S2B','After S2B','Ground truth')):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(rgb)
            elif col == 3:
                ax.imshow(gt, cmap='gray',vmin=0,vmax=1)
            else:
                ax.imshow(rgb)
                ax.imshow(a if col==1 else b,cmap='RdYlGn',vmin=0,vmax=1,alpha=0.53)
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
        axes[row,0].set_ylabel(f"#{i:03d}  ΔIoU {item['delta_iou']:+.3f}\n"
                               f"FP BG -{item['fp_removed_bg']} | TP FG {item['tp_change_fg']:+d}",
                               fontsize=9,rotation=90,labelpad=12)
        fig.text(0.01,1-(row+0.03)/count, f"#{i:03d}", fontsize=9)
    fig.suptitle(f"S2B — GT usado apenas para auditoria (threshold={threshold:g})",fontsize=13)
    fig.tight_layout(rect=(0.03,0.015,1,0.98))
    fig.savefig(output,dpi=135,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory',type=Path,help='pasta que contem features.npz, correspondence.npz, correspondence_bg.npz')
    parser.add_argument('--root',type=Path,default=Path.cwd(),help='raiz do repositorio (padrao: diretorio atual)')
    parser.add_argument('--gt-root',type=Path,default=None,help='pasta ground_truth; padrao: datasets/icoseg/ground_truth')
    parser.add_argument('--gt-index',type=Path,default=None,help='gt_index.json; padrao: <root>/gt_index.json')
    parser.add_argument('--threshold',type=float,default=0.5,help='limiar ilustrativo dos fg_scores de E2/E2B (default 0.5)')
    parser.add_argument('--max-recall-loss',type=float,default=0.05,help='perda maxima de TPR permitida para bons exemplos (default 0.05)')
    parser.add_argument('--top',type=int,default=6,help='quantidade de melhores exemplos no painel')
    parser.add_argument('--out',type=Path,default=None,help='pasta de relatorios (padrao <directory>/analysis_s2b)')
    args=parser.parse_args()
    if not 0 < args.threshold < 1: parser.error('--threshold deve estar entre 0 e 1')
    if args.top < 1: parser.error('--top deve ser >= 1')
    root=args.root.resolve()
    directory=args.directory.resolve()
    gt_root=(args.gt_root or root/'datasets'/'icoseg'/'ground_truth').resolve()
    gt_index=(args.gt_index or root/'gt_index.json').resolve()
    output=(args.out or directory/'analysis_s2b').resolve()
    inputs=[directory/'features.npz',directory/'correspondence.npz',directory/'correspondence_bg.npz']
    missing=[str(path) for path in inputs if not path.is_file()]
    if missing: parser.error('Arquivos ausentes: '+', '.join(missing))
    if not gt_root.is_dir() and not gt_index.is_file():
        parser.error(f'Nao achei ground truth em {gt_root} nem indice {gt_index}. Informe --gt-root.')
    output.mkdir(parents=True,exist_ok=True)
    with np.load(inputs[0],allow_pickle=True) as feat, \
         np.load(inputs[1],allow_pickle=True) as before, \
         np.load(inputs[2],allow_pickle=True) as after:
        a=np.asarray(before['fg_scores_2d'],dtype=np.float32)
        b=np.asarray(after['fg_scores_2d'],dtype=np.float32)
        paths=[str(s) for s in before['image_paths']]
        resize_info=feat['resize_info'] if 'resize_info' in feat.files else None
        if a.shape!=b.shape or len(paths)!=len(a):
            raise ValueError(f'Arquivos NPZ incompatíveis: a={a.shape}, b={b.shape}, imgs={len(paths)}')
        if 'image_paths' in after.files and not np.array_equal(before['image_paths'],after['image_paths']):
            raise ValueError('A ordem de image_paths difere entre correspondences (nao misturar execucoes)!')
        if 'image_paths' in feat.files and not np.array_equal(before['image_paths'],feat['image_paths']):
            raise ValueError('A ordem de image_paths difere entre E1 e E2!')
        if resize_info is not None and len(resize_info)!=len(paths):
            raise ValueError('resize_info inconsistente com o numero de imagens')
        candidates,specific=load_gt_candidates(root,gt_root,gt_index)
        rows=[]
        errors=[]
        for i, raw in enumerate(paths):
            img=choose_path(raw,root)
            if img is None:
                errors.append(f'#{i:03d}: imagem nao encontrada: {raw}')
                continue
            gt_path,reason=find_gt(img,raw,candidates,specific)
            if gt_path is None:
                errors.append(f'#{i:03d}: {reason}: {img.name}')
                continue
            with Image.open(img) as im:
                wh=im.size
            with Image.open(gt_path) as im:
                mask=np.asarray(im.convert('L').resize(wh,Image.Resampling.NEAREST))>128
            if not np.any(mask) or not np.any(~mask):
                errors.append(f'#{i:03d}: GT vazio ou completamente FG: {gt_path}')
                continue
            r=resize_info[i] if resize_info is not None else None
            before_pixels=pixel_scores(a[i],wh,r)
            after_pixels=pixel_scores(b[i],wh,r)
            m0=calc_metrics(before_pixels,mask,args.threshold)
            m1=calc_metrics(after_pixels,mask,args.threshold)
            row={'index':i,'image':img.name,'image_path':str(img),'gt_path':str(gt_path),
                 'gt_fg_fraction':round(float(mask.mean()),5),
                 'iou_before':round(m0['iou'],6),'iou_after':round(m1['iou'],6),
                 'delta_iou':round(m1['iou']-m0['iou'],6),
                 'fpr_bg_before':round(m0['fpr'],6),'fpr_bg_after':round(m1['fpr'],6),
                 'delta_fpr_bg':round(m1['fpr']-m0['fpr'],6),
                 'tpr_fg_before':round(m0['tpr'],6),'tpr_fg_after':round(m1['tpr'],6),
                 'delta_tpr_fg':round(m1['tpr']-m0['tpr'],6),
                 'fp_removed_bg':m0['fp']-m1['fp'],
                 'tp_change_fg':m1['tp']-m0['tp'],
                 'mean_bg_before':round(m0['mean_bg'],6),'mean_bg_after':round(m1['mean_bg'],6),
                 'mean_fg_before':round(m0['mean_fg'],6),'mean_fg_after':round(m1['mean_fg'],6),
                 'delta_fg_bg_separation':round((m1['mean_fg']-m1['mean_bg'])-(m0['mean_fg']-m0['mean_bg']),6),
                 'eligible':int(m1['iou']>m0['iou'] and m1['fpr']<m0['fpr']
                                and m1['tpr']>=m0['tpr']-args.max_recall_loss)}
            rows.append(row)
        if not rows:
            raise RuntimeError('Nao foi possivel avaliar nenhuma imagem. Motivos:\n'+'\n'.join(errors))
        ranked=sorted(rows,key=lambda x:(-x['eligible'],-x['delta_iou'],x['delta_fpr_bg'],x['index']))
        csvfile=output/'metrics_per_image.csv'
        with csvfile.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader();writer.writerows(ranked)
        candidates_best=[row for row in ranked if row['eligible']]
        selection=candidates_best[:args.top]
        if selection:
            draw_contact_sheet(selection,a,b,resize_info,output/'top_candidates.png',args.threshold)
        diag=directory/'background_suppression.json'
        diagnostics=json.loads(diag.read_text(encoding='utf-8')) if diag.is_file() else None
        report={'directory':str(directory),'n_npz':len(paths),'n_evaluated':len(rows),
                'n_missing_or_ambiguous':len(errors),'threshold':args.threshold,
                'n_positive_delta_iou':sum(r['delta_iou']>0 for r in rows),
                'n_eligible':len(candidates_best),
                'mean_iou_before':float(np.mean([r['iou_before'] for r in rows])),
                'mean_iou_after':float(np.mean([r['iou_after'] for r in rows])),
                'mean_delta_iou':float(np.mean([r['delta_iou'] for r in rows])),
                'diagnostics':diagnostics,'selected_indices':[r['index'] for r in selection],
                'warnings':errors,
                'note':'Scores de E2/E2B sao por-patch e normalizados independentemente; threshold=0.5 e diagnostico, nao mascara final da E3-E5. GT usado SOMENTE para auditoria post-hoc.'}
        (output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f'\nColecao: {len(paths)} imagens | com GT valido: {len(rows)} | sem GT: {len(errors)}')
        print(f'IoU medio (E2 threshold {args.threshold:g}) {report["mean_iou_before"]:.4f} -> {report["mean_iou_after"]:.4f} (delta {report["mean_delta_iou"]:+.4f})')
        if diagnostics:
            print('lambda_effective:', diagnostics.get('lambda_effective','indisponivel'))
        print(f'\nCandidatos: deltaIoU>0, FP background reduzidos, queda de TPR FG <= {args.max_recall_loss:.1%}')
        if candidates_best:
            print(f'{"idx":>4} {"deltaIoU":>10} {"FP_BG_removed":>15} {"deltaTPR":>10}  imagem')
            for r in candidates_best[:min(args.top,10)]:
                print(f'{r["index"]:4d} {r["delta_iou"]:+10.4f} {r["fp_removed_bg"]:15d} {r["delta_tpr_fg"]:+10.4f}  {r["image"]}')
        else:
            print('Nenhuma imagem atendeu aos 3 criterios. Nao force a selecao de um exemplo.')
        for error in errors: print('AVISO:',error,file=sys.stderr)
        print('\nCSV:',csvfile)
        print('Resumo:',output/'summary.json')
        if selection: print('Painel:',output/'top_candidates.png')
        print('\nOBS: GT nao altera pipeline; avaliacao e posterior a inferencia.')

if __name__=='__main__':
    main()
