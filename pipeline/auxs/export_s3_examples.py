#!/usr/bin/env python3
"""Exporta bons exemplos visuais da etapa S3 a partir de cosaliency.npz.

Roda apenas pos-processamento: SEM executar DINOv2, spectral clustering ou SAM2.
Os arrays sao aqueles gravados pelo step3_cosaliency.py:
  cluster_labels   -> mapa de clusters espectrais por patch
  fg_mask_coarse   -> decisao FG/BG diretamente a partir dos clusters
  fg_mask_patch    -> mascara S3 apos refinamento intra-imagem/bypass (entrada E4)

Executar a partir da raiz do repositorio:
  python pipeline/export_s3_examples.py --features-root features/icoseg38_topk250 --filter Kite --top 12
  python pipeline/export_s3_examples.py --features-root features/icoseg38_topk250 --top 15

GT (quando encontrado) serve SOMENTE para ranquear exemplos de slides. Nao e
necessario para gerar as figuras. Usa a mascara coarse como criterio principal.

Saidas em s3_best_examples/ (ou --out): ranking.csv, gallery_01.png,
selected_pairs/rank01_..._clusters_overlay.png, _coarse_mask.png,
_coarse_overlay.png, _s3_mask_overlay.png, etc.

Dependencias: numpy, Pillow, matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

ImageResampling = Image.Resampling if hasattr(Image, 'Resampling') else Image


def choose_image(raw: str, root: Path) -> Path | None:
    p = Path(str(raw))
    options = [p, root/p]
    norm = str(raw).replace('\\', '/')
    if len(norm) > 2 and norm[1:3] == ':/':
        norm = '/mnt/' + norm[0].lower() + norm[2:]
    options.append(Path(norm))
    for marker in ('datasets/icoseg/', 'icoseg/'):
        if marker in norm:
            tail = norm.split(marker, 1)[1]
            options.extend([root/'datasets'/'icoseg'/tail, root/'icoseg'/tail])
    for p in options:
        if p.is_file():
            return p.resolve()
    return None


def gt_candidates(root: Path, gt_root: Path, index_path: Path):
    candidates = defaultdict(set)
    mapping = {}
    if gt_root.is_dir():
        for p in gt_root.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.png','.bmp','.tiff','.tif'}:
                candidates[p.stem].add(p.resolve())
                for suff in ('_gt','_mask','_seg','_ground_truth','_label'):
                    if p.stem.lower().endswith(suff):
                        candidates[p.stem[:-len(suff)]].add(p.resolve())
    if index_path.is_file():
        obj = json.loads(index_path.read_text(encoding='utf-8'))
        for stem, raw_gt in obj.get('gt_index', {}).items():
            path = choose_image(raw_gt, root)
            if path is not None:
                candidates[stem].add(path)
        for raw_img, raw_gt in obj.get('mappings', {}).items():
            path = choose_image(raw_gt, root)
            if path is not None:
                mapping[str(raw_img).replace('\\','/')] = path
    return candidates, mapping


def lookup_gt(image: Path, raw_image: str, candidates, mapping):
    raw = str(raw_image).replace('\\','/')
    if raw in mapping:
        return mapping[raw]
    matches = [v for k,v in mapping.items() if Path(k).name == image.name]
    if len(set(matches)) == 1:
        return matches[0]
    options = list(candidates.get(image.stem, []))
    if len(options) == 1:
        return options[0]
    in_album = [p for p in options if p.parent.name == image.parent.name]
    if len(in_album) == 1:
        return in_album[0]
    return None


def mapped_patch_to_pixel(patch: np.ndarray, wh: tuple[int,int], resize_row, categorical=False) -> np.ndarray:
    """Reconstitui o alinhamento da E3: canvas -> crop de padding -> original.

    Para as mascaras binarias usamos BILINEAR + limiar, como upsample_masks
    do script original. Para rotulos discretos, NEAREST evita misturar IDs.
    """
    interp = ImageResampling.NEAREST if categorical else ImageResampling.BILINEAR
    if categorical:
        tile = Image.fromarray(np.asarray(patch, dtype=np.int32), mode='I')
    else:
        tile = Image.fromarray((np.clip(patch,0,1)*255).astype(np.uint8), mode='L')
    if resize_row is not None:
        rh,rw,pt,pl,ch,cw = [int(v) for v in resize_row[:6]]
        tile = tile.resize((cw,ch), interp)
        tile = tile.crop((pl,pt,pl+rw,pt+rh))
    tile = tile.resize(wh, interp)
    arr = np.asarray(tile)
    return arr if categorical else arr > 127


def iou(pred: np.ndarray, gt: np.ndarray):
    union=np.count_nonzero(pred | gt)
    return float(np.count_nonzero(pred & gt)/union) if union else float('nan')


def entropy_normalized(label: np.ndarray, n_clusters: int) -> float:
    counts = np.bincount(label.ravel().astype(np.int64), minlength=n_clusters).astype(np.float64)
    vals = counts[counts>0]
    if len(vals)<2:
        return 0.
    prob=vals/vals.sum()
    return float(-np.sum(prob*np.log(prob))/np.log(max(2,n_clusters)))


def get_features_resize(folder: Path, image_paths: list[str]):
    p=folder/'features.npz'
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=True) as z:
        if 'image_paths' in z.files and [str(x) for x in z['image_paths']] != image_paths:
            raise ValueError('image_paths de E1 e E3 estao em ordens diferentes')
        if 'resize_info' in z.files:
            r=z['resize_info']
            if len(r)!=len(image_paths):
                raise ValueError('resize_info tem numero incorreto de imagens')
            return r
    return None


def list_cases(feat_root: Path, root: Path, filt: str | None, gt_paths, verbose=False):
    cases=[]; warnings=[]; count_collections=0
    for p in sorted(feat_root.rglob('cosaliency.npz')):
        folder=p.parent
        if filt and filt.lower() not in folder.name.lower():
            continue
        count_collections+=1
        try:
            with np.load(p,allow_pickle=True) as z:
                needed=['cluster_labels','fg_mask_coarse','fg_mask_patch','image_paths']
                missing=[k for k in needed if k not in z.files]
                if missing:
                    warnings.append(f'{p}: faltando {missing}')
                    continue
                coarse=np.asarray(z['fg_mask_coarse'],dtype=np.float32)
                refined=np.asarray(z['fg_mask_patch'],dtype=np.float32)
                labels=np.asarray(z['cluster_labels'],dtype=np.int32)
                paths=[str(v) for v in z['image_paths']]
                triggered=(z['empty_fg_bypass_triggered'].astype(bool)
                           if 'empty_fg_bypass_triggered' in z.files else np.zeros(len(paths),bool))
                n_clusters=int(z['n_clusters_used']) if 'n_clusters_used' in z.files else int(labels.max()+1)
                if labels.shape != (len(paths), coarse.shape[1]*coarse.shape[2]):
                    if labels.shape != coarse.shape:
                        raise ValueError(f'shape de labels incompatível: {labels.shape} vs {coarse.shape}')
                    labels=labels.reshape(len(paths),-1)
                if coarse.shape!=refined.shape or len(coarse)!=len(paths):
                    raise ValueError('fg_mask_coarse / fg_mask_patch / image_paths incompativeis')
            resize=get_features_resize(folder,paths)
            for i,raw in enumerate(paths):
                img=choose_image(raw,root)
                if img is None:
                    warnings.append(f'{folder.name} #{i}: nao encontrei imagem {raw}')
                    continue
                with Image.open(img) as im:
                    wh=im.size
                rr=resize[i] if resize is not None else None
                coarse_pix=mapped_patch_to_pixel(coarse[i],wh,rr)
                refined_pix=mapped_patch_to_pixel(refined[i],wh,rr)
                lmap=labels[i].reshape(coarse.shape[1:])
                entropy=entropy_normalized(lmap,n_clusters)
                fg_fraction=float(np.mean(coarse_pix))
                gt=lookup_gt(img,raw,*gt_paths)
                if gt is not None:
                    with Image.open(gt) as im:
                        gt_mask=np.asarray(im.convert('L').resize(wh,ImageResampling.NEAREST))>128
                    if np.any(gt_mask) and np.any(~gt_mask):
                        iou_coarse=iou(coarse_pix,gt_mask)
                        iou_refined=iou(refined_pix,gt_mask)
                    else:
                        gt=None
                        iou_coarse=iou_refined=float('nan')
                else:
                    iou_coarse=iou_refined=float('nan')
                # Ranking de ilustracoes, nao metrica cientifica do metodo.
                # Priorizar coarse corretamente localizado; diversidade de clusters ajuda.
                if gt is not None:
                    score=0.75*iou_coarse+0.25*iou_refined+0.035*entropy
                else:
                    # Fallback sem GT: mascara com area plausivel e clusters discerniveis.
                    score=0.20*entropy+0.10*max(0.,1.-abs(fg_fraction-0.22)/0.22)
                if fg_fraction<.008 or fg_fraction>.78:
                    score-=.30
                if bool(triggered[i]):
                    score-=.06  # evitar exemplo cujo resultado final foi fallback de E2B
                cases.append(dict(collection=folder.name, index=i, image=img.name,
                    image_path=str(img), original_path=raw,
                    npz_path=str(p), feature_path=str(folder/'features.npz'),
                    gt_path=str(gt) if gt else '', rank_score=float(score),
                    iou_coarse=float(iou_coarse),iou_s3_final=float(iou_refined),
                    n_clusters=n_clusters, diversity=entropy,
                    fg_fraction=fg_fraction,bypass=bool(triggered[i])))
        except Exception as ex:
            warnings.append(f'{p}: {type(ex).__name__}: {ex}')
    if verbose:
        print(f'Colecoes com cosaliency.npz encontradas: {count_collections}; exemplos validos: {len(cases)}')
    return cases, warnings


def overlay_mask(rgb: Image.Image, mask: np.ndarray, alpha: float=.50) -> Image.Image:
    a=np.asarray(rgb.convert('RGB'),dtype=np.float32)
    mask3=mask[...,None].astype(np.float32)
    tint=np.zeros_like(a);tint[:]=[42,213,96]
    blend=a*(1.-alpha*mask3)+tint*(alpha*mask3)
    # background discreto, sem alterar semantica binaria
    return Image.fromarray(blend.clip(0,255).astype(np.uint8),'RGB')


def cluster_rgb(label: np.ndarray, nc: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pal=(plt.get_cmap('tab10',max(10,nc))(np.arange(max(10,nc)))[:,:3]*255).astype('uint8')
    return Image.fromarray(pal[np.clip(label,0,len(pal)-1)],'RGB')


def export_case(item, output: Path, root: Path):
    d=Path(item['npz_path']);i=item['index']
    with np.load(d,allow_pickle=True) as z:
        coarse=np.asarray(z['fg_mask_coarse'][i],dtype=np.float32)
        refined=np.asarray(z['fg_mask_patch'][i],dtype=np.float32)
        labels=np.asarray(z['cluster_labels'][i],dtype=np.int32).reshape(coarse.shape)
        if 'cluster_is_fg' in z.files:
            cluster_fg=[int(x) for x in np.flatnonzero(z['cluster_is_fg'])]
        else: cluster_fg=[]
        nc=int(z['n_clusters_used']) if 'n_clusters_used' in z.files else int(labels.max()+1)
    with np.load(d, allow_pickle=True) as z:
        raw_paths = [str(x) for x in z['image_paths']]
    resize=get_features_resize(d.parent,raw_paths)
    rr=resize[i] if resize is not None else None
    with Image.open(item['image_path']) as im:
        rgb=im.convert('RGB');wh=rgb.size
    labels_pix=mapped_patch_to_pixel(labels,wh,rr,categorical=True)
    coarse_pix=mapped_patch_to_pixel(coarse,wh,rr)
    refined_pix=mapped_patch_to_pixel(refined,wh,rr)
    patches=cluster_rgb(labels_pix,nc)
    clusters_overlay=Image.blend(rgb,patches,.64)
    coarse_overlay=overlay_mask(rgb,coarse_pix)
    refined_overlay=overlay_mask(rgb,refined_pix)
    prefix=f"rank{item['rank']:02d}_{item['collection'][:45]}_img{i:03d}_{Path(item['image']).stem[:20]}"
    prefix=re.sub(r'[^A-Za-z0-9_-]','_',prefix)
    paths={
       'original': output/f'{prefix}_original.png',
       'clusters': output/f'{prefix}_clusters.png',
       'clusters_overlay': output/f'{prefix}_clusters_overlay.png',
       'coarse_mask': output/f'{prefix}_coarse_mask.png',
       'coarse_overlay': output/f'{prefix}_coarse_overlay.png',
       's3_mask': output/f'{prefix}_s3_mask.png',
       's3_mask_overlay': output/f'{prefix}_s3_mask_overlay.png',
    }
    rgb.save(paths['original'])
    patches.save(paths['clusters'])
    clusters_overlay.save(paths['clusters_overlay'])
    Image.fromarray(coarse_pix.astype(np.uint8)*255,mode='L').save(paths['coarse_mask'])
    coarse_overlay.save(paths['coarse_overlay'])
    Image.fromarray(refined_pix.astype(np.uint8)*255,mode='L').save(paths['s3_mask'])
    refined_overlay.save(paths['s3_mask_overlay'])
    return {k:str(v) for k,v in paths.items()},cluster_fg


def gallery(cases,out, per_page=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for pg, start in enumerate(range(0,len(cases),per_page),1):
        batch=cases[start:start+per_page]
        fig, axes=plt.subplots(len(batch),5,figsize=(18,3.3*len(batch)),squeeze=False)
        for row,x in enumerate(batch):
            panels=[('original','Original'),('clusters_overlay','Clusters (global)'),
                    ('coarse_overlay','Coarse spectral FG'),('s3_mask_overlay','S3 output (refined)'),
                    ('gt_path','GT (selection only)')]
            for col,(key,title) in enumerate(panels):
                ax=axes[row,col]
                if key=='gt_path':
                    if x['gt_path']:
                        with Image.open(x['gt_path']) as im:
                            arr=np.asarray(im.convert('L'))
                        ax.imshow(arr,cmap='gray',vmin=0,vmax=255)
                    else:ax.text(.5,.5,'GT indisponível',ha='center',va='center',transform=ax.transAxes)
                else:
                    with Image.open(x['files'][key]) as im:arr=np.asarray(im.convert('RGB'))
                    ax.imshow(arr)
                if col==0:
                    t=f"#{x['rank']:02d} {x['collection'][:28]}\n"
                    if x['gt_path']:
                        t+=f"IoU coarse {x['iou_coarse']:.0%}; S3 {x['iou_s3_final']:.0%}\n"
                    else:t+='Ranking visual (GT unavailable)\n'
                    title=t+title
                ax.set_title(title,fontsize=8)
                ax.axis('off')
        fig.suptitle('S3: clusters espectrais e máscaras (GT apenas para escolha)',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.97))
        dest=out/f'gallery_{pg:02d}.png'
        fig.savefig(dest,dpi=120,bbox_inches='tight')
        plt.close(fig)
        print('Galeria:',dest)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features-root',type=Path,default=Path('features/icoseg38_topk250'))
    ap.add_argument('--root',type=Path,default=Path.cwd(),help='raiz do repositorio')
    ap.add_argument('--filter',default=None,help='filtro por nome da colecao: Kite, Pandas, etc.')
    ap.add_argument('--top',type=int,default=12)
    ap.add_argument('--out',type=Path,default=Path('s3_best_examples'))
    ap.add_argument('--gt-root',type=Path,default=None,help='padrao: datasets/icoseg/ground_truth')
    ap.add_argument('--gt-index',type=Path,default=None,help='padrao: gt_index.json')
    args=ap.parse_args()
    if args.top<1:ap.error('--top precisa ser >= 1')
    root=args.root.resolve(); feat_root=args.features_root.resolve();out=args.out.resolve()
    if not feat_root.is_dir():ap.error(f'Nao encontrei {feat_root}')
    gt_root=(args.gt_root or root/'datasets/icoseg/ground_truth').resolve()
    gt_index=(args.gt_index or root/'gt_index.json').resolve()
    cand,mapping=gt_candidates(root,gt_root,gt_index)
    cases,warnings=list_cases(feat_root,root,args.filter,(cand,mapping),verbose=True)
    if not cases:
        ap.error(f'Nenhuma imagem util encontrada. Verifique se existem cosaliency.npz em {feat_root} e se o filtro --filter esta correto')
    cases.sort(key=lambda x:(x['gt_path']!='',x['rank_score']),reverse=True)
    out.mkdir(parents=True,exist_ok=True)
    selected=cases[:args.top]
    pairs=out/'selected_pairs';pairs.mkdir(exist_ok=True)
    manifest=[]
    for rank,x in enumerate(selected,1):
        x['rank']=rank
        files, fg_clusters=export_case(x,pairs,root)
        x['files']=files
        x['foreground_cluster_ids']=fg_clusters
        manifest.append(x)
    cols=['collection','index','image','rank_score','iou_coarse','iou_s3_final',
          'fg_fraction','diversity','bypass','gt_path','image_path','npz_path']
    with (out/'ranking.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        writer.writeheader();writer.writerows(cases)
    (out/'selected_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'warnings.txt').write_text('\n'.join(warnings),encoding='utf-8')
    gallery(selected,out)
    print('\n== MELHORES EXEMPLOS ==')
    for item in selected:
        print(f"#{item['rank']:02d} {item['collection']} / {item['image']} "
              f"IoU_coarse={item['iou_coarse']:.3f} IoU_S3={item['iou_s3_final']:.3f} "
              f"FG={item['fg_fraction']:.1%}")
        print('  Clusters:',item['files']['clusters_overlay'])
        print('  Coarse:  ',item['files']['coarse_overlay'])
    print('\nRanking:',out/'ranking.csv')
    print('Pares:',pairs)
    if warnings:print(f'Avisos ({len(warnings)}):',out/'warnings.txt')

if __name__=='__main__':main()
