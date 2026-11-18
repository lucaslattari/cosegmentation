from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from pipeline_runner import (
    production_options,
    run_pipeline,
    run_step1_analysis,
    run_step2_analysis,
    run_step3_analysis,
    run_step4_analysis,
    run_correction,
)


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RUN_MODES = {"pipeline", "step1", "step2", "step3", "step4"}
STEP1_MODELS = {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"}
STEP1_IMAGE_SIZES = {448, 518, 672}
STEP1_RESIZE_MODES = {"keep_ar", "square"}
STEP1_FEATURE_MODES = {"raw", "l2", "pca128_l2", "pca256_l2"}
SAM_MODELS = {
    "facebook/sam2.1-hiera-tiny",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-large",
    "facebook/sam2-hiera-tiny",
    "facebook/sam2-hiera-small",
    "facebook/sam2-hiera-base-plus",
    "facebook/sam2-hiera-large",
}
SAM_PROMPT_MODES = {"box_only", "box_points", "box_points_mask"}
E2_ORIENTATIONS = {"cls", "spatial"}
E2_BG_SUPPRESSION_MODES = {"none", "recurrence"}
E3_EMPTY_FG_BYPASS_MODES = {"none", "e2_topk"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_dir(run_id: str) -> Path:
    if not run_id or any(c not in "0123456789abcdef" for c in run_id):
        abort(404)
    path = RUNS_DIR / run_id
    if not path.is_dir():
        abort(404)
    return path


def read_json(path: Path, default: dict | None = None) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default or {}


def read_status(path: Path) -> dict:
    return read_json(
        path / "status.json",
        {"state": "preparing", "step": 0, "progress": 0, "message": "Preparando execução..."},
    )


def parse_step1_options() -> tuple[dict, str | None]:
    try:
        img_size = int(request.form.get("img_size", "518"))
    except ValueError:
        return {}, "Tamanho de imagem inválido."

    options = {
        "protocol": "semantic",
        "model": request.form.get("model", "dinov2_vits14"),
        "img_size": img_size,
        "resize": request.form.get("resize", "keep_ar"),
        "feature_mode": request.form.get("feature_mode", "l2"),
    }
    checks = [
        (options["model"] in STEP1_MODELS, "Modelo DINOv2 inválido."),
        (options["img_size"] in STEP1_IMAGE_SIZES, "Resolução inválida."),
        (options["resize"] in STEP1_RESIZE_MODES, "Modo de redimensionamento inválido."),
        (options["feature_mode"] in STEP1_FEATURE_MODES, "Modo de features inválido."),
    ]
    return options, next((message for valid, message in checks if not valid), None)


def parse_step2_options() -> tuple[dict, str | None]:
    try:
        top_k = int(request.form.get("top_k", "250"))
        temperature = float(request.form.get("temperature", "0.05"))
        degree_alpha = float(request.form.get("degree_alpha", "0.10"))
    except ValueError:
        return {}, "Os parâmetros da Etapa 2 precisam ser numéricos."

    orientation = request.form.get("orientation", "spatial")
    checks = [
        (1 <= top_k <= 500, "top_k deve estar entre 1 e 500."),
        (0.001 <= temperature <= 1.0, "temperature deve estar entre 0.001 e 1.0."),
        (0.0 <= degree_alpha <= 1.0, "degree_alpha deve estar entre 0 e 1."),
        (orientation in E2_ORIENTATIONS, "Heurística de orientação da Etapa 2 inválida."),
    ]
    return {
        "top_k": top_k,
        "temperature": temperature,
        "orientation": orientation,
        "degree_alpha": degree_alpha,
    }, next((message for valid, message in checks if not valid), None)


def parse_step3_options() -> tuple[dict, str | None]:
    try:
        k_neighbors = int(request.form.get("k_neighbors", "15"))
        n_spectral = int(request.form.get("n_spectral", "8"))
        n_clusters = int(request.form.get("n_clusters", "8"))
        saliency_alpha = float(request.form.get("saliency_alpha", "0.5"))
        bg_lambda = float(request.form.get("bg_lambda", "0.50"))
        bg_gamma = float(request.form.get("bg_gamma", "0.00"))
        empty_fg_top_fraction = float(request.form.get("empty_fg_top_fraction", "0.20"))
    except ValueError:
        return {}, "Os parâmetros das Etapas 2B/3 precisam ser numéricos."

    bg_suppression = request.form.get("bg_suppression", "recurrence")
    empty_fg_bypass = request.form.get("empty_fg_bypass", "e2_topk")
    checks = [
        (1 <= k_neighbors <= 100, "k_neighbors deve estar entre 1 e 100."),
        (2 <= n_spectral <= 64, "n_spectral deve estar entre 2 e 64."),
        (2 <= n_clusters <= 32, "n_clusters deve estar entre 2 e 32."),
        (0.0 <= saliency_alpha <= 1.0, "saliency_alpha deve estar entre 0 e 1."),
        (bg_suppression in E2_BG_SUPPRESSION_MODES, "Modo de supressão E2B inválido."),
        (bg_lambda >= 0.0, "lambda da E2B deve ser não negativo."),
        (0.0 <= bg_gamma <= 1.0, "gamma da E2B deve estar entre 0 e 1."),
        (empty_fg_bypass in E3_EMPTY_FG_BYPASS_MODES, "Modo de fallback empty-FG inválido."),
        (0.0 < empty_fg_top_fraction <= 1.0, "A fração do fallback empty-FG deve estar em (0, 1]."),
    ]
    error = next((message for valid, message in checks if not valid), None)
    return {
        "k_neighbors": k_neighbors,
        "n_spectral": n_spectral,
        "n_clusters": n_clusters,
        "saliency_alpha": saliency_alpha,
        "bg_suppression": bg_suppression,
        "bg_lambda": bg_lambda,
        "bg_gamma": bg_gamma,
        "empty_fg_bypass": empty_fg_bypass,
        "empty_fg_top_fraction": empty_fg_top_fraction,
    }, error


def parse_step4_options() -> tuple[dict, str | None]:
    model = request.form.get("sam_model", "facebook/sam2.1-hiera-large")
    prompt_mode = request.form.get("prompt_mode", "box_points_mask")
    try:
        n_iter = int(request.form.get("sam_n_iter", "2"))
    except ValueError:
        return {}, "O número de iterações do SAM precisa ser inteiro."
    checks = [
        (model in SAM_MODELS, "Modelo SAM 2 inválido."),
        (prompt_mode in SAM_PROMPT_MODES, "Modo de prompting inválido."),
        (1 <= n_iter <= 5, "O número de iterações do SAM deve estar entre 1 e 5."),
    ]
    return {
        "sam_model": model,
        "prompt_mode": prompt_mode,
        "sam_n_iter": n_iter,
    }, next((message for valid, message in checks if not valid), None)


def pairing_key(filename: str) -> str:
    """Normaliza nomes como foto_01.jpg, foto_01_mask.png e foto-01-gt.png."""
    stem = Path(secure_filename(filename)).stem.lower()
    stem = re.sub(r"(?:[_\-. ](?:mask|gt|ground[_\- ]?truth|segmentation))+$", "", stem)
    return re.sub(r"[^a-z0-9]+", "", stem)


def pair_ground_truths(images, masks) -> tuple[dict[int, int], str | None]:
    if not masks:
        return {}, None

    mask_by_key: dict[str, int] = {}
    for index, mask in enumerate(masks):
        key = pairing_key(mask.filename)
        if not key:
            return {}, f"Não foi possível interpretar o nome do ground truth '{mask.filename}'."
        if key in mask_by_key:
            return {}, f"Há mais de um ground truth correspondente à chave '{key}'."
        mask_by_key[key] = index

    pairs: dict[int, int] = {}
    missing = []
    image_keys: set[str] = set()
    for index, image in enumerate(images):
        key = pairing_key(image.filename)
        if not key or key in image_keys:
            return {}, "Os nomes das imagens precisam produzir identificadores únicos para o pareamento."
        image_keys.add(key)
        if key not in mask_by_key:
            missing.append(secure_filename(image.filename))
        else:
            pairs[index] = mask_by_key[key]

    extra = sorted(set(mask_by_key) - image_keys)
    if missing or extra:
        details = []
        if missing:
            details.append("sem GT: " + ", ".join(missing[:8]))
        if extra:
            details.append("GT sem imagem: " + ", ".join(extra[:8]))
        return {}, "Não foi possível parear todos os arquivos (" + "; ".join(details) + ")."
    return pairs, None


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/runs")
def create_run():
    mode = request.form.get("mode", "pipeline")
    if mode not in RUN_MODES:
        return render_template("index.html", error="Tipo de execução inválido."), 400

    files = [f for f in request.files.getlist("images") if f.filename]
    valid_files = [
        f for f in files
        if Path(secure_filename(f.filename)).suffix.lower() in ALLOWED_EXTENSIONS
    ]
    if len(valid_files) != len(files):
        return render_template("index.html", error="Há imagens com formato não suportado."), 400
    if len(valid_files) < 2:
        return render_template("index.html", error="Envie pelo menos duas imagens válidas."), 400

    options: dict = production_options() if mode == "pipeline" else {}
    if mode in {"step1", "step2", "step3", "step4"}:
        options, error = parse_step1_options()
        if error:
            return render_template("index.html", error=error), 400
        if mode in {"step2", "step3", "step4"}:
            step2_options, error = parse_step2_options()
            if error:
                return render_template("index.html", error=error), 400
            options.update(step2_options)
        if mode in {"step3", "step4"}:
            step3_options, error = parse_step3_options()
            if error:
                return render_template("index.html", error=error), 400
            options.update(step3_options)
        if mode == "step4":
            step4_options, error = parse_step4_options()
            if error:
                return render_template("index.html", error=error), 400
            options.update(step4_options)

    masks = request.files.getlist("ground_truths")
    submitted_masks = [f for f in masks if f.filename]
    valid_masks = [
        f for f in submitted_masks
        if Path(secure_filename(f.filename)).suffix.lower() in ALLOWED_EXTENSIONS
    ]
    if len(valid_masks) != len(submitted_masks):
        return render_template("index.html", error="Há ground truths com formato não suportado."), 400
    gt_pairs, error = pair_ground_truths(valid_files, valid_masks)
    if error:
        return render_template("index.html", error=error), 400

    run_id = uuid.uuid4().hex
    path = RUNS_DIR / run_id
    input_dir = path / "input"
    gt_dir = path / "ground_truth"
    output_dir = path / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir()

    used_names: set[str] = set()
    saved_images: list[Path] = []
    for index, file in enumerate(valid_files, start=1):
        name = secure_filename(file.filename) or f"image_{index}.png"
        if name in used_names:
            name = f"{Path(name).stem}_{index}{Path(name).suffix}"
        used_names.add(name)
        destination = input_dir / name
        file.save(destination)
        saved_images.append(destination)

    gt_index_path = None
    if valid_masks:
        gt_dir.mkdir()
        saved_masks: list[Path] = []
        used_gt_names: set[str] = set()
        for index, file in enumerate(valid_masks, start=1):
            name = secure_filename(file.filename) or f"mask_{index}.png"
            if name in used_gt_names:
                name = f"{Path(name).stem}_{index}{Path(name).suffix}"
            used_gt_names.add(name)
            destination = gt_dir / name
            file.save(destination)
            saved_masks.append(destination)
        gt_index_path = path / "gt_index.json"
        mappings = {
            str(saved_images[image_index].resolve()): str(saved_masks[mask_index].resolve())
            for image_index, mask_index in gt_pairs.items()
        }
        stem_index = {
            saved_images[image_index].stem: str(saved_masks[mask_index].resolve())
            for image_index, mask_index in gt_pairs.items()
        }
        gt_index_path.write_text(
            json.dumps(
                {"mappings": mappings, "gt_index": stem_index},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    metadata = {
        "run_id": run_id,
        "mode": mode,
        "collection": request.form.get("collection", "Minha coleção").strip() or "Minha coleção",
        "collection_source": "uploaded_folder",
        "e1_protocol": str(options.get("protocol", "semantic")),
        "configuration_source": "production_preset" if mode == "pipeline" else "audit_form",
        "n_images": len(valid_files),
        "has_ground_truth": bool(valid_masks),
        "n_ground_truths": len(valid_masks),
        "gt_index": str(gt_index_path.resolve()) if gt_index_path else None,
        "options": options,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    target = {
        "step1": run_step1_analysis,
        "step2": run_step2_analysis,
        "step3": run_step3_analysis,
        "step4": run_step4_analysis,
    }.get(mode, run_pipeline)
    worker = threading.Thread(target=target, args=(path,), daemon=True)
    worker.start()
    return redirect(url_for("show_run", run_id=run_id))


@app.get("/runs/<run_id>")
def show_run(run_id: str):
    path = run_dir(run_id)
    metadata = read_json(path / "metadata.json")
    return render_template("run.html", run_id=run_id, metadata=metadata)


def _reference_candidates(path: Path, run_id: str) -> list[dict]:
    """Imagens disponíveis pra escolher como referência do scribble — reaproveita
    o mesmo resultado que já é exibido em 'Resultados' (segmented_e5/e4)."""
    return pipeline_results(path, run_id)


@app.get("/runs/<run_id>/correct")
def show_correction(run_id: str):
    path = run_dir(run_id)
    metadata = read_json(path / "metadata.json")
    if metadata.get("mode") != "pipeline":
        abort(404)
    cosaliency_path = path / "output" / "cosaliency.npz"
    if not cosaliency_path.exists():
        return render_template(
            "correct.html", run_id=run_id, metadata=metadata, images=[],
            error="A Etapa 3 (cosaliency.npz) ainda não terminou nesta execução — "
                  "espere o pipeline completar antes de corrigir.",
        )
    images = _reference_candidates(path, run_id)
    return render_template("correct.html", run_id=run_id, metadata=metadata, images=images, error=None)


@app.post("/runs/<run_id>/correct")
def create_correction(run_id: str):
    path = run_dir(run_id)
    metadata = read_json(path / "metadata.json")
    if metadata.get("mode") != "pipeline":
        abort(404)

    reference_image = request.form.get("reference_image", "").strip()
    fg_data_url = request.form.get("fg_scribble_data", "")
    bg_data_url = request.form.get("bg_scribble_data", "")

    if not reference_image:
        return render_template("correct.html", run_id=run_id, metadata=metadata,
                              images=_reference_candidates(path, run_id),
                              error="Escolha uma imagem de referência."), 400
    if not fg_data_url and not bg_data_url:
        return render_template("correct.html", run_id=run_id, metadata=metadata,
                              images=_reference_candidates(path, run_id),
                              error="Desenhe pelo menos uma marcação (FG ou BG) antes de enviar."), 400

    def decode_scribble(data_url: str, label: str) -> bytes:
        match = re.match(r"^data:image/png;base64,(.+)$", data_url or "")
        if not match:
            raise ValueError(f"Não foi possível ler o traço de {label} enviado.")
        try:
            return base64.b64decode(match.group(1))
        except (binascii.Error, ValueError):
            raise ValueError(f"Não foi possível decodificar o traço de {label} enviado.")

    fg_path = bg_path = None
    try:
        if fg_data_url:
            fg_path = path / "scribble_fg.png"
            fg_path.write_bytes(decode_scribble(fg_data_url, "FG"))
        if bg_data_url:
            bg_path = path / "scribble_bg.png"
            bg_path.write_bytes(decode_scribble(bg_data_url, "BG"))
    except ValueError as exc:
        return render_template("correct.html", run_id=run_id, metadata=metadata,
                              images=_reference_candidates(path, run_id),
                              error=str(exc)), 400

    correction_meta = {
        "reference_image": reference_image,
        "has_fg": fg_path is not None,
        "has_bg": bg_path is not None,
    }
    (path / "correction_meta.json").write_text(
        json.dumps(correction_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    worker = threading.Thread(
        target=run_correction,
        args=(path, reference_image, fg_path, bg_path),
        daemon=True,
    )
    worker.start()
    return redirect(url_for("show_run", run_id=run_id) + "#correction")


def _preferred_segmented_dir(base_dir: Path) -> Path | None:
    """Prefere E5, mas só quando ela já contém máscaras exportadas.

    Durante a execução, ``segmented_e5`` pode ser criada antes de todos os
    arquivos finais existirem. Testar apenas ``is_dir()`` faria a API trocar
    cedo demais para uma pasta ainda vazia. Enquanto isso, E4 continua sendo
    um preview válido; assim que a primeira máscara E5 aparece, ela passa a
    ter prioridade.
    """
    for dirname in ("segmented_e5", "segmented_e4"):
        candidate = base_dir / dirname
        if candidate.is_dir() and any(candidate.glob("*_mask.png")):
            return candidate
    return None


def correction_results(path: Path, run_id: str) -> list[dict]:
    result_dir = _preferred_segmented_dir(path / "output" / "correction")
    results = []
    if result_dir is not None:
        for mask in sorted(result_dir.glob("*_mask.png")):
            stem = mask.name.removesuffix("_mask.png")
            original = next((p for p in (path / "input").iterdir() if p.stem == stem), None)
            results.append({
                "name": stem,
                "stage": result_dir.name,
                "original": url_for("run_file", run_id=run_id, filename=f"input/{original.name}") if original else None,
                "mask": url_for("run_file", run_id=run_id, filename=f"output/correction/{result_dir.name}/{mask.name}"),
                "overlay": url_for("run_file", run_id=run_id, filename=f"output/correction/{result_dir.name}/{stem}_overlay.jpg"),
                "cutout": url_for("run_file", run_id=run_id, filename=f"output/correction/{result_dir.name}/{stem}_cutout.png"),
            })
    return results


def correction_evaluation(path: Path, run_id: str) -> dict | None:
    metrics_path = path / "output" / "correction" / "evaluation" / "metrics.json"
    metrics = read_json(metrics_path)
    aggregate = metrics.get("aggregate", {})
    if not aggregate:
        return None

    def mean(metric: str):
        value = aggregate.get(metric, {})
        return value.get("mean") if isinstance(value, dict) else None

    iou = aggregate.get("iou", {})
    return {
        "n_images": aggregate.get("n_images", 0),
        "accuracy": mean("accuracy"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "iou": mean("iou"),
        "iou_std": iou.get("std") if isinstance(iou, dict) else None,
        "metrics_url": url_for("run_file", run_id=run_id, filename="output/correction/evaluation/metrics.json"),
    }


def correction_report(path: Path, run_id: str) -> dict | None:
    report_path = path / "output" / "correction" / "cosaliency_corrected.json"
    payload = read_json(report_path)
    if not payload:
        return None
    correction_dir = path / "output" / "correction"
    debug_filenames = {
        "scribble_fg_raw_debug_url": "scribble_fg_raw_debug.png",
        "scribble_fg_debug_url": "scribble_fg_debug.png",
        "scribble_bg_raw_debug_url": "scribble_bg_raw_debug.png",
        "scribble_bg_debug_url": "scribble_bg_debug.png",
    }
    for url_key, filename in debug_filenames.items():
        if (correction_dir / filename).exists():
            payload[url_key] = url_for("run_file", run_id=run_id, filename=f"output/correction/{filename}")
    return payload


@app.get("/api/runs/<run_id>/correction")
def correction_status(run_id: str):
    path = run_dir(run_id)
    status = read_json(
        path / "correction_status.json",
        {"state": "idle", "step": 0, "total_steps": 0, "progress": 0, "message": ""},
    )
    status["results"] = correction_results(path, run_id)
    status["evaluation"] = correction_evaluation(path, run_id)
    status["report"] = correction_report(path, run_id)
    response = jsonify(status)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response



def output_image_groups(path: Path, run_id: str) -> list[dict]:
    """Lista recursivamente imagens geradas em output, agrupadas por pasta."""
    output_dir = path / "output"
    if not output_dir.is_dir():
        return []

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    grouped: dict[str, list[dict]] = {}

    for image in sorted(
        (file for file in output_dir.rglob("*") if file.is_file() and file.suffix.lower() in image_extensions),
        key=lambda file: file.relative_to(output_dir).as_posix().lower(),
    ):
        relative = image.relative_to(output_dir)
        group_key = relative.parent.as_posix() if relative.parent != Path(".") else "output"
        grouped.setdefault(group_key, []).append({
            "name": image.stem,
            "filename": image.name,
            "relative_path": relative.as_posix(),
            "url": url_for(
                "run_file",
                run_id=run_id,
                filename=f"output/{relative.as_posix()}",
            ),
        })

    return [
        {"name": group_name, "images": images}
        for group_name, images in grouped.items()
    ]

def pipeline_results(path: Path, run_id: str) -> list[dict]:
    result_dir = _preferred_segmented_dir(path / "output")
    results = []
    if result_dir is not None:
        for mask in sorted(result_dir.glob("*_mask.png")):
            stem = mask.name.removesuffix("_mask.png")
            original = next((p for p in (path / "input").iterdir() if p.stem == stem), None)
            results.append({
                "name": stem,
                "stage": result_dir.name,
                "original": url_for("run_file", run_id=run_id, filename=f"input/{original.name}") if original else None,
                "mask": url_for("run_file", run_id=run_id, filename=f"output/{result_dir.name}/{mask.name}"),
                "overlay": url_for("run_file", run_id=run_id, filename=f"output/{result_dir.name}/{stem}_overlay.jpg"),
                "cutout": url_for("run_file", run_id=run_id, filename=f"output/{result_dir.name}/{stem}_cutout.png"),
            })
    return results


def pipeline_evaluation(path: Path, run_id: str) -> dict | None:
    """Expõe o resumo pixel-wise produzido por pipeline/evaluate.py."""
    evaluation_dir = path / "output" / "evaluation"
    metrics_path = evaluation_dir / "metrics.json"
    metrics = read_json(metrics_path)
    aggregate = metrics.get("aggregate", {})
    if not aggregate:
        return None

    def mean(metric: str):
        value = aggregate.get(metric, {})
        return value.get("mean") if isinstance(value, dict) else None

    iou = aggregate.get("iou", {})
    payload = {
        "n_images": aggregate.get("n_images", 0),
        "accuracy": mean("accuracy"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "iou": mean("iou"),
        "iou_std": iou.get("std") if isinstance(iou, dict) else None,
        "metrics_url": url_for(
            "run_file", run_id=run_id, filename="output/evaluation/metrics.json"
        ),
    }
    summary_path = evaluation_dir / "metrics_summary.txt"
    if summary_path.exists():
        payload["summary_url"] = url_for(
            "run_file", run_id=run_id, filename="output/evaluation/metrics_summary.txt"
        )
    return payload


def audit_results(path: Path, run_id: str, mode: str) -> dict | None:
    if mode == "step4":
        quality_dir = path / "output" / "step4_quality"
        quality_file = quality_dir / "step4_quality.json"
        report_name = "step4_quality_report.md"
        plot_names = [
            "viz_sam_confidence.png",
            "viz_e3_vs_e4_comparison.png",
            "viz_fg_fraction.png",
            "viz_gt_iou_comparison.png",
            "viz_per_image_gain.png",
            "viz_gt_overlay.png",
        ]
    elif mode == "step3":
        quality_dir = path / "output" / "step3_quality"
        quality_file = quality_dir / "step3_quality.json"
        report_name = "step3_quality_report.md"
        plot_names = [
            "viz_mask_quality.png",
            "viz_cluster_map.png",
            "viz_coarse_vs_refined.png",
            "viz_fg_fraction.png",
            "viz_saliency_decomposition.png",
            "viz_saliency_spatial_map.png",
            "viz_gt_iou_curve.png",
            "viz_oracle_clusters.png",
            "viz_mapping_accuracy.png",
        ]
    elif mode == "step2":
        quality_dir = path / "output" / "step2_quality"
        quality_file = quality_dir / "step2_quality.json"
        report_name = "step2_quality_report.md"
        plot_names = [
            "viz_score_polarization.png",
            "viz_pair_agreement.png",
            "viz_leave_one_pair_stability.png",
            "viz_fg_fraction_distribution.png",
            "viz_gt_iou_curve.png",
            "viz_orientation_accuracy.png",
            "viz_orientation_heuristics_comparison.png",
        ]
    else:
        quality_dir = path / "output" / "feature_quality"
        quality_file = quality_dir / "feature_quality.json"
        report_name = "feature_quality_report.md"
        plot_names = [
            "cls_similarity_heatmap.png",
            "local_similarity_heatmap.png",
            "cycle_top1_heatmap.png",
            "cycle_reconstruction_heatmap.png",
            "cls_outlier_scores.png",
            "cluster_distribution.png",
            "fg_bg_score_hist.png",
            "oracle_cluster_preview.png",
        ]
    if not quality_file.exists():
        return None
    payload = read_json(quality_file)
    payload["plots"] = [
        {
            "name": name,
            "url": url_for("run_file", run_id=run_id, filename=f"output/{quality_dir.name}/{name}"),
        }
        for name in plot_names if (quality_dir / name).exists()
    ]
    if mode == "step4":
        compare_dir = quality_dir / "segmented_with_gt"
        compare_files = sorted(compare_dir.glob("*_compare.png")) if compare_dir.is_dir() else []
        for compare in compare_files:
            payload["plots"].append({
                "name": f"Comparação: {compare.stem.removesuffix('_compare')}",
                "url": url_for(
                    "run_file",
                    run_id=run_id,
                    filename=f"output/{quality_dir.name}/segmented_with_gt/{compare.name}",
                ),
            })
    payload["report_url"] = url_for("run_file", run_id=run_id, filename=f"output/{quality_dir.name}/{report_name}")
    payload["stage"] = {"step1": 1, "step2": 2, "step3": 3, "step4": 4}.get(mode)
    return payload


@app.get("/api/runs/<run_id>")
def run_status(run_id: str):
    path = run_dir(run_id)
    metadata = read_json(path / "metadata.json")
    status = read_status(path)
    if metadata.get("mode") in {"step1", "step2", "step3", "step4"}:
        status["quality"] = audit_results(path, run_id, metadata["mode"])
    else:
        status["results"] = pipeline_results(path, run_id)
        status["evaluation"] = pipeline_evaluation(path, run_id)

    response = jsonify(status)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/runs/<run_id>/files/<path:filename>")
def run_file(run_id: str, filename: str):
    return send_from_directory(run_dir(run_id), filename)


@app.get("/runs/<run_id>/log")
def run_log(run_id: str):
    path = run_dir(run_id)
    log_file = path / "pipeline.log"
    content = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    return app.response_class(content, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
