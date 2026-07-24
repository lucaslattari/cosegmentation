from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEBAPP_DIR.parent
PIPELINE_DIR = PROJECT_DIR / "pipeline"
ABLATION_DIR = PROJECT_DIR / "ablation"


def write_status(run_dir: Path, state: str, step: int, total_steps: int, message: str) -> None:
    payload = {
        "state": state,
        "step": step,
        "total_steps": total_steps,
        "progress": int(step / total_steps * 100) if total_steps else 0,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = run_dir / "status.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, run_dir / "status.json")


def execute(command: list[str], log_file: Path, cwd: Path) -> None:
    with log_file.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"O comando terminou com código {code}. Consulte pipeline.log.")


def fail_run(run_dir: Path, log_file: Path, total_steps: int, exc: Exception) -> None:
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[ERRO] {exc}\n")
    current = 0
    status_file = run_dir / "status.json"
    if status_file.exists():
        try:
            current = json.loads(status_file.read_text(encoding="utf-8")).get("step", 0)
        except (OSError, json.JSONDecodeError):
            pass
    write_status(run_dir, "failed", current, total_steps, str(exc))


def run_pipeline(run_dir: Path) -> None:
    input_dir = (run_dir / "input").resolve()
    output_dir = (run_dir / "output").resolve()
    log_file = run_dir / "pipeline.log"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    gt_index = metadata.get("gt_index")
    python = sys.executable
    total_steps = 6 if gt_index else 5

    steps = [
        (1, "Extraindo features com DINOv2...", [python, "step1_feature_extraction.py", "--collection", str(input_dir), "--output", str(output_dir), "--model", "dinov2_vits14", "--img_size", "518", "--resize", "keep_ar", "--feature_mode", "l2"]),
        (2, "Calculando correspondências entre imagens...", [python, "step2_correspondence.py", "--features", str(output_dir / "features.npz"), "--output", str(output_dir), "--top_k", "10", "--temperature", "0.05"]),
        (3, "Calculando co-saliência global...", [python, "step3_cosaliency.py", "--features", str(output_dir / "features.npz"), "--correspondence", str(output_dir / "correspondence.npz"), "--output", str(output_dir), "--n_clusters", "5", "--k_neighbors", "10", "--n_spectral", "8", "--no_refine"]),
        (4, "Refinando máscaras com SAM 2...", [python, "step4_sam_refinement.py", "--features", str(output_dir / "features.npz"), "--cosaliency", str(output_dir / "cosaliency.npz"), "--output", str(output_dir), "--model", "facebook/sam2.1-hiera-small", "--prompt_mode", "box_points_mask", "--n_iter", "2", "--export"]),
        (5, "Executando refinamento iterativo...", [python, "step5_iterative.py", "--features", str(output_dir / "features.npz"), "--refined", str(output_dir / "refined_masks.npz"), "--output", str(output_dir), "--model", "facebook/sam2.1-hiera-small", "--prompt_mode", "box_points_mask", "--max_iter", "4", "--converge_thresh", "0.95", "--export"]),
    ]
    if gt_index:
        steps.append((
            6,
            "Avaliando máscaras finais contra o ground truth...",
            [
                python,
                "evaluate.py",
                "--masks_dir", str(output_dir / "segmented_e5"),
                "--gt_index", str(gt_index),
                "--output", str(output_dir / "evaluation"),
                "--collection", metadata.get("collection", "Minha coleção"),
                "--no_flip",
                "--visualize",
            ],
        ))

    try:
        if not PIPELINE_DIR.is_dir():
            raise RuntimeError(f"Pasta pipeline não encontrada em: {PIPELINE_DIR}")
        write_status(run_dir, "running", 0, total_steps, "Execução iniciada...")
        for number, message, command in steps:
            write_status(run_dir, "running", number - 1, total_steps, message)
            execute(command, log_file, PIPELINE_DIR)
            write_status(run_dir, "running", number, total_steps, f"Etapa {number} concluída.")
        final_message = (
            "Cosegmentação e avaliação concluídas!"
            if gt_index else "Cosegmentação concluída!"
        )
        write_status(run_dir, "completed", total_steps, total_steps, final_message)
    except Exception as exc:
        fail_run(run_dir, log_file, total_steps, exc)


def run_correction(run_dir: Path, reference_image: str, scribble_fg: Path | None, scribble_bg: Path | None) -> None:
    """
    Reaproveita features.npz (E1) e cosaliency.npz (E3) já calculados.
    Aplica UMA correção global (scribbles de FG e/ou BG numa única imagem,
    propagados por similaridade de features a todas as outras) e reprocessa
    a partir da Etapa 4 — não recalcula nada de E1-E3.

    Saída em output/correction/ — não sobrescreve o resultado original em
    output/, dá pra comparar os dois lado a lado.
    """
    output_dir = (run_dir / "output").resolve()
    correction_dir = output_dir / "correction"
    correction_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "correction.log"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    gt_index = metadata.get("gt_index")
    python = sys.executable
    total_steps = 4 if gt_index else 3

    features_npz = output_dir / "features.npz"
    cosaliency_npz = output_dir / "cosaliency.npz"
    corrected_npz = correction_dir / "cosaliency_corrected.npz"

    correction_cmd = [
        python, "apply_global_correction.py",
        "--features", str(features_npz),
        "--cosaliency", str(cosaliency_npz),
        "--reference_image", reference_image,
        "--output", str(corrected_npz),
    ]
    if scribble_fg:
        correction_cmd += ["--scribble_mask_fg", str(scribble_fg)]
    if scribble_bg:
        correction_cmd += ["--scribble_mask_bg", str(scribble_bg)]
    sam_cmd = [
        python, "step4_sam_refinement.py",
        "--features", str(features_npz),
        "--cosaliency", str(corrected_npz),
        "--output", str(correction_dir),
        "--model", "facebook/sam2.1-hiera-small",
        "--prompt_mode", "box_points_mask",
        "--n_iter", "2",
        "--export",
    ]
    iterative_cmd = [
        python, "step5_iterative.py",
        "--features", str(features_npz),
        "--refined", str(correction_dir / "refined_masks.npz"),
        "--output", str(correction_dir),
        "--model", "facebook/sam2.1-hiera-small",
        "--prompt_mode", "box_points_mask",
        "--max_iter", "4",
        "--converge_thresh", "0.95",
        "--export",
    ]

    try:
        if not (PIPELINE_DIR / "apply_global_correction.py").is_file():
            raise RuntimeError("Script de correção não encontrado.")
        write_status_correction(run_dir, "running", 0, total_steps, "Aplicando correção global...")
        execute(correction_cmd, log_file, PIPELINE_DIR)
        write_status_correction(run_dir, "running", 1, total_steps, "Reprocessando SAM 2 a partir da Etapa 4...")
        execute(sam_cmd, log_file, PIPELINE_DIR)
        write_status_correction(run_dir, "running", 2, total_steps, "Refinamento iterativo...")
        execute(iterative_cmd, log_file, PIPELINE_DIR)
        if gt_index:
            eval_cmd = [
                python, "evaluate.py",
                "--masks_dir", str(correction_dir / "segmented_e5"),
                "--gt_index", str(gt_index),
                "--output", str(correction_dir / "evaluation"),
                "--collection", metadata.get("collection", "Minha coleção") + " (corrigido)",
                "--no_flip",
            ]
            write_status_correction(run_dir, "running", 3, total_steps, "Avaliando resultado corrigido...")
            execute(eval_cmd, log_file, PIPELINE_DIR)
        write_status_correction(run_dir, "completed", total_steps, total_steps, "Correção aplicada e reprocessada!")
    except Exception as exc:
        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"\n[ERRO] {exc}\n")
        write_status_correction(run_dir, "failed", 0, total_steps, str(exc))


def write_status_correction(run_dir: Path, state: str, step: int, total_steps: int, message: str) -> None:
    payload = {
        "state": state,
        "step": step,
        "total_steps": total_steps,
        "progress": int(step / total_steps * 100) if total_steps else 0,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = run_dir / "correction_status.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, run_dir / "correction_status.json")


def run_step1_analysis(run_dir: Path) -> None:
    input_dir = (run_dir / "input").resolve()
    output_dir = (run_dir / "output").resolve()
    quality_dir = output_dir / "feature_quality"
    log_file = run_dir / "pipeline.log"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    options = metadata.get("options", {})
    python = sys.executable
    total_steps = 2

    extraction = [
        python, "step1_feature_extraction.py",
        "--collection", str(input_dir),
        "--output", str(output_dir),
        "--model", options.get("model", "dinov2_vits14"),
        "--img_size", str(options.get("img_size", 518)),
        "--resize", options.get("resize", "keep_ar"),
        "--feature_mode", options.get("feature_mode", "l2"),
    ]
    analysis = [
        python, str(ABLATION_DIR / "analyze_step1_features.py"),
        "--features", str(output_dir / "features.npz"),
        "--output", str(quality_dir),
    ]
    gt_index = metadata.get("gt_index")
    if gt_index:
        analysis.extend(["--gt_index", str(gt_index)])

    try:
        if not (PIPELINE_DIR / "step1_feature_extraction.py").is_file():
            raise RuntimeError("Script da Etapa 1 não encontrado.")
        if not (ABLATION_DIR / "analyze_step1_features.py").is_file():
            raise RuntimeError("Analisador da Etapa 1 não encontrado.")
        write_status(run_dir, "running", 0, total_steps, "Extraindo features com DINOv2...")
        execute(extraction, log_file, PIPELINE_DIR)
        write_status(run_dir, "running", 1, total_steps, "Calculando métricas e gerando diagnóstico...")
        execute(analysis, log_file, PROJECT_DIR)
        write_status(run_dir, "completed", 2, total_steps, "Auditoria da Etapa 1 concluída!")
    except Exception as exc:
        fail_run(run_dir, log_file, total_steps, exc)


def run_step2_analysis(run_dir: Path) -> None:
    input_dir = (run_dir / "input").resolve()
    output_dir = (run_dir / "output").resolve()
    quality_dir = output_dir / "step2_quality"
    log_file = run_dir / "pipeline.log"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    options = metadata.get("options", {})
    python = sys.executable
    total_steps = 3

    extraction = [
        python, "step1_feature_extraction.py",
        "--collection", str(input_dir), "--output", str(output_dir),
        "--model", options.get("model", "dinov2_vits14"),
        "--img_size", str(options.get("img_size", 518)),
        "--resize", options.get("resize", "keep_ar"),
        "--feature_mode", options.get("feature_mode", "l2"),
    ]
    correspondence = [
        python, "step2_correspondence.py",
        "--features", str(output_dir / "features.npz"),
        "--output", str(output_dir),
        "--top_k", str(options.get("top_k", 10)),
        "--temperature", str(options.get("temperature", 0.05)),
    ]
    analysis = [
        python, str(ABLATION_DIR / "analyze_step2_correspondence.py"),
        "--correspondence", str(output_dir / "correspondence.npz"),
        "--features", str(output_dir / "features.npz"),
        "--output", str(quality_dir),
    ]
    gt_index = metadata.get("gt_index")
    if gt_index:
        analysis.extend(["--gt_index", str(gt_index)])

    try:
        required = [
            PIPELINE_DIR / "step1_feature_extraction.py",
            PIPELINE_DIR / "step2_correspondence.py",
            ABLATION_DIR / "analyze_step2_correspondence.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Scripts necessários não encontrados: " + ", ".join(missing))
        write_status(run_dir, "running", 0, total_steps, "Extraindo features com DINOv2...")
        execute(extraction, log_file, PIPELINE_DIR)
        write_status(run_dir, "running", 1, total_steps, "Calculando correspondências cross-image...")
        execute(correspondence, log_file, PIPELINE_DIR)
        write_status(run_dir, "running", 2, total_steps, "Calculando métricas e diagnosticando a Etapa 2...")
        execute(analysis, log_file, PROJECT_DIR)
        write_status(run_dir, "completed", 3, total_steps, "Auditoria da Etapa 2 concluída!")
    except Exception as exc:
        fail_run(run_dir, log_file, total_steps, exc)


def run_step3_analysis(run_dir: Path) -> None:
    input_dir = (run_dir / "input").resolve()
    output_dir = (run_dir / "output").resolve()
    quality_dir = output_dir / "step3_quality"
    log_file = run_dir / "pipeline.log"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    options = metadata.get("options", {})
    python = sys.executable
    total_steps = 4

    extraction = [
        python, "step1_feature_extraction.py",
        "--collection", str(input_dir), "--output", str(output_dir),
        "--model", options.get("model", "dinov2_vits14"),
        "--img_size", str(options.get("img_size", 518)),
        "--resize", options.get("resize", "keep_ar"),
        "--feature_mode", options.get("feature_mode", "l2"),
    ]
    correspondence = [
        python, "step2_correspondence.py",
        "--features", str(output_dir / "features.npz"), "--output", str(output_dir),
        "--top_k", str(options.get("top_k", 10)),
        "--temperature", str(options.get("temperature", 0.05)),
    ]
    cosaliency = [
        python, "step3_cosaliency.py",
        "--features", str(output_dir / "features.npz"),
        "--correspondence", str(output_dir / "correspondence.npz"),
        "--output", str(output_dir),
        "--k_neighbors", str(options.get("k_neighbors", 10)),
        "--n_spectral", str(options.get("n_spectral", 8)),
        "--n_clusters", str(options.get("n_clusters", 5)),
        "--fg_threshold", str(options.get("fg_threshold", 0.5)),
        "--saliency_alpha", str(options.get("saliency_alpha", 0.5)),
    ]
    cosaliency.append("--refine" if options.get("refine", False) else "--no_refine")
    if not options.get("adaptive_clusters", True):
        cosaliency.append("--no_adaptive_clusters")

    analysis = [
        python, str(ABLATION_DIR / "analyze_step3_cosaliency.py"),
        "--cosaliency", str(output_dir / "cosaliency.npz"),
        "--features", str(output_dir / "features.npz"),
        "--output", str(quality_dir),
    ]
    gt_index = metadata.get("gt_index")
    if gt_index:
        analysis.extend(["--gt_index", str(gt_index)])

    try:
        required = [
            PIPELINE_DIR / "step1_feature_extraction.py",
            PIPELINE_DIR / "step2_correspondence.py",
            PIPELINE_DIR / "step3_cosaliency.py",
            ABLATION_DIR / "analyze_step3_cosaliency.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Scripts necessários não encontrados: " + ", ".join(missing))
        messages = [
            "Extraindo features com DINOv2...",
            "Calculando correspondências cross-image...",
            "Calculando co-saliência e clustering espectral...",
            "Calculando métricas e diagnosticando a Etapa 3...",
        ]
        commands = [extraction, correspondence, cosaliency, analysis]
        write_status(run_dir, "running", 0, total_steps, messages[0])
        for index, (message, command) in enumerate(zip(messages, commands)):
            write_status(run_dir, "running", index, total_steps, message)
            execute(command, log_file, PIPELINE_DIR if index < 3 else PROJECT_DIR)
        write_status(run_dir, "completed", total_steps, total_steps, "Auditoria da Etapa 3 concluída!")
    except Exception as exc:
        fail_run(run_dir, log_file, total_steps, exc)


def run_step4_analysis(run_dir: Path) -> None:
    input_dir = (run_dir / "input").resolve()
    output_dir = (run_dir / "output").resolve()
    quality_dir = output_dir / "step4_quality"
    log_file = run_dir / "pipeline.log"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    options = metadata.get("options", {})
    python = sys.executable
    total_steps = 5

    extraction = [
        python, "step1_feature_extraction.py",
        "--collection", str(input_dir), "--output", str(output_dir),
        "--model", options.get("model", "dinov2_vits14"),
        "--img_size", str(options.get("img_size", 518)),
        "--resize", options.get("resize", "keep_ar"),
        "--feature_mode", options.get("feature_mode", "l2"),
    ]
    correspondence = [
        python, "step2_correspondence.py",
        "--features", str(output_dir / "features.npz"), "--output", str(output_dir),
        "--top_k", str(options.get("top_k", 10)),
        "--temperature", str(options.get("temperature", 0.05)),
    ]
    cosaliency = [
        python, "step3_cosaliency.py",
        "--features", str(output_dir / "features.npz"),
        "--correspondence", str(output_dir / "correspondence.npz"),
        "--output", str(output_dir),
        "--k_neighbors", str(options.get("k_neighbors", 10)),
        "--n_spectral", str(options.get("n_spectral", 8)),
        "--n_clusters", str(options.get("n_clusters", 5)),
        "--fg_threshold", str(options.get("fg_threshold", 0.5)),
        "--saliency_alpha", str(options.get("saliency_alpha", 0.5)),
    ]
    cosaliency.append("--refine" if options.get("refine", False) else "--no_refine")
    if not options.get("adaptive_clusters", True):
        cosaliency.append("--no_adaptive_clusters")

    sam = [
        python, "step4_sam_refinement.py",
        "--features", str(output_dir / "features.npz"),
        "--cosaliency", str(output_dir / "cosaliency.npz"),
        "--output", str(output_dir),
        "--model", options.get("sam_model", "facebook/sam2.1-hiera-small"),
        "--prompt_mode", options.get("prompt_mode", "box_points_mask"),
        "--n_iter", str(options.get("sam_n_iter", 2)),
        "--export",
    ]
    analysis = [
        python, str(ABLATION_DIR / "analyze_step4_sam_refinement.py"),
        "--refined", str(output_dir / "refined_masks.npz"),
        "--cosaliency", str(output_dir / "cosaliency.npz"),
        "--output", str(quality_dir),
    ]
    gt_index = metadata.get("gt_index")
    if gt_index:
        analysis.extend(["--gt_index", str(gt_index)])

    try:
        required = [
            PIPELINE_DIR / "step1_feature_extraction.py",
            PIPELINE_DIR / "step2_correspondence.py",
            PIPELINE_DIR / "step3_cosaliency.py",
            PIPELINE_DIR / "step4_sam_refinement.py",
            ABLATION_DIR / "analyze_step4_sam_refinement.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Scripts necessários não encontrados: " + ", ".join(missing))

        messages = [
            "Extraindo features com DINOv2...",
            "Calculando correspondências cross-image...",
            "Calculando co-saliência e clustering espectral...",
            "Refinando as máscaras com SAM 2...",
            "Calculando métricas e diagnosticando a Etapa 4...",
        ]
        commands = [extraction, correspondence, cosaliency, sam, analysis]
        write_status(run_dir, "running", 0, total_steps, messages[0])
        for index, (message, command) in enumerate(zip(messages, commands)):
            write_status(run_dir, "running", index, total_steps, message)
            execute(command, log_file, PIPELINE_DIR if index < 4 else PROJECT_DIR)
        write_status(run_dir, "completed", total_steps, total_steps, "Auditoria da Etapa 4 concluída!")
    except Exception as exc:
        fail_run(run_dir, log_file, total_steps, exc)
