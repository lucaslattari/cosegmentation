from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEBAPP_DIR.parent
PIPELINE_DIR = PROJECT_DIR / "pipeline"
ABLATION_DIR = PROJECT_DIR / "ablation"

# -----------------------------------------------------------------------------
# Configuração de produção
# -----------------------------------------------------------------------------
# Mantida em um único lugar dentro do webapp para o modo Pipeline completo e
# para a correção global não divergirem entre si. Os valores espelham o preset
# validado experimentalmente, mas pipeline/run_pipeline.sh permanece independente.
PRODUCTION = {
    # No webapp a coleção já chega resolvida como uma pasta enviada pelo usuário.
    # Portanto, o protocolo de E1 é o genérico/semântico; protocolos como
    # icoseg38 continuam pertencendo ao orquestrador experimental em shell.
    "e1_protocol": "semantic",
    "e1_model": "dinov2_vits14",
    "e1_img_size": 518,
    "e1_resize": "keep_ar",
    "e1_feature_mode": "l2",
    "e2_top_k": 250,
    "e2_temperature": 0.05,
    "e2_orientation": "spatial",
    "e2_degree_alpha": 0.10,
    "e2b_mode": "recurrence",
    "e2b_lambda": 0.50,
    "e2b_gamma": 0.00,
    "e2b_border_width": 2,
    "e2b_top_fg_fraction": 0.10,
    "e3_n_clusters": 8,
    "e3_k_neighbors": 15,
    "e3_n_spectral": 8,
    "e3_saliency_alpha": 0.5,
    "e3_fg_decision_mode": "saliency",
    "e3_empty_fg_bypass": "e2_topk",
    "e3_empty_fg_top_fraction": 0.20,
    "e4_model": "facebook/sam2.1-hiera-large",
    "e4_prompt_mode": "box_points_mask",
    "e4_n_iter": 2,
    "e5_max_iter": 20,
    "e5_converge_thresh": 0.99,
    "e5_n_fg_pts": 8,
    "e5_n_bg_pts": 8,
}


def write_status(
    run_dir: Path,
    state: str,
    step: int,
    total_steps: int,
    message: str,
    **extra,
) -> None:
    payload = {
        "state": state,
        "step": step,
        "total_steps": total_steps,
        "progress": int(step / total_steps * 100) if total_steps else 0,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(extra)
    tmp = run_dir / "status.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, run_dir / "status.json")


def write_status_correction(
    run_dir: Path,
    state: str,
    step: int,
    total_steps: int,
    message: str,
    **extra,
) -> None:
    payload = {
        "state": state,
        "step": step,
        "total_steps": total_steps,
        "progress": int(step / total_steps * 100) if total_steps else 0,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(extra)
    tmp = run_dir / "correction_status.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, run_dir / "correction_status.json")


def execute(command: list[str], log_file: Path, cwd: Path) -> None:
    with log_file.open("a", encoding="utf-8") as log:
        log.write("\n$ " + shlex.join(command) + "\n")
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


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Scripts necessários não encontrados: " + ", ".join(missing))


def _validate_npz(path: Path, label: str) -> None:
    """Valida um NPZ de forma semelhante ao shell, importando numpy só no run."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} não foi gerado ou está vazio: {path}")
    try:
        import numpy as np

        with np.load(path, allow_pickle=True) as data:
            if not data.files:
                raise RuntimeError(f"{label} é um NPZ vazio: {path}")
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith(label):
            raise
        raise RuntimeError(f"{label} não é um NPZ válido: {path}") from exc


def _validate_step3_output(path: Path, options: dict) -> None:
    """Além do NPZ, confere o metadata do fallback empty-FG usado na E3."""
    _validate_npz(path, "Saída da E3")
    desired_mode = str(options.get("empty_fg_bypass", PRODUCTION["e3_empty_fg_bypass"]))
    desired_top = float(options.get("empty_fg_top_fraction", PRODUCTION["e3_empty_fg_top_fraction"]))
    try:
        import numpy as np

        with np.load(path, allow_pickle=True) as data:
            files = set(data.files)
            if "empty_fg_bypass_mode" not in files:
                if desired_mode != "none":
                    raise RuntimeError(
                        "A saída da E3 não registra empty_fg_bypass_mode, embora o bypass esteja habilitado."
                    )
                return
            mode = str(data["empty_fg_bypass_mode"].item())
            top = float(data["empty_fg_top_fraction"].item()) if "empty_fg_top_fraction" in files else 0.20
            if mode != desired_mode:
                raise RuntimeError(
                    f"A E3 registrou empty_fg_bypass_mode={mode}, mas o run exige {desired_mode}."
                )
            if desired_mode == "e2_topk" and abs(top - desired_top) > 1e-8:
                raise RuntimeError(
                    f"A E3 registrou empty_fg_top_fraction={top}, mas o run exige {desired_top}."
                )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Não foi possível validar o metadata da E3: {path}") from exc


def _validate_mask_dir(path: Path, label: str) -> None:
    if not path.is_dir() or not any(path.glob("*_mask.png")):
        raise RuntimeError(f"{label} não contém máscaras exportadas: {path}")


def _validate_pipeline_output(step: str, output_dir: Path, options: dict) -> None:
    expected_npz = {
        "E1": "features.npz",
        "E2": "correspondence.npz",
        "E2B": "correspondence_bg.npz",
        "E3": "cosaliency.npz",
        "E4": "refined_masks.npz",
        "E5": "iterative_masks.npz",
    }
    filename = expected_npz.get(step)
    if filename is None:
        raise RuntimeError(f"Validador não definido para a etapa {step}.")
    if step == "E3":
        _validate_step3_output(output_dir / filename, options)
    else:
        _validate_npz(output_dir / filename, f"Saída da {step}")
    if step == "E4":
        _validate_mask_dir(output_dir / "segmented_e4", "Exportação da E4")
    elif step == "E5":
        _validate_mask_dir(output_dir / "segmented_e5", "Exportação da E5")


def _step1_command(python: str, input_dir: Path, output_dir: Path, options: dict) -> list[str]:
    return [
        python,
        "step1_feature_extraction.py",
        "--collection", str(input_dir),
        "--protocol", str(options.get("protocol", PRODUCTION["e1_protocol"])),
        "--output", str(output_dir),
        "--model", str(options.get("model", PRODUCTION["e1_model"])),
        "--img_size", str(options.get("img_size", PRODUCTION["e1_img_size"])),
        "--resize", str(options.get("resize", PRODUCTION["e1_resize"])),
        "--feature_mode", str(options.get("feature_mode", PRODUCTION["e1_feature_mode"])),
    ]


def _step2_command(python: str, output_dir: Path, options: dict) -> list[str]:
    return [
        python,
        "step2_correspondence.py",
        "--features", str(output_dir / "features.npz"),
        "--output", str(output_dir),
        "--top_k", str(options.get("top_k", PRODUCTION["e2_top_k"])),
        "--temperature", str(options.get("temperature", PRODUCTION["e2_temperature"])),
        "--orientation", str(options.get("orientation", PRODUCTION["e2_orientation"])),
        "--degree_alpha", str(options.get("degree_alpha", PRODUCTION["e2_degree_alpha"])),
    ]


def _step2b_command(python: str, output_dir: Path, options: dict) -> list[str] | None:
    mode = str(options.get("bg_suppression", PRODUCTION["e2b_mode"]))
    if mode == "none":
        return None
    return [
        python,
        "step2b_background_suppression.py",
        "--features", str(output_dir / "features.npz"),
        "--correspondence", str(output_dir / "correspondence.npz"),
        "--output", str(output_dir),
        "--mode", mode,
        "--lambda-bg", str(options.get("bg_lambda", PRODUCTION["e2b_lambda"])),
        "--gamma", str(options.get("bg_gamma", PRODUCTION["e2b_gamma"])),
        "--border-width", str(options.get("bg_border_width", PRODUCTION["e2b_border_width"])),
        "--top-fg-fraction", str(options.get("bg_top_fg_fraction", PRODUCTION["e2b_top_fg_fraction"])),
    ]


def _correspondence_for_e3(output_dir: Path, options: dict) -> Path:
    if str(options.get("bg_suppression", PRODUCTION["e2b_mode"])) == "none":
        return output_dir / "correspondence.npz"
    return output_dir / "correspondence_bg.npz"


def _step3_command(python: str, output_dir: Path, options: dict) -> list[str]:
    return [
        python,
        "step3_cosaliency.py",
        "--features", str(output_dir / "features.npz"),
        "--correspondence", str(_correspondence_for_e3(output_dir, options)),
        "--output", str(output_dir),
        "--n_clusters", str(options.get("n_clusters", PRODUCTION["e3_n_clusters"])),
        "--k_neighbors", str(options.get("k_neighbors", PRODUCTION["e3_k_neighbors"])),
        "--n_spectral", str(options.get("n_spectral", PRODUCTION["e3_n_spectral"])),
        "--saliency_alpha", str(options.get("saliency_alpha", PRODUCTION["e3_saliency_alpha"])),
        "--fg_decision_mode", str(options.get("fg_decision_mode", PRODUCTION["e3_fg_decision_mode"])),
        "--empty_fg_bypass", str(options.get("empty_fg_bypass", PRODUCTION["e3_empty_fg_bypass"])),
        "--empty_fg_top_fraction", str(options.get("empty_fg_top_fraction", PRODUCTION["e3_empty_fg_top_fraction"])),
    ]


def _step4_command(python: str, output_dir: Path, options: dict, *, cosaliency: Path | None = None) -> list[str]:
    return [
        python,
        "step4_sam_refinement.py",
        "--features", str(output_dir / "features.npz"),
        "--cosaliency", str(cosaliency or (output_dir / "cosaliency.npz")),
        "--output", str(output_dir),
        "--model", str(options.get("sam_model", PRODUCTION["e4_model"])),
        "--prompt_mode", str(options.get("prompt_mode", PRODUCTION["e4_prompt_mode"])),
        "--n_iter", str(options.get("sam_n_iter", PRODUCTION["e4_n_iter"])),
        "--export",
    ]


def _step5_command(python: str, output_dir: Path, options: dict) -> list[str]:
    return [
        python,
        "step5_iterative.py",
        "--features", str(output_dir / "features.npz"),
        "--refined", str(output_dir / "refined_masks.npz"),
        "--output", str(output_dir),
        "--model", str(options.get("sam_model", PRODUCTION["e4_model"])),
        "--prompt_mode", str(options.get("prompt_mode", PRODUCTION["e4_prompt_mode"])),
        "--max_iter", str(options.get("max_iter", PRODUCTION["e5_max_iter"])),
        "--converge_thresh", str(options.get("converge_thresh", PRODUCTION["e5_converge_thresh"])),
        "--n_fg_pts", str(options.get("n_fg_pts", PRODUCTION["e5_n_fg_pts"])),
        "--n_bg_pts", str(options.get("n_bg_pts", PRODUCTION["e5_n_bg_pts"])),
        "--no_mask_history",
        "--export",
    ]


def production_options() -> dict:
    """Retorna um snapshot completo do preset de produção usado pelo webapp."""
    return {
        "protocol": PRODUCTION["e1_protocol"],
        "model": PRODUCTION["e1_model"],
        "img_size": PRODUCTION["e1_img_size"],
        "resize": PRODUCTION["e1_resize"],
        "feature_mode": PRODUCTION["e1_feature_mode"],
        "top_k": PRODUCTION["e2_top_k"],
        "temperature": PRODUCTION["e2_temperature"],
        "orientation": PRODUCTION["e2_orientation"],
        "degree_alpha": PRODUCTION["e2_degree_alpha"],
        "bg_suppression": PRODUCTION["e2b_mode"],
        "bg_lambda": PRODUCTION["e2b_lambda"],
        "bg_gamma": PRODUCTION["e2b_gamma"],
        "bg_border_width": PRODUCTION["e2b_border_width"],
        "bg_top_fg_fraction": PRODUCTION["e2b_top_fg_fraction"],
        "n_clusters": PRODUCTION["e3_n_clusters"],
        "k_neighbors": PRODUCTION["e3_k_neighbors"],
        "n_spectral": PRODUCTION["e3_n_spectral"],
        "saliency_alpha": PRODUCTION["e3_saliency_alpha"],
        "fg_decision_mode": PRODUCTION["e3_fg_decision_mode"],
        "empty_fg_bypass": PRODUCTION["e3_empty_fg_bypass"],
        "empty_fg_top_fraction": PRODUCTION["e3_empty_fg_top_fraction"],
        "sam_model": PRODUCTION["e4_model"],
        "prompt_mode": PRODUCTION["e4_prompt_mode"],
        "sam_n_iter": PRODUCTION["e4_n_iter"],
        "max_iter": PRODUCTION["e5_max_iter"],
        "converge_thresh": PRODUCTION["e5_converge_thresh"],
        "n_fg_pts": PRODUCTION["e5_n_fg_pts"],
        "n_bg_pts": PRODUCTION["e5_n_bg_pts"],
    }


def _options_from_metadata(metadata: dict) -> dict:
    """Mescla defaults atuais com o snapshot salvo no início do run."""
    options = production_options()
    saved = metadata.get("options")
    if isinstance(saved, dict):
        options.update(saved)
    return options


def run_pipeline(run_dir: Path) -> None:
    input_dir = (run_dir / "input").resolve()
    output_dir = (run_dir / "output").resolve()
    log_file = run_dir / "pipeline.log"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    gt_index = metadata.get("gt_index")
    python = sys.executable

    # O app salva o preset no metadata antes de iniciar a thread. Usar esse
    # snapshot garante que a execução corresponda exatamente ao que ficou
    # registrado naquele run, mesmo que PRODUCTION mude posteriormente.
    options = _options_from_metadata(metadata)

    core_steps: list[tuple[str, str, list[str] | None, Path]] = [
        ("E1", "Extraindo features com DINOv2...", _step1_command(python, input_dir, output_dir, options), PIPELINE_DIR),
        ("E2", "Calculando correspondências entre imagens...", _step2_command(python, output_dir, options), PIPELINE_DIR),
        ("E2B", "Suprimindo background recorrente (E2B)...", _step2b_command(python, output_dir, options), PIPELINE_DIR),
        ("E3", "Calculando co-saliência global...", _step3_command(python, output_dir, options), PIPELINE_DIR),
        ("E4", "Refinando máscaras com SAM 2...", _step4_command(python, output_dir, options), PIPELINE_DIR),
        ("E5", "Executando refinamento iterativo...", _step5_command(python, output_dir, options), PIPELINE_DIR),
    ]
    core_steps = [item for item in core_steps if item[2] is not None]

    core_total = len(core_steps)
    total_steps = core_total + (1 if gt_index else 0)
    try:
        _require_files([
            PIPELINE_DIR / "step1_feature_extraction.py",
            PIPELINE_DIR / "step2_correspondence.py",
            PIPELINE_DIR / "step3_cosaliency.py",
            PIPELINE_DIR / "step4_sam_refinement.py",
            PIPELINE_DIR / "step5_iterative.py",
        ] + (
            [PIPELINE_DIR / "step2b_background_suppression.py"]
            if str(options.get("bg_suppression", PRODUCTION["e2b_mode"])) != "none"
            else []
        ) + ([PIPELINE_DIR / "evaluate.py"] if gt_index else []))

        write_status(
            run_dir,
            "running",
            0,
            total_steps,
            "Execução iniciada sobre a coleção enviada, com o preset de produção registrado no run...",
            evaluation_state="pending" if gt_index else "not_requested",
        )

        for index, (step_name, message, command, cwd) in enumerate(core_steps, start=1):
            assert command is not None
            write_status(
                run_dir, "running", index - 1, total_steps, message,
                evaluation_state="pending" if gt_index else "not_requested",
            )
            execute(command, log_file, cwd)
            _validate_pipeline_output(step_name, output_dir, options)
            write_status(
                run_dir, "running", index, total_steps, f"{step_name} concluída e validada.",
                evaluation_state="pending" if gt_index else "not_requested",
            )

        if gt_index:
            eval_cmd = [
                python,
                "evaluate.py",
                "--masks_dir", str(output_dir / "segmented_e5"),
                "--gt_index", str(gt_index),
                "--output", str(output_dir / "evaluation"),
                "--collection", metadata.get("collection", "Minha coleção"),
                "--visualize",
            ]
            write_status(
                run_dir, "running", core_total, total_steps,
                "Avaliando máscaras finais contra o ground truth...",
                evaluation_state="running",
            )
            try:
                execute(eval_cmd, log_file, PIPELINE_DIR)
            except Exception as eval_exc:
                warning = f"A cosegmentação E1–E5 foi concluída, mas a avaliação com GT falhou: {eval_exc}"
                with log_file.open("a", encoding="utf-8") as log:
                    log.write(f"\n[AVISO] {warning}\n")
                # A aplicação principal terminou corretamente. Para não marcar
                # visualmente a avaliação como concluída, fechamos o run com
                # seis etapas de seis e registramos a falha separadamente.
                write_status(
                    run_dir, "completed", core_total, core_total, warning,
                    evaluation_state="failed", warning=str(eval_exc),
                )
                return

            write_status(
                run_dir, "completed", total_steps, total_steps,
                "Cosegmentação e avaliação concluídas!",
                evaluation_state="completed",
            )
            return

        write_status(
            run_dir, "completed", core_total, core_total, "Cosegmentação concluída!",
            evaluation_state="not_requested",
        )
    except Exception as exc:
        fail_run(run_dir, log_file, total_steps, exc)


def run_correction(run_dir: Path, reference_image: str, scribble_fg: Path | None, scribble_bg: Path | None) -> None:
    """Reaproveita E1/E3, corrige a co-saliência e reroda E4/E5 com os defaults de produção."""
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

    options = _options_from_metadata(metadata)
    sam_cmd = [
        python, "step4_sam_refinement.py",
        "--features", str(features_npz),
        "--cosaliency", str(corrected_npz),
        "--output", str(correction_dir),
        "--model", str(options.get("sam_model", PRODUCTION["e4_model"])),
        "--prompt_mode", str(options.get("prompt_mode", PRODUCTION["e4_prompt_mode"])),
        "--n_iter", str(options.get("sam_n_iter", PRODUCTION["e4_n_iter"])),
        "--export",
    ]
    iterative_cmd = [
        python, "step5_iterative.py",
        "--features", str(features_npz),
        "--refined", str(correction_dir / "refined_masks.npz"),
        "--output", str(correction_dir),
        "--model", str(options.get("sam_model", PRODUCTION["e4_model"])),
        "--prompt_mode", str(options.get("prompt_mode", PRODUCTION["e4_prompt_mode"])),
        "--max_iter", str(options.get("max_iter", PRODUCTION["e5_max_iter"])),
        "--converge_thresh", str(options.get("converge_thresh", PRODUCTION["e5_converge_thresh"])),
        "--n_fg_pts", str(options.get("n_fg_pts", PRODUCTION["e5_n_fg_pts"])),
        "--n_bg_pts", str(options.get("n_bg_pts", PRODUCTION["e5_n_bg_pts"])),
        "--no_mask_history",
        "--export",
    ]

    try:
        _require_files([
            PIPELINE_DIR / "apply_global_correction.py",
            PIPELINE_DIR / "step4_sam_refinement.py",
            PIPELINE_DIR / "step5_iterative.py",
        ] + ([PIPELINE_DIR / "evaluate.py"] if gt_index else []))
        write_status_correction(
            run_dir, "running", 0, total_steps, "Aplicando correção global...",
            evaluation_state="pending" if gt_index else "not_requested",
        )
        execute(correction_cmd, log_file, PIPELINE_DIR)
        _validate_npz(corrected_npz, "Co-saliência corrigida")

        write_status_correction(
            run_dir, "running", 1, total_steps, "Reprocessando SAM 2 com o preset registrado no run...",
            evaluation_state="pending" if gt_index else "not_requested",
        )
        execute(sam_cmd, log_file, PIPELINE_DIR)
        _validate_npz(correction_dir / "refined_masks.npz", "Saída corrigida da E4")
        _validate_mask_dir(correction_dir / "segmented_e4", "Exportação corrigida da E4")

        write_status_correction(
            run_dir, "running", 2, total_steps, "Refinamento iterativo com o preset registrado no run...",
            evaluation_state="pending" if gt_index else "not_requested",
        )
        execute(iterative_cmd, log_file, PIPELINE_DIR)
        _validate_npz(correction_dir / "iterative_masks.npz", "Saída corrigida da E5")
        _validate_mask_dir(correction_dir / "segmented_e5", "Exportação corrigida da E5")

        if gt_index:
            eval_cmd = [
                python, "evaluate.py",
                "--masks_dir", str(correction_dir / "segmented_e5"),
                "--gt_index", str(gt_index),
                "--output", str(correction_dir / "evaluation"),
                "--collection", metadata.get("collection", "Minha coleção") + " (corrigido)",
            ]
            write_status_correction(
                run_dir, "running", 3, total_steps, "Avaliando resultado corrigido...",
                evaluation_state="running",
            )
            try:
                execute(eval_cmd, log_file, PIPELINE_DIR)
            except Exception as eval_exc:
                warning = f"A correção e o reprocessamento E4/E5 foram concluídos, mas a avaliação falhou: {eval_exc}"
                with log_file.open("a", encoding="utf-8") as log:
                    log.write(f"\n[AVISO] {warning}\n")
                write_status_correction(
                    run_dir, "completed", 3, 3, warning,
                    evaluation_state="failed", warning=str(eval_exc),
                )
                return

        write_status_correction(
            run_dir, "completed", total_steps, total_steps, "Correção aplicada e reprocessada!",
            evaluation_state="completed" if gt_index else "not_requested",
        )
    except Exception as exc:
        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"\n[ERRO] {exc}\n")
        write_status_correction(run_dir, "failed", 0, total_steps, str(exc))


def run_step1_analysis(run_dir: Path) -> None:
    input_dir = (run_dir / "input").resolve()
    output_dir = (run_dir / "output").resolve()
    quality_dir = output_dir / "feature_quality"
    log_file = run_dir / "pipeline.log"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    options = metadata.get("options", {})
    python = sys.executable
    total_steps = 2

    extraction = _step1_command(python, input_dir, output_dir, options)
    analysis = [
        python, str(ABLATION_DIR / "analyze_step1_features.py"),
        "--features", str(output_dir / "features.npz"),
        "--output", str(quality_dir),
    ]
    gt_index = metadata.get("gt_index")
    if gt_index:
        analysis.extend(["--gt_index", str(gt_index)])

    try:
        _require_files([PIPELINE_DIR / "step1_feature_extraction.py", ABLATION_DIR / "analyze_step1_features.py"])
        commands = [
            ("Extraindo features com DINOv2...", extraction, PIPELINE_DIR),
            ("Calculando métricas e gerando diagnóstico...", analysis, PROJECT_DIR),
        ]
        write_status(run_dir, "running", 0, total_steps, commands[0][0])
        for index, (message, command, cwd) in enumerate(commands, start=1):
            write_status(run_dir, "running", index - 1, total_steps, message)
            execute(command, log_file, cwd)
        write_status(run_dir, "completed", total_steps, total_steps, "Auditoria da Etapa 1 concluída!")
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

    extraction = _step1_command(python, input_dir, output_dir, options)
    correspondence = _step2_command(python, output_dir, options)
    analysis = [
        python, str(ABLATION_DIR / "analyze_step2_correspondence.py"),
        "--correspondence", str(output_dir / "correspondence.npz"),
        "--features", str(output_dir / "features.npz"),
        "--output", str(quality_dir),
    ]
    gt_index = metadata.get("gt_index")
    if gt_index:
        analysis.extend(["--gt_index", str(gt_index)])

    commands = [
        ("Extraindo features com DINOv2...", extraction, PIPELINE_DIR),
        ("Calculando correspondências cross-image...", correspondence, PIPELINE_DIR),
        ("Calculando métricas e diagnosticando a Etapa 2...", analysis, PROJECT_DIR),
    ]
    total_steps = len(commands)
    try:
        _require_files([
            PIPELINE_DIR / "step1_feature_extraction.py",
            PIPELINE_DIR / "step2_correspondence.py",
            ABLATION_DIR / "analyze_step2_correspondence.py",
        ])
        write_status(run_dir, "running", 0, total_steps, commands[0][0])
        for index, (message, command, cwd) in enumerate(commands, start=1):
            write_status(run_dir, "running", index - 1, total_steps, message)
            execute(command, log_file, cwd)
        write_status(run_dir, "completed", total_steps, total_steps, "Auditoria da Etapa 2 concluída!")
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

    commands: list[tuple[str, list[str], Path]] = [
        ("Extraindo features com DINOv2...", _step1_command(python, input_dir, output_dir, options), PIPELINE_DIR),
        ("Calculando correspondências cross-image...", _step2_command(python, output_dir, options), PIPELINE_DIR),
    ]
    step2b = _step2b_command(python, output_dir, options)
    if step2b:
        commands.append(("Aplicando supressão adaptativa de background (E2B)...", step2b, PIPELINE_DIR))
    commands.append(("Calculando co-saliência e clustering espectral...", _step3_command(python, output_dir, options), PIPELINE_DIR))

    analysis = [
        python, str(ABLATION_DIR / "analyze_step3_cosaliency.py"),
        "--cosaliency", str(output_dir / "cosaliency.npz"),
        "--features", str(output_dir / "features.npz"),
        "--output", str(quality_dir),
    ]
    gt_index = metadata.get("gt_index")
    if gt_index:
        analysis.extend(["--gt_index", str(gt_index)])
    commands.append(("Calculando métricas e diagnosticando a Etapa 3...", analysis, PROJECT_DIR))

    total_steps = len(commands)
    try:
        required = [
            PIPELINE_DIR / "step1_feature_extraction.py",
            PIPELINE_DIR / "step2_correspondence.py",
            PIPELINE_DIR / "step3_cosaliency.py",
            ABLATION_DIR / "analyze_step3_cosaliency.py",
        ]
        if step2b:
            required.append(PIPELINE_DIR / "step2b_background_suppression.py")
        _require_files(required)
        write_status(run_dir, "running", 0, total_steps, commands[0][0])
        for index, (message, command, cwd) in enumerate(commands, start=1):
            write_status(run_dir, "running", index - 1, total_steps, message)
            execute(command, log_file, cwd)
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

    commands: list[tuple[str, list[str], Path]] = [
        ("Extraindo features com DINOv2...", _step1_command(python, input_dir, output_dir, options), PIPELINE_DIR),
        ("Calculando correspondências cross-image...", _step2_command(python, output_dir, options), PIPELINE_DIR),
    ]
    step2b = _step2b_command(python, output_dir, options)
    if step2b:
        commands.append(("Aplicando supressão adaptativa de background (E2B)...", step2b, PIPELINE_DIR))
    commands.extend([
        ("Calculando co-saliência e clustering espectral...", _step3_command(python, output_dir, options), PIPELINE_DIR),
        ("Refinando as máscaras com SAM 2...", _step4_command(python, output_dir, options), PIPELINE_DIR),
    ])

    analysis = [
        python, str(ABLATION_DIR / "analyze_step4_sam_refinement.py"),
        "--refined", str(output_dir / "refined_masks.npz"),
        "--cosaliency", str(output_dir / "cosaliency.npz"),
        "--output", str(quality_dir),
    ]
    gt_index = metadata.get("gt_index")
    if gt_index:
        analysis.extend(["--gt_index", str(gt_index)])
    commands.append(("Calculando métricas e diagnosticando a Etapa 4...", analysis, PROJECT_DIR))

    total_steps = len(commands)
    try:
        required = [
            PIPELINE_DIR / "step1_feature_extraction.py",
            PIPELINE_DIR / "step2_correspondence.py",
            PIPELINE_DIR / "step3_cosaliency.py",
            PIPELINE_DIR / "step4_sam_refinement.py",
            ABLATION_DIR / "analyze_step4_sam_refinement.py",
        ]
        if step2b:
            required.append(PIPELINE_DIR / "step2b_background_suppression.py")
        _require_files(required)
        write_status(run_dir, "running", 0, total_steps, commands[0][0])
        for index, (message, command, cwd) in enumerate(commands, start=1):
            write_status(run_dir, "running", index - 1, total_steps, message)
            execute(command, log_file, cwd)
        write_status(run_dir, "completed", total_steps, total_steps, "Auditoria da Etapa 4 concluída!")
    except Exception as exc:
        fail_run(run_dir, log_file, total_steps, exc)
