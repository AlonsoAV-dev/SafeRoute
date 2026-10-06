"""Compara ventanas y modelos con selección cronológica y caché reproducible.

Desde Backend: python -m app.flujo_entrenamiento_sidpol.optimizacion.ejecutar
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.flujo_entrenamiento_sidpol.config import TrainingConfig
from app.flujo_entrenamiento_sidpol.modelado import _fit, _models, _sample
from app.flujo_entrenamiento_sidpol.optimizacion.evaluacion import ProbabilityCalibration, metrics, tune_joint_thresholds
from app.flujo_entrenamiento_sidpol.optimizacion.modelos import create_model
from app.flujo_entrenamiento_sidpol.optimizacion.muestreo import select_training_rows
from app.flujo_entrenamiento_sidpol.optimizacion.variables import WindowFeatures
from app.flujo_entrenamiento_sidpol.segmentacion import MONTHS, cargar_matrices


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Experiment:
    def __init__(self, path):
        self.settings = json.loads(path.read_text(encoding="utf-8"))
        self.config = TrainingConfig.load((path.parent / self.settings["source_config"]).resolve())
        self.output = (path.parent / self.settings["output"]).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache = self.output / "cache"
        self.cache.mkdir(exist_ok=True)
        self.baseline = json.loads((self.config.output / "resumen_entrenamiento.json").read_text(encoding="utf-8"))
        self.audit = json.loads((self.config.output / "auditoria_fuente.json").read_text(encoding="utf-8"))
        normalized = self.config.output / "delitos_geolocalizados.csv"
        if normalized.stat().st_mtime_ns < max(self.config.source.stat().st_mtime_ns, self.config.weights_file.stat().st_mtime_ns):
            raise ValueError("La fuente normalizada está desactualizada; ejecute primero el preprocesamiento SIDPOL")
        files = [self.config.source, self.config.weights_file, self.config.output / "tramos_osm.csv",
                 self.config.output / "auditoria_fuente.json", *sorted(self.config.output.glob("matriz_*.npz"))]
        provenance = {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files}
        fingerprint = hashlib.sha256(json.dumps({"settings": self.settings, "files": provenance}, sort_keys=True).encode()).hexdigest()
        definition_path = self.output / "definicion.json"
        revisions = []
        if definition_path.exists():
            old_definition = json.loads(definition_path.read_text(encoding="utf-8"))
            revisions = old_definition.get("pending_candidate_revisions", [])
            if old_definition["fingerprint"] != fingerprint:
                old_protocol = {k: v for k, v in old_definition["settings"].items() if k != "candidates"}
                new_protocol = {k: v for k, v in self.settings.items() if k != "candidates"}
                saved_path = self.output / "comparacion.json"
                completed = json.loads(saved_path.read_text(encoding="utf-8")).get("candidates", {}) if saved_path.exists() else {}
                current_specs = {c["name"]: c for c in self.settings["candidates"]}
                compatible = old_protocol == new_protocol and old_definition["files"] == provenance
                compatible &= all(current_specs.get(name) == result["spec"] for name, result in completed.items())
                if not compatible:
                    raise ValueError("Cambió la fuente, el protocolo o un candidato completado. Use un directorio de experimento nuevo")
                revisions.append({"previous_fingerprint": old_definition["fingerprint"],
                                  "previous_candidates": old_definition["settings"]["candidates"],
                                  "completed_candidates": list(completed)})
        write_json(definition_path, {"fingerprint": fingerprint, "settings": self.settings, "files": provenance,
                                     "pending_candidate_revisions": revisions})
        self.train_months = [MONTHS.index(m) for m in self.baseline["splits"]["train"]]
        self.stages = {key: [MONTHS.index(m) for m in self.settings[f"{key}_months"]]
                       for key in ("calibration", "selection", "retrospective")}
        if not max(self.train_months) < min(self.stages["calibration"]) <= max(self.stages["calibration"]) < min(self.stages["selection"]) <= max(self.stages["selection"]) < min(self.stages["retrospective"]):
            raise ValueError("Los cortes deben ser cronológicos y no solaparse")
        if any(self.audit["monthly"][MONTHS[m]]["cobertura"] < self.settings["minimum_history_coverage"]
               for months in self.stages.values() for m in months):
            raise ValueError("Un mes de evaluación no alcanza la cobertura mínima")
        tramos = pd.read_csv(self.config.output / "tramos_osm.csv")
        self.builder = WindowFeatures(tramos, cargar_matrices(self.config.output), self.config, self.audit,
                                      self.settings["windows"], self.settings["minimum_history_coverage"], self.settings["high_threshold"])
        self.results_path = self.output / "comparacion.json"
        self.results = json.loads(self.results_path.read_text(encoding="utf-8")) if self.results_path.exists() else {
            "settings": self.settings, "source": str(self.config.source),
            "train_months": [MONTHS[m] for m in self.train_months],
            "feature_names": self.builder.names, "feature_groups": self.builder.groups,
            "target_definition": "0 sin eventos registrados; 1 gravedad positiva inferior a 3; 2 gravedad >=3 por factor de longitud",
            "selection_criterion": "media de F1 medio y F1 alto en mayo-agosto 2024; umbrales ajustados solo en enero-abril 2024",
            "evaluation_status": "2026 es evaluación retrospectiva previamente examinada, no una nueva prueba ciega",
            "candidates": {},
        }
        self.results["settings"] = self.settings

    def target(self, month, turn):
        return self.builder.target(month, turn, self.settings["high_threshold"])

    def cache_training(self, months=None, tag="train", record_in_results=True):
        months = self.train_months if months is None else months
        complete = self.cache / f"{tag}_complete.json"
        if complete.exists():
            return
        rng = np.random.default_rng(self.settings["random_state"])
        choices = []
        n = 0
        for month in months:
            history = self.builder.months(month, 12)
            activity = self.builder.sum_rows(self.builder.matrices["count"], history) > 0
            for turn in range(4):
                y = self.target(month, turn)
                selected, weights = select_training_rows(
                    y, activity, rng, self.settings["negative_ratio"],
                    self.settings["minimum_negatives"], self.settings["active_negative_fraction"],
                )
                choices.append((month, turn, selected, weights))
                n += len(selected)
        x = np.lib.format.open_memmap(self.cache / f"{tag}_x.npy", mode="w+", dtype=np.float32, shape=(n, len(self.builder.names)))
        y = np.empty(n, dtype=np.int8)
        weights = np.empty(n, dtype=np.float32)
        month_ids = np.empty(n, dtype=np.int16)
        offset = 0
        for month, turn, indices, inverse in choices:
            size = len(indices)
            x[offset:offset + size] = self.builder.block(month, turn)[indices]
            y[offset:offset + size] = self.target(month, turn)[indices]
            weights[offset:offset + size] = inverse
            month_ids[offset:offset + size] = month
            offset += size
            if turn == 3:
                print(f"Caché {tag} {MONTHS[month]}: {offset:,}/{n:,}", flush=True)
        x.flush()
        del x
        np.savez(self.cache / f"{tag}_metadata.npz", y=y, weights=weights, months=month_ids)
        write_json(complete, {"rows": n, "class_counts": np.bincount(y, minlength=3).tolist(), "months": [MONTHS[m] for m in months]})
        if record_in_results:
            self.results[f"{tag}_sampling"] = json.loads(complete.read_text(encoding="utf-8"))
            write_json(self.results_path, self.results)

    def cache_stage(self, stage):
        marker = self.cache / f"{stage}_complete.json"
        if marker.exists():
            return
        months = self.stages[stage]
        n = len(months) * 4 * self.builder.n_segments
        x = np.lib.format.open_memmap(self.cache / f"{stage}_x.npy", mode="w+", dtype=np.float32, shape=(n, len(self.builder.names)))
        y = np.empty(n, dtype=np.int8)
        offset = 0
        for month in months:
            for turn in range(4):
                end = offset + self.builder.n_segments
                x[offset:end] = self.builder.block(month, turn)
                y[offset:end] = self.target(month, turn)
                offset = end
            print(f"Caché {stage}: {MONTHS[month]}", flush=True)
        x.flush()
        del x
        np.save(self.cache / f"{stage}_y.npy", y)
        write_json(marker, {"rows": n, "months": [MONTHS[m] for m in months]})

    def fit(self, spec, tag="train", months=None):
        if spec.get("legacy"):
            x, y = _sample(self.builder.legacy, self.train_months if months is None else months,
                           self.settings["high_threshold"], self.settings["random_state"])
            model = _models(self.config)["random_forest" if spec["family"] == "rf" else "xgboost"]
            model.set_params(n_jobs=self.settings["threads"])
            _fit(model, x, y)
        else:
            self.cache_training(months=months, tag=tag)
            metadata = np.load(self.cache / f"{tag}_metadata.npz")
            source = np.load(self.cache / f"{tag}_x.npy", mmap_mode="r")
            columns = self.builder.groups[spec["features"]]
            x = np.ascontiguousarray(source[:, columns])
            del source
            y = metadata["y"]
            weight = metadata["weights"] ** spec.get("weight_power", 0.0)
            if spec.get("half_life_months"):
                age = max(metadata["months"]) - metadata["months"]
                weight *= np.exp2(-age / spec["half_life_months"])
            weight /= weight.mean()
            model = create_model(spec, self.settings["threads"], self.settings["random_state"])
            model.fit(x, y, sample_weight=weight)
            metadata.close()
        del x, y
        gc.collect()
        return model

    def probability(self, model, group, stage):
        self.cache_stage(stage)
        x = np.load(self.cache / f"{stage}_x.npy", mmap_mode="r")
        p = np.empty((len(x), 3), dtype=np.float32)
        columns = self.builder.groups[group]
        chunk = self.builder.n_segments
        for start in range(0, len(x), chunk):
            p[start:start + chunk] = model.predict_proba(np.ascontiguousarray(x[start:start + chunk, columns]))
        del x
        return p

    def per_month(self, stage, p, cutoffs):
        y = np.load(self.cache / f"{stage}_y.npy", mmap_mode="r")
        size = self.builder.n_segments * 4
        return {MONTHS[month]: metrics(y[i * size:(i + 1) * size], p[i * size:(i + 1) * size], cutoffs)
                for i, month in enumerate(self.stages[stage])}

    def compare(self):
        self.cache_training()
        for stage in ("calibration", "selection"):
            self.cache_stage(stage)
        y_cal = np.load(self.cache / "calibration_y.npy")
        y_selection = np.load(self.cache / "selection_y.npy")
        for spec in self.settings["candidates"]:
            name = spec["name"]
            if name in self.results["candidates"]:
                continue
            start = time.monotonic()
            print(f"ENTRENANDO {name}", flush=True)
            model = self.fit(spec)
            joblib.dump(model, self.output / f"modelo_{name}.joblib", compress=3)
            p_cal = self.probability(model, spec["features"], "calibration")
            p_selection = self.probability(model, spec["features"], "selection")
            variants = {}
            calibrator = ProbabilityCalibration().fit(y_cal, p_cal, self.settings["random_state"])
            joblib.dump(calibrator, self.output / f"calibrador_{name}.joblib")
            for variant in ("raw", "calibrated"):
                cal = p_cal if variant == "raw" else calibrator.predict_proba(p_cal)
                selection = p_selection if variant == "raw" else calibrator.predict_proba(p_selection)
                cutoffs = tune_joint_thresholds(y_cal, cal)
                evaluation = metrics(y_selection, selection, cutoffs)
                variants[variant] = {"cutoffs": cutoffs, "selection": evaluation,
                                     "selection_by_month": self.per_month("selection", selection, cutoffs)}
                c = evaluation["clases"]
                print(f"SELECCIÓN {name}/{variant}: F1 medio={c['medio']['f1']:.4f}, alto={c['alto']['f1']:.4f}, media={evaluation['f1_medio_alto']:.4f}", flush=True)
            selected_variant = max(variants, key=lambda v: variants[v]["selection"]["f1_medio_alto"])
            self.results["candidates"][name] = {"spec": spec, "variants": variants, "selected_variant": selected_variant,
                                                "elapsed_seconds": round(time.monotonic() - start, 1)}
            write_json(self.results_path, self.results)
            del model, calibrator, p_cal, p_selection, cal, selection
            gc.collect()
        ranking = sorted(self.results["candidates"], key=lambda name: self.score(name), reverse=True)
        self.results["ranking_validation"] = ranking
        self.results["selected"] = ranking[0]
        self.results["selected_per_family"] = {
            family: next(name for name in ranking if self.results["candidates"][name]["spec"]["family"] == family)
            for family in ("rf", "xgb", "hurdle")
        }
        write_json(self.results_path, self.results)
        print(f"SELECCIONADO POR VALIDACIÓN: {ranking[0]}", flush=True)

    def score(self, name):
        candidate = self.results["candidates"][name]
        return candidate["variants"][candidate["selected_variant"]]["selection"]["f1_medio_alto"]

    def retrospective(self):
        self.cache_stage("retrospective")
        selected = set(self.results["selected_per_family"].values()) | {"base_rf", "base_xgb", self.results["selected"]}
        y = np.load(self.cache / "retrospective_y.npy")
        for name in sorted(selected):
            candidate = self.results["candidates"][name]
            if "retrospective" in candidate:
                continue
            print(f"EVALUANDO 2026 {name}", flush=True)
            model = joblib.load(self.output / f"modelo_{name}.joblib")
            probability = self.probability(model, candidate["spec"]["features"], "retrospective")
            variant = candidate["selected_variant"]
            if variant == "calibrated":
                calibrator = joblib.load(self.output / f"calibrador_{name}.joblib")
                probability = calibrator.predict_proba(probability)
            cutoffs = candidate["variants"][variant]["cutoffs"]
            candidate["retrospective"] = metrics(y, probability, cutoffs)
            candidate["retrospective_by_month"] = self.per_month("retrospective", probability, cutoffs)
            np.save(self.output / f"probabilidades_2026_{name}.npy", probability)
            if name in ("base_rf", "base_xgb"):
                original_key = "random_forest" if name == "base_rf" else "xgboost"
                raw = self.probability(model, "legacy", "retrospective") if variant == "calibrated" else probability
                original_cutoffs = self.baseline["assessments"][original_key]["decision_thresholds"]
                candidate["retrospective_original_cutoffs"] = metrics(y, raw, original_cutoffs)
                candidate["retrospective_original_by_month"] = self.per_month("retrospective", raw, original_cutoffs)
            print(f"RESULTADO 2026 {name}: {candidate['retrospective']['f1_medio_alto']:.4f}", flush=True)
            write_json(self.results_path, self.results)
            del model, probability
            gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[3] / "config_optimizacion_sidpol.json")
    parser.add_argument("--phase", choices=("all", "compare", "retrospective"), default="all")
    args = parser.parse_args()
    experiment = Experiment(args.config.resolve())
    if args.phase in ("all", "compare"):
        experiment.compare()
    if args.phase in ("all", "retrospective"):
        experiment.retrospective()
    print(experiment.results_path, flush=True)


if __name__ == "__main__":
    main()
