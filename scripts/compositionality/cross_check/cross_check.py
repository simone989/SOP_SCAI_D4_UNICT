"""Multicore cross-validation of oracle SVO runs with filtering and tracing.

This script replicates the logic of the `/cross_validate_oracles` endpoint while
keeping the multicore strategy from
`compositionality/V2/output/compare_3_files_multicore.py`.

Two usage modes are supported:

1. Provide a JSON input already shaped like the API request via `--input`.
2. Let the script read the three default oracle result files:
     - compositionality/V3/Oracolo_0/results.json
     - compositionality/V3/Oracolo_1/results.json
     - compositionality/V3/Oracolo_2/results_top_k_3.json

     (You can override these with `--oracle-files path0 path1 path2`).

Optional post-processing:

* Apply the legacy CSV filtering thresholds directly to the in-memory matches.
* Attach the original Threat A/B/C + composed path from the Oracolo_0 output
  (or any other composition file) to help with reporting.

Usage example:

python main.py \
    --output ./cross_validation_results.json \
    --max-results 5000 \
    --jobs 12 \
    --mean-sim 0.55 --sim-verb 0.45 \
    --composition-file compositionality/V3/Oracolo_0/results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

LOGGER = logging.getLogger("cross_check")
MODEL_NAME = "stsb-roberta-base-v2"
DEFAULT_ORACLE_FILES = [
    "results.json",
    "result_threshold_0_55.json",
    "Oracolo_2/results_top_k_3.json",
]
DEFAULT_COMPOSITION_FILE = DEFAULT_ORACLE_FILES[0]


@dataclass
class ParsedEntry:
    run_index: int
    threat_index: int
    subject: str
    verb: str
    obj: str
    payload: Dict[str, Any]

    @property
    def normalized(self) -> Tuple[str, str, str]:
        return (self.subject, self.verb, self.obj)


def _clean_token(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _normalize_term(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _parse_runs(data: Dict[str, Any]) -> List[List[ParsedEntry]]:
    runs_raw = data.get("oracle_runs")
    if not runs_raw or not isinstance(runs_raw, list):
        raise ValueError("Input JSON must include an 'oracle_runs' list with at least two runs.")

    parsed_runs: List[List[ParsedEntry]] = []
    for run_idx, run in enumerate(runs_raw):
        if not isinstance(run, list) or not run:
            raise ValueError(f"Run {run_idx} must be a non-empty list of SVO entries.")
        parsed_entries: List[ParsedEntry] = []
        for threat_idx, item in enumerate(run):
            if not isinstance(item, dict):
                raise ValueError(f"Run {run_idx} entry {threat_idx} must be an object.")
            svo = item.get("svo_representation", item)
            subject = _normalize_term(svo.get("subject"))
            verb = _normalize_term(svo.get("verb"))
            obj = _normalize_term(svo.get("object"))
            if not (subject and verb and obj):
                raise ValueError(
                    f"Run {run_idx} entry {threat_idx} is missing subject/verb/object fields."
                )
            payload = {
                "subject": subject,
                "verb": verb,
                "object": obj,
            }
            if "original_threat" in item:
                payload["original_threat"] = item["original_threat"]
            parsed_entries.append(
                ParsedEntry(
                    run_index=run_idx,
                    threat_index=threat_idx,
                    subject=subject,
                    verb=verb,
                    obj=obj,
                    payload=payload,
                )
            )
        parsed_runs.append(parsed_entries)

    if len(parsed_runs) < 2:
        raise ValueError("Provide at least two oracle runs to compare.")
    return parsed_runs


def _wrap_svo_entry(subject: Optional[str], verb: Optional[str], obj: Optional[str]) -> Optional[Dict[str, Any]]:
    subject = _clean_token(subject)
    verb = _clean_token(verb)
    obj = _clean_token(obj)
    if not (subject and verb and obj):
        return None
    return {"svo_representation": {"subject": subject, "verb": verb, "object": obj}}


def _extract_from_composition_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    composition = record.get("composition_details") or {}
    svo = composition.get("composed_threat_svo") or {}
    return _wrap_svo_entry(svo.get("subject"), svo.get("verb"), svo.get("object"))


def _extract_from_data_block(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = block.get("data") if isinstance(block, dict) else None
    if not isinstance(records, list):
        return []
    extracted: List[Dict[str, Any]] = []
    for record in records:
        entry = _extract_from_composition_record(record)
        if entry:
            extracted.append(entry)
    return extracted


def _extract_from_combinations(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    combos = block.get("combinations") if isinstance(block, dict) else None
    if not isinstance(combos, list):
        return []
    extracted: List[Dict[str, Any]] = []
    for combo in combos:
        if not isinstance(combo, dict):
            continue
        entry = _wrap_svo_entry(combo.get("subject"), combo.get("verb"), combo.get("object"))
        if entry:
            extracted.append(entry)
    return extracted


def _extract_generic_block(block: Any) -> List[Dict[str, Any]]:
    if not isinstance(block, dict):
        return []
    aggregated: List[Dict[str, Any]] = []
    if "data" in block:
        aggregated.extend(_extract_from_data_block(block))
    if "json" in block and isinstance(block["json"], dict):
        aggregated.extend(_extract_from_combinations(block["json"]))
    if "combinations" in block:
        aggregated.extend(_extract_from_combinations(block))
    if "svo_representation" in block and isinstance(block["svo_representation"], dict):
        svo_map = block["svo_representation"]
        entry = _wrap_svo_entry(
            svo_map.get("subject"),
            svo_map.get("verb"),
            svo_map.get("object"),
        )
        if entry:
            aggregated.append(entry)
    return aggregated


def _load_oracle_file(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Oracle file not found: {path}")
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for block in payload:
            entries.extend(_extract_generic_block(block))
    elif isinstance(payload, dict):
        entries.extend(_extract_generic_block(payload))
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    if not entries:
        raise ValueError(f"No SVO combinations extracted from {path}")

    LOGGER.info("Loaded %d SVO combinations from %s", len(entries), path)
    return entries


def _filter_matches(
    matches: Sequence[Dict[str, Any]],
    *,
    sim_subject: Optional[float] = None,
    sim_verb: Optional[float] = None,
    sim_object: Optional[float] = None,
    mean_sim: Optional[float] = None,
    std_sim: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Replicates the CSV-based filtering rules on in-memory matches."""
    def _passes(match: Dict[str, Any]) -> bool:
        if sim_subject is not None and match["similarity_subject"] < sim_subject:
            return False
        if sim_verb is not None and match["similarity_verb"] < sim_verb:
            return False
        if sim_object is not None and match["similarity_object"] < sim_object:
            return False
        if mean_sim is not None and match["mean_similarity"] < mean_sim:
            return False
        if std_sim is not None and match["std_similarity"] > std_sim:
            return False
        return True

    return [match for match in matches if _passes(match)]


def _iter_composition_records(payload: Any) -> Sequence[Dict[str, Any]]:
    """Yield every composition record regardless of the surrounding schema."""
    if isinstance(payload, list):
        aggregated: List[Dict[str, Any]] = []
        for block in payload:
            aggregated.extend(_iter_composition_records(block))
        return aggregated
    if not isinstance(payload, dict):
        return []

    aggregated = []
    if "data" in payload and isinstance(payload["data"], list):
        aggregated.extend([record for record in payload["data"] if isinstance(record, dict)])
    if "valid_compositions" in payload and isinstance(payload["valid_compositions"], list):
        aggregated.extend([record for record in payload["valid_compositions"] if isinstance(record, dict)])
    if "compositions" in payload and isinstance(payload["compositions"], list):
        aggregated.extend([record for record in payload["compositions"] if isinstance(record, dict)])
    if "json" in payload and isinstance(payload["json"], dict):
        aggregated.extend(_iter_composition_records(payload["json"]))
    return aggregated


def _load_composition_lookup(path: str) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Build a lookup table from normalized SVO → list of detailed compositions."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Composition file not found: {path}")
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    lookup: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for record in _iter_composition_records(payload):
        composition = record.get("composition_details") or {}
        svo = composition.get("composed_threat_svo") or {}
        triple = (
            _normalize_term(svo.get("subject")),
            _normalize_term(svo.get("verb")),
            _normalize_term(svo.get("object")),
        )
        if not all(triple):
            continue
        lookup.setdefault(triple, []).append(record)

    LOGGER.info("Loaded %d composition traces from %s", sum(len(v) for v in lookup.values()), path)
    return lookup


def _attach_compositions(
    matches: Sequence[Dict[str, Any]],
    lookup: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    composition_run_index: int,
) -> None:
    """Attach the original Oracolo composition(s) to matches for reporting."""
    for match in matches:
        try:
            target_pos = match["run_indices"].index(composition_run_index)
        except ValueError:
            continue

        threat_payload = match["threats"][target_pos]
        triple = (
            _normalize_term(threat_payload.get("subject")),
            _normalize_term(threat_payload.get("verb")),
            _normalize_term(threat_payload.get("object")),
        )
        composed = lookup.get(triple)
        if composed:
            match["composition_trace"] = composed


def _collapse_to_run(
    matches: Sequence[Dict[str, Any]],
    target_run_index: int,
) -> List[Dict[str, Any]]:
    """Keep only the threats originating from the selected run, deduplicated."""

    collapsed: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for match in matches:
        try:
            target_pos = match["run_indices"].index(target_run_index)
        except ValueError:
            continue

        key = (target_run_index, match["threat_indices"][target_pos])
        reduced: Dict[str, Any] = {
            "run_index": target_run_index,
            "threat_index": match["threat_indices"][target_pos],
            "threat": match["threats"][target_pos],
            "similarity_subject": match["similarity_subject"],
            "similarity_verb": match["similarity_verb"],
            "similarity_object": match["similarity_object"],
            "mean_similarity": match["mean_similarity"],
            "std_similarity": match["std_similarity"],
            "composition_trace": match.get("composition_trace"),
        }
        partners = []
        for idx, run_idx in enumerate(match["run_indices"]):
            if run_idx == target_run_index:
                continue
            partners.append(
                {
                    "run_index": run_idx,
                    "threat_index": match["threat_indices"][idx],
                    "threat": match["threats"][idx],
                }
            )
        if partners:
            reduced["supporting_runs"] = partners

        existing = collapsed.get(key)
        if not existing or reduced["mean_similarity"] > existing["mean_similarity"]:
            collapsed[key] = reduced

    return list(collapsed.values())


def _collect_role_terms(runs: Sequence[Sequence[ParsedEntry]]) -> Dict[str, List[str]]:
    roles = {"subject": set(), "verb": set(), "object": set()}
    for run in runs:
        for entry in run:
            roles["subject"].add(entry.subject)
            roles["verb"].add(entry.verb)
            roles["object"].add(entry.obj)
    return {role: sorted(terms) for role, terms in roles.items() if terms}


def _encode_terms(role_terms: Dict[str, List[str]], model_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    embedder = SentenceTransformer(model_name)
    embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for role, terms in role_terms.items():
        vectors = embedder.encode(terms, convert_to_numpy=True, show_progress_bar=False)
        embeddings[role] = {term: vector for term, vector in zip(terms, np.asarray(vectors))}
    return embeddings


def _mean_pairwise_cosine(vectors: List[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(vectors) - 1):
        for j in range(i + 1, len(vectors)):
            total += float(
                cosine_similarity(
                    vectors[i].reshape(1, -1),
                    vectors[j].reshape(1, -1),
                )[0, 0]
            )
            count += 1
    return total / count if count else 0.0


def _role_similarity(values: List[str], embedding_map: Dict[str, np.ndarray]) -> float:
    vectors = [embedding_map[value] for value in values if value in embedding_map]
    return _mean_pairwise_cosine(vectors)


def _score_combination(
    combo: Sequence[ParsedEntry],
    embeddings: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    subjects = [entry.subject for entry in combo]
    verbs = [entry.verb for entry in combo]
    objects = [entry.obj for entry in combo]

    subj_sim = _role_similarity(subjects, embeddings.get("subject", {}))
    verb_sim = _role_similarity(verbs, embeddings.get("verb", {}))
    obj_sim = _role_similarity(objects, embeddings.get("object", {}))
    sims = np.array([subj_sim, verb_sim, obj_sim], dtype=float)

    return {
        "run_indices": [entry.run_index for entry in combo],
        "threat_indices": [entry.threat_index for entry in combo],
        "threats": [entry.payload for entry in combo],
        "similarity_subject": subj_sim,
        "similarity_verb": verb_sim,
        "similarity_object": obj_sim,
        "mean_similarity": float(np.mean(sims)),
        "std_similarity": float(np.std(sims)),
    }


def _process_anchor(
    anchor_entry: ParsedEntry,
    tail_runs: Sequence[Sequence[ParsedEntry]],
    embeddings: Dict[str, Dict[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    local_results: List[Dict[str, Any]] = []
    if not tail_runs:
        # Only two runs case handled elsewhere; guard to avoid empty product.
        combo = [anchor_entry]
        local_results.append(_score_combination(combo, embeddings))
        return local_results

    for combo_tail in itertools.product(*tail_runs):
        combo = (anchor_entry, *combo_tail)
        local_results.append(_score_combination(combo, embeddings))
    return local_results


def compute_matches_parallel(
    runs: Sequence[Sequence[ParsedEntry]],
    embeddings: Dict[str, Dict[str, np.ndarray]],
    n_jobs: int,
) -> Tuple[List[Dict[str, Any]], int]:
    anchor_run = runs[0]
    tail_runs = runs[1:]

    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(_process_anchor)(entry, tail_runs, embeddings)
        for entry in tqdm(anchor_run, total=len(anchor_run), desc="Anchors")
    )

    flat_results = [item for sublist in results for item in sublist]
    evaluated = len(flat_results)
    return flat_results, evaluated


def build_metadata(runs: Sequence[Sequence[ParsedEntry]], evaluated: int, returned: int) -> Dict[str, Any]:
    total_candidates = sum(len(run) for run in runs)
    total_combinations = math.prod(len(run) for run in runs)
    unique_subjects = len({entry.subject for run in runs for entry in run})
    unique_verbs = len({entry.verb for run in runs for entry in run})
    unique_objects = len({entry.obj for run in runs for entry in run})

    return {
        "total_runs": len(runs),
        "total_candidates": total_candidates,
        "combinations_evaluated": evaluated,
        "returned_matches": returned,
        "unique_subjects": unique_subjects,
        "unique_verbs": unique_verbs,
        "unique_objects": unique_objects,
        "total_combinations": total_combinations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate oracle runs using multicore cosine similarity.")
    parser.add_argument("--input", help="Path to JSON file already containing oracle_runs (API format).")
    parser.add_argument(
        "--oracle-files",
        nargs="+",
        help="Paths to raw oracle result files; defaults to the three Oracolo_* outputs when --input is absent.",
    )
    parser.add_argument("--output", help="Destination JSON file for the computed matches and metadata.")
    parser.add_argument("--max-results", type=int, default=None, help="Optional cap on the number of matches returned.")
    parser.add_argument("--model", default=MODEL_NAME, help="SentenceTransformer model to use (default: stsb-roberta-base-v2).")
    ##default=max(mp.cpu_count() - 2, 1),
    parser.add_argument("--jobs", type=int, default=max(mp.cpu_count() - 2, 1), help="Parallel jobs (default: cpu_count-1).")
    parser.add_argument(
        "--sim-subject",
        type=float,
        default=0.30,
        help="Minimum similarity for the subject role (default: 0.30).",
    )
    parser.add_argument(
        "--sim-verb",
        type=float,
        default=0.30,
        help="Minimum similarity for the verb role (default: 0.30).",
    )
    parser.add_argument(
        "--sim-object",
        type=float,
        default=0.30,
        help="Minimum similarity for the object role (default: 0.30).",
    )
    parser.add_argument(
        "--mean-sim",
        type=float,
        default=0.50,
        help="Minimum mean similarity across roles (default: 0.50).",
    )
    parser.add_argument(
        "--std-sim",
        type=float,
        default=0.15,
        help="Maximum allowed std. deviation across role similarities (default: 0.15).",
    )
    parser.add_argument(
        "--composition-file",
        help="Optional Oracolo output with composition details to attach (defaults to Oracolo_0 if available).",
    )
    parser.add_argument(
        "--composition-run-index",
        type=int,
        default=0,
        help="Which run index corresponds to the composition file (default: 0 / first run).",
    )
    parser.add_argument(
        "--skip-composition-trace",
        action="store_true",
        help="Skip the original threat lookup even if a composition file is provided.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...).")
    return parser.parse_args()


def _load_runs(args: argparse.Namespace) -> List[List[ParsedEntry]]:
    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _parse_runs(payload)

    oracle_files = args.oracle_files or DEFAULT_ORACLE_FILES
    if len(oracle_files) < 2:
        raise ValueError("Provide at least two oracle files or supply an --input payload.")

    aggregated_runs = []
    for path in oracle_files:
        aggregated_runs.append(_load_oracle_file(path))

    payload = {"oracle_runs": aggregated_runs}
    return _parse_runs(payload)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    runs = _load_runs(args)
    LOGGER.info("Loaded %d runs with %s candidates.", len(runs), sum(len(run) for run in runs))

    role_terms = _collect_role_terms(runs)
    if not role_terms:
        raise RuntimeError("No valid SVO terms found for embedding.")
    embeddings = _encode_terms(role_terms, args.model)

    matches, evaluated = compute_matches_parallel(runs, embeddings, args.jobs)
    matches.sort(key=lambda item: item["mean_similarity"], reverse=True)

    threshold_kwargs = {
        "sim_subject": args.sim_subject,
        "sim_verb": args.sim_verb,
        "sim_object": args.sim_object,
        "mean_sim": args.mean_sim,
        "std_sim": args.std_sim,
    }
    if any(value is not None for value in threshold_kwargs.values()):
        before = len(matches)
        matches = _filter_matches(matches, **threshold_kwargs)
        LOGGER.info("Applied filtering thresholds; %d -> %d matches remain.", before, len(matches))

    if args.max_results is not None:
        matches = matches[: args.max_results]

    composition_lookup: Optional[Dict[Tuple[str, str, str], List[Dict[str, Any]]]] = None
    composition_path: Optional[str] = None
    if not args.skip_composition_trace:
        composition_path = args.composition_file or DEFAULT_COMPOSITION_FILE
        if composition_path:
            try:
                composition_lookup = _load_composition_lookup(composition_path)
            except FileNotFoundError:
                LOGGER.warning("Composition file %s not found; skipping trace attachment.", composition_path)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Invalid JSON in composition file %s: %s", composition_path, exc)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to load composition file %s: %s", composition_path, exc)
    if composition_lookup:
        _attach_compositions(matches, composition_lookup, args.composition_run_index)

    matches = _collapse_to_run(matches, args.composition_run_index)
    LOGGER.info(
        "Kept %d unique threats from run %d after collapsing combinations.",
        len(matches),
        args.composition_run_index,
    )

    metadata = build_metadata(runs, evaluated, len(matches))
    metadata["filter_thresholds"] = {k: v for k, v in threshold_kwargs.items() if v is not None}
    metadata["composition_trace"] = {
        "enabled": bool(composition_lookup),
        "run_index": args.composition_run_index if composition_lookup else None,
        "source_file": composition_path if composition_lookup else None,
    }
    output = {"matches": matches, "metadata": metadata}

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        LOGGER.info("Saved results to %s", args.output)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
