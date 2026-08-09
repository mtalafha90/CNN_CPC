"""B6: competition-only structured weak labels from radiology reports.

B6 converts each training report into target-wise states and fixed soft labels
without fitting on the 58 gold labels. Gold studies are retained only in the
audit output and are excluded from the B6 training-target export.

The B6 parser is deliberately conservative:

* multilingual, accent-insensitive concept matching;
* target-local negation, normality and uncertainty handling;
* non-diagnostic history/indication suppression;
* explicit conflict detection for genuinely opposing definite evidence;
* four states: positive / negated / uncertain / unmentioned;
* fixed probabilities and confidence weights, not gold-fitted calibration;
* review queue for mentioned-but-uncertain/conflicting cells;
* compatibility output for the later MRI student model.

No external model, external language resource or internet service is required.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import gold_mask, load_train_csv, normalize_report
from .report_labels import (
    LEXICON,
    NEG_RE,
    OA_TARGETS,
    POSITIVE_MODIFIERS,
    STATE_NEGATED,
    STATE_POSITIVE,
    STATE_UNCERTAIN,
    STATE_UNMENTIONED,
    predict_target as legacy_predict_target,
)

B6_VERSION = "1.2"

EXTRA_LEXICON: dict[str, tuple[str, ...]] = {
    "ACL": (
        "ligament croise anterieur",
        "legamento crociato anteriore",
        "ligamento cruzado anterior",
        "vorderes kreuzband",
        "voorste kruisband",
        "on capraz bag",
        "prednji krizni ligament",
        "prednji ukrsteni ligament",
    ),
    "MCL": (
        "ligament collateral medial",
        "legamento collaterale mediale",
        "ligamento colateral medial",
        "mediales kollateralband",
        "mediale knieband",
        "medial kollateral ligament",
        "ic yan bag",
    ),
    "Medial Meniscus": (
        "menisque medial",
        "menisco mediale",
        "menisco medial",
        "innenmeniskus",
        "mediale meniscus",
        "medial meniskus",
    ),
    "Lateral Meniscus": (
        "menisque lateral",
        "menisco laterale",
        "menisco lateral",
        "aussenmeniskus",
        "laterale meniscus",
        "lateral meniskus",
    ),
    "Medial OA": (
        "medial osteoarthritis",
        "medial osteoarthrosis",
        "medial gonarthrosis",
        "medial gonartrosis",
        "medial arthrosis",
        "medial artrosis",
        "artrosi compartimento mediale",
        "artrose compartimento medial",
    ),
    "Lateral OA": (
        "lateral osteoarthritis",
        "lateral osteoarthrosis",
        "lateral gonarthrosis",
        "lateral gonartrosis",
        "lateral arthrosis",
        "lateral artrosis",
        "artrosi compartimento laterale",
        "artrose compartimento lateral",
    ),
    "PF OA": (
        "patellofemoral osteoarthritis",
        "patellofemoral osteoarthrosis",
        "patellofemoral arthrosis",
        "patellofemoral artrosis",
        "artrosi femoro rotulea",
        "artrosi femororotulea",
        "artrose patelofemoral",
    ),
    "Effusion": (
        "versamento articolare",
        "derrame articular",
        "epanchement articulaire",
        "gelenkerguss",
        "gewrichtseffusie",
        "eklem efüzyonu",
        "eklem efüzyon",
    ),
    "Synovitis": (
        "sinovite",
        "sinovitis",
        "sinovit",
        "synovial inflammation",
    ),
    "Baker's": (
        "baker cyst",
        "baker's cyst",
        "popliteal cyst",
        "kyste de baker",
        "quiste de baker",
        "cisti di baker",
        "baker zyste",
        "baker kisti",
    ),
    "Contusion": (
        "bone bruise",
        "bone contusion",
        "marrow contusion",
        "bone marrow edema",
        "bone marrow oedema",
        "edema osseo",
        "edema oseo",
        "oedeme osseux",
        "knochenmarkodem",
        "kemik kontuzyonu",
    ),
    "Fracture": (
        "fracture",
        "fractura",
        "frattura",
        "fraktur",
        "breuk",
        "kirik",
        "prijelom",
        "prelom",
    ),
}

STRUCTURAL_TARGETS = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"}

UNCERTAIN_RE = re.compile(
    r"\b(?:"
    r"possible|possibly|probable|probably|questionable|equivocal|suspect(?:ed)?|"
    r"suspicious for|cannot exclude|can not exclude|cannot rule out|can not rule out|"
    r"may represent|may be|might represent|might be|"
    r"posible|probable|sospech(?:a|oso|osa)|sugestiv[oa]|no se puede excluir|no puede descartarse|"
    r"possible|probable|suspect|evocateur|ne peut exclure|a exclure|"
    r"moglich|wahrscheinlich|verdacht|nicht auszuschliessen|"
    r"mogelijk|waarschijnlijk|verdacht|niet uit te sluiten|"
    r"olasi|supheli|dusundurur|dislanamaz|r/o|rule out|"
    r"possibile|probabile|sospett[oa]|non si puo escludere|"
    r"possivel|provavel|suspeit[oa]|nao se pode excluir|"
    r"moguc|moguce|vjerojat|sumnj|ne moze se iskljuciti"
    r")\b",
    re.I,
)

NON_DIAGNOSTIC_RE = re.compile(
    r"\b(?:"
    r"indication|clinical indication|clinical history|clinical information|history of|"
    r"reason for exam|reason for examination|diagnostic question|diagnostische vraagstelling|"
    r"antecedentes clinicos|anamnese|vraagstelling|assess for|evaluate for|evaluation for|"
    r"concern for|query|quid"
    r")\b",
    re.I,
)

NORMAL_RE = re.compile(
    r"\b(?:"
    r"intact|normal|unremarkable|preserved|continuous|continuity preserved|without tear|"
    r"sans rupture|sans lesion|integre|intacte|"
    r"integro|integra|sin rotura|sin desgarro|"
    r"intakt|unauffallig|erhalten|"
    r"intacte|normaal|behouden|"
    r"intatto|integro|senza rottura|"
    r"integro|sem rotura|sem ruptura|"
    r"intakt|ocuvan|uredan"
    r")\b",
    re.I,
)

STRUCTURAL_PATHOLOGY_RE = re.compile(
    r"tear|ruptur|lesion|injur|sprain|degener|damage|defect|fissur|rotur|desgar|"
    r"riss|yirt|lezion|lacer|avuls|discontinu|distors|"
    r"rottur|lesao|ruptura|entorse|yirtil|kopma|hasar|puknu|ostecen|deficien|"
    r"fray|macerat|attenuat|thinn",
    re.I,
)

NEGATED_STRUCTURAL_FINDING_RE = re.compile(
    r"\b(?:no|without|absent|negative for|sin|sans|kein|keine|ohne|geen|zonder|yok|degil|bez)"
    r"\b[^.;:>]{0,70}\b(?:tear|ruptur|lesion|injur|sprain|damage|defect|fissur|rotur|desgar|"
    r"riss|yirt|lezion|lacer|avuls|discontinu|rottur|lesao|ruptura|entorse|yirtil|kopma|hasar)\w*",
    re.I,
)

LOSS_OF_NORMAL_RE = re.compile(
    r"\b(?:complete\s+)?loss\s+of\s+(?:the\s+)?normal\b|"
    r"\babsence\s+of\s+(?:the\s+)?normal\b|"
    r"\bno\s+normal\s+intact\s+fibers?\b|"
    r"\bloss\s+of\s+(?:normal\s+)?(?:fiber|fibers|signal|morphology|continuity|architecture)\b",
    re.I,
)


@dataclass(frozen=True)
class B6Prediction:
    probability: float
    confidence: float
    mentioned: bool
    state: str
    evidence: str = ""
    reason: str = ""


def _aliases(target: str) -> tuple[str, ...]:
    values = list(LEXICON.get(target, ())) + list(EXTRA_LEXICON.get(target, ()))
    normalized = {normalize_report(value) for value in values if normalize_report(value)}
    return tuple(sorted(normalized, key=len, reverse=True))


def _clause(text: str, start: int, stop: int, radius: int = 180) -> str:
    """Return a punctuation/arrow-bounded clause around one target mention."""
    left = max(0, start - radius)
    right = min(len(text), stop + radius)
    snippet = text[left:right]
    local_start = start - left
    local_stop = stop - left

    before = max(
        snippet.rfind(".", 0, local_start),
        snippet.rfind(";", 0, local_start),
        snippet.rfind(">", 0, local_start),
    )
    after_candidates = [
        pos
        for pos in (
            snippet.find(".", local_stop),
            snippet.find(";", local_stop),
            snippet.find(">", local_stop),
        )
        if pos >= 0
    ]
    clause_start = before + 1 if before >= 0 else 0
    clause_stop = min(after_candidates) if after_candidates else len(snippet)
    return snippet[clause_start:clause_stop].strip()


def _target_focus(clause: str, phrase: str, *, before: int = 80, after: int = 100) -> str:
    """Tight target-centered context to prevent neighboring anatomy bleed."""
    index = clause.find(phrase)
    if index < 0:
        return clause[: before + after + len(phrase)]
    return clause[max(0, index - before) : min(len(clause), index + len(phrase) + after)]


def _left_context(focus: str, phrase: str, width: int = 70) -> str:
    index = focus.find(phrase)
    if index < 0:
        return focus[:width]
    return focus[max(0, index - width) : index]


def _classify_mention(target: str, clause: str, phrase: str) -> tuple[str, str]:
    focus = _target_focus(clause, phrase)
    left = _left_context(focus, phrase)

    # A mention inside indication/history is not diagnostic evidence. If the
    # report later contains a real finding for the same target, that definite
    # finding will win over this uncertain observation.
    if NON_DIAGNOSTIC_RE.search(left):
        return STATE_UNCERTAIN, "non_diagnostic_context"

    uncertain = bool(UNCERTAIN_RE.search(focus))
    negated_target = bool(NEG_RE.search(left))

    if target in STRUCTURAL_TARGETS:
        negated_finding = bool(NEGATED_STRUCTURAL_FINDING_RE.search(focus))
        # Remove directly negated tear/injury phrases before searching for other
        # abnormalities. This lets "mucoid degeneration without tear" remain a
        # positive abnormality while "no ACL tear" remains negative.
        residual = NEGATED_STRUCTURAL_FINDING_RE.sub(" ", focus)
        pathology = bool(
            STRUCTURAL_PATHOLOGY_RE.search(residual)
            or POSITIVE_MODIFIERS.search(residual)
            or LOSS_OF_NORMAL_RE.search(focus)
        )
        normal = bool(NORMAL_RE.search(focus))

        if pathology and uncertain:
            return STATE_UNCERTAIN, "uncertainty_scope"
        # Explicit structural abnormality outranks generic words such as intact,
        # preserved or normal when both occur in the same local statement, e.g.
        # "sprain with intact fibers" or "tear; tibial insertion intact".
        if pathology:
            return STATE_POSITIVE, "explicit_structural_abnormality"
        if negated_finding:
            return STATE_NEGATED, "explicit_negated_structural_finding"
        if negated_target or normal:
            return STATE_NEGATED, "explicit_normal_or_negated"
        if uncertain:
            return STATE_UNCERTAIN, "uncertainty_scope"
        return STATE_UNCERTAIN, "anatomy_mentioned_without_finding"

    if uncertain:
        return STATE_UNCERTAIN, "uncertainty_scope"
    if negated_target:
        return STATE_NEGATED, "explicit_negation"
    return STATE_POSITIVE, "explicit_pathology_mention"


def _state_to_values(state: str, *, conflict: bool = False) -> tuple[float, float]:
    if conflict:
        return 0.50, 0.20
    if state == STATE_POSITIVE:
        return 0.97, 0.90
    if state == STATE_NEGATED:
        return 0.03, 0.90
    if state == STATE_UNCERTAIN:
        return 0.50, 0.25
    return 0.50, 0.0


def predict_target_b6(text: str, target: str) -> B6Prediction:
    """Return a fixed, auditable structured weak label for one report/target."""
    norm = normalize_report(text)
    if not norm:
        return B6Prediction(0.50, 0.0, False, STATE_UNMENTIONED, "", "empty_report")

    observations: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for phrase in _aliases(target):
        for match in re.finditer(re.escape(phrase), norm, re.I):
            clause = _clause(norm, match.start(), match.end())
            state, reason = _classify_mention(target, clause, phrase)
            item = (state, clause[:360], reason)
            if item not in seen:
                observations.append(item)
                seen.add(item)

    if not observations and target in OA_TARGETS:
        legacy = legacy_predict_target(norm, target)
        if legacy.state != STATE_UNMENTIONED:
            state = legacy.state
            probability, confidence = _state_to_values(state)
            return B6Prediction(
                probability,
                confidence,
                True,
                state,
                "oa_context_parser",
                "compartment_aware_oa_context",
            )

    if not observations:
        return B6Prediction(0.50, 0.0, False, STATE_UNMENTIONED, "", "no_target_evidence")

    definite = {state for state, _, _ in observations if state in {STATE_POSITIVE, STATE_NEGATED}}

    if len(definite) > 1:
        probability, confidence = _state_to_values(STATE_UNCERTAIN, conflict=True)
        evidence = " || ".join(item[1] for item in observations[:3])
        return B6Prediction(
            probability,
            confidence,
            True,
            STATE_UNCERTAIN,
            evidence,
            "conflicting_definite_evidence",
        )

    if definite:
        state = next(iter(definite))
        selected = next(item for item in observations if item[0] == state)
        probability, confidence = _state_to_values(state)
        return B6Prediction(probability, confidence, True, state, selected[1], selected[2])

    probability, confidence = _state_to_values(STATE_UNCERTAIN)
    return B6Prediction(
        probability,
        confidence,
        True,
        STATE_UNCERTAIN,
        observations[0][1],
        observations[0][2],
    )


def build_b6_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"].astype(str)})
    out["is_gold"] = gold_mask(df).to_numpy(dtype=bool)
    reports = df["Report"].fillna("").astype(str)
    out["has_report"] = reports.map(lambda text: bool(normalize_report(text))).to_numpy(dtype=bool)

    for target in TARGETS:
        predictions = [predict_target_b6(text, target) for text in reports]
        out[target] = np.asarray([item.probability for item in predictions], dtype=np.float32)
        out[f"{target}__confidence"] = np.asarray(
            [item.confidence for item in predictions], dtype=np.float32
        )
        out[f"{target}__state"] = [item.state for item in predictions]
        out[f"{target}__mentioned"] = [item.mentioned for item in predictions]
        out[f"{target}__reason"] = [item.reason for item in predictions]
        out[f"{target}__evidence"] = [item.evidence for item in predictions]
    return out


def _target_audit(frame: pd.DataFrame, target: str, *, min_confidence: float) -> dict:
    state_col = f"{target}__state"
    conf_col = f"{target}__confidence"
    reason_col = f"{target}__reason"
    report_only = ~frame["is_gold"].astype(bool)
    subset = frame.loc[report_only]
    states = subset[state_col].value_counts().reindex(
        [STATE_POSITIVE, STATE_NEGATED, STATE_UNCERTAIN, STATE_UNMENTIONED], fill_value=0
    )
    usable = subset[state_col].isin([STATE_POSITIVE, STATE_NEGATED]) & subset[conf_col].ge(min_confidence)
    positive = subset[state_col].eq(STATE_POSITIVE) & subset[conf_col].ge(min_confidence)
    negative = subset[state_col].eq(STATE_NEGATED) & subset[conf_col].ge(min_confidence)
    n = max(1, len(subset))
    mentioned = subset[f"{target}__mentioned"].astype(bool)
    reasons = subset.loc[mentioned, reason_col].value_counts()
    return {
        "states": {str(key): int(value) for key, value in states.items()},
        "high_confidence_positive": int(positive.sum()),
        "high_confidence_negative": int(negative.sum()),
        "usable_cells": int(usable.sum()),
        "usable_fraction_report_only": float(usable.sum() / n),
        "mean_confidence_when_mentioned": float(subset.loc[mentioned, conf_col].mean())
        if mentioned.any()
        else 0.0,
        "reason_counts_when_mentioned": {str(key): int(value) for key, value in reasons.items()},
    }


def _review_queue(frame: pd.DataFrame, raw: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    report_by_uid = raw.set_index("StudyInstanceUID")["Report"].fillna("").astype(str)
    rows: list[dict] = []
    for target in TARGETS:
        uncertain = frame[f"{target}__state"].eq(STATE_UNCERTAIN) & frame[f"{target}__mentioned"].astype(bool)
        for _, row in frame.loc[uncertain].iterrows():
            uid = str(row["StudyInstanceUID"])
            rows.append(
                {
                    "StudyInstanceUID": uid,
                    "is_gold": bool(row["is_gold"]),
                    "target": target,
                    "state": row[f"{target}__state"],
                    "confidence": float(row[f"{target}__confidence"]),
                    "reason": row[f"{target}__reason"],
                    "evidence": row[f"{target}__evidence"],
                    "report": report_by_uid.get(uid, "")[:1600],
                }
            )
    review = pd.DataFrame(rows)
    if review.empty:
        return pd.DataFrame(
            columns=["StudyInstanceUID", "is_gold", "target", "state", "confidence", "reason", "evidence", "report"]
        )
    priority = review["reason"].eq("conflicting_definite_evidence").astype(int)
    review = review.assign(_priority=priority).sort_values(
        ["_priority", "confidence"], ascending=[False, True]
    )
    return review.drop(columns="_priority").head(max_rows).reset_index(drop=True)


def run_b6_export(
    train_csv: str | Path,
    *,
    out_root: str | Path = "runs/b6_report_labels",
    min_confidence: float = 0.75,
    max_review: int = 1000,
) -> dict:
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0,1]")
    if max_review < 0:
        raise ValueError("max_review must be >=0")

    df = load_train_csv(train_csv)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    structured = build_b6_frame(df)
    structured.to_csv(out / "structured_labels.csv", index=False)

    report_only = structured.loc[~structured["is_gold"].astype(bool)].copy()
    training_columns = ["StudyInstanceUID"]
    for target in TARGETS:
        training_columns.extend([target, f"{target}__confidence", f"{target}__state"])
    report_only[training_columns].to_csv(out / "training_targets.csv", index=False)

    review = _review_queue(structured, df, max_rows=max_review)
    review.to_csv(out / "review_queue.csv", index=False)

    audit = {
        "b6_version": B6_VERSION,
        "n_studies": int(len(structured)),
        "n_gold_audit_only": int(structured["is_gold"].sum()),
        "n_report_only_training": int((~structured["is_gold"].astype(bool)).sum()),
        "n_reports_present": int(structured["has_report"].sum()),
        "min_confidence_for_usable_cell": float(min_confidence),
        "external_models": False,
        "external_data": False,
        "gold_fitted_calibration": False,
        "gold_rows_in_training_targets": 0,
        "targets": {
            target: _target_audit(structured, target, min_confidence=min_confidence)
            for target in TARGETS
        },
        "review_queue_rows": int(len(review)),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    policy = {
        "experiment": "B6",
        "version": B6_VERSION,
        "purpose": "structured multilingual report weak labels for later MRI training",
        "states": [STATE_POSITIVE, STATE_NEGATED, STATE_UNCERTAIN, STATE_UNMENTIONED],
        "fixed_soft_labels": {
            STATE_POSITIVE: {"probability": 0.97, "confidence": 0.90},
            STATE_NEGATED: {"probability": 0.03, "confidence": 0.90},
            STATE_UNCERTAIN: {"probability": 0.50, "confidence": 0.25},
            STATE_UNMENTIONED: {"probability": 0.50, "confidence": 0.0},
            "conflict": {"probability": 0.50, "confidence": 0.20},
        },
        "gold_usage": "audit only; excluded from training_targets.csv",
        "unmentioned_is_negative": False,
        "external_models": False,
        "external_data": False,
        "next_gate": "inspect audit/review, then measure fixed B6 precision on gold before B7",
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    print(out / "structured_labels.csv")
    print(out / "training_targets.csv")
    print(out / "review_queue.csv")
    print(out / "audit.json")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b6")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-root", default="runs/b6_report_labels")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--max-review", type=int, default=1000)
    args = parser.parse_args()
    run_b6_export(
        args.train_csv,
        out_root=args.out_root,
        min_confidence=args.min_confidence,
        max_review=args.max_review,
    )


if __name__ == "__main__":
    main()
