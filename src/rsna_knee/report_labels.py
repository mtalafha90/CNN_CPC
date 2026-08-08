from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import normalize_report

NEG_RE = re.compile(
    r"\b(no|without|absent|negative for|sin|ausencia de|sans|pas de|kein|keine|ohne|"
    r"niet|geen|zonder|yok|degil|nema|bez)\b",
    re.I,
)
POSITIVE_MODIFIERS = re.compile(
    r"tear|ruptur|lesion|injur|sprain|degener|damage|defect|fissur|rotur|desgar|riss|yirt|lezion",
    re.I,
)

LEXICON = {
    "ACL": ["acl", "anterior cruciate ligament", "ligamento cruzado anterior", "vorderes kreuzband", "voorste kruisband", "on capraz bag"],
    "MCL": ["mcl", "medial collateral ligament", "ligamento colateral medial", "mediales kollateralband", "mediale knieband"],
    "Medial Meniscus": ["medial meniscus", "menisco medial", "menisque medial", "innenmeniskus", "mediale meniscus", "medial meniskus"],
    "Lateral Meniscus": ["lateral meniscus", "menisco lateral", "menisque lateral", "aussenmeniskus", "laterale meniscus", "lateral meniskus"],
    # OA uses the compartment-aware parser below. These phrases remain useful
    # documentation of explicit forms observed in the corpus.
    "Medial OA": ["medial osteoarthritis", "medial gonarthrosis", "artrosis compartimento medial", "mediale gonarthrose", "medial kompartman osteoartrit"],
    "Lateral OA": ["lateral osteoarthritis", "lateral gonarthrosis", "artrosis compartimento lateral", "laterale gonarthrose", "lateral kompartman osteoartrit"],
    "PF OA": ["patellofemoral osteoarthritis", "patellofemoral oa", "patellofemoral arthrosis", "artrosis patelofemoral", "patellofemorale arthrose"],
    "Effusion": ["joint effusion", "effusion", "derrame articular", "epanchement articulaire", "gelenkerguss", "gewrichtseffusie", "eklem efüzyonu", "efüzyon"],
    "Synovitis": ["synovitis", "sinovitis", "synovite", "sinovit"],
    "Baker's": ["baker cyst", "baker's cyst", "popliteal cyst", "quiste de baker", "kyste de baker", "baker zyste", "baker kisti"],
    "Contusion": ["bone contusion", "bone bruise", "marrow contusion", "bone marrow edema", "bone marrow oedema", "edema oseo", "knochenmarkodem", "kemik kontuzyonu"],
    "Fracture": ["fracture", "fractura", "fraktur", "breuk", "kirik", "prijelom"],
}

STATE_POSITIVE = "positive"
STATE_NEGATED = "negated"
STATE_UNCERTAIN = "uncertain"
STATE_UNMENTIONED = "unmentioned"
STATES = (STATE_POSITIVE, STATE_NEGATED, STATE_UNCERTAIN, STATE_UNMENTIONED)

OA_TARGETS = {"Medial OA", "Lateral OA", "PF OA"}

# Context is deliberately anatomical/compartment specific. We do not map a
# generic "degenerative knee" statement to all three OA outputs.
OA_CONTEXT_PATTERNS = {
    "Medial OA": (
        r"\bmedial compartment\b",
        r"\bmedial tibiofemoral\b",
        r"\bmedial femorotibial\b",
        r"\bfemorotibial medial\b",
        r"\bmediaal femorotibiaal\b",
        r"\bcompartimento medial\b",
        r"\bcompartiment medial\b",
        r"\bmedial kompartman\b",
        r"\bmediale(?:s|n)? kompartiment\b",
        r"\bmedial femoral condyle\b",
        r"\bmedial tibial plateau\b",
        r"\bmediale femorale condyl\b",
    ),
    "Lateral OA": (
        r"\blateral compartment\b",
        r"\blateral tibiofemoral\b",
        r"\blateral femorotibial\b",
        r"\bfemorotibial lateral\b",
        r"\blateraal femorotibiaal\b",
        r"\bcompartimento lateral\b",
        r"\bcompartiment lateral\b",
        r"\blateral kompartman\b",
        r"\blaterale(?:s|n)? kompartiment\b",
        r"\blateral femoral condyle\b",
        r"\blateral tibial plateau\b",
        r"\blaterale femorale condyl\b",
    ),
    "PF OA": (
        r"\bpatellofemoral(?: compartment| joint)?\b",
        r"\bpatello ?femoral\b",
        r"\bpatelofemoral\b",
        r"\bfemoropatellar\b",
        r"\bfemoropatelar\b",
        r"\bfemoropatellair\b",
        r"\bpatellofemorale\b",
        r"\bpatellofemoraal\b",
        r"\bpatellar cartilage\b",
        r"\btrochlear cartilage\b",
        r"\bpatellar facet\b",
        r"\btrochlea\b",
        r"\btrochlear\b",
    ),
}
OA_CONTEXT_REGEX = {
    target: tuple(re.compile(pattern, re.I) for pattern in patterns)
    for target, patterns in OA_CONTEXT_PATTERNS.items()
}
OA_ALL_CONTEXT_REGEX = tuple(
    (target, regex)
    for target, regexes in OA_CONTEXT_REGEX.items()
    for regex in regexes
)

# These are cartilage/OA findings rather than generic "degeneration", because a
# generic degenerative term may refer to a meniscus, tendon, or ligament.
OA_DISEASE_RE = re.compile(
    r"\b(?:"
    r"oa|"
    r"osteoarthritis|osteoarthrosis|osteoarthrose|osteoartritis|osteoartrit|osteoartroz|"
    r"arthrosis|arthrose|artrosis|artrose|gonarthrosis|gonartrosis|gonarthrose|gonartroz|"
    r"chondrosis|chondromalacia|chondropathy|kondromalazi|"
    r"osteophytes?|osteophytosis|osteofytes?|osteofyt\w*|"
    r"joint space narrowing|"
    r"kraakbeenlijden|kraakbeenverlies|kraakbeendefect\w*|"
    r"(?:full[- ]thickness|high[- ]grade|moderate|severe|advanced|mild|low[- ]grade)?\s*"
    r"(?:articular )?cartilage (?:loss|thinning|fissur\w*|defect\w*|wear\w*|damage\w*|degener\w*)|"
    r"(?:full[- ]thickness|high[- ]grade|moderate|severe|advanced|mild|low[- ]grade)?\s*"
    r"chondral (?:defect\w*|loss|fissur\w*|injur\w*|lesion\w*)"
    r")\b",
    re.I,
)
OA_NORMAL_RE = re.compile(
    r"\b(?:"
    r"no (?:focal )?(?:chondrosis|chondral (?:injury|lesion|defect)|cartilage (?:loss|defect|abnormality))|"
    r"without (?:focal )?(?:chondrosis|chondral (?:injury|lesion|defect)|cartilage (?:loss|defect|abnormality))|"
    r"(?:articular )?cartilage (?:appears |is )?(?:intact|normal|preserved|maintained|unremarkable)|"
    r"(?:patellar and trochlear )?cartilage appears congruent without chondral lesion"
    r")\b",
    re.I,
)
OA_TRICOMPARTMENTAL_RE = re.compile(
    r"\b(?:tricompartmental|tri-compartmental|three[- ]compartment)"
    r"[^.;]{0,80}\b(?:osteoarthritis|osteoarthrosis|arthrosis|chondrosis|osteophytes?)\b",
    re.I,
)


@dataclass
class RulePrediction:
    probability: float
    confidence: float
    mentioned: bool
    state: str = STATE_UNMENTIONED


def _oa_context_windows(norm: str, target: str) -> list[str]:
    """Return target-local text without leaking into a neighboring compartment.

    Reports often use headings such as ``MEDIAL COMPARTMENT:`` followed by
    several sentences. A sentence-only matcher misses those findings, while an
    unrestricted radius can accidentally borrow lateral findings for the medial
    target. The window therefore starts slightly before the target marker and
    stops at the next marker belonging to a different compartment.
    """
    hits: list[tuple[int, int, str]] = []
    for marker_target, regex in OA_ALL_CONTEXT_REGEX:
        for match in regex.finditer(norm):
            hits.append((match.start(), match.end(), marker_target))
    hits.sort(key=lambda item: (item[0], item[1]))

    windows: list[str] = []
    for i, (start, end, marker_target) in enumerate(hits):
        if marker_target != target:
            continue

        previous_other_end = 0
        for _, previous_end, previous_target in reversed(hits[:i]):
            if previous_target != target:
                previous_other_end = previous_end
                break

        next_other_start = len(norm)
        for next_start, _, next_target in hits[i + 1 :]:
            if next_target != target:
                next_other_start = next_start
                break

        window_start = max(previous_other_end, start - 160)
        window_end = min(next_other_start, end + 850)
        windows.append(norm[window_start:window_end])
    return windows


def _oa_disease_is_negated(window: str, start: int) -> bool:
    left = window[max(0, start - 55) : start]
    return bool(
        re.search(
            r"\b(?:no|without|absent|negative for|sin|sans|kein|keine|ohne|geen|zonder|yok|degil|bez)"
            r"\b[^.;:]{0,35}$",
            left,
            re.I,
        )
    )


def _predict_oa_target(text: str, target: str) -> RulePrediction:
    norm = normalize_report(text)
    if OA_TRICOMPARTMENTAL_RE.search(norm):
        return RulePrediction(0.92, 0.80, True, STATE_POSITIVE)

    windows = _oa_context_windows(norm, target)
    if not windows:
        return RulePrediction(0.50, 0.03, False, STATE_UNMENTIONED)

    positive = False
    negated = False
    for window in windows:
        if OA_NORMAL_RE.search(window):
            negated = True
        for match in OA_DISEASE_RE.finditer(window):
            if _oa_disease_is_negated(window, match.start()):
                negated = True
            else:
                positive = True

    if positive:
        return RulePrediction(0.92, 0.80, True, STATE_POSITIVE)
    if negated:
        return RulePrediction(0.06, 0.80, True, STATE_NEGATED)

    # Merely mentioning a compartment is not evidence for OA.
    return RulePrediction(0.50, 0.03, False, STATE_UNMENTIONED)


def predict_target(text: str, target: str) -> RulePrediction:
    if target in OA_TARGETS:
        return _predict_oa_target(text, target)

    norm = normalize_report(text)
    any_mention = negated = positive = False
    for phrase in sorted(set(LEXICON[target]), key=len, reverse=True):
        for m in re.finditer(re.escape(normalize_report(phrase)), norm, re.I):
            any_mention = True
            left = norm[max(0, m.start() - 70):m.start()]
            if NEG_RE.search(left):
                negated = True
                continue
            local = norm[max(0, m.start() - 45):min(len(norm), m.end() + 90)]
            if target in {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"}:
                if POSITIVE_MODIFIERS.search(local):
                    positive = True
            else:
                positive = True
    if positive:
        return RulePrediction(0.92, 0.80, True, STATE_POSITIVE)
    if any_mention and negated:
        return RulePrediction(0.06, 0.80, True, STATE_NEGATED)
    if any_mention:
        return RulePrediction(0.50, 0.12, True, STATE_UNCERTAIN)
    return RulePrediction(0.50, 0.03, False, STATE_UNMENTIONED)


def predict_report(text: str) -> tuple[np.ndarray, np.ndarray]:
    pred = [predict_target(text, t) for t in TARGETS]
    return (
        np.array([x.probability for x in pred], np.float32),
        np.array([x.confidence for x in pred], np.float32),
    )


def label_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probs = np.zeros((len(df), len(TARGETS)), np.float32)
    conf = np.zeros_like(probs)
    for i, text in enumerate(df["Report"].fillna("").astype(str)):
        probs[i], conf[i] = predict_report(text)
    return probs, conf


def state_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Return the rule state per study and target as an ``[n, 12]`` array."""
    states = np.empty((len(df), len(TARGETS)), dtype=object)
    for i, text in enumerate(df["Report"].fillna("").astype(str)):
        for j, target in enumerate(TARGETS):
            states[i, j] = predict_target(text, target).state
    return states


def combine_gold_and_pseudo(
    df: pd.DataFrame,
    pseudo_probs: np.ndarray,
    pseudo_conf: np.ndarray,
    gold_weight: float = 8.0,
):
    """Combine pseudo-labels with official labels at the target-cell level.

    A partially annotated row must not turn every missing target into a negative.
    Only finite official target cells override the teacher and receive the gold
    weight; all other cells retain their pseudo-label and confidence.
    """
    targets = pseudo_probs.astype(np.float32).copy()
    weights = pseudo_conf.astype(np.float32).copy()
    gold = df[TARGETS].to_numpy(dtype=np.float32)
    mask = np.isfinite(gold)
    targets[mask] = gold[mask]
    weights[mask] = float(gold_weight)
    return targets, weights
