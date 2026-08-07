from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import normalize_report

NEG_RE = re.compile(r"\b(no|without|absent|negative for|sin|ausencia de|sans|pas de|kein|keine|ohne|niet|geen|zonder|yok|degil|nema|bez)\b", re.I)
POSITIVE_MODIFIERS = re.compile(r"tear|ruptur|lesion|injur|sprain|degener|damage|defect|fissur|rotur|desgar|riss|yirt|lezion", re.I)

LEXICON = {
    "ACL": ["acl", "anterior cruciate ligament", "ligamento cruzado anterior", "vorderes kreuzband", "voorste kruisband", "on capraz bag"],
    "MCL": ["mcl", "medial collateral ligament", "ligamento colateral medial", "mediales kollateralband", "mediale knieband"],
    "Medial Meniscus": ["medial meniscus", "menisco medial", "menisque medial", "innenmeniskus", "mediale meniscus", "medial meniskus"],
    "Lateral Meniscus": ["lateral meniscus", "menisco lateral", "menisque lateral", "aussenmeniskus", "laterale meniscus", "lateral meniskus"],
    "Medial OA": ["medial osteoarthritis", "medial gonarthrosis", "artrosis compartimento medial", "mediale gonarthrose", "medial kompartman osteoartrit"],
    "Lateral OA": ["lateral osteoarthritis", "lateral gonarthrosis", "artrosis compartimento lateral", "laterale gonarthrose", "lateral kompartman osteoartrit"],
    "PF OA": ["patellofemoral osteoarthritis", "patellofemoral oa", "patellofemoral arthrosis", "artrosis patelofemoral", "patellofemorale arthrose"],
    "Effusion": ["joint effusion", "effusion", "derrame articular", "epanchement articulaire", "gelenkerguss", "gewrichtseffusie", "eklem efüzyonu", "efüzyon"],
    "Synovitis": ["synovitis", "sinovitis", "synovite", "sinovit"],
    "Baker's": ["baker cyst", "baker's cyst", "popliteal cyst", "quiste de baker", "kyste de baker", "baker zyste", "baker kisti"],
    "Contusion": ["bone contusion", "bone bruise", "marrow contusion", "bone marrow edema", "bone marrow oedema", "edema oseo", "knochenmarkodem", "kemik kontuzyonu"],
    "Fracture": ["fracture", "fractura", "fraktur", "breuk", "kirik", "prijelom"],
}

@dataclass
class RulePrediction:
    probability: float
    confidence: float
    mentioned: bool


def predict_target(text: str, target: str) -> RulePrediction:
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
        return RulePrediction(0.92, 0.80, True)
    if any_mention and negated:
        return RulePrediction(0.06, 0.80, True)
    if any_mention:
        return RulePrediction(0.50, 0.12, True)
    return RulePrediction(0.50, 0.03, False)


def predict_report(text: str) -> tuple[np.ndarray, np.ndarray]:
    pred = [predict_target(text, t) for t in TARGETS]
    return np.array([x.probability for x in pred], np.float32), np.array([x.confidence for x in pred], np.float32)


def label_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probs = np.zeros((len(df), len(TARGETS)), np.float32)
    conf = np.zeros_like(probs)
    for i, text in enumerate(df["Report"].fillna("").astype(str)):
        probs[i], conf[i] = predict_report(text)
    return probs, conf


def combine_gold_and_pseudo(df: pd.DataFrame, pseudo_probs: np.ndarray, pseudo_conf: np.ndarray, gold_weight: float = 8.0):
    targets = pseudo_probs.astype(np.float32).copy()
    weights = pseudo_conf.astype(np.float32).copy()
    mask = df[TARGETS].notna().any(axis=1).to_numpy()
    if mask.any():
        targets[mask] = df.loc[mask, TARGETS].fillna(0).astype(np.float32).to_numpy()
        weights[mask] = gold_weight
    return targets, weights
