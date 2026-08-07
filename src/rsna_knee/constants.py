from __future__ import annotations

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
TARGET_SLUGS = [
    "acl",
    "mcl",
    "medial_meniscus",
    "lateral_meniscus",
    "medial_oa",
    "lateral_oa",
    "pf_oa",
    "effusion",
    "synovitis",
    "bakers",
    "contusion",
    "fracture",
]
DISPLAY_TO_SLUG = dict(zip(TARGETS, TARGET_SLUGS))
SLUG_TO_DISPLAY = dict(zip(TARGET_SLUGS, TARGETS))
SUBMISSION_COLUMNS = ["StudyInstanceUID", *TARGETS]
N_TARGETS = len(TARGETS)

DUAL_STREAMS = [
    "sagittal_fluid",
    "sagittal_structural",
    "coronal_fluid",
    "coronal_structural",
    "axial_fluid",
    "axial_structural",
]
N_STREAMS = len(DUAL_STREAMS)
