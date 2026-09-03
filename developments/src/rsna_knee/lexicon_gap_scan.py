"""Reports that plainly discuss arthritis, and the words B6 cannot place.

B6 does not find osteoarthritis by matching a name. It pairs a **compartment**
against a **disease term**, which is a careful design and the reason its OA
lexicon entries are documentation rather than working patterns:

```text
OA_DISEASE_RE          osteoarthrosis, chondropathy, cartilage loss, osteophytes, ...
OA_CONTEXT_PATTERNS    medial compartment, lateral tibial plateau, trochlea, ...
                       -> both must be present, and near each other
```

The disease vocabulary is broad and multilingual. The compartment vocabulary is
not. It is English and Germanic with a little Spanish, and the gap is visible in
a single report:

```text
the report says     "tróclea femoral", "cartílago rotuliano"
normalised          "troclea femoral", "cartilago rotuliano"
the pattern says    \\btrochlea\\b, \\bpatellar cartilage\\b
```

`trochlea` does not match `troclea`. A grade 4 trochlear lesion is invisible to
the patellofemoral compartment, and every OA call B6 then makes on that report
rests on whichever compartment word it *did* find.

## What this measures

Reports where a disease term fires but no compartment does. Those are reports
that discuss cartilage damage which B6 cannot attribute anywhere, and the words
around the disease term are the vocabulary the patterns are missing.

It does not guess at what to add. It counts what is there, in the corpus, in
whatever language the radiologist wrote, and leaves the reading to a person.

## What it is blind to

The window only exists where `OA_DISEASE_RE` already matched. A report whose
*disease* vocabulary is missing produces no window at all, so its compartment
words are never counted and never appear as candidates.

That is not hypothetical. `condropatia`, `condromalacia`, `lesion condral` and
`osteofitos` are all absent from the disease pattern, so a Spanish report using
them is invisible here however clearly it names the compartment -- which is
exactly why `troclea` places nothing in this scan despite a read study turning on
it. **The Spanish gap is on the disease side, and this tool cannot see it.**
Measuring that needs a separate scan over reports with no disease match at all.

## Why this is safe to do now

It changes nothing. No label moves, no export is written, no threshold is
chosen. It reads reports and reports word counts, so it can be run and argued
about without spending the expert studies on it -- which matters, because
whatever vocabulary is added afterwards must be justified by what the corpus
contains rather than by what improves a score on 58 studies.

## It reads patient text

Windows of report text reach the output file. Local only.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .data import load_train_csv, normalize_report
from .report_labels import OA_ALL_CONTEXT_REGEX, OA_CONTEXT_REGEX, OA_DISEASE_RE, OA_TARGETS

SCAN_VERSION = "lexicon_gap_v1"

# Enough context around a disease term to see which anatomy it belongs to.
WINDOW = 90

# Words too common to be anatomy, in the languages this corpus actually uses.
# Kept explicit rather than clever: a stop list nobody can read is a stop list
# nobody can correct.
STOPWORDS = {
    "de", "del", "la", "el", "los", "las", "y", "en", "con", "sin", "por", "para",
    "un", "una", "al", "se", "su", "es", "no", "the", "of", "and", "in", "with",
    "a", "to", "is", "at", "on", "or", "van", "het", "een", "der", "die", "das",
    "und", "mit", "im", "ve", "bir", "i", "u", "je", "na", "se", "que",
}


def unplaced_mentions(
    reports: pd.Series, *, window: int = WINDOW
) -> tuple[pd.DataFrame, dict]:
    """Every disease mention that no compartment pattern can attribute.

    A mention is *placed* when some compartment pattern matches within the same
    window, and unplaced when none does. Unplaced mentions are where the
    vocabulary runs out.
    """
    rows: list[dict] = []
    placed = unplaced = 0
    for uid, text in reports.items():
        norm = normalize_report(str(text or ""))
        if not norm:
            continue
        for match in OA_DISEASE_RE.finditer(norm):
            low = max(0, match.start() - window)
            high = min(len(norm), match.end() + window)
            around = norm[low:high]
            hits = sorted({
                target
                for target, regex in OA_ALL_CONTEXT_REGEX
                if regex.search(around)
            })
            if hits:
                placed += 1
                continue
            unplaced += 1
            rows.append(
                {
                    "StudyInstanceUID": str(uid),
                    "disease_term": match.group(0),
                    "window": around,
                }
            )
    frame = pd.DataFrame(rows, columns=["StudyInstanceUID", "disease_term", "window"])
    total = placed + unplaced
    return frame, {
        "disease_mentions": total,
        "placed_in_a_compartment": placed,
        "unplaced": unplaced,
        "unplaced_fraction": (unplaced / total) if total else 0.0,
        "studies_with_an_unplaced_mention": int(frame["StudyInstanceUID"].nunique()),
        "window": int(window),
    }


def candidate_vocabulary(frame: pd.DataFrame, *, top: int = 40) -> list[dict]:
    """The words that keep appearing beside a disease term B6 cannot place.

    Counted by how many *studies* each word appears in rather than by raw
    frequency, so one verbose report cannot invent a candidate on its own.
    """
    per_word: dict[str, set[str]] = {}
    for row in frame.itertuples(index=False):
        uid = str(row.StudyInstanceUID)
        for word in re.findall(r"[a-z]{4,}", str(row.window)):
            if word in STOPWORDS:
                continue
            per_word.setdefault(word, set()).add(uid)
    counted = Counter({word: len(uids) for word, uids in per_word.items()})
    return [
        {"word": word, "studies": count} for word, count in counted.most_common(top)
    ]


def pattern_coverage(reports: pd.Series) -> dict:
    """How often each compartment pattern fires at all, across the corpus.

    A pattern that never matches is not enforcing anything, and the list is long
    enough that nobody has checked.
    """
    normalised = [normalize_report(str(text or "")) for text in reports]
    result: dict[str, dict[str, int]] = {}
    for target in sorted(OA_TARGETS):
        counts = {}
        for regex in OA_CONTEXT_REGEX[target]:
            counts[regex.pattern] = sum(1 for norm in normalised if regex.search(norm))
        result[target] = dict(sorted(counts.items(), key=lambda pair: -pair[1]))
    return result


# Patterns proposed for a B6 v1.3, each traceable to something the corpus shows.
# Declared here so a change is argued from counts rather than written straight
# into the parser. Nothing imports these; only `simulate` reads them.
# What "patellar" must not be followed by. Reading thirty windows, every wrong
# placement was this and nothing else: the adjective attached to a structure
# that is not cartilage.
#
#   "patellar bursitis"        beside generic knee osteoarthritis
#   "patellar plicae"          beside generic knee osteoarthritis
#   "patellar tendons: mild tendinosis"
#   "patellar enthesopathy"
#
# The guard is a lookahead on the match rather than a filter on the window,
# because several *correct* placements mention a patellar tendon in the same
# sentence -- "moderate chondromalacia patella. mild patellar tendinosis" is a
# patellofemoral finding whatever else the sentence says.
NOT_CARTILAGE = r"(?!\s+(?:tendon|tendin|enthesop|bursit|plica|ligament|retinacul))"

CANDIDATE_PATTERNS: dict[str, tuple[str, ...]] = {
    # "chondromalacia patella" is the standard English phrase for patellofemoral
    # cartilage damage and matches nothing today: PF OA wants "patellofemoral",
    # "patellar cartilage", "patellar facet" or "trochlea". Bare patella and
    # patellar sit beside 108 and 148 unplaceable mentions.
    "PF OA": (
        r"\bpatella\b",
        r"\bpatellar\b" + NOT_CARTILAGE,
        r"\bpatellae\b",
        # \btrochlea\b cannot match the plural, and reports write "medial and
        # lateral trochleas".
        r"\btrochleas\b",
        r"\btrochlear\b",
        # Romance and Turkish forms of the same anatomy. These place nothing in
        # this scan, because the reports using them fail the *disease* pattern
        # and so never produce a window at all.
        r"\brotulian\w*\b",
        r"\brotula\b",
        r"\btroclea\w*\b",
    ),
    # "medial and lateral compartment(s)" defeats \bmedial compartment\b, which
    # currently yields Lateral positive and Medial silent on one sentence.
    "Medial OA": (
        r"\bmedial(?: and lateral)? compartments?\b",
        r"\bmedial(?:e|es|en)? femorotibial\w*\b",
        r"\bmediyal\b",
        r"\bcondilo femoral medial\b",
        r"\bplatillo tibial medial\b",
    ),
    "Lateral OA": (
        r"\b(?:medial and )?lateral compartments?\b",
        r"\blateral(?:e|es|en)? femorotibial\w*\b",
        r"\bcondilo femoral lateral\b",
        r"\bplatillo tibial lateral\b",
    ),
}


def pattern_examples(
    reports: pd.Series,
    target: str,
    pattern: str,
    *,
    window: int = WINDOW,
    limit: int = 25,
) -> pd.DataFrame:
    """The windows one proposed pattern would place or widen, for reading.

    A count cannot say whether widening is right. `\bpatellar\b` beside a
    cartilage term is the patellofemoral joint; beside "patellar tendon" it is a
    tendon and the widen is wrong. Only the text settles it.
    """
    regex = re.compile(pattern, re.I)
    rows: list[dict] = []
    for uid, text in reports.items():
        norm = normalize_report(str(text or ""))
        if not norm:
            continue
        for match in OA_DISEASE_RE.finditer(norm):
            low = max(0, match.start() - window)
            high = min(len(norm), match.end() + window)
            around = norm[low:high]
            if not regex.search(around):
                continue
            current = {t for t, r in OA_ALL_CONTEXT_REGEX if r.search(around)}
            if target in current:
                continue
            rows.append(
                {
                    "StudyInstanceUID": str(uid),
                    "effect": "places" if not current else "widens",
                    "already": ", ".join(sorted(current)),
                    "disease_term": match.group(0),
                    "window": around,
                }
            )
            if len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(
        rows,
        columns=["StudyInstanceUID", "effect", "already", "disease_term", "window"],
    )


def simulate(
    reports: pd.Series,
    additions: dict[str, tuple[str, ...]] | None = None,
    *,
    window: int = WINDOW,
) -> dict:
    """What each proposed pattern would newly place, and what it would disturb.

    Two numbers per pattern, and the second is the one that decides:

    ```text
    newly places   a disease mention no compartment could reach before
    widens         a mention already placed, which now gains another compartment
    ```

    A pattern that only places is a repair. A pattern that mostly widens is
    changing existing calls, which is a different and much riskier proposition:
    those cells already carry a label the model has trained on.

    Counted per study, so a verbose report cannot carry a pattern on its own.
    """
    additions = CANDIDATE_PATTERNS if additions is None else additions
    compiled = {
        (target, pattern): re.compile(pattern, re.I)
        for target, patterns in additions.items()
        for pattern in patterns
    }

    places: dict[tuple[str, str], set[str]] = {key: set() for key in compiled}
    widens: dict[tuple[str, str], set[str]] = {key: set() for key in compiled}
    newly_placed_studies: set[str] = set()

    for uid, text in reports.items():
        norm = normalize_report(str(text or ""))
        if not norm:
            continue
        for match in OA_DISEASE_RE.finditer(norm):
            low = max(0, match.start() - window)
            high = min(len(norm), match.end() + window)
            around = norm[low:high]
            current = {
                target
                for target, regex in OA_ALL_CONTEXT_REGEX
                if regex.search(around)
            }
            for (target, pattern), regex in compiled.items():
                if not regex.search(around):
                    continue
                if not current:
                    places[(target, pattern)].add(str(uid))
                    newly_placed_studies.add(str(uid))
                elif target not in current:
                    widens[(target, pattern)].add(str(uid))

    rows = [
        {
            "target": target,
            "pattern": pattern,
            "newly_places_studies": len(places[(target, pattern)]),
            "widens_studies": len(widens[(target, pattern)]),
        }
        for (target, pattern) in compiled
    ]
    return {
        "patterns": sorted(rows, key=lambda row: -row["newly_places_studies"]),
        "studies_newly_placed": len(newly_placed_studies),
        "window": int(window),
    }


def scan(
    *,
    data_root: str | Path,
    window: int = WINDOW,
    top: int = 40,
    out_root: str | Path | None = None,
) -> dict:
    train = load_train_csv(Path(data_root) / "train.csv")
    reports = pd.Series(
        train["Report"].fillna("").astype(str).to_numpy(),
        index=train["StudyInstanceUID"].astype(str),
    )

    frame, summary = unplaced_mentions(reports, window=window)
    result = {
        "version": SCAN_VERSION,
        "studies": int(len(reports)),
        **summary,
        "candidate_vocabulary": candidate_vocabulary(frame, top=top),
        "pattern_coverage": pattern_coverage(reports),
        "proposed": simulate(reports, window=window),
    }
    if out_root is not None:
        out = Path(out_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        frame.to_csv(out / "unplaced_mentions.csv", index=False)
        result["out_root"] = str(out)
    return result


def _report(result: dict) -> None:
    print()
    print(f"  studies scanned                    {result['studies']:>8,}")
    print(f"  disease mentions found             {result['disease_mentions']:>8,}")
    print(f"  placed in a compartment            {result['placed_in_a_compartment']:>8,}")
    print(
        f"  no compartment matched             {result['unplaced']:>8,}"
        f"   {result['unplaced_fraction'] * 100:5.1f}%"
    )
    print(f"  studies with an unplaced mention   {result['studies_with_an_unplaced_mention']:>8,}")

    print()
    print("  words beside a disease term B6 could not place, by study count")
    for item in result["candidate_vocabulary"]:
        print(f"    {item['word']:<28}{item['studies']:>7,}")

    print()
    print("  compartment patterns that never fire in the corpus")
    dead = [
        (target, pattern)
        for target, counts in result["pattern_coverage"].items()
        for pattern, count in counts.items()
        if count == 0
    ]
    for target, pattern in dead:
        print(f"    {target:<14}{pattern}")
    if not dead:
        print("    (none -- every pattern matches something)")

    proposed = result.get("proposed")
    if proposed:
        print()
        print(
            f"  What the proposed v1.3 patterns would do "
            f"({proposed['studies_newly_placed']:,} studies newly placed)"
        )
        print(f"    {'target':<12}{'pattern':<44}{'places':>8}{'widens':>8}")
        for row in proposed["patterns"]:
            if not row["newly_places_studies"] and not row["widens_studies"]:
                continue
            print(
                f"    {row['target']:<12}{row['pattern'][:42]:<44}"
                f"{row['newly_places_studies']:>8,}{row['widens_studies']:>8,}"
            )
        print(
            "\n    places = reached a mention nothing could reach, a repair.\n"
            "    widens = added a compartment to a mention already placed, which\n"
            "             changes a label the model has already trained on."
        )

    print()
    print(
        "  These are counts, not a proposal. A word appearing often beside an\n"
        "  unplaceable disease term is a candidate to read, not a pattern to add."
    )
    if "out_root" in result:
        print(f"\n  windows written to {result['out_root']}/unplaced_mentions.csv")
        print("  They contain report text. Local only -- do not commit them.")


def main() -> None:
    parser = argparse.ArgumentParser(
        "Find the compartment vocabulary B6's OA patterns are missing"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out-root", default=None)
    parser.add_argument(
        "--show",
        nargs=2,
        metavar=("TARGET", "PATTERN"),
        default=None,
        help="print the windows one proposed pattern would place or widen",
    )
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    if args.show:
        train = load_train_csv(Path(args.data_root) / "train.csv")
        reports = pd.Series(
            train["Report"].fillna("").astype(str).to_numpy(),
            index=train["StudyInstanceUID"].astype(str),
        )
        frame = pattern_examples(
            reports, args.show[0], args.show[1], window=args.window, limit=args.limit
        )
        print()
        for row in frame.itertuples(index=False):
            print(f"  [{row.effect}{(' after ' + row.already) if row.already else ''}]  {row.disease_term}")
            print(f"    {row.window}")
            print()
        print(f"  {len(frame)} shown. Report text -- local only.")
        return

    _report(
        scan(
            data_root=args.data_root,
            window=args.window,
            top=args.top,
            out_root=args.out_root,
        )
    )


if __name__ == "__main__":
    main()
