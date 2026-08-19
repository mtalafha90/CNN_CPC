"""Every frozen-encoder guard must respect deliberate fine-tuning.

These guards were added when the encoder was always frozen, and they are spread
across the module: at startup, inside the batch loop, at the end of each epoch,
after training, and in the checkpoint loader. Enabling fine-tuning meant finding
all of them, and they were found one failed run at a time.

This test reads the source and insists that every fingerprint comparison sits
next to the flag that says whether the encoder was allowed to move, so a new
guard cannot silently reintroduce the problem.
"""

from __future__ import annotations

import inspect
import re

from rsna_knee import phase9_matched_supervision_training as trainer

# A guard is "aware" if it mentions the fine-tuning flag, the derived
# encoder_moved decision, or compares against the final rather than initial
# fingerprint.
AWARE = ("encoder_trainable_stages", "encoder_moved", "final_sha")


def _lines_comparing_fingerprints() -> list[tuple[int, str]]:
    source = inspect.getsource(trainer).splitlines()
    pattern = re.compile(r"(encoder_state_sha256|encoder_sha_initial|initial_sha)")
    found = []
    for number, line in enumerate(source, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if pattern.search(stripped) and ("!=" in stripped or "==" in stripped):
            found.append((number, stripped))
    return found


def test_the_guards_have_not_disappeared():
    """Guard the guard: the scan must still find real comparisons."""
    assert len(_lines_comparing_fingerprints()) >= 3


def test_every_fingerprint_guard_knows_about_fine_tuning():
    source = inspect.getsource(trainer).splitlines()
    unaware = []
    for number, line in _lines_comparing_fingerprints():
        # allow the flag to appear on the same line or the two lines above,
        # which covers `if <flag> and <comparison>:` and guarded blocks
        window = " ".join(source[max(0, number - 3):number])
        if not any(token in window for token in AWARE):
            unaware.append(f"line {number}: {line}")
    assert not unaware, (
        "these fingerprint guards would fire on a legitimately fine-tuned "
        "encoder:\n" + "\n".join(unaware)
    )


def test_no_guard_forbids_all_encoder_gradients():
    """The in-loop check must exempt parameters that were deliberately freed."""
    source = inspect.getsource(trainer)
    assert "detected an encoder gradient" not in source, (
        "a guard rejects any encoder gradient, which blocks fine-tuning "
        "part-way into a run"
    )
    assert "frozen encoder parameter" in source


def test_the_loader_checks_the_weights_it_actually_loaded():
    """A checkpoint holds the encoder as it ended, not as it began."""
    source = inspect.getsource(trainer.load_phase9_checkpoint)
    assert "final_sha" in source
    reconstruct = [l for l in source.splitlines() if "encoder_state_sha256" in l]
    assert reconstruct and all("initial_sha" not in l for l in reconstruct)
