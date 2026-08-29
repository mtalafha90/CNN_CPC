# Runtime environments

## Local B42 development

Use the project Conda environment, then install the repository editable:

```bash
conda activate rsna-knee
pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

The editable package must expose both the top-level compatibility modules and
`rsna_knee`. Do not rely on an ad-hoc `PYTHONPATH` for a release or submission
artifact.

## Kaggle B42/B49 inference

The hidden-safe endpoints are PyTorch CUDA workflows. Use two GPUs when the
endpoint requires it; B42 and B49 were validated with **T4 x2**. Do not replace
the frozen inference module with a generic single-GPU loop.

Some RSNA DICOMs use compressed transfer syntaxes. Before importing the model
or decoding data, verify the required offline decoders. The completed Kaggle
path used an attached offline GDCM/Python package and installed `python-gdcm`;
internet installation is not part of the runtime contract.

```python
from pydicom.pixels import get_decoder

required = {
    "JPEG Lossless P14": "1.2.840.10008.1.2.4.57",
    "JPEG Lossless SV1": "1.2.840.10008.1.2.4.70",
    "JPEG2000 Lossless": "1.2.840.10008.1.2.4.90",
    "JPEG2000": "1.2.840.10008.1.2.4.91",
}
for name, uid in required.items():
    decoder = get_decoder(uid)
    assert decoder.is_available, f"Missing decoder: {name} ({decoder.missing_dependencies})"
```

If this check fails, stop and repair the offline runtime dependency. Do not
fall back to a different decoder, altered preprocessing, or partial submission.

## Verification before a release

Run the focused B42 regression tests in the project environment:

```bash
pip install -e ".[test]"
python -m pytest -q \
  developments/tests/test_b42_constant_area_aspect_sparse.py \
  developments/tests/test_b42_train_versus_submission_preprocessing.py
```

For a Kaggle candidate, also run the endpoint's visible-study numerical
equivalence check before the hidden rerun. The appropriate frozen notebook
instructions remain with that endpoint's submission documentation.
