# External four-image knee MRI validation fixture

This directory is a **technical smoke-validation fixture**, not a competition validation set and not a source of leaderboard/scientific AUC.

The materialization workflow downloads four openly licensed Wikimedia Commons knee MRI images, keeps the original JPEG files under `source_jpgs/`, and wraps each single published slice into a seven-frame synthetic DICOM so the production DICOM/preprocessing code can be exercised with the normal directory contract:

```text
fixtures/external_validation/
  validation.csv
  validation_series.csv
  validation_images/
    EXTVAL_ACL_001/...
    EXTVAL_MEDMEN_001/...
    EXTVAL_BAKER_001/...
    EXTVAL_REFERENCE_001/...
  source_jpgs/
  sources.csv
  materialization.json
```

The repeated frames are **not an original MRI volume**. They exist only to test decoding, metadata routing, resizing, 2.5D construction, missing-stream masking, and model plumbing.

## Four sources

1. `EXTVAL_ACL_001` — anterior cruciate ligament rupture, sagittal PD-weighted MRI. Author: Hellerhoff. License: CC BY-SA 3.0. Source: `https://commons.wikimedia.org/wiki/File:MRT_VKB-Riss_PDW.jpg`.
2. `EXTVAL_MEDMEN_001` — grade 2 medial meniscal tear, coronal proton-density MRI. Authors: Nicolas Lefevre, Jean Francois Naouri, Serge Herman, Antoine Gerometta, Shahnaz Klouche, Yoann Bohu. License: CC BY 4.0. Source: `https://commons.wikimedia.org/wiki/File:Proton_density_MRI_of_a_grade_2_medial_meniscal_tear.jpg`.
3. `EXTVAL_BAKER_001` — Baker cyst in a patient with ACL rupture. Author: Hellerhoff. License: CC BY-SA 3.0. Source: `https://commons.wikimedia.org/wiki/File:MRT_Bakerzyste.jpg`.
4. `EXTVAL_REFERENCE_001` — sagittal PD TSE FS knee MRI with no pathology label supplied by the source. Author: Ptrump16. License: CC BY-SA 4.0. Source: `https://commons.wikimedia.org/wiki/File:Knee_MRI_PD_TSE_FS_Sagittal.jpg`.

`validation.csv` contains only source-supported positive target cells. Unspecified targets remain `NaN`; the absence of a finding in a Wikimedia caption is **not** converted into a negative label.

## Materialize manually

If the binary fixture has not yet been generated in your clone:

```bash
source .venv/bin/activate
pip install pillow
PYTHONPATH=src python scripts/materialize_external_validation.py \
  --output fixtures/external_validation \
  --overwrite
```

Then run the validation preflight:

```bash
rsna-knee preflight \
  --data-root fixtures/external_validation \
  --split validation \
  --sample-size 4 \
  --out runs/external_validation_preflight.json
```

This fixture must never be merged into the competition training data or used to choose final competition hyperparameters.
