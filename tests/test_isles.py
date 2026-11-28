"""Tests of the ISLES'24 perfusion-volume extractor and demonstration node
(plan docs/registrations/v2_imaging_plan.md). Synthetic maps only; no ISLES'24 or registry data."""
import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from mtop import isles as I  # noqa: E402

VOX_MM = (2.0, 2.0, 2.0)          # 0.008 mL per voxel
VOX_ML = 0.008


def _maps(stray=False, small_brain=False):
    """Brain block with normal tissue (Tmax 2 s, CBF 50), a hypoperfused block of 1000 voxels
    (Tmax 10 s, CBF 40) and inside it a core of 100 voxels (CBF 10 = 20 % of the reference)."""
    t = np.zeros((40, 40, 20), np.float32)
    c = np.zeros_like(t)
    b = (slice(5, 15), slice(5, 15), slice(5, 15)) if small_brain else (slice(5, 35), slice(5, 35), slice(2, 18))
    t[b], c[b] = 2.0, 50.0
    t[10:20, 10:20, 5:15], c[10:20, 10:20, 5:15] = 10.0, 40.0
    c[12:17, 12:17, 6:10] = 10.0
    if stray:
        t[30, 30, 10], c[30, 30, 10] = 10.0, 40.0       # one isolated hypoperfused voxel
    return t, c


def test_volumes_match_the_definitions():
    t, c = _maps()
    v = I.volumes(t, c, VOX_ML, I.settings()["extractor"])
    assert v["ctp_tmax6_ml"] == pytest.approx(8.0)
    assert v["ctp_core_ml"] == pytest.approx(0.8)
    assert v["ctp_mismatch_ml"] == pytest.approx(7.2)
    assert v["ctp_mismatch_ratio"] == pytest.approx(10.0)
    assert v["ctp_mismatch_rel"] == pytest.approx(90.0)
    assert v["reference_cbf"] == pytest.approx(50.0)


def test_invalid_voxels_are_ignored():
    t, c = _maps()
    t[0, 0, 0], c[0, 0, 0] = 50.0, 5.0       # Tmax above the valid range
    t[1, 1, 1], c[1, 1, 1] = np.nan, 5.0     # not finite
    t[2, 2, 2], c[2, 2, 2] = 10.0, 0.0       # CBF 0 (outside the brain)
    v = I.volumes(t, c, VOX_ML, I.settings()["extractor"])
    assert v["ctp_tmax6_ml"] == pytest.approx(8.0)
    assert v["ctp_core_ml"] == pytest.approx(0.8)


def test_core_missing_without_enough_reference_tissue():
    t, c = _maps(small_brain=True)
    v = I.volumes(t, c, VOX_ML, I.settings()["extractor"])
    assert np.isnan(v["ctp_core_ml"]) and np.isnan(v["ctp_mismatch_ratio"])
    assert v["ctp_tmax6_ml"] > 0


def test_cluster_cleaning_only_in_sensitivity_analysis():
    t, c = _maps(stray=True)
    ex = I.settings()["extractor"]
    assert I.volumes(t, c, VOX_ML, ex)["ctp_tmax6_ml"] == pytest.approx(8.008)
    assert I.volumes(t, c, VOX_ML, ex, clean=True)["ctp_tmax6_ml"] == pytest.approx(8.0)


def test_grids_must_match():
    with pytest.raises(ValueError):
        I.volumes(np.zeros((4, 4, 4)), np.zeros((4, 4, 5)), VOX_ML, I.settings()["extractor"])


def _save(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, np.diag(list(VOX_MM) + [1.0])), str(path))


def _tree(root, n=2):
    for k in range(1, n + 1):
        cid = f"sub-strokecase{k:04d}"
        t, c = _maps()
        pm = root / "raw_data" / cid / "ses-0001" / "perfusion-maps"
        _save(t, pm / f"{cid}_ses-0001_tmax.nii.gz")
        _save(c, pm / f"{cid}_ses-0001_cbf.nii.gz")
        les = np.zeros((40, 40, 20), np.float32)
        les[10:20, 10:20, 5:5 + k] = 1                   # 100*k voxels
        _save(les, root / "derivatives" / cid / "ses-0002" / f"{cid}_ses-0002_lesion-msk.nii.gz")
        ph = root / "phenotype" / cid / "ses-0001"
        ph.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"age": [70 + k], "sex": ["M"], "nihss_admission": [12]}).to_csv(
            ph / f"{cid}_ses-0001_demographic_baseline.csv", index=False)
        pd.DataFrame({"mrs_3month": [k]}).to_csv(ph / f"{cid}_ses-0001_outcome.csv", index=False)


def test_extract_reads_a_bids_tree(tmp_path):
    _tree(tmp_path)
    vol = I.extract(tmp_path).set_index("case")
    assert list(vol.index) == ["sub-strokecase0001", "sub-strokecase0002"]
    assert (vol["error"] == "").all()
    assert vol["ctp_tmax6_ml"].tolist() == pytest.approx([8.0, 8.0])
    assert vol["final_infarct_ml"].tolist() == pytest.approx([0.8, 1.6])


def test_headers_show_names_and_missing_counts_only(tmp_path):
    _tree(tmp_path)
    h = I.headers(tmp_path)
    assert h["cases"] == 2
    assert set(h["columns"]) == {"age", "sex", "nihss_admission", "mrs_3month"}
    assert set(h) == {"cases", "columns", "missing", "tmax_value_ranges"}   # no clinical values
    assert h["tmax_value_ranges"][0]["tmax_p5_p50_p95"][1] == pytest.approx(2.0)


def test_site_csv_is_read_by_the_network_template(tmp_path):
    from mtop.fed.schema import read_site_csv

    _tree(tmp_path)
    cfg = I.settings()
    clin, rep = I.map_clinical(I.clinical(tmp_path, cfg), cfg)
    assert rep["mapped"] == {"age": "age", "sex_male": "sex", "nihss_adm": "nihss_admission", "mrs90": "mrs_3month"}
    site = I.build_site(I.extract(tmp_path, cfg), clin, cfg)
    p = tmp_path / "site.csv"
    site.to_csv(p, index=False)
    df, report = read_site_csv(p)
    assert report["rows"] == 2 and report["with_outcome"] == 2
    assert df["age"].tolist() == [71.0, 72.0] and df["sex_male"].tolist() == [1.0, 1.0]
    assert df["ctp_core_ml"].tolist() == pytest.approx([0.8, 0.8])
    assert report["future_recorded"]["ctp_tmax6_ml"] == 2
    assert report["unrecognised_values"] == {}


def test_agreement_is_aggregate_and_reproducible():
    rng = np.random.default_rng(0)
    core = rng.gamma(2.0, 10.0, 60)
    vol = pd.DataFrame({"ctp_core_ml": core, "final_infarct_ml": core * 1.3 + rng.normal(0, 3, 60)})
    a, b = I.agreement(vol), I.agreement(vol)
    assert a == b and a["n"] == 60 and a["usable"] is True
    assert a["spearman_ci95"][0] <= a["spearman"] <= a["spearman_ci95"][1]
    assert I.agreement(vol.head(5))["n"] == 5


# ---------------------------------------------------------------- DICOM input
pydicom = pytest.importorskip("pydicom")


def _dicom_series(arr, folder, slope=1.0, intercept=0.0, rgb=False, uid="1.2.3"):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    folder.mkdir(parents=True, exist_ok=True)
    stored = np.round((arr - intercept) / slope).astype(np.uint16)
    for k in range(arr.shape[2]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = f"{uid}.{k + 1}"
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"im{k:03d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SeriesInstanceUID = uid
        ds.Rows, ds.Columns = arr.shape[0], arr.shape[1]
        ds.PixelSpacing = [VOX_MM[0], VOX_MM[1]]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, (arr.shape[2] - 1 - k) * VOX_MM[2]]      # written in reverse order
        ds.InstanceNumber = k + 1
        ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        if rgb:
            ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
            ds.PixelData = np.zeros((arr.shape[0], arr.shape[1], 3), np.uint8).tobytes()
        else:
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.PixelData = np.ascontiguousarray(stored[:, :, arr.shape[2] - 1 - k]).tobytes()
        ds.save_as(str(folder / f"im{k:03d}.dcm"), enforce_file_format=True)


def test_dicom_series_gives_the_same_volumes_as_nifti(tmp_path):
    t, c = _maps()
    _dicom_series(t, tmp_path / "tmax", slope=0.01, uid="1.2.3.1")       # Tmax stored in 1/100 s
    _dicom_series(c, tmp_path / "cbf", uid="1.2.3.2")
    arr, vox = I.load_map(tmp_path / "tmax")
    assert vox == pytest.approx(VOX_ML)
    assert np.allclose(arr, t, atol=0.006)
    out = I.perfusion(tmp_path / "tmax", tmp_path / "cbf")
    assert out["ctp_tmax6_ml"] == pytest.approx(8.0) and out["ctp_core_ml"] == pytest.approx(0.8)
    assert out["ctp_mismatch_ratio"] == pytest.approx(10.0)


def test_colour_dicom_is_refused(tmp_path):
    t, _ = _maps()
    _dicom_series(t[:, :, :3], tmp_path / "shot", rgb=True)
    with pytest.raises(ValueError, match="colour"):
        I.load_map(tmp_path / "shot")
