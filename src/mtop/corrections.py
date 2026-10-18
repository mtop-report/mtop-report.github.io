"""Corrections to the registry export, kept apart from the export itself (local, FNO only).

A correction says: in the record of patient P, registry column C read `old_value` and should read
`new_value`, for this reason, entered by this person at this time. `python -m mtop clean` applies
the active corrections to a copy of the export before cleaning, so

* the registry file itself is never overwritten;
* every change can be traced and undone (reverting keeps the row, with who reverted it and when);
* a correction is applied only while the export still holds the value it was made against. When
  the source has meanwhile been corrected the same way it is reported as "already in the source",
  and when the source holds something else it is a conflict and is not applied.

A "confirmed" entry records that a value flagged for review was checked and is correct; it
changes nothing in the data.

Files (all under data/studio/, which git ignores): corrections.csv, audit.csv (append-only log of
every action) and password.json (scrypt hash of the studio password). They contain patient IDs
and must stay at FNO.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["id", "patient_id", "column", "old_value", "new_value", "kind", "reason", "issue", "author", "created",
          "status", "reverted_by", "reverted_at"]
AUDIT_FIELDS = ["time", "author", "action", "id", "patient_id", "column", "old_value", "new_value", "reason"]


def studio_dir() -> Path:
    """data/studio, or MTOP_STUDIO_DIR (read at call time, so tests can point elsewhere)."""
    env = os.environ.get("MTOP_STUDIO_DIR")
    if env:
        return Path(env)
    from .paths import DATA
    return DATA / "studio"


def corrections_file() -> Path:
    return studio_dir() / "corrections.csv"


def audit_file() -> Path:
    return studio_dir() / "audit.csv"


def password_file() -> Path:
    return studio_dir() / "password.json"


def _now() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat(sep=" ")


# ----------------------------------------------------------------------------------------- values
def norm_value(v) -> str:
    """One text form per recorded value, for comparing what the export holds with what a
    correction was made against: empty -> "", 25.0 -> "25", dates -> YYYY-MM-DD (with time if any)."""
    if v is None:
        return ""
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        if pd.isna(v):
            return ""
        t = pd.Timestamp(v)
        return t.strftime("%Y-%m-%d") if (t.hour, t.minute, t.second) == (0, 0, 0) else t.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, dt.time):
        return v.strftime("%H:%M")
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{int(v)}" if float(v).is_integer() else f"{float(v):g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    s = str(v).strip()
    return "" if s.lower() in ("nan", "nat", "none") else s


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    vals = series.dropna()
    if len(vals) == 0:
        return "text"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    num = pd.to_numeric(vals.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return "number" if num.notna().mean() > 0.9 else "text"


def typed_value(new: str, kind: str):
    """A correction's text as the export would hold it (number, date or text); "" = empty."""
    s = (new or "").strip()
    if s == "":
        return np.nan if kind != "text" else None
    if kind == "number":
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return s
    if kind == "date":
        t = pd.to_datetime(s, errors="coerce", dayfirst=bool(re.match(r"^\d{1,2}\.", s)))
        return s if pd.isna(t) else t
    return s


# ----------------------------------------------------------------------------------------- store
def load_corrections() -> pd.DataFrame:
    f = corrections_file()
    if not f.exists():
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    for c in FIELDS:
        if c not in df.columns:
            df[c] = ""
    return df[FIELDS]


def _write(df: pd.DataFrame) -> None:
    f = corrections_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    df[FIELDS].to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    tmp.replace(f)


def audit(action: str, author: str, **kw) -> None:
    f = audit_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AUDIT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": _now(), "author": author, "action": action, **{k: kw.get(k, "") for k in AUDIT_FIELDS[3:]}})


def load_audit() -> pd.DataFrame:
    f = audit_file()
    return pd.read_csv(f, dtype=str, keep_default_na=False) if f.exists() else pd.DataFrame(columns=AUDIT_FIELDS)


def add_correction(patient_id, column: str, old_value, new_value, reason: str, author: str,
                   kind: str = "correction", issue: str = "") -> str:
    """Record a correction (or, kind="confirmed", that a flagged value is right). Returns its id.
    An earlier active entry for the same patient and column is superseded (reverted)."""
    if not str(author or "").strip():
        raise ValueError("author is required")
    if kind not in ("correction", "confirmed"):
        raise ValueError(kind)
    df = load_corrections()
    pid = str(int(float(patient_id)))
    old, new = norm_value(old_value), norm_value(new_value) if kind == "correction" else norm_value(old_value)
    if kind == "correction" and old == new:
        raise ValueError("the new value equals the recorded value")
    same = (df["patient_id"] == pid) & (df["column"] == column) & (df["status"] == "active")
    for i in df.index[same]:
        df.loc[i, ["status", "reverted_by", "reverted_at"]] = ["superseded", author, _now()]
    cid = uuid.uuid4().hex[:10]
    row = {"id": cid, "patient_id": pid, "column": column, "old_value": old, "new_value": new, "kind": kind,
           "reason": reason or "", "issue": issue or "", "author": author, "created": _now(), "status": "active",
           "reverted_by": "", "reverted_at": ""}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write(df)
    audit("confirm" if kind == "confirmed" else "correct", author, id=cid, patient_id=pid, column=column,
          old_value=old, new_value=new, reason=reason or "")
    return cid


def revert_correction(cid: str, author: str) -> None:
    if not str(author or "").strip():
        raise ValueError("author is required")
    df = load_corrections()
    hit = df.index[(df["id"] == cid) & (df["status"] == "active")]
    if not len(hit):
        raise KeyError(cid)
    i = hit[0]
    df.loc[i, ["status", "reverted_by", "reverted_at"]] = ["reverted", author, _now()]
    _write(df)
    r = df.loc[i]
    audit("revert", author, id=cid, patient_id=r["patient_id"], column=r["column"], old_value=r["old_value"],
          new_value=r["new_value"], reason="")


# ----------------------------------------------------------------------------------------- apply
def apply_corrections(raw: pd.DataFrame, corrections: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the active corrections to a copy of the export. Returns (corrected raw, report) where the
    report has one row per active entry with result: applied | already in the source | conflict |
    patient not found | unknown column | confirmed."""
    corr = load_corrections() if corrections is None else corrections
    out = raw.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rows = []
    if corr is None or not len(corr):
        return out, pd.DataFrame(columns=FIELDS + ["result", "found"])
    ids = pd.to_numeric(out["Patient ID"], errors="coerce")
    kinds = {}
    for _, r in corr[corr["status"] == "active"].iterrows():
        res = dict(r)
        col = r["column"]
        hit = out.index[ids == float(r["patient_id"])] if str(r["patient_id"]).strip() else []
        if col not in out.columns:
            res.update(result="unknown column", found="")
        elif not len(hit):
            res.update(result="patient not found", found="")
        else:
            i = hit[0]
            cur = norm_value(out.at[i, col])
            res["found"] = cur
            if r["kind"] == "confirmed":
                res["result"] = "confirmed" if cur == r["old_value"] else "conflict"
            elif cur == r["old_value"]:
                if col not in kinds:
                    kinds[col] = _column_kind(raw[col] if col in raw.columns else out[col])
                val = typed_value(r["new_value"], kinds[col])
                if out[col].dtype != object and isinstance(val, str):
                    out[col] = out[col].astype(object)
                if pd.api.types.is_datetime64_any_dtype(out[col]) and not isinstance(val, pd.Timestamp):
                    out[col] = out[col].astype(object)
                out.at[i, col] = val
                res["result"] = "applied"
            elif cur == r["new_value"]:
                res["result"] = "already in the source"
            else:
                res["result"] = "conflict"
        rows.append(res)
    return out, pd.DataFrame(rows, columns=FIELDS + ["result", "found"])


# ----------------------------------------------------------------------------------------- password
def set_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("use at least 8 characters")
    salt = secrets.token_bytes(16)
    n, r, p = 2 ** 14, 8, 1
    h = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32)
    f = password_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"scheme": "scrypt", "n": n, "r": r, "p": p, "salt": salt.hex(), "hash": h.hex(),
                             "set": _now()}), encoding="utf-8")
    try:
        os.chmod(f, 0o600)
    except OSError:
        pass


def password_is_set() -> bool:
    return password_file().exists()


def check_password(password: str) -> bool:
    f = password_file()
    if not f.exists():
        return False
    d = json.loads(f.read_text(encoding="utf-8"))
    h = hashlib.scrypt((password or "").encode("utf-8"), salt=bytes.fromhex(d["salt"]), n=int(d["n"]),
                       r=int(d["r"]), p=int(d["p"]), dklen=32)
    return hmac.compare_digest(h.hex(), d["hash"])
