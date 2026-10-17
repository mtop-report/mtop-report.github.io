"""Secure aggregation by pairwise masking (Bonawitz et al., CCS 2017, without dropout recovery).

For every job each centre makes a fresh X25519 key pair and signs the public half with its
long-term Ed25519 identity key. Every pair of centres (i, j) derives a shared 256-bit seed
(X25519 + HKDF-SHA256) that the coordinator cannot compute. In round r, centre i sends

    y_i = x_i + sum_{j > i} PRG(seed_ij, r) - sum_{j < i} PRG(seed_ij, r)   (mod 2^64)

where x_i is its vector of integer sums and PRG is the ChaCha20 key stream. Each y_i alone is
uniformly random; in the sum over all centres every mask appears once with + and once with -,
so the coordinator learns sum_i x_i exactly and nothing else. Real numbers travel as fixed-point
integers round(v * 2^24); negative values wrap around modulo 2^64.

If a centre leaves in the middle of a round its masks do not cancel and the round fails; the
job then stops (a new job needs fresh approval at every centre). This also blocks the
"difference attack" of asking the same question twice with different groups of centres.
"""
from __future__ import annotations

import base64
import hashlib
import struct

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import PROTOCOL

RAW = dict(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def pack(vec: np.ndarray) -> str:
    """uint64/int64 vector -> base64 (little endian)."""
    return b64(np.ascontiguousarray(vec).view(np.uint64).astype("<u8").tobytes())


def unpack(s: str) -> np.ndarray:
    return np.frombuffer(unb64(s), dtype="<u8").astype(np.uint64)


def to_fixed(x, bits: int) -> np.ndarray:
    return np.rint(np.asarray(x, float) * float(1 << bits)).astype(np.int64)


def from_fixed(v: np.ndarray, bits: int) -> np.ndarray:
    return np.asarray(v, np.int64).astype(float) / float(1 << bits)


def fingerprint(public_raw: bytes) -> str:
    h = hashlib.sha256(public_raw).hexdigest()[:16]
    return ":".join(h[i:i + 4] for i in range(0, 16, 4))


class Identity:
    """Long-term signing key of a centre (Ed25519). Its fingerprint is what other centres pin."""

    def __init__(self, key: Ed25519PrivateKey | None = None):
        self.key = key or Ed25519PrivateKey.generate()
        self.public = self.key.public_key().public_bytes(**RAW)

    @classmethod
    def load_or_create(cls, path) -> "Identity":
        from pathlib import Path

        p = Path(path)
        if p.exists():
            return cls(serialization.load_pem_private_key(p.read_bytes(), password=None))
        ident = cls()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(ident.key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption()))
        try:
            p.chmod(0o600)
        except OSError:
            pass
        return ident

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.public)

    def sign(self, msg: bytes) -> bytes:
        return self.key.sign(msg)


def _key_message(job_id: str, site_id: str, x_pub: bytes) -> bytes:
    return f"{PROTOCOL}|job-key|{job_id}|{site_id}|".encode() + x_pub


def verify_job_key(job_id: str, site_id: str, x_pub: bytes, identity_pub: bytes, sig: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(identity_pub).verify(sig, _key_message(job_id, site_id, x_pub))
        return True
    except (InvalidSignature, ValueError):
        return False


class JobKeys:
    """A centre's fresh key pair for one job and the pairwise seeds derived from it."""

    def __init__(self, job_id: str, site_id: str, identity: Identity):
        self.job_id, self.site_id = job_id, site_id
        self._sk = X25519PrivateKey.generate()
        self.public = self._sk.public_key().public_bytes(**RAW)
        self.signature = identity.sign(_key_message(job_id, site_id, self.public))
        self.seeds: dict[str, bytes] = {}

    def offer(self) -> dict:
        return {"x25519": b64(self.public), "sig": b64(self.signature)}

    def agree(self, peers: dict[str, dict], pinned: dict[str, str] | None = None) -> None:
        """peers: site_id -> {x25519, sig, identity} (base64). Verifies every signature and, when
        `pinned` (site_id -> identity fingerprint) is given, the identities themselves."""
        self.seeds = {}
        for sid, p in peers.items():
            if sid == self.site_id:
                continue
            ident = unb64(p["identity"])
            if pinned is not None and sid in pinned and pinned[sid] != fingerprint(ident):
                raise PermissionError(f"identity of {sid} does not match the pinned fingerprint")
            x_pub = unb64(p["x25519"])
            if not verify_job_key(self.job_id, sid, x_pub, ident, unb64(p["sig"])):
                raise PermissionError(f"key of {sid} is not signed by its identity")
            shared = self._sk.exchange(X25519PublicKey.from_public_bytes(x_pub))
            a, b = sorted((self.site_id, sid))
            info = f"{PROTOCOL}|pair|{self.job_id}|{a}|{b}".encode()
            self.seeds[sid] = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared)

    def mask(self, vec: np.ndarray, round_id: int) -> np.ndarray:
        """Add the pairwise masks for this round to an int64/uint64 vector (mod 2^64)."""
        out = np.ascontiguousarray(vec).view(np.uint64).copy() if vec.dtype == np.int64 else vec.astype(np.uint64)
        for sid, seed in self.seeds.items():
            stream = prg(seed, round_id, len(out))
            if self.site_id < sid:
                out += stream
            else:
                out -= stream
        return out


def prg(seed: bytes, round_id: int, n: int) -> np.ndarray:
    nonce = b"\x00" * 4 + struct.pack("<Q", int(round_id)) + b"\x00" * 4
    enc = Cipher(algorithms.ChaCha20(seed, nonce), mode=None).encryptor()
    return np.frombuffer(enc.update(b"\x00" * (8 * n)), dtype="<u8").astype(np.uint64)


def aggregate(masked: list[np.ndarray]) -> np.ndarray:
    """Sum of masked vectors modulo 2^64, read back as signed integers."""
    total = np.zeros_like(masked[0], dtype=np.uint64)
    for v in masked:
        total += v
    return total.view(np.int64)
