"""Batch interface to the vendored MadGraph/SMEFTsim tree matrix element."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

from SMEFT.build_backend import ROOT, ensure_backend, sha256, source_fingerprint


_DOUBLE = ctypes.POINTER(ctypes.c_double)
_INT = ctypes.POINTER(ctypes.c_int)
_ERROR_SIZE = 2048


class NativeHBBackend:
    """Only cHB varies; all electroweak inputs and widths stay fixed.

    ``evaluate`` returns coefficients a,b of 1+x*a+x*x*b, where
    x = C_HB / (Lambda / TeV)**2. The native reference uses Lambda=1000 GeV.
    The input order is arbitrary; PDG codes identify the charged leptons.
    The C++ library serializes access to the generated model singleton.
    """

    def __init__(self, *, build_dir: str | Path | None = None):
        provenance = json.loads((ROOT / "generated/provenance.json").read_text())
        card = ROOT / "cards/param_card.dat"
        if sha256(card) != provenance.get("param_card_sha256"):
            raise ValueError("parameter card differs from the certified fixed-input backend; regenerate and validate before using new inputs")
        library = ensure_backend(build_dir)
        self._library = ctypes.CDLL(str(library))
        self._library.hb_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self._library.hb_init.restype = ctypes.c_int
        self._library.hb_evaluate.argtypes = [ctypes.c_size_t, _DOUBLE, _INT, _DOUBLE, _DOUBLE,
                                             ctypes.c_char_p, ctypes.c_size_t]
        self._library.hb_evaluate.restype = ctypes.c_int
        self._library.hb_matrix_elements.argtypes = [ctypes.c_size_t, _DOUBLE, _INT, ctypes.c_double,
                                                    _DOUBLE, ctypes.c_char_p, ctypes.c_size_t]
        self._library.hb_matrix_elements.restype = ctypes.c_int
        error = ctypes.create_string_buffer(_ERROR_SIZE)
        if self._library.hb_init(str(card).encode(), error, len(error)):
            raise RuntimeError(error.value.decode(errors="replace"))
        self.metadata = {
            "backend": "MadGraph5_aMC standalone_cpp / SMEFTsim",
            "generated": provenance,
            "source_sha256": source_fingerprint(), "param_card_sha256": sha256(card),
            "build": json.loads(library.with_suffix(".json").read_text()),
            "scope": "LO SMEFT/SM decay-ME deformation of generator baseline; fixed widths; unchanged production",
            "operator": "(H^dagger H) B_mu_nu B^mu_nu",
            "lagrangian": "L_SM + C_HB/Lambda^2 * O_HB",
            "polynomial_variable": "x = C_HB/(Lambda/TeV)^2",
            "polynomial": "|M_SM+x*M_HB|^2/|M_SM|^2 = 1+x*linear+x^2*quadratic",
            "channels": "massless direct 4e, 4mu, 2e2mu; identical exchange included",
            "eft_order": "one dimension-six insertion in amplitude, including its squared contribution",
        }

    @staticmethod
    def _inputs(momenta, pdg_ids):
        raw_p = np.asarray(momenta)
        raw_ids = np.asarray(pdg_ids)
        if raw_p.ndim != 3 or raw_p.shape[1:] != (4, 4):
            raise ValueError("momenta must have shape (N,4,4) in E,px,py,pz order")
        if raw_p.dtype.kind not in "fiu":
            raise ValueError("momenta must contain real numbers")
        if raw_ids.shape != raw_p.shape[:2] or raw_ids.dtype.kind not in "iu":
            raise ValueError("pdg_ids must be an integer array of shape (N,4)")
        if not np.all(np.isin(raw_ids, [11, -11, 13, -13])):
            raise ValueError("the native backend supports direct electrons and muons only")
        if not np.all(np.isfinite(raw_p)):
            raise ValueError("momenta must be finite")
        return np.ascontiguousarray(raw_p, dtype=np.float64), np.ascontiguousarray(raw_ids, dtype=np.int32)

    def evaluate(self, momenta, pdg_ids) -> tuple[np.ndarray, np.ndarray]:
        p, ids = self._inputs(momenta, pdg_ids)
        linear, quadratic = np.empty(len(p)), np.empty(len(p))
        error = ctypes.create_string_buffer(_ERROR_SIZE)
        code = self._library.hb_evaluate(len(p), p.ctypes.data_as(_DOUBLE), ids.ctypes.data_as(_INT),
                                         linear.ctypes.data_as(_DOUBLE), quadratic.ctypes.data_as(_DOUBLE),
                                         error, len(error))
        if code:
            raise ValueError(error.value.decode(errors="replace"))
        if not np.all(np.isfinite(linear) & np.isfinite(quadratic)):
            raise ValueError("native matrix element returned nonfinite coefficients")
        tolerance = 1.e-10 * np.maximum(1., np.maximum(linear * linear, np.abs(quadratic)))
        if np.any(quadratic < -tolerance) or np.any(linear * linear > 4 * quadratic + tolerance):
            raise ValueError("matrix-element polynomial fails amplitude positivity")
        quadratic = np.maximum(quadratic, 0.)
        return linear, quadratic

    def matrix_elements(self, momenta, pdg_ids, coefficient: float) -> np.ndarray:
        """Direct evaluation at x, useful for independent polynomial checks."""
        if not np.isfinite(coefficient):
            raise ValueError("coefficient must be finite")
        p, ids = self._inputs(momenta, pdg_ids)
        output = np.empty(len(p))
        error = ctypes.create_string_buffer(_ERROR_SIZE)
        code = self._library.hb_matrix_elements(
            len(p), p.ctypes.data_as(_DOUBLE), ids.ctypes.data_as(_INT), float(coefficient),
            output.ctypes.data_as(_DOUBLE), error, len(error))
        if code:
            raise ValueError(error.value.decode(errors="replace"))
        if not np.all(np.isfinite(output) & (output >= 0)):
            raise ValueError("native matrix element returned an invalid squared amplitude")
        return output
