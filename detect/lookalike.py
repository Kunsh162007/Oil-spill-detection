"""Physics-based look-alike rejection. THE DIFFERENTIATOR.

Deliberately NOT a neural network. A small logistic model over interpretable
physical measurements, so every rejection can be explained out loud:

    "rejected: wind 1.2 m/s, below the 3 m/s detection threshold"

A judge can follow that. Nobody can follow a network hunch.

Two layers, in order:

  1. HARD PHYSICAL GATES. Conditions under which SAR physically cannot
     distinguish oil from water. These are not learned and cannot be
     out-voted by the model, because no amount of training data changes the
     fact that a mirror-calm sea is dark for reasons that have nothing to do
     with oil.
  2. A weighted logistic score over the remaining evidence.

Both layers report per-feature contributions so the UI can show its working.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.contracts import (
    WIND_HIGH_CUT_MS,
    WIND_LOW_CUT_MS,
    WindContext,
)
from detect.polygonize import RegionFeatures

log = logging.getLogger(__name__)


@dataclass
class LookalikeVerdict:
    """The outcome for one region, with its reasoning attached."""

    p_oil: float
    rejected_reason: str | None
    contributions: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    gate_hit: str | None = None

    @property
    def is_oil(self) -> bool:
        return self.rejected_reason is None


def build_features(region: RegionFeatures, wind: WindContext) -> dict[str, float]:
    """Physical feature vector for one region.

    Every entry is something a person can point at on a slide. Nothing here
    is a learned embedding.
    """
    damping = region.damping_ratio
    # Damping expressed in dB is the physically meaningful quantity: oil
    # typically suppresses VV backscatter by roughly 3-12 dB relative to the
    # surrounding sea.
    damping_db = (
        -10.0 * math.log10(damping)
        if damping and math.isfinite(damping) and damping > 0
        else 0.0
    )

    return {
        "wind_window_score": float(wind.window_score),
        "wind_speed_ms": float(wind.speed_ms),
        "damping_db": float(damping_db),
        "elongation": float(min(region.elongation, 50.0)),
        "compactness": float(region.compactness),
        "texture_homogeneity": float(region.texture_homogeneity),
        "texture_contrast": float(region.texture_contrast),
        "texture_variance": float(region.texture_variance),
        "area_km2": float(region.area_km2),
        "log_area": float(math.log10(max(region.area_km2, 1e-3))),
        "vh_vv_ratio": float(region.vh_vv_ratio) if region.vh_vv_ratio is not None else float("nan"),
        "has_vh": 0.0 if region.vh_vv_ratio is None else 1.0,
        "mean_confidence": float(region.mean_confidence),
    }


# --------------------------------------------------------------------------
# Layer 1 - hard physical gates
# --------------------------------------------------------------------------

@dataclass
class GateConfig:
    """Thresholds for the non-negotiable physical checks."""

    min_wind_ms: float = WIND_LOW_CUT_MS
    max_wind_ms: float = WIND_HIGH_CUT_MS
    min_damping_db: float = 1.0     # below this the patch is barely darker than sea
    min_area_km2: float = 0.05
    require_wind: bool = True


def apply_gates(
    features: dict[str, float], gates: GateConfig
) -> tuple[str, str] | None:
    """Return (gate_name, human_reason) if a hard physical gate rejects.

    Ordered most-decisive first so the printed reason is the one that
    actually matters, not the first one that happens to trip.
    """
    wind = features["wind_speed_ms"]

    if gates.require_wind and not math.isfinite(wind):
        return (
            "no_wind",
            "no wind data for this location and time - cannot assess (CLAUDE.md rule 3)",
        )

    if wind < gates.min_wind_ms:
        return (
            "wind_too_low",
            f"wind {wind:.1f} m/s is below the {gates.min_wind_ms:.1f} m/s floor - "
            f"calm water is indistinguishable from oil on SAR",
        )

    if wind > gates.max_wind_ms:
        return (
            "wind_too_high",
            f"wind {wind:.1f} m/s is above the {gates.max_wind_ms:.1f} m/s ceiling - "
            f"oil mixes into the wave field and cannot be resolved",
        )

    if features["damping_db"] < gates.min_damping_db:
        return (
            "insufficient_damping",
            f"only {features['damping_db']:.1f} dB darker than surrounding sea - "
            f"below the {gates.min_damping_db:.1f} dB floor for a real slick",
        )

    if features["area_km2"] < gates.min_area_km2:
        return (
            "too_small",
            f"area {features['area_km2']:.3f} km2 is below the "
            f"{gates.min_area_km2:.2f} km2 floor - indistinguishable from speckle",
        )

    return None


# --------------------------------------------------------------------------
# Layer 2 - weighted logistic score
# --------------------------------------------------------------------------

def _bump(x: float, lo: float, peak_lo: float, peak_hi: float, hi: float) -> float:
    """Trapezoidal membership, 0-1. Used where more is NOT monotonically better."""
    if not math.isfinite(x) or x <= lo or x >= hi:
        return 0.0
    if x < peak_lo:
        return (x - lo) / (peak_lo - lo)
    if x <= peak_hi:
        return 1.0
    return (hi - x) / (hi - peak_hi)


def transform_features(f: dict[str, float]) -> dict[str, float]:
    """Map raw physics onto 0-1 evidence terms, each signed towards oil.

    The transforms carry the domain knowledge; the coefficients only weigh
    them. That split is what keeps the model explainable - each term below
    has a one-line physical justification.
    """
    t: dict[str, float] = {}

    # Wind, signed: +1 mid-window, -1 at the very edge. Being INSIDE the
    # window is a precondition that apply_gates already enforces, so a raw
    # 0-1 score here would hand every look-alike a large positive term for
    # merely occurring on an ordinary day. Centring it means marginal wind
    # actively lowers confidence, which is the honest reading.
    t["wind"] = 2.0 * f["wind_window_score"] - 1.0

    # Oil damps VV by roughly 3-12 dB. Far more than that is usually a
    # radar shadow or a no-data hole, not a slick.
    t["damping"] = _bump(f["damping_db"], 1.0, 5.0, 12.0, 20.0)

    # A long thin taper is the signature of a moving vessel discharging.
    t["elongation"] = min(f["elongation"] / 12.0, 1.0)

    # High compactness (round) leans algal bloom or rain cell, not a dump.
    t["roundness_penalty"] = min(max(f["compactness"], 0.0), 1.0)

    # Mineral oil presents a smoother, more homogeneous damped surface than
    # patchy biogenic film.
    t["homogeneity"] = min(max(f["texture_homogeneity"], 0.0), 1.0)

    # Internal waves and rain cells show strong periodic/granular contrast.
    t["contrast_penalty"] = min(f["texture_contrast"] / 50.0, 1.0)

    # Very small features are dominated by residual speckle.
    t["size"] = min(max((f["log_area"] + 1.0) / 2.0, 0.0), 1.0)

    # Cross-pol: oil suppresses VH towards the noise floor harder than most
    # look-alikes do. Only contributes on dual-pol scenes.
    if f["has_vh"] > 0.5 and math.isfinite(f["vh_vv_ratio"]):
        t["vh_vv"] = 1.0 - min(max(f["vh_vv_ratio"] / 0.35, 0.0), 1.0)
    else:
        t["vh_vv"] = 0.0

    return t


# Log-odds weights. Hand-set priors, used until scripts/train.py fits real
# ones from the Yang et al. look-alike clusters. Signs are physics, not fit.
PRIOR_WEIGHTS: dict[str, float] = {
    "wind": 1.60,
    "damping": 2.40,
    "elongation": 1.80,
    "roundness_penalty": -2.20,
    "homogeneity": 1.40,
    "contrast_penalty": -2.60,
    "size": 0.40,
    "vh_vv": 0.90,
}
PRIOR_BIAS = -4.40


class LookalikeModel:
    """Interpretable logistic model over physical features.

    Ships with hand-set priors so the stage works before any training has
    happened, and loads fitted weights from disk when scripts/train.py has
    produced them.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        bias: float = PRIOR_BIAS,
        gates: GateConfig | None = None,
        threshold: float = 0.5,
        source: str = "prior",
    ) -> None:
        self.weights = dict(weights or PRIOR_WEIGHTS)
        self.bias = float(bias)
        self.gates = gates or GateConfig()
        self.threshold = float(threshold)
        self.source = source

    def score(self, features: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Return (p_oil, per-term log-odds contributions)."""
        terms = transform_features(features)
        contributions = {
            name: self.weights.get(name, 0.0) * value for name, value in terms.items()
        }
        logit = self.bias + sum(contributions.values())
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        contributions["_bias"] = self.bias
        return p, contributions

    def classify(
        self, region: RegionFeatures, wind: WindContext
    ) -> LookalikeVerdict:
        """Full verdict for one region: gates first, then the score."""
        features = build_features(region, wind)

        gate = apply_gates(features, self.gates)
        if gate is not None:
            gate_name, reason = gate
            # A gated region still gets a score, so the UI can show how close
            # it came - but the rejection stands regardless of the number.
            p, contributions = self.score(features)
            return LookalikeVerdict(
                p_oil=round(min(p, 0.2), 4),
                rejected_reason=reason,
                contributions=contributions,
                features=features,
                gate_hit=gate_name,
            )

        p, contributions = self.score(features)
        if p < self.threshold:
            reason = self._explain_low_score(p, contributions)
            return LookalikeVerdict(
                p_oil=round(p, 4),
                rejected_reason=reason,
                contributions=contributions,
                features=features,
            )

        return LookalikeVerdict(
            p_oil=round(p, 4),
            rejected_reason=None,
            contributions=contributions,
            features=features,
        )

    def _explain_low_score(self, p: float, contributions: dict[str, float]) -> str:
        """Name the single most damaging piece of evidence, in plain words."""
        negatives = {
            k: v for k, v in contributions.items() if k != "_bias" and v < 0
        }
        weakest_positive = min(
            ((k, v) for k, v in contributions.items() if k != "_bias" and v >= 0),
            key=lambda kv: kv[1],
            default=None,
        )

        phrases = {
            "roundness_penalty": "shape is round and blob-like rather than a discharge streak",
            "contrast_penalty": "surface texture is granular/periodic, typical of rain cells or internal waves",
            "wind": "wind is at the edge of the detection window",
            "damping": "backscatter suppression is weak for mineral oil",
            "elongation": "shape is not elongated like a vessel discharge",
            "homogeneity": "surface texture is patchy, more typical of biogenic film",
            "size": "feature is small enough to be residual speckle",
            "vh_vv": "cross-pol response does not match mineral oil",
        }

        if negatives:
            worst = min(negatives.items(), key=lambda kv: kv[1])[0]
            detail = phrases.get(worst, worst)
        elif weakest_positive is not None:
            detail = phrases.get(weakest_positive[0], weakest_positive[0])
        else:
            detail = "combined physical evidence is weak"

        return f"P(oil) {p:.2f} below {self.threshold:.2f} threshold - {detail}"

    def save(self, path: str | Path) -> Path:
        import json

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "weights": self.weights,
                    "bias": self.bias,
                    "threshold": self.threshold,
                    "source": self.source,
                    "gates": self.gates.__dict__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> LookalikeModel:
        import json

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Look-alike model not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            weights=data["weights"],
            bias=data["bias"],
            gates=GateConfig(**data.get("gates", {})),
            threshold=data.get("threshold", 0.5),
            source=data.get("source", "trained"),
        )

    @classmethod
    def load_or_prior(cls, path: str | Path | None, threshold: float = 0.5) -> LookalikeModel:
        """Load fitted weights if present, otherwise fall back to priors.

        The fallback is announced loudly - a silently-untrained model that
        looks trained is exactly the kind of thing that survives to demo day.
        """
        if path:
            p = Path(path)
            if p.exists():
                model = cls.load(p)
                log.info("Look-alike model loaded from %s (source=%s)", p, model.source)
                return model
        log.warning(
            "No fitted look-alike model at %s - using hand-set physical priors. "
            "Run scripts/train.py --stage lookalike to fit real weights.",
            path,
        )
        return cls(threshold=threshold, source="prior")
