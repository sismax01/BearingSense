# =============================================================================
# BearingSense — Bearing Kinematics & Characteristic Defect Frequencies
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Bearing kinematics and characteristic defect frequency calculations.

Implements analytical rolling-element bearing kinematics for deep-groove ball
bearings, with exact pre-configured specifications for the Case Western Reserve
University (CWRU) Bearing Data Center test stand bearings:
  - Drive End (DE): SKF 6205-2RS JEM
  - Fan End (FE): SKF 6203-2RS JEM

References:
  - CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter/bearing-information
  - Smith, W. A., & Randall, R. B. (2015). Rolling element bearing diagnostics
    using the Case Western Reserve University data: A benchmark study.
    Mechanical Systems and Signal Processing, 64-65, 100-131.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict


@dataclass(frozen=True)
class BearingSpec:
    """Geometric specification of a rolling-element ball bearing.

    Attributes:
        designation: Human-readable bearing model designation.
        num_balls: Number of rolling elements (N_b).
        ball_diameter_in: Rolling element diameter in inches (d).
        pitch_diameter_in: Bearing pitch diameter in inches (D).
        contact_angle_deg: Contact angle in degrees (alpha).
    """

    designation: str
    num_balls: int
    ball_diameter_in: float
    pitch_diameter_in: float
    contact_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.num_balls < 3:
            raise ValueError(f"num_balls must be >= 3, got {self.num_balls}")
        if self.ball_diameter_in <= 0 or self.pitch_diameter_in <= 0:
            raise ValueError("Ball diameter and pitch diameter must be positive.")
        if self.ball_diameter_in >= self.pitch_diameter_in:
            raise ValueError("Ball diameter must be smaller than pitch diameter.")

    @property
    def diameter_ratio(self) -> float:
        """Ratio of ball diameter to pitch diameter (d / D) * cos(alpha)."""
        alpha_rad = math.radians(self.contact_angle_deg)
        return (self.ball_diameter_in / self.pitch_diameter_in) * math.cos(alpha_rad)

    @property
    def bpfo_multiplier(self) -> float:
        """Ball Pass Frequency Outer race (BPFO) order (multiple of shaft speed)."""
        return (self.num_balls / 2.0) * (1.0 - self.diameter_ratio)

    @property
    def bpfi_multiplier(self) -> float:
        """Ball Pass Frequency Inner race (BPFI) order (multiple of shaft speed)."""
        return (self.num_balls / 2.0) * (1.0 + self.diameter_ratio)

    @property
    def ftf_multiplier(self) -> float:
        """Fundamental Train Frequency (FTF / cage speed) order."""
        return 0.5 * (1.0 - self.diameter_ratio)

    @property
    def bsf_multiplier(self) -> float:
        """Ball Spin Frequency (BSF) order (frequency of ball rotation about its own axis)."""
        ratio = self.diameter_ratio
        return (self.pitch_diameter_in / (2.0 * self.ball_diameter_in)) * (1.0 - ratio * ratio)

    @property
    def bdf_multiplier(self) -> float:
        """Ball Defect Frequency (BDF = 2 * BSF) order, since a ball defect strikes both races per spin."""
        return 2.0 * self.bsf_multiplier

    def fault_multipliers(self) -> Dict[str, float]:
        """Return dictionary of characteristic fault frequency orders."""
        return {
            "BPFO": self.bpfo_multiplier,
            "BPFI": self.bpfi_multiplier,
            "BSF": self.bsf_multiplier,
            "BDF": self.bdf_multiplier,
            "FTF": self.ftf_multiplier,
        }


# SISMAX Note: Official CWRU Drive End bearing (SKF 6205-2RS JEM)
SKF_6205_DRIVE_END = BearingSpec(
    designation="SKF 6205-2RS JEM (CWRU Drive End)",
    num_balls=9,
    ball_diameter_in=0.3126,
    pitch_diameter_in=1.5370,
    contact_angle_deg=0.0,
)

# SISMAX Note: Official CWRU Fan End bearing (SKF 6203-2RS JEM)
SKF_6203_FAN_END = BearingSpec(
    designation="SKF 6203-2RS JEM (CWRU Fan End)",
    num_balls=8,
    ball_diameter_in=0.2656,
    pitch_diameter_in=1.1220,
    contact_angle_deg=0.0,
)


def compute_fault_frequencies(
    rpm: float,
    bearing: BearingSpec = SKF_6205_DRIVE_END,
) -> Dict[str, float]:
    """Compute characteristic bearing defect frequencies in Hz for a given shaft RPM.

    Args:
        rpm: Rotational speed of the shaft in revolutions per minute (must be > 0).
        bearing: BearingSpec instance defining bearing geometry.

    Returns:
        Dictionary mapping frequency identifiers ('fr', 'BPFO', 'BPFI', 'BSF', 'BDF', 'FTF')
        to their exact frequencies in Hz.

    Raises:
        ValueError: If rpm <= 0.
    """
    if rpm <= 0:
        raise ValueError(f"Shaft speed rpm must be strictly positive, got {rpm}")

    fr = rpm / 60.0
    multipliers = bearing.fault_multipliers()
    return {
        "fr": fr,
        "BPFO": fr * multipliers["BPFO"],
        "BPFI": fr * multipliers["BPFI"],
        "BSF": fr * multipliers["BSF"],
        "BDF": fr * multipliers["BDF"],
        "FTF": fr * multipliers["FTF"],
    }
