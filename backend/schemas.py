from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class GeneratorCreate(BaseModel):
    name: str
    plant_type: str = Field(pattern="^(Solar|Wind|Hydro)$")
    state: str
    capacity_mw: float
    wallet_address: Optional[str] = None


class GenerationSubmit(BaseModel):
    generator_id: str
    energy_generated_mwh: float
    weather_factor: float = 1.0
    rec_quantity: Optional[float] = None  # defaults to energy_generated_mwh if omitted
    reuse_generation_id: Optional[str] = None  # for duplicate-generation simulation/testing


class TransferRequest(BaseModel):
    rec_id: str
    sender: str
    receiver: str
    quantity: float


class SimulationConfig(BaseModel):
    interval_seconds: Optional[float] = Field(default=None, gt=0)
    # ge=0 only -- the upper bound is normalized below rather than rejected,
    # so an accidental percentage value (e.g. 15 meaning 15%) degrades
    # gracefully instead of failing validation.
    fraud_probability: Optional[float] = Field(default=None, ge=0)
    tamper_enabled: Optional[bool] = None
    tamper_probability: Optional[float] = Field(default=None, ge=0)

    @field_validator("fraud_probability", "tamper_probability")
    @classmethod
    def _normalize_probability(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v > 1:  # tolerate "15" meaning 15% instead of 0.15
            v = v / 100
        return min(max(v, 0.0), 1.0)


class RevokeRequest(BaseModel):
    actor: str = "REGULATOR_MAIN"
    reason: Optional[str] = None


class VerifyRecRequest(BaseModel):
    verifier_company: Optional[str] = Field(default=None, max_length=120)
    verifier_user: Optional[str] = Field(default=None, max_length=120)


class VerificationReportRequest(BaseModel):
    verification_id: str


class GraphStatusUpdate(BaseModel):
    status: str = Field(pattern="^(UNDER_INVESTIGATION|RESOLVED|FALSE_POSITIVE|OPEN|ACTIVE)$")
