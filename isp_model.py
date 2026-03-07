from dataclasses import dataclass, field
from math import ceil

from config_loader import load_config

CONFIG = load_config()
ISP_CFG = CONFIG["isp"]
TIERS_CFG = CONFIG["tiers"]


@dataclass
class ISP:
    slices: dict = field(default_factory=lambda: ISP_CFG["slices"])
    tiers: dict = field(default_factory=lambda: TIERS_CFG)

    @property
    def number_of_slices(self) -> int:
        return len(self.slices)

    @property
    def number_of_tiers(self) -> int:
        return len(self.tiers)

    def capacity_prbs(self, slice_name: str) -> int:
        return int(self.slices[slice_name]["capacity_prbs"])

    def cost_per_prb(self, slice_name: str) -> float:
        return float(self.slices[slice_name]["cost_per_prb"])

    def efficiency_mhz_per_prb(self, slice_name: str) -> float:
        return float(self.slices[slice_name]["efficiency_mhz_per_prb"])

    def required_prbs(self, slice_name: str, bandwidth_mhz: float) -> int:
        efficiency = self.efficiency_mhz_per_prb(slice_name=slice_name)
        if efficiency <= 0:
            raise ValueError(f"Invalid PRB efficiency for {slice_name}: {efficiency}")
        return ceil(bandwidth_mhz / efficiency)

    def max_bandwidth_mhz(self, slice_name: str) -> float:
        return self.capacity_prbs(slice_name) * self.efficiency_mhz_per_prb(slice_name)

    def target_rate_mbps(self, tier_name: str) -> float:
        return float(self.tiers[tier_name]["target_rate_mbps"])

    def weight(self, tier_name: str) -> int:
        return int(self.tiers[tier_name]["weight"])

    def price(self, tier_name: str) -> float:
        return float(self.tiers[tier_name]["price"])
