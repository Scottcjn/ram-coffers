"""DeepSeek MoE inference across ninth-generation console hardware.

A fleet of PlayStation 5s, Xbox Series X/S consoles, and the salvage boards
built from the same silicon (AMD 4700S/4800S, BC-250), running one DeepSeek
model between them under the RAM Coffers memory-tier model.

The pieces, in the order they are used:

``hardware``    console SKUs, per-unit downbinning, memory coffers, backends
``model``       DeepSeek profiles and the sizes derived from them
``planner``     what goes on which console, and what that should cost
``protocol``    the G9XC wire format
``transport``   persistent multiplexed connections
``node``        the console-side worker
``dispatch``    per-layer expert fan-out and reduction
``coordinator`` shelf and fleet drivers
``inventory``   fleet files, probing, and plan/config generation
"""

from .hardware import (BOARD_TO_SKU, SKUS, ComputeBackend, ConsoleSKU,
                       ConsoleUnit, Downbin, EffectiveCapability, MemoryTier,
                       Runtime, StorageSpec, FleetSummary, fleet_summary,
                       sku_for)
from .model import (DEEPSEEK_TINY, DEEPSEEK_V3, DEEPSEEK_V4_FLASH,
                    DEEPSEEK_V4_PRO, PROFILES, AttentionConfig,
                    HybridAttentionConfig, MLAConfig, ModelProfile, MoEConfig,
                    QuantSpec, profile_for)
from .planner import (PlanningError, SplitPlan, StagePlan, UnitPlan,
                      describe_plan, plan_split)

__version__ = "0.1.0"

__all__ = [
    "BOARD_TO_SKU", "SKUS", "ComputeBackend", "ConsoleSKU", "ConsoleUnit",
    "Downbin", "EffectiveCapability", "MemoryTier", "Runtime", "StorageSpec",
    "FleetSummary", "fleet_summary", "sku_for",
    "DEEPSEEK_TINY", "DEEPSEEK_V3", "DEEPSEEK_V4_FLASH", "DEEPSEEK_V4_PRO",
    "PROFILES", "AttentionConfig", "HybridAttentionConfig", "MLAConfig",
    "ModelProfile", "MoEConfig", "QuantSpec", "profile_for",
    "PlanningError", "SplitPlan", "StagePlan", "UnitPlan", "describe_plan",
    "plan_split",
    "__version__",
]
