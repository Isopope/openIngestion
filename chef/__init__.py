"""openingestion.chef — CHOMP step 1: parsing & extraction."""
from openingestion.chef.base import BaseChef
from openingestion.chef.mineru_chef import MinerUChef

# DoclingChef is loaded lazily — requires: pip install openingestion[docling]
def __getattr__(name: str):
    if name == "DoclingChef":
        from openingestion.chef.docling_chef import DoclingChef
        return DoclingChef
    raise AttributeError(f"module 'openingestion.chef' has no attribute {name!r}")

__all__ = ["BaseChef", "DoclingChef", "MinerUChef"]
