import json
from pathlib import Path
from typing import Dict, Tuple

def save_merges(merges: Dict[Tuple[int, int], int], save_path: str):
    """Save merges dictionary to JSON file"""
    # Convert tuple keys to strings for JSON serialization
    serializable_merges = {f"{k[0]},{k[1]}": v for k, v in merges.items()}
    
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "merges.json", "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f, ensure_ascii=False, indent=2)

def load_merges(load_path: str) -> Dict[Tuple[int, int], int]:
    """Load merges dictionary from JSON file"""
    with open(Path(load_path) / "merges.json", "r", encoding="utf-8") as f:
        serialized_merges = json.load(f)
    
    # Convert string keys back to tuples
    merges = {tuple(map(int, k.split(","))): v for k, v in serialized_merges.items()}
    return merges 