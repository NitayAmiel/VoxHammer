# Shared globals for attention capture. Import and mutate these from scripts.

from typing import Optional
import torch

# Collector list. When STORE_ATTN is True, modules append dicts like:
# {"type": "cross"|"self", "attn": Tensor[B, Lq, Lk]}

class AttentionCollector:
    def __init__(self):
        self.store_attn: bool = False
        self.attn: Optional[torch.Tensor] = None
        self.layer_idx: int = 0
        self.percentage_of_layers_to_store: float = 1.0
        self.total_layers: int = 960
        self.layers_are_stored: int = 0
    
    def set_store_attn(self, store_attn: bool) -> None:
        self.store_attn = store_attn
    
    def get_store_attn(self) -> bool:
        return self.store_attn
    
    def add_attn_average(self, attn: torch.Tensor) -> None:
        if not self.store_attn:
            return
        self.layer_idx += 1
        if self.layer_idx < self.total_layers * (1-self.percentage_of_layers_to_store):
            return
        if self.attn is None:
            self.attn = attn
        else:
            self.attn += attn        
        self.layers_are_stored += 1
        
    def get(self) -> Optional[torch.Tensor]:
        
        return self.attn / self.layers_are_stored
    
    def get_num_of_layers(self) -> int:
        return self.layer_idx
    
    def set_percentage_of_layers_to_store(self, percentage_of_layers_to_store: float) -> None:
        self.percentage_of_layers_to_store = percentage_of_layers_to_store
    
    def get_percentage_of_layers_to_store(self) -> float:
        return self.percentage_of_layers_to_store

    def get_layers_are_stored(self) -> int:
        return self.layers_are_stored


ATTN_COLLECT = AttentionCollector()

