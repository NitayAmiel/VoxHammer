# Shared globals for attention capture. Import and mutate these from scripts.

from typing import Optional
import torch
import math

# Collector list. When STORE_ATTN is True, modules append dicts like:
# {"type": "cross"|"self", "attn": Tensor[B, Lq, Lk]}

class AttentionCollector:
    def __init__(self, steps: int = 25):
        self.store_attn: bool = False
        self.attn: Optional[torch.Tensor] = None
        self.inject_attn_flag: bool = False
        self.inject_attn_col: int = 0
        self.inject_attn_percentage: float = 0.0
        self.attn_inject: list[torch.Tensor] = []
        self.layer_idx: int = 0
        self.percentage_of_layers_to_store: float = 1.0
        self.percentage_of_layers_to_inject: float = 0.0
        self.store_attn_each_layer: bool = False
        self.layers_are_stored: int = 0
        self.noise: Optional[torch.Tensor] = None
        self.layers_in_step: int = 24
        self.number_of_steps: int = steps
        self.total_layers: int = steps * self.layers_in_step
        self.alpha_fade: float = 0.0
        self.threshold: float = 0.65
        self.cond_old = None
        self.only_on_self: bool = False
    
    def set_noise(self, noise: torch.Tensor) -> None:
        self.noise = noise
    def get_noise(self) -> Optional[torch.Tensor]:
        print(f"using predefined noise")
        return self.noise
    
    def set_cond_old(self, cond_old) -> None:
        self.cond_old = cond_old
    def get_cond_old(self):
        return self.cond_old
    
    def reset_cond_old(self) -> None:
        self.cond_old = None
    
    def set_store_attn(self, store_attn: bool) -> None:
        self.store_attn = store_attn
    
    def get_store_attn(self, on_self: bool = False) -> bool:
        if self.only_on_self != on_self:
            return False
        return self.store_attn
    
    def set_layer_idx(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
    
    def add_attn_average(self, attn: torch.Tensor) -> None:
        if not self.store_attn and not self.store_attn_each_layer and not self.inject_attn_flag:
            return

        if self.store_attn_each_layer:
            self.attn_inject.append(attn)
            return
        
        # if self.layer_idx < self.total_layers * (1-self.percentage_of_layers_to_store):
        #     return
        
        if self.attn is None:
            self.attn = attn
        else:
            self.attn += attn        
        self.layers_are_stored += 1
    
    def add_attn_from_k_q(self, k: torch.Tensor, q: torch.Tensor) -> None:
        if not self.store_attn and not self.store_attn_each_layer:
            return
        scale = 1.0 / math.sqrt(q.shape[-1])
        q32 = (q * scale).to(torch.float32)
        k32 = k.to(torch.float32)
        scores = torch.einsum('blhc,bkhc->bhlk', q32, k32)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn_tmp = torch.softmax(scores, dim=-1).detach().to('cpu')  # [B,L,Lk]
        self.add_attn_average(attn_tmp)
        
    def get(self) -> Optional[torch.Tensor]:
        
        return self.attn / self.layers_are_stored
    
    def get_num_of_layers(self) -> int:
        return self.layer_idx
    
    def update_layer_idx(self, on_self: bool = False) -> None:
        if self.only_on_self != on_self:
            return
        if self.store_attn_each_layer or self.inject_attn_flag or self.store_attn:
            self.layer_idx += 1 
    
    def set_percentage_of_layers_to_store(self, percentage_of_layers_to_store: float) -> None:
        self.percentage_of_layers_to_store = percentage_of_layers_to_store
    
    def get_percentage_of_layers_to_store(self) -> float:
        return self.percentage_of_layers_to_store

    def get_layers_are_stored(self) -> int:
        return self.layers_are_stored


    def set_inject_attn(self, inject_attn_flag: bool, inject_attn_col: int = -1, reset_layer_idx: bool = True) -> None:
        if reset_layer_idx:
            self.layer_idx = 0
        self.inject_attn_flag = inject_attn_flag
        if inject_attn_col != -1:
            self.inject_attn_col = inject_attn_col

    def get_inject_attn_flag(self, on_self: bool = False) -> bool:
        if self.only_on_self != on_self:
            return False
        return self.inject_attn_flag
    
    def set_store_attn_each_layer(self, store_attn_each_layer: bool) -> None:
        self.store_attn_each_layer = store_attn_each_layer
    
    def get_store_attn_each_layer(self, on_self: bool = False) -> bool:
        if self.only_on_self != on_self:
            return False
        return self.store_attn_each_layer
    
    def set_only_on_self(self, only_on_self: bool) -> None:
        self.only_on_self = only_on_self
    
    def get_only_on_self(self) -> bool:
        return self.only_on_self
    
    def get_inject_attn_col(self) -> int:
        return self.inject_attn_col
    
    def set_percentage_of_layers_to_inject(self, inject_attn_percentage: float) -> None:
        # high indicate inject more
        # assert inject_attn_percentage <= 0.5 ,"inject_attn_percentage must be leq than 0.5"
        self.percentage_of_layers_to_inject = inject_attn_percentage
    
    def should_inject_attn(self, layer_idx: int) -> bool:
        number_of_steps = self.total_layers // self.layers_in_step
        assert self.total_layers % self.layers_in_step == 0, "total_layers must be divisible by layers_in_step"
        current_step = layer_idx // self.layers_in_step
        return current_step <= number_of_steps * (self.percentage_of_layers_to_inject) and self.inject_attn_flag
        # current_step > number_of_steps * self.percentage_of_layers_to_inject and 
    
    def should_belnd_latents(self, layer_idx: int) -> bool:
        # number_of_steps = self.total_layers // self.layers_in_step
        # assert self.total_layers % self.layers_in_step == 0, "total_layers must be divisible by layers_in_step"
        current_step = layer_idx // self.layers_in_step
        assert  current_step <=  self.number_of_steps, "current_step must be less than number_of_steps "
        return current_step <= self.number_of_steps *1# self.percentage_of_layers_to_inject
    
    def apply_mixed_attention(self, attn_new: torch.Tensor,
                            V: torch.Tensor
                            ) -> torch.Tensor:
        """
        M1, M2: [B, H, Q, K] attention maps (e.g., [1, 16, 4096, 77])
        V:      [B, H, K, C] value tensor (e.g., [1, 16, 77, C])
        
        Returns:
        out: [B, H, Q, C] = (mixed attention) @ V
        """
        attn_to_inject = self.attn_inject[self.layer_idx-1]
        assert attn_new.shape == attn_to_inject.shape, "attn_new and attn_to_inject must have the same shape"
        B, H, Q, K = attn_new.shape
        assert V.shape[:3] == (B, K, H), f"V must be [B,H,K,C], got {V.shape}"
        if not self.should_inject_attn(self.layer_idx):
            V32 = V.float()
            return torch.einsum('bhqk,bkhc->bhqc', attn_new, V32).to(V.dtype)
        device = attn_new.device
        key_idx = self.inject_attn_col
        # Build a boolean mask over keys (length K): True -> take from M1, else from M2
        mask = torch.zeros(K, dtype=torch.bool, device=device)
        mask[key_idx] = True
        # Broadcast mask to [B,H,Q,K]
        mask_b = mask.view(1, 1, 1, K)
        M = attn_to_inject.clone()
        M[:,:,:,key_idx] = attn_new[:,:,:,key_idx]
        M = M.float()
        den = M.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        M = M / den

        # Mix: take selected key columns from M1, the rest from M2
        # M = torch.where(mask_b, attn_new, attn_to_inject)  # [B,H,Q,K]
        V32 = V.float()
        # Apply attention to V -> [B,H,Q,C]
        out = torch.einsum('bhqk,bkhc->bhqc', M, V32).to(V.dtype)
        return out
    def get_alpha_fade(self) -> None:
        current_step = self.layer_idx // self.layers_in_step
        final_step = int(self.number_of_steps * (1-self.percentage_of_layers_to_inject))
        alpha = current_step / final_step if final_step > 0 else 0.0
        return 0.0
        return alpha if alpha <= 1.0 else 1.0

    def combine_attn(self, attn_new: torch.Tensor) -> torch.Tensor:
        betta = self.get_alpha_fade()
        if not self.should_inject_attn(self.layer_idx):
            return attn_new
        else:
            return self.attn_inject[self.layer_idx-1]
        attn_to_inject = self.attn_inject[self.layer_idx-1]
        assert attn_new.shape == attn_to_inject.shape, "attn_new and attn_to_inject must have the same shape"
        B, H, Q, K = attn_new.shape
        device = attn_new.device
        key_idx = self.inject_attn_col
        # Build a boolean mask over keys (length K): True -> take from M1, else from M2
        mask = torch.zeros(K, dtype=torch.bool, device=device)
        mask[key_idx] = True
        M = attn_to_inject.clone()
        alpha = self.get_alpha_fade()
        M[:,:,:,key_idx] = (1-alpha) * attn_new[:,:,:,key_idx] + alpha * M[:,:,:,key_idx]
        return M

    def get_average_inject_attn(self, till_current_layer: bool = False) -> torch.Tensor:
        sum  = torch.zeros_like(self.attn_inject[0])
        for i, attn in enumerate(self.attn_inject):
            if till_current_layer and i >= self.layer_idx:
                break
            sum += attn
        return sum / len(self.attn_inject)

    def update_mask_and_create_sample(self) -> torch.Tensor:
        average_new_attn = self.get()
        print(f"number of layers stored: {self.layers_are_stored}")
        average_old_attn = self.get_average_inject_attn()
        average_new_attn = average_new_attn.mean(dim=1).squeeze(0) # [ L, Lk]
        average_old_attn = average_old_attn.mean(dim =1).squeeze(0) # [ L, Lk]
        average_new_attn = normalize_tensor(average_new_attn[:,self.inject_attn_col])
        average_old_attn = normalize_tensor(average_old_attn[:,self.inject_attn_col])
        mask_new = (average_new_attn > self.threshold)
        mask_old = (average_old_attn > self.threshold)
        mask = mask_new | mask_old
        return mask
    
    def zero_all_flags(self) -> None:
        state = {
            "store_attn": self.store_attn,
            "store_attn_each_layer": self.store_attn_each_layer,
            "inject_attn_flag": self.inject_attn_flag
        }
        self.store_attn = False
        self.store_attn_each_layer = False
        self.inject_attn_flag = False
        return state
    
    def set_all_flags(self, state) -> None:
        self.store_attn = state["store_attn"]
        self.store_attn_each_layer = state["store_attn_each_layer"]
        self.inject_attn_flag = state["inject_attn_flag"]



ATTN_COLLECT = AttentionCollector()


def compute_attention_for_injection_from_k_q(k: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    query = q.to(torch.float32)
    key = k.to(torch.float32)
    query = query.permute(0, 2, 1, 3)   # [N, H, L, C]
    key = key.permute(0, 2, 1, 3)   # [N, H, L, C]
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, 0.0, train=True)
    return attn_weight

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention_for_injection(query_: torch.Tensor = None, key_: torch.Tensor = None, value_: torch.Tensor = None, attn_weight_param: torch.Tensor = None) -> torch.Tensor:
    attn_weight = compute_attention_for_injection_from_k_q(key_, query_) if attn_weight_param is None else attn_weight_param
    assert value_ is not None, "value_ must be provided"
    value = value_.to(torch.float32)
    value = value.permute(0, 2, 1, 3)   # [N, H, L, C]
    out = attn_weight @ value
    out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    out = out.to(value_.dtype)
    return out

def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm