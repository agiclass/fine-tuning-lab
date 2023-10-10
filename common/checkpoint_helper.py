import os
import torch
from peft import PeftModel

def load_lora_checkpoint(model,checkpoint_path, logger=None, merge=False):
    adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, checkpoint_path)
        if logger:
            logger.info(f"load checkpoint (adapter) from: {checkpoint_path}")
    else:
        sd_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(sd_path):
            model.load_state_dict(
                torch.load(sd_path), 
                strict=False
            )
            if logger:
                logger.info(f"load checkpoint (state_dict) from: {checkpoint_path}")
        else:
            if logger:
                logger.error(f"no checkpoint found at: {checkpoint_path}")
    
    if merge:
        model = model.merge_and_unload()
        
    return model
