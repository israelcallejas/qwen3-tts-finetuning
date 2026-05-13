import torch
import os
from safetensors.torch import load_file
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

BASE_PATH = r"C:\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base" 
CKPT_FILE = r"C:\Qwen3-TTS\output_es-ES\checkpoint-epoch-x\model.safetensors" # x > 0

def check_encoders():
    base_model = Qwen3TTSModel.from_pretrained(BASE_PATH, dtype=torch.bfloat16)
    base_model.model.eval()
    base_state = base_model.model.speaker_encoder.state_dict()
    
    if not os.path.exists(CKPT_FILE):
        print(f"missing file: {CKPT_FILE}")
        return
        
    ckpt_full_state = load_file(CKPT_FILE)
    ckpt_state = {
        k.replace("speaker_encoder.", ""): v 
        for k, v in ckpt_full_state.items() 
        if k.startswith("speaker_encoder.")
    }
    
    if not ckpt_state:
        print("no speaker_encoder keys found in checkpoint.")
        return

    base_keys = set(base_state.keys())
    ckpt_keys = set(ckpt_state.keys())
    
    missing_in_ckpt = base_keys - ckpt_keys
    missing_in_base = ckpt_keys - base_keys
    
    max_diff = 0.0
    exact_matches = 0
    common_keys = base_keys.intersection(ckpt_keys)
    
    for key in common_keys:
        base_tensor = base_state[key]
        ckpt_tensor = ckpt_state[key]
        
        diff = (base_tensor.float() - ckpt_tensor.float()).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            
        if torch.equal(base_tensor, ckpt_tensor):
            exact_matches += 1

    print(f"total base keys = {len(base_keys)}")
    print(f"keys missing in checkpoint = {len(missing_in_ckpt)}")
    if missing_in_ckpt:
        print(f" ! {missing_in_ckpt}")
    print(f"keys in checkpoint but not base = {len(missing_in_base)}")
    if missing_in_base:
        print(f" ! {missing_in_base}")
    print(f"exact matches (bfloat16) = {exact_matches} / {len(common_keys)}")
    print(f"max abs difference (float32) = {max_diff:.8f}")

if __name__ == "__main__":
    check_encoders()
