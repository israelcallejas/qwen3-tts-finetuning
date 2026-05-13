def train():
    # (...)
    config = AutoConfig.from_pretrained(MODEL_PATH)
    if qwen3tts.model.speaker_encoder is None:
        base_path = r"C:\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base" # or 0.6B
        base_model = Qwen3TTSModel.from_pretrained(
            base_path, 
            torch_dtype=torch.bfloat16
        )
        qwen3tts.model.speaker_encoder = base_model.model.speaker_encoder
        del base_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    train_data = open(args.train_jsonl).readlines()
    # (...)
