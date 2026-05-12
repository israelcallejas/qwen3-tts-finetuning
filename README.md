# qwen3-tts-finetuning
make fine-tuning run without crashing during speaker embedding extraction; the current fix focuses on `sft_12hz.py` to solve this error:

```python
TypeError: 'NoneType' object is not callable
```

the crash happens because `model.speaker_encoder` is `None` during training.
