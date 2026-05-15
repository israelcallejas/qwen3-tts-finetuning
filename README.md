# qwen3-tts-finetuning
make fine-tuning run without crashing during speaker embedding extraction; the current fix focuses on `sft_12hz.py` to solve this error:

```python
TypeError: 'NoneType' object is not callable
```

the crash happens because `model.speaker_encoder` is `None` during training.

https://github.com/user-attachments/assets/b0520b19-b49f-478c-a5ab-6afc56f83260

https://github.com/user-attachments/assets/1c7e3a43-fa8c-4a64-b4d7-c7931db8733d

https://github.com/user-attachments/assets/c9e95faf-a738-493c-80b8-6b81c5295ebf



