## Llama3-Chat_Vector-kor_llava : Llama3 기반 한국어 LLAVA 모델
---

### Update Logs
- 2024.06.27: [🤗Llama3 기반 한국어 LLAVA 모델 공개](nebchi/Llama3-Chat_Vector-kor_llava)

### Reference Models:
1) beomi/Llama-3-KoEn-8B(https://huggingface.co/beomi/Llama-3-KoEn-8B)
2) xtuner/llava-llama-3-8b-transformers(https://huggingface.co/xtuner/llava-llama-3-8b-transformers)
3) Chat-Vector(https://arxiv.org/abs/2310.04799)

**Model Developers**: nebchi

## Model Description
* 이번 Kor-LLAVA 모델은 대량의 한국어 코퍼스로 사전학습한 LLAMA 모델과 LLAVA 모델을 Chat_Vector를 활용하여, 이미지에 대하여, 텍스트로 설명이 가능한 VLM 모델 입니다.

<p align="left" width="100%">
<img src="assert/Seoul_city.png" alt="NLP Logo" style="width: 40%;">
</p>

### Running the model on GPU
```python
import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, TextStreamer

model_id = "nebchi/Llama3-Chat_Vector-kor_llava"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype='auto', 
    device_map='auto',
    revision='a38aac3', 
)

processor = AutoProcessor.from_pretrained(model_id)

tokenizer = processor.tokenizer
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
streamer = TextStreamer(tokenizer)

prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\n이 이미지에 대해서 설명해주세요.<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n이 이미지에는")
image_file = "https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyNDAzMzFfNDYg%2FMDAxNzExODMzNTM4MTU5.Xf8te7rReNi4aXtFAsjjdeCsXDv1Tr4Be5pOsuofd0Mg.i8UclMMaD91i0MEMEXXKsgloQKZQbJfVJQeqK_2UC8Yg.PNG%2F359d2185%25A3%25ADc597%25A3%25AD49a3%25A3%25ADb102%25A3%25ADdf25158be59f.png&type=sc960_832"

raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,  
    eos_token_id=terminators,
    no_repeat_ngram_size=3, 
    temperature=0.7,  
    top_p=0.9,  
    streamer=streamer
)
print(processor.decode(output[0][2:], skip_special_tokens=False))
```

### results
```python
이 이미지에는 도시의 모습이 잘 보여집니다. 도시 내부에는 여러 건물과 건물들이 있고, 도시를 연결하는 도로와 교통 시스템이 잘 발달되어 있습니다. 이 도시의 특징은 높고 광범위한 건물들과 교통망을 갖춘 것이 좋습니다.
```
---
**Citation**

```bibtex
@misc {Llama3-Chat_Vector-kor_llava,
	author       = { {nebchi} },
	title        = { Llama3-Chat_Vector-kor_llava },
	year         = 2024,
	url          = { https://huggingface.co/nebchi/Llama3-Chat_Vector-kor_llava },
	publisher    = { Hugging Face }
}
