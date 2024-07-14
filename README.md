# Llama3-Chat_Vector-kor_llava
저는 Beomi가 만든 한국어 챗 벡터 LLAVA 모델과 Toshi456이 만든 일본어 챗 벡터 LLAVA 모델을 참조하여 한국어 LLAVA 모델을 구현했습니다.

### Reference Models:
1) beomi/Llama-3-KoEn-8B-xtuner-llava-preview(https://huggingface.co/beomi/Llama-3-KoEn-8B-xtuner-llava-preview)
2) toshi456/chat-vector-llava-v1.5-7b-ja(https://huggingface.co/toshi456/chat-vector-llava-v1.5-7b-ja)
3) [xtuner/llava-llama-3-8b-transformers](https://huggingface.co/xtuner/llava-llama-3-8b-transformers)

```

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
