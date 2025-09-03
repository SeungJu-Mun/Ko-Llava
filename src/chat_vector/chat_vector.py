import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration, TextStreamer
from PIL import Image
import requests
import os
from typing import List, Optional
from config.env_config import EnvironmentConfig

env = EnvironmentConfig.load() 

class LlavaChatVector:
    """
    ChatVector를 이용한 멀티모달 모델과 한국어 모델 결합 클래스
    """
    
    def __init__(self, 
                 base_model_name: str = env.LLAMA_MODEL_NAME,
                 ko_model_name: str = env.LLAVA_MODEL_NAME,
                 cache_dir: str = env.CACHE_DIR,
                 output_dir: str = env.OUTPUT_DIR):
        """
        ChatVector 초기화
        
        Args:
            base_model_name: 베이스 멀티모달 모델명
            ko_model_name: 한국어 사전학습 모델명
            cache_dir: 캐시 디렉토리
            output_dir: 결합된 모델 저장 경로
        """
        self.base_model_name = base_model_name
        self.ko_model_name = ko_model_name
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        # 제외할 레이어 정의
        self.skip_layers = [
            "model.embed_tokens.weight",
            "lm_head.weight", 
            "vision_tower.vision_model.embeddings.class_embedding",
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "vision_tower.vision_model.pre_layrnorm.weight"
        ]
        
        self.base_model = None
        self.ko_model = None
        self.merged_model = None
        self.processor = None
        self.tokenizer = None
        self.streamer = None
    
    def load_models(self):
        """베이스 모델과 한국어 모델을 로드"""
        print("Loading base multimodal model...")
        self.base_model = LlavaForConditionalGeneration.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir
        )
        
        print("Loading Korean pretrained model...")
        self.ko_model = AutoModelForCausalLM.from_pretrained(
            self.ko_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            cache_dir=self.cache_dir
        )
        
        print("Models loaded successfully!")
    
    def merge_models(self):
        """ChatVector를 이용해 모델들을 결합"""
        if self.base_model is None or self.ko_model is None:
            raise ValueError("Models must be loaded first. Call load_models() first.")
        
        print("Merging models using ChatVector...")
        
        # 새로운 모델 상태 사전 생성
        new_state_dict = self.base_model.state_dict()
        
        # 가중치 업데이트
        for k, v in self.ko_model.state_dict().items():
            # 스킵할 레이어나 layernorm 레이어는 제외
            if any(skip in k for skip in self.skip_layers) or ("layernorm" in k):
                continue
                
            if k in new_state_dict:
                # ChatVector 계산: base_model[k] - ko_model[k]  
                chat_vector = self.base_model.state_dict()[k] - self.ko_model.state_dict().get(k, torch.zeros_like(v))
                # 새로운 가중치: ko_model[k] + chat_vector
                new_v = v + chat_vector.to(v.device)
                # 베이스 모델에 새로운 가중치 적용
                with torch.no_grad():
                    new_state_dict[k].copy_(new_v)
        
        print("Model merging completed!")
    
    def save_merged_model(self):
        """결합된 모델 저장"""
        if self.base_model is None:
            raise ValueError("No model to save. Merge models first.")
        
        print(f"Saving merged model to {self.output_dir}...")
        self.base_model.save_pretrained(self.output_dir)
        print("Model saved successfully!")
    
    def load_merged_model(self, revision: str = 'a38aac3'):
        """저장된 결합 모델 로드"""
        print(f"Loading merged model from {self.output_dir}...")
        self.merged_model = LlavaForConditionalGeneration.from_pretrained(
            self.output_dir,
            torch_dtype='auto',
            device_map='auto',
            revision=revision
        )
        
        # 프로세서와 토크나이저 설정
        self.processor = AutoProcessor.from_pretrained(self.base_model_name)
        self.tokenizer = self.processor.tokenizer
        
        # 텍스트 스트리머 설정
        self.streamer = TextStreamer(self.tokenizer)
        
        print("Merged model loaded successfully!")
    
    def generate_response(self, 
                         prompt: str, 
                         image_url: str,
                         max_new_tokens: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         no_repeat_ngram_size: int = 3) -> str:
        """
        이미지와 텍스트 프롬프트를 받아 응답 생성
        
        Args:
            prompt: 텍스트 프롬프트
            image_url: 이미지 URL
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성 조절
            top_p: top-p 샘플링 값
            no_repeat_ngram_size: n-gram 반복 방지
            
        Returns:
            생성된 응답 텍스트
        """
        if self.merged_model is None:
            raise ValueError("Merged model not loaded. Call load_merged_model() first.")
        
        # 이미지 로드
        raw_image = Image.open(requests.get(image_url, stream=True).raw)
        
        # 입력 전처리
        inputs = self.processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
        
        # 종료 토큰 설정
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # 응답 생성
        with torch.no_grad():
            output = self.merged_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                eos_token_id=terminators,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature,
                top_p=top_p,
                streamer=self.streamer
            )
        
        # 응답 디코딩
        response = self.processor.decode(output[0][2:], skip_special_tokens=False)
        return response
    
    def create_prompt(self, user_message: str, assistant_start: str = "이 이미지에는") -> str:
        """표준 프롬프트 템플릿 생성"""
        return (f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{user_message}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_start}")
    
    def full_pipeline(self):
        """전체 파이프라인 실행 (모델 로드 -> 결합 -> 저장)"""
        self.load_models()
        self.merge_models()
        self.save_merged_model()
        print("Full ChatVector pipeline completed!")