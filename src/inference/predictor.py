"""
Ko-LLaVA 추론 모듈

병합된 한국어 LLaVA 모델을 사용하여 이미지-텍스트 추론을 수행합니다.
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, TextStreamer
from PIL import Image
import requests
from typing import Optional
from pathlib import Path

from src.config.settings import EnvironmentConfig
from src.utils.logger import setup_logger

env = EnvironmentConfig.load()

class KoLLaVAPredictor:
    """한국어 LLaVA 모델 추론 클래스"""
    
    def __init__(self, 
                 model_path: str = env.OUTPUT_DIR,
                 base_model_name: str = env.LLAMA_MODEL_NAME):
        """
        추론기 초기화
        
        Args:
            model_path: 병합된 모델이 저장된 경로
            base_model_name: 프로세서 로드를 위한 베이스 모델명
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        
        # 로거 설정
        self.logger = setup_logger("ko-llava-predictor", level="INFO")
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.streamer = None
    
    def load_model(self, revision: str = 'main'):
        """병합된 모델을 로드합니다."""
        self.logger.info(f"🔄 Loading merged Ko-LLaVA model from {self.model_path}...")
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype='auto',
            device_map='auto',
            revision=revision
        )
        
        # 프로세서와 토크나이저 설정
        self.processor = AutoProcessor.from_pretrained(self.base_model_name)
        self.tokenizer = self.processor.tokenizer
        
        # 텍스트 스트리머 설정
        self.streamer = TextStreamer(self.tokenizer)
        
        self.logger.info("✅ Ko-LLaVA model loaded successfully!")
    
    def predict(self, 
                prompt: str, 
                image_url: str,
                max_new_tokens: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.9,
                no_repeat_ngram_size: int = 3,
                stream: bool = False) -> str:
        """
        이미지와 텍스트 프롬프트로 응답을 생성합니다.
        
        Args:
            prompt: 텍스트 프롬프트
            image_url: 이미지 URL 또는 로컬 파일 경로
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성 조절
            top_p: top-p 샘플링 값
            no_repeat_ngram_size: n-gram 반복 방지
            stream: 스트리밍 출력 여부
            
        Returns:
            생성된 응답 텍스트
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # 이미지 로드
        if image_url.startswith(('http://', 'https://')):
            raw_image = Image.open(requests.get(image_url, stream=True).raw)
        else:
            raw_image = Image.open(image_url)
        
        # 입력 전처리
        inputs = self.processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
        
        # 종료 토큰 설정
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # 응답 생성
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                eos_token_id=terminators,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature,
                top_p=top_p,
                streamer=self.streamer if stream else None
            )
        
        # 응답 디코딩 (입력 부분 제외)
        response = self.processor.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        return response.strip()
    
    def create_prompt(self, user_message: str, assistant_start: str = "이 이미지에는") -> str:
        """표준 한국어 프롬프트 템플릿을 생성합니다."""
        return (f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{user_message}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_start}")
    
    def batch_predict(self, prompts_and_images: list, **generate_kwargs) -> list:
        """
        여러 입력에 대해 배치 추론을 수행합니다.
        
        Args:
            prompts_and_images: [(prompt, image_url), ...] 형태의 리스트
            **generate_kwargs: 생성 파라미터들
            
        Returns:
            응답 리스트
        """
        results = []
        for prompt, image_url in prompts_and_images:
            try:
                response = self.predict(prompt, image_url, **generate_kwargs)
                results.append(response)
            except Exception as e:
                self.logger.error(f"❌ Error processing {image_url}: {e}")
                results.append(f"Error: {str(e)}")
        return results
    
    def cleanup(self):
        """메모리 정리를 위해 모델을 해제합니다."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        self.logger.info("🧹 Predictor cleaned up from memory!")
