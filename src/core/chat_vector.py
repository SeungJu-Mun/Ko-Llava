"""
ChatVector 구현 모듈

ChatVector 논문의 핵심 아이디어를 구현하여 
한국어 Llama 모델의 지식을 LLaVA 모델에 전이합니다.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
from typing import List, Optional
from pathlib import Path

from src.config.settings import EnvironmentConfig
from src.utils.logger import setup_logger

env = EnvironmentConfig.load() 

class ChatVectorMerger:
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
        
        # 로거 설정
        self.logger = setup_logger("chat-vector", level="INFO")
        
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
    
    def load_models(self):
        """베이스 모델과 한국어 모델을 로드합니다."""
        self.logger.info("🔄 Loading base multimodal model...")
        self.base_model = LlavaForConditionalGeneration.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir
        )
        
        self.logger.info("🔄 Loading Korean pretrained model...")
        self.ko_model = AutoModelForCausalLM.from_pretrained(
            self.ko_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            cache_dir=self.cache_dir
        )
        
        self.logger.info("✅ Models loaded successfully!")
    
    def merge_models(self, alpha: float = 1.0):
        """
        ChatVector를 이용해 모델들을 결합합니다.
        
        Args:
            alpha: 결합 강도 조절 파라미터 (기본값: 1.0)
        """
        if self.base_model is None or self.ko_model is None:
            raise ValueError("Models must be loaded first. Call load_models() first.")
        
        self.logger.info(f"🔄 Merging models using ChatVector (alpha={alpha})...")
        
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
                # 새로운 가중치: ko_model[k] + alpha * chat_vector
                new_v = v + alpha * chat_vector.to(v.device)
                # 베이스 모델에 새로운 가중치 적용
                with torch.no_grad():
                    new_state_dict[k].copy_(new_v)
        
        self.logger.info("✅ Model merging completed!")
    
    def save_merged_model(self):
        """결합된 모델을 저장합니다."""
        if self.base_model is None:
            raise ValueError("No model to save. Merge models first.")
        
        self.logger.info(f"💾 Saving merged model to {self.output_dir}...")
        self.base_model.save_pretrained(self.output_dir)
        self.logger.info("✅ Model saved successfully!")
    
    def run_full_pipeline(self, alpha: float = 1.0):
        """전체 파이프라인을 실행합니다 (모델 로드 -> 결합 -> 저장)"""
        self.logger.info("🚀 Starting ChatVector full pipeline...")
        self.load_models()
        self.merge_models(alpha=alpha)
        self.save_merged_model()
        self.logger.info("🎉 ChatVector pipeline completed!")
        
    def cleanup_models(self):
        """메모리 정리를 위해 모델들을 해제합니다."""
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
        if self.ko_model is not None:
            del self.ko_model  
            self.ko_model = None
        if self.merged_model is not None:
            del self.merged_model
            self.merged_model = None
        torch.cuda.empty_cache()
        self.logger.info("🧹 Models cleaned up from memory!")