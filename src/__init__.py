"""
Ko-LLaVA: ChatVector를 활용한 한국어 멀티모달 모델

ChatVector 논문의 아이디어를 활용하여 한국어 Llama 모델의 지식을
LLaVA 멀티모달 모델에 전이시키는 프로젝트입니다.
"""

__version__ = "0.1.0"

# 필수 의존성 없이도 로거와 설정은 사용 가능
from .config import EnvironmentConfig
from .utils import setup_logger, get_logger

# torch 등이 설치된 경우에만 임포트
try:
    from .core import ChatVectorMerger
    from .inference import KoLLaVAPredictor
    
    __all__ = [
        "ChatVectorMerger",
        "KoLLaVAPredictor", 
        "EnvironmentConfig",
        "setup_logger",
        "get_logger"
    ]
except ImportError:
    # 의존성이 없는 경우 기본 기능만 제공
    __all__ = [
        "EnvironmentConfig",
        "setup_logger",
        "get_logger"
    ]
