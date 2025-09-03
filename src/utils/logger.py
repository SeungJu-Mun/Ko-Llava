"""
로깅 유틸리티

프로젝트 전반에서 사용할 로깅 설정을 제공합니다.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = "ko-llava",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    로거를 설정합니다.
    
    Args:
        name: 로거 이름
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로 (선택사항)
        format_string: 로그 포맷 문자열 (선택사항)
        
    Returns:
        설정된 로거 인스턴스
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 기본 포맷 설정 (이모지와 색상 지원)
    if format_string is None:
        format_string = '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 추가 (선택사항)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
