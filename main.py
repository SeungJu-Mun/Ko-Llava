"""
Ko-LLaVA 메인 실행 스크립트

ChatVector를 사용한 한국어 멀티모달 모델 데모
"""

from src import ChatVectorMerger, KoLLaVAPredictor, EnvironmentConfig, setup_logger

# 로거 설정 (로그 파일도 함께 기록)
logger = setup_logger("ko-llava-main", level="INFO", log_file="logs/ko-llava.log")

def run_merging_pipeline():
    """모델 병합 파이프라인을 실행합니다."""
    logger.info("🚀 Starting ChatVector merging pipeline...")
    
    merger = ChatVectorMerger()
    try:
        merger.run_full_pipeline(alpha=1.0)
        logger.info("✅ Merging pipeline completed successfully!")
    except Exception as e:
        logger.error(f"❌ Merging pipeline failed: {e}")
        raise
    finally:
        merger.cleanup_models()

def run_inference_demo():
    """추론 데모를 실행합니다."""
    logger.info("🔄 Starting inference demo...")
    
    predictor = KoLLaVAPredictor()
    try:
        # 모델 로드
        predictor.load_model()
        
        # 테스트 케이스들
        test_cases = [
            {
                "name": "이미지 설명",
                "prompt": predictor.create_prompt("이 이미지에 대해서 자세히 설명해주세요."),
                "image_url": "https://cdn-uploads.huggingface.co/production/uploads/5e56829137cb5b49818287ea/NWfoArWI4UPAxpEnolkwT.jpeg"
            },
            {
                "name": "한국 장소 식별", 
                "prompt": predictor.create_prompt("이곳은 대한민국의 어디인가요?"),
                "image_url": "https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyNDAzMzFfNDYg%2FMDAxNzExODMzNTM4MTU5.Xf8te7rReNi4aXtFAsjjdeCsXDv1Tr4Be5pOsuofd0Mg.i8UclMMaD91i0MEMEXXKsgloQKZQbJfVJQeqK_2UC8Yg.PNG%2F359d2185%25A3%25ADc597%25A3%25AD49a3%25A3%25ADb102%25A3%25ADdf25158be59f.png&type=sc960_832"
            }
        ]
        
        # 각 테스트 케이스 실행
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n=== 테스트 {i}: {test_case['name']} ===")
            
            try:
                response = predictor.predict(
                    prompt=test_case["prompt"],
                    image_url=test_case["image_url"],
                    max_new_tokens=512,
                    temperature=0.7
                )
                logger.info(f"✨ 응답:\n{response}\n")
                
            except Exception as e:
                logger.error(f"❌ 테스트 {i} 실패: {e}")
                
    except Exception as e:
        logger.error(f"❌ Inference demo failed: {e}")
        raise
    finally:
        predictor.cleanup()

def run_interactive_mode():
    """대화형 모드를 실행합니다."""
    logger.info("🎮 Starting interactive mode...")
    
    predictor = KoLLaVAPredictor()
    predictor.load_model()
    
    print("\n" + "="*50)
    print("🇰🇷 Ko-LLaVA 대화형 모드")
    print("이미지 URL과 질문을 입력하여 한국어 응답을 받아보세요!")
    print("종료하려면 'quit' 입력")
    print("="*50 + "\n")
    
    try:
        while True:
            try:
                user_input = input("\n💬 질문을 입력하세요: ").strip()
                if user_input.lower() in ['quit', 'exit', '종료']:
                    break
                
                if not user_input:
                    print("⚠️  질문을 입력해주세요.")
                    continue
                
                image_url = input("🖼️  이미지 URL을 입력하세요: ").strip()
                if not image_url:
                    print("⚠️  이미지 URL이 필요합니다.")
                    continue
                
                print("\n🤖 응답 생성 중...")
                prompt = predictor.create_prompt(user_input)
                response = predictor.predict(prompt, image_url, stream=True)
                print(f"\n✨ 응답:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                logger.error(f"❌ 오류 발생: {e}")
                
    finally:
        predictor.cleanup()

def main():
    """메인 함수"""
    logger.info("🇰🇷 Ko-LLaVA 시작!")
    
    # 사용자 선택
    print("\n" + "="*50)
    print("🇰🇷 Ko-LLaVA: ChatVector 기반 한국어 멀티모달 모델")
    print("="*50)
    print("1. 모델 병합 (최초 1회만 실행)")
    print("2. 추론 데모 실행")
    print("3. 대화형 모드")
    print("="*50)
    
    choice = input("\n선택하세요 (1-3): ").strip()
    
    try:
        if choice == "1":
            run_merging_pipeline()
        elif choice == "2":
            run_inference_demo()
        elif choice == "3":
            run_interactive_mode()
        else:
            logger.info("ℹ️  추론 데모를 실행합니다...")
            run_inference_demo()
            
    except Exception as e:
        logger.error(f"❌ 실행 중 오류 발생: {e}")
    
    logger.info("👋 Ko-LLaVA 종료!")


if __name__ == "__main__":
    main()
