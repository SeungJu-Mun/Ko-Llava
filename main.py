from src.chat_vector import ChatVector
from src.utils.environment import EnvironmentConfig

env = EnvironmentConfig.load()

def main():
    # ChatVector 인스턴스 생성
    chat_vector = ChatVector(
        base_model_name=env.LLAMA_MODEL_NAME,
        ko_model_name=env.LLAVA_MODEL_NAME, 
        cache_dir=env.CACHE_DIR,
        output_dir=env.OUTPUT_DIR
    )
    
    # 1. 모델 학습/결합 과정 (최초 1회만 실행)
    print("=== ChatVector Model Merging ===")
    # chat_vector.full_pipeline()  # 주석 해제하여 최초 1회 실행
    
    # 2. 결합된 모델 로드
    print("\n=== Loading Merged Model ===")
    chat_vector.load_merged_model()
    
    # 3. 테스트 예제 1 - 이미지 설명
    print("\n=== Test 1: Image Description ===")
    prompt1 = chat_vector.create_prompt("이 이미지에 대해서 설명해주세요.")
    image_url1 = "https://cdn-uploads.huggingface.co/production/uploads/5e56829137cb5b49818287ea/NWfoArWI4UPAxpEnolkwT.jpeg"
    
    response1 = chat_vector.generate_response(prompt1, image_url1)
    print(f"Response 1:\n{response1}\n")
    
    # 4. 테스트 예제 2 - 장소 식별
    print("\n=== Test 2: Location Identification ===")
    prompt2 = chat_vector.create_prompt("대한민국의 어디를 나타내는걸까요?")
    image_url2 = "https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyNDAzMzFfNDYg%2FMDAxNzExODMzNTM4MTU5.Xf8te7rReNi4aXtFAsjjdeCsXDv1Tr4Be5pOsuofd0Mg.i8UclMMaD91i0MEMEXXKsgloQKZQbJfVJQeqK_2UC8Yg.PNG%2F359d2185%25A3%25ADc597%25A3%25AD49a3%25A3%25ADb102%25A3%25ADdf25158be59f.png&type=sc960_832"
    
    response2 = chat_vector.generate_response(prompt2, image_url2)
    print(f"Response 2:\n{response2}\n")
    
    # 5. 대화형 테스트 (옵션)
    print("\n=== Interactive Mode (optional) ===")
    interactive_mode = False  # True로 변경하여 대화형 모드 활성화
    
    if interactive_mode:
        while True:
            try:
                user_input = input("\n질문을 입력하세요 (종료: 'quit'): ")
                if user_input.lower() == 'quit':
                    break
                
                image_url = input("이미지 URL을 입력하세요: ")
                if not image_url:
                    print("이미지 URL이 필요합니다.")
                    continue
                
                prompt = chat_vector.create_prompt(user_input)
                response = chat_vector.generate_response(prompt, image_url)
                print(f"\n응답:\n{response}")
                
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()