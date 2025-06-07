# Eye-State-Detection-Program
Eye State Detection Program (눈 뜸/감음 상태 감지)

본 프로젝트는 MediaPipe와 TensorFlow를 활용하여 웹캠으로 사용자의 눈 상태(감김/뜸)를 실시간으로 분류하는 프로그램입니다.

## 📌 주요 기능
- 웹캠 실시간 얼굴/눈 감지 (MediaPipe Face Mesh)
- CNN 기반 눈 상태 분류 (Open/Closed)
- 실시간 결과 시각화 (OpenCV + 텍스트 출력)

##  사용된 기술
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras

<img width="960" alt="눈 감" src="https://github.com/user-attachments/assets/f729d325-0ba2-4115-9724-ffe6b171c7c6" />
<img width="932" alt="눈ㅁㅁ" src="https://github.com/user-attachments/assets/00dc244a-64e1-4707-8a1f-ae8841458677" />


    

##  실행 방법

```bash
pip install -r requirements.txt
python detect_eye_state.py


