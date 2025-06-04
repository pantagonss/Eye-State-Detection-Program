import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model('model/eye_model.h5')

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 눈 인덱스 (MediaPipe 기준)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# 눈 이미지 추출 함수
def extract_eye(frame, landmarks, eye_indices):
    h, w = frame.shape[:2]
    eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x, y, x2, y2 = cv2.boundingRect(np.array(eye_points))

    x = max(0, x)
    y = max(0, y)
    x2 = min(w, x2)
    y2 = min(h, y2)

    eye_img = frame[y:y2, x:x2]
    if eye_img.size == 0:
        print("❌ 눈 이미지 추출 실패:", x, y, x2, y2)
        return None

    eye_img = cv2.resize(cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY), (24, 24))
    eye_img = eye_img / 255.0
    return eye_img.reshape(1, 24, 24, 1)

# 웹캠 실행
cap = cv2.VideoCapture(0)

# 해상도 설정 (선택)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # 눈 랜드마크 시각화
        for i in LEFT_EYE_IDX:
            cx = int(landmarks[i].x * frame.shape[1])
            cy = int(landmarks[i].y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

        for i in RIGHT_EYE_IDX:
            cx = int(landmarks[i].x * frame.shape[1])
            cy = int(landmarks[i].y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 2, (255, 0, 255), -1)

        # 눈 이미지 추출
        left_eye = extract_eye(frame, landmarks, LEFT_EYE_IDX)
        right_eye = extract_eye(frame, landmarks, RIGHT_EYE_IDX)

        if left_eye is not None:
            left = (left_eye[0] * 255).astype(np.uint8).reshape(24, 24)
            cv2.imshow("Left Eye", left)

        if right_eye is not None:
            right = (right_eye[0] * 255).astype(np.uint8).reshape(24, 24)
            cv2.imshow("Right Eye", right)

        if left_eye is not None and right_eye is not None:
            pred_left = model.predict(left_eye, verbose=0)[0][0]
            pred_right = model.predict(right_eye, verbose=0)[0][0]

            # 예측 확률 출력
            cv2.putText(frame, f"L: {pred_left:.2f}, R: {pred_right:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 눈 상태 판별
            eye_state = "Open" if (pred_left > 0.5 and pred_right > 0.5) else "Closed"
            color = (0, 255, 0) if eye_state == "Open" else (0, 0, 255)
            cv2.putText(frame, f"Eyes: {eye_state}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    else:
        cv2.putText(frame, "Face NOT detected", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Eye State Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
