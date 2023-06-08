import cv2
import dlib

# 이미지 파일 경로
image_path = "asset/cat/강동원5.jpg"

# 이미지 불러오기
image = cv2.imread(image_path)

# Dlib의 얼굴 검출기 로드
detector = dlib.get_frontal_face_detector()

# 이미지에서 얼굴 검출
faces = detector(image)

# 얼굴 영역 추출 및 저장
for face in faces:
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    face_image = image[y1:y2, x1:x2]
    cv2.imshow("Face", face_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()