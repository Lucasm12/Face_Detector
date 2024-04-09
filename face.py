import cv2

# Carrega o classificador pré-treinado para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicia a captura de vídeo da webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Captura frame por frame
    ret, frame = video_capture.read()

    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos na imagem
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Desenha um retângulo ao redor de cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exibe o frame resultante
    cv2.imshow('Video', frame)

    # Condição de saída: pressione 'q' para fechar a janela
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o objeto de captura de vídeo e fecha a janela
video_capture.release()
cv2.destroyAllWindows()
