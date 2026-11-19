import os
import cv2


class FaceDetector:

    def __init__(
        self,
        model_path="models/face_detection_yunet_2023mar.onnx",
        min_detection_confidence=0.6,
    ):
        # Resolve o caminho do modelo para funcionar tanto no Windows quanto no Linux/Armbian
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            alt_path = os.path.join(base_dir, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path

        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence

        # Inicializa o detector YuNet nativo do OpenCV
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),  # Tamanho inicial (atualizado no detect)
            score_threshold=self.min_detection_confidence,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect(self, frame):
        """Recebe um frame BGR do OpenCV.

        Retorna:
        [
            {
                "bbox": (x, y, w, h),
                "confidence": float
            }
        ]
        """
        if frame is None:
            return []

        frame_height, frame_width = frame.shape[:2]

        # Atualiza a resolução de entrada do YuNet para a resolução atual do frame
        self.detector.setInputSize((frame_width, frame_height))

        # Executa a detecção nativa do OpenCV
        _, results = self.detector.detect(frame)

        faces = []

        if results is not None:
            for face in results:
                # O YuNet retorna os 4 primeiros valores como [x, y, w, h]
                x, y, w, h = map(int, face[:4])

                # O índice 14 contém a pontuação de confiança (score)
                confidence = float(face[14])

                # Evita coordenadas negativas
                x = max(0, x)
                y = max(0, y)

                # Garante que o retângulo não ultrapasse os limites da imagem
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)

                faces.append(
                    {"bbox": (x, y, w, h), "confidence": confidence}
                )

        return faces

    def crop_face(self, frame, face, margin=0.20):
        x, y, w, h = face["bbox"]

        frame_height, frame_width = frame.shape[:2]

        margin_x = int(w * margin)
        margin_y = int(h * margin)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)

        x2 = min(frame_width, x + w + margin_x)
        y2 = min(frame_height, y + h + margin_y)

        cropped_face = frame[y1:y2, x1:x2]

        return cropped_face

    def close(self):
        # Mantido para compatibilidade com chamadas antigas no app.py
        pass