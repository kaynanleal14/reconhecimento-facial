import os
import cv2
import numpy as np


class FaceEmbedder:

    def __init__(
        self, model_path="models/face_recognition_sface_2021dec.onnx"
    ):
        # Resolve o caminho do modelo para Windows e Linux
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            alt_path = os.path.join(base_dir, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path

        self.model_path = model_path
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path, config=""
        )

        print("Modelo SFace (ONNX) carregado via OpenCV nativo")

    def generate(self, face):
        """Gera o vetor de embedding de 128 elementos a partir do corte do rosto."""
        if face is None or face.size == 0:
            return np.zeros(128, dtype=np.float32)

        # Redimensiona o recorte do rosto para 112x112
        face_resized = cv2.resize(face, (112, 112))

        # Extrai as características faciais
        embedding = self.recognizer.feature(face_resized)
        embedding = embedding.flatten()

        # Normalização L2
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding