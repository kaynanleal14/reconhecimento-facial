import os
import cv2
import numpy as np


class Camera:

    def __init__(self, camera_index=0, test_image_path="test.jpg"):
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        self.use_mock = False

        # Se a câmera física não abrir, ativa o modo de simulação
        if not self.capture.isOpened():
            print(
                "[INFO] Câmera física não encontrada. Ativando modo de simulação."
            )
            self.use_mock = True

            # Resolve o caminho do arquivo 'test.jpg' na raiz do projeto
            if not os.path.isabs(test_image_path) and not os.path.exists(
                test_image_path
            ):
                base_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..")
                )
                alt_path = os.path.join(base_dir, test_image_path)
                if os.path.exists(alt_path):
                    test_image_path = alt_path

            if os.path.exists(test_image_path):
                self.mock_frame = cv2.imread(test_image_path)
            else:
                # Caso não tenha a foto 'test.jpg', gera uma imagem sintética
                self.mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    self.mock_frame,
                    "Modo Simulacao: Adicione 'test.jpg' na raiz",
                    (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

    def read(self):
        """Retorna o frame atual (ou None se falhar).

        No modo simulação, retorna a imagem estática continuamente.
        """
        if self.use_mock:
            if self.mock_frame is not None:
                return self.mock_frame.copy()
            return None

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def release(self):
        if not self.use_mock and self.capture.isOpened():
            self.capture.release()