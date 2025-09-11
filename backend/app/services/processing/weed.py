import cv2
import numpy as np

def placeholder_annotate(img: np.ndarray) -> np.ndarray:
    # Demonstração: desenha uma caixa no centro como se fosse uma área de mato
    out = img.copy()
    h, w = out.shape[:2]
    x1, y1 = int(w*0.3), int(h*0.3)
    x2, y2 = int(w*0.7), int(h*0.7)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0,0,255), 3)
    cv2.putText(out, "possivel mato (demo)", (x1, max(30, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    return out
