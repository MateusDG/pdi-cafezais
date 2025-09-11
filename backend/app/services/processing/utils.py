import cv2
import numpy as np

def imread(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {path}")
    return img

def imwrite(path: str, img):
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Falha ao salvar imagem em {path}")
