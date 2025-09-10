from celery import Celery
from .core.config import settings

celery_app = Celery(
    "pdi_cafezais",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

@celery_app.task(name="run_inference_task")
def run_inference_task(run_id: int):
    # TODO: chamar pipeline de PDI/ML (stub)
    return {"run_id": run_id, "status": "done"}


@celery_app.task(name="run_inference_geo", queue="ml")
def run_inference_geo(image_path: str | None = None):
    # Apenas operações PDI/Geo leves (ex.: NDVI, contornos, vetorização)
    try:
        from .modules.preprocessing.ndvi import ndvi
        import numpy as np
        # Dummy NDVI só para validar pipeline
        red = np.array([[10, 20], [30, 40]], dtype=float)
        nir = np.array([[20, 30], [40, 50]], dtype=float)
        m = ndvi(red, nir).mean().item()
        return {"tier": "geo", "ndvi_mean": m}
    except Exception as e:
        return {"tier": "geo", "error": str(e)}

@celery_app.task(name="run_inference_onnx", queue="ml")
def run_inference_onnx(model_path: str = "ml_models/weights/model.onnx"):
    # Inferência com ONNX Runtime (leve)
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        # Dummy input (depende do modelo real)
        inputs = { sess.get_inputs()[0].name: np.zeros((1, 3, 224, 224), dtype=np.float32) }
        out = sess.run(None, inputs)
        return {"tier": "ml-onnx", "outputs_len": len(out)}
    except Exception as e:
        return {"tier": "ml-onnx", "error": str(e)}

@celery_app.task(name="run_inference_torch", queue="ml")
def run_inference_torch(weights: str = "ml_models/weights/yolov8n.pt"):
    # Inferência com PyTorch/Ultralytics (pesado)
    try:
        from ultralytics import YOLO
        m = YOLO(weights)
        # Dummy inference on a zeros image (adjust path in production)
        import numpy as np, cv2
        img = (np.zeros((256,256,3), dtype=np.uint8))
        results = m.predict(source=img, verbose=False)
        return {"tier": "train+infer", "detections": len(results[0].boxes) if results else 0}
    except Exception as e:
        return {"tier": "train+infer", "error": str(e)}
