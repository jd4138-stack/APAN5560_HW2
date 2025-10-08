# helper_lib/api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .model import get_model
from .checkpoints import load_checkpoint

app = FastAPI(title="HelperLib Classifier API")

# CIFAR-10 stats & labels
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD  = [0.2470, 0.2435, 0.2616]
_LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

class ClassifierService:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model("CNN", num_classes=10)
        # restore weights
        _ = load_checkpoint(self.model, optimizer=None, checkpoint_path=checkpoint_path, device=self.device)
        self.model.to(self.device).eval()
        # preprocess (resize to 64×64 to match the CNN)
        self.tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])

    @torch.no_grad()
    def predict_bytes(self, image_bytes: bytes, topk: int = 3):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=1)[0]
        p, i = probs.topk(topk)
        return {"topk": [{"label": _LABELS[j.item()], "index": int(j), "prob": float(pp)} for pp, j in zip(p, i)]}

# Load checkpoint path from env or default
_CHECKPOINT = os.getenv("CLASSIFIER_CKPT", "checkpoints/best.pt")
_svc: ClassifierService | None = None

@app.on_event("startup")
def _startup():
    global _svc
    if not os.path.exists(_CHECKPOINT):
        raise RuntimeError(f"Checkpoint not found: {_CHECKPOINT}. Train first or set CLASSIFIER_CKPT.")
    _svc = ClassifierService(checkpoint_path=_CHECKPOINT)

@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": _CHECKPOINT}

@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 3):
    if _svc is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    content = await file.read()
    return _svc.predict_bytes(content, topk=topk)
