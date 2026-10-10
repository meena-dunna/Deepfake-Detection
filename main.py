import torch
import timm
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from torchvision import transforms

app = FastAPI()

# Allow frontend (Vercel) requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# LAZY LOAD MODEL (important for Render memory)
# =========================
model = None

def get_model():
    global model
    if model is None:
        try:
            model = timm.create_model('legacy_xception', pretrained=False)

            in_features = model.get_classifier().in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features, 1)
            )

            model.load_state_dict(
                torch.load("model/final_best_model.pth", map_location=device)
            )

            model.to(device)
            model.eval()

            print("Model loaded successfully!")

        except Exception as e:
            print("Model loading failed:", e)
            model = None

    return model


# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/")
def home():
    return {"message": "Deepfake API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    prediction = "Real" if probability > 0.5 else "Fake"

    return {
        "raw_probability": probability,
        "prediction": prediction
    }
