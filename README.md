# Deepfake Detection API

A FastAPI-based web service for detecting deepfake images using a pre-trained deep learning model.

## Features

- **Image Classification**: Classify uploaded images as Real or Fake
- **RESTful API**: Simple HTTP endpoints for easy integration
- **GPU Support**: Automatic GPU detection and utilization
- **Fast Inference**: Optimized for real-time predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jagadeesh0191/Deepfake-Detection.git
cd Deepfake-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model file `model/final_best_model.pth` in the correct location.

## Usage

### Running the Server

Start the FastAPI server:
```bash
python main.py
```

The server will start on `http://127.0.0.1:8000`

### API Endpoints

#### POST /predict

Upload an image file to get a deepfake prediction.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "raw_probability": 0.123456,
  "prediction": "Fake"
}
```

**Example using curl:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

## Model Details

- **Architecture**: Xception (legacy version)
- **Framework**: PyTorch with timm library
- **Input Size**: 299x299 pixels
- **Output**: Binary classification (Real/Fake)
- **Threshold**: 0.5 (probability > 0.5 = Real, else Fake)

## Requirements

- Python 3.7+
- PyTorch
- FastAPI
- timm (PyTorch Image Models)
- Pillow
- CUDA (optional, for GPU acceleration)

## Project Structure

```
Deepfake-Detection/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── model/
│   └── final_best_model.pth  # Pre-trained model weights
└── README.md            # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details