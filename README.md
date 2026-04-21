# MNIST Neural Network & Real-Time Visualizer

Handwritten digit classifier trained with TensorFlow/Keras and an interactive real-time neural network visualizer built with Pygame.

## Objective
Train a dense neural network to classify handwritten digits (0–9) from the MNIST dataset and visualize the network's internal activations in real time as it processes each image.

## Model

Trained on 70,000 grayscale images (28×28 px) from the MNIST dataset.

### Architecture
```
Flatten (28×28 → 784)
Dense(200, activation='relu')
Dense(200, activation='relu')
Dense(200, activation='sigmoid')
Dense(10,  activation='softmax')   ← digit probabilities 0–9
```
- Optimizer: Adam
- Loss: sparse_categorical_crossentropy
- Epochs: 10 — Batch size: 100
- Split: 90% train / 10% validation + 10,000 test

### Results
| Metric | Score |
|--------|-------|
| Validation Accuracy | 98.95% |
| Test Accuracy | **97.40%** |
| Test Loss | 0.10 |

## Visualizer

`MNIST_visualizer.py` loads the trained model and displays three panels in real time:

- **Left** — MNIST digit image scaled up (8px per pixel)
- **Center** — Neural network with 5 columns (input, 3 hidden layers, output). Neurons glow from dark blue to bright cyan based on their activation level; connections brighten accordingly.
- **Right** — Probability bars for each digit (0–9). Green = correct prediction, red = wrong, blue = others.

### Controls
| Key | Action |
|-----|--------|
| `SPACE` / `→` | Next image |
| `←` | Previous image |
| `R` | Random image |
| `Q` / `ESC` | Quit |

Auto-advance every 2.6 seconds.

## Project Structure
```
├── MNIST_model.ipynb     # Training notebook
├── MNIST_visualizer.py   # Real-time Pygame visualizer
└── mnist_model.keras     # Trained model weights
```

## How to Run

```bash
git clone https://github.com/Flakkito/MNIST.git
cd MNIST
pip install tensorflow pygame
python MNIST_visualizer.py
```

> **Note:** The visualizer uses `tf.keras.datasets.mnist` directly — no need for `tensorflow_datasets`.

## Tech Stack
- Python, TensorFlow / Keras
- NumPy, Matplotlib, Seaborn, Scikit-learn
- Pygame
