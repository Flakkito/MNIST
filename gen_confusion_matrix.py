import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

model = tf.keras.models.load_model("mnist_model.keras")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted", fontsize=13)
plt.ylabel("Actual", fontsize=13)
plt.title("Confusion Matrix — MNIST Test Set (10,000 samples)", fontsize=14)
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=150)
print("Saved to images/confusion_matrix.png")
