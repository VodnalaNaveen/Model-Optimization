# 🧠 Keras Model Optimization: Compression + Quantization

This project demonstrates how to **compress** and **quantize** a Keras model to reduce its file size and improve inference performance for deployment.

---

## 📌 Techniques Covered

### 🔧 Model Compression
- Utilizes `tf.lite.TFLiteConverter` with optimization flags.
- Reduces model size by removing unused float precision.
- Weight data types remain as `float32`.

### 🧮 Model Quantization
- Applies **Post-Training Quantization** to convert weights from `float32` → `int8`.
- Enables faster inference and a smaller memory footprint.

### 🔍 Model Inspection (with [Netron](https://netron.app/))
- Verified model structures visually.
- Confirmed:
  - Compressed model retains `float32` weights.
  - Quantized model uses `int8` weights.

---

## 📊 Results

| Metric         | Normal Model | Compressed Model | Quantized Model |
|----------------|----------------|------------------|------------------|
| Model Size     | 33kb            | 3kb              | 4kb              |
| Accuracy       | 95%            | 51%              | 61%              |
| Loss       | 0.2070            | 1.1449              | 0.6830              |
| Inference Time | 11.9768 sec            | 0.0012 sec              | 0.0042 sec              |
| Weight Dtype   | `float32`      | `float32`        | `int8`           |

---

## 🗂️ File Structure

├── normal.py # Load the original model

├── compress_model.py # Apply model compression

├── quantize_model.py # Perform post-training quantization using TFLite

├── evaluate.ipynb # Evaluate performance of all three model versions

└── README.md # Project documentation


---

## ▶️ How to Run

Each script can be executed using command-line parser arguments:

```bash
python compress_model.py --weights_path 'normal.h5' --compressed_model_path 'C:\Users\vodna\OneDrive\Desktop\inno\DL\compresed_model'
python quantize_model.py --weights_path 'normal.h5' --quantized_model_path 'C:\Users\vodna\OneDrive\Desktop\inno\DL\compresed_model'
```


## 📚 References

- [Keras Model Optimization Documentation](https://www.tensorflow.org/model_optimization)
- [Netron - Visualize ML Models](https://netron.app/)



