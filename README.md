# Pedestrian-Attention-Recognition-Deep-Learning-Project
CNN-based pedestrian attention recognition system using real-world image data and PyTorch.

## Project Goals
- **Detect pedestrian attentiveness in real time** from images.
- Issue a warning if a pedestrian is classified as distracted.
- Build a robust preprocessing pipeline for gaze-based classification.
- Handle highly imbalanced real-world data effectively.

---

## Dataset
We use the **MPIIGaze** dataset:
- **Size:** 213,659 RGB images from 15 participants in natural settings.
- **Diversity:** Multiple head poses, lighting conditions, and backgrounds.
- **Task adaptation:** Converted into a binary classification dataset.

**Labeling Strategy:**
1. Extract 2D head pose & gaze direction (pitch, yaw) from `annotation.txt`.
2. Convert to 3D unit vectors.
3. Compute angular difference.
4. **Attentive** if difference ≤ 45°, **Distracted** otherwise.

**Class Balancing:**
- Original: 176,618 distracted vs. 37,040 attentive → **highly imbalanced**.
- Balanced via random downsampling → 37,040 per class (total 74,080 samples).

**Preprocessing:**
- Convert to grayscale.
- Resize to 224×224 (later 64×64 for speed).
- Store in `/attentive/` and `/distracted/` folders.
- Upload final dataset to Google Drive for Colab training.

---

## Key Challenges & Solutions
| Challenge | Solution |
|-----------|----------|
| Dataset documentation referenced `.mat` files without image paths. | Located correct gaze/head pose data in nested `pXX/dayXX/annotation.txt` files. |
| Too few “attentive” samples with strict 15° threshold. | Increased threshold to 45° after manual inspection. |
| Preprocessing 200k+ images in Colab was slow. | Switched to local preprocessing in VS Code with multithreading. |

---

## Model Architecture
Primary model:
- **Input:** Grayscale images.
- **Layers:** 3× Conv + ReLU + Pool (3×3 filters, consistent stride & padding).
- **Skip connection** to mitigate vanishing gradients.
- **Two auxiliary loss functions** to aid convergence.
- **Dropout (0.5)** for regularization.

**Hyperparameters:**
- Batch size: 64
- Learning rate: 0.001
- Auxiliary loss weight: 0.4
- Epochs: 10
- Optimizer: `BCEWithLogitsLoss`
- Momentum: 0.9
- Sample size for experiments: 5000 images

---

## Results
| Metric | Value |
|--------|-------|
| Test Error | 0.5010 |
| Test Loss  | 1.2488 |
| Training Time | 4776.04s |

**Note:** Results are from preliminary experiments with a reduced dataset size (5000 samples) to iterate quickly.

---

## Future Work
- Train on the **full dataset** for higher accuracy.
- Test with additional datasets like **GazeCapture** or **OpenFace**.
- Integrate temporal modeling (LSTMs/GRUs) for video-based prediction.
- Deploy as a **real-time attention detection module** in an autonomous driving simulation.
