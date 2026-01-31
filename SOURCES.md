# Sources & Attributions

This document lists all libraries, datasets, and external resources used in the Eigenfaces Demo project.

---

## Python Libraries

| Library | Purpose | License |
|---------|---------|---------|
| [Streamlit](https://streamlit.io/) | Interactive web interface and UI components | Apache 2.0 |
| [NumPy](https://numpy.org/) | Linear algebra, array operations, dot products, norms | BSD-3-Clause |
| [Matplotlib](https://matplotlib.org/) | Plotting face images, graphs, and visualizations | PSF-based |
| [Scikit-Learn](https://scikit-learn.org/) | PCA implementation, dataset loading, train/test split | BSD-3-Clause |
| [OpenCV (opencv-python)](https://opencv.org/) | Real-time face detection using Haar Cascades | Apache 2.0 |
| [Pillow (PIL)](https://pillow.readthedocs.io/) | Image uploads, format conversion, preprocessing | HPND |

---

## Dataset

### Olivetti Faces Dataset

- **Source:** AT&T Laboratories Cambridge (originally "ORL Database of Faces")
- **Loaded via:** `sklearn.datasets.fetch_olivetti_faces()`
- **Details:**
  - **Total Images:** 400
  - **Subjects:** 40 individuals
  - **Images per Subject:** 10
  - **Resolution:** 64×64 pixels (grayscale)
  - **Format:** Flattened vectors of 4,096 features
- **Usage in Project:** Training PCA model, testing recognition algorithm
- **License:** Free for academic and research purposes
- **Reference:** [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)

---

## Face Detection (Haar Cascades)

- **Used for:** Webcam face detection and auto-cropping
- **Source:** OpenCV pre-trained classifiers
- **Cascade File:** `haarcascade_frontalface_default.xml`
- **Original Paper:** Viola, P., & Jones, M. (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features." *CVPR*.
- **Reference:** [OpenCV Cascade Classifier Tutorial](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

---

## Academic References

### Primary Paper

> **Turk, M., & Pentland, A. (1991).** "Eigenfaces for Recognition."  
> *Journal of Cognitive Neuroscience*, 3(1), 71-86.  
> [IEEE Xplore](https://ieeexplore.ieee.org/document/139758)

### Related Works

- **Sirovich, L., & Kirby, M. (1987).** "Low-dimensional procedure for the characterization of human faces." *Journal of the Optical Society of America A*, 4(3), 519-524.
  - *Foundational work on applying PCA to face images*

- **Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997).** "Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 19(7), 711-720.
  - *Comparison with Linear Discriminant Analysis (LDA)*

---

## Icons & UI Elements

- **Streamlit Components:** Built-in UI elements (sliders, tabs, expanders, metrics)
- **Matplotlib Colormaps:** `gray`, `seismic` (for diverging difference maps)

---

## Additional Resources

### Mathematical Concepts

- **Principal Component Analysis (PCA):** Karhunen-Loève expansion / Hotelling transform
- **Eigenvalue Decomposition:** Standard linear algebra technique
- **Euclidean Distance:** Used for face matching and reconstruction error calculation

### Educational Context

This demo is designed for educational purposes to teach:
- Dimensionality reduction
- Linear algebra applications in computer vision
- The evolution from classical to modern face recognition

---

## License

This project is open-source under the **MIT License**.

See [LICENSE](LICENSE) for details.
