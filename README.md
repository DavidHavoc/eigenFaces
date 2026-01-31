# Eigenfaces Demo: Educational Facial Recognition App 

An interactive Streamlit application that breaks down and visualizes the **Eigenfaces** facial recognition algorithm (Turk & Pentland, 1991).

This project is designed for students and enthusiasts to understand the mathematics behind early computer vision, specifically how **Principal Component Analysis (PCA)** is used to reconstruct and recognize human faces.

## Features

The app is divided into 7 educational tabs:

0.  **Training Deconstruction**: See exactly how the algorithm learns from training data:
    *   Calculate the Average Face
    *   Compute Difference Faces
    *   Build the Covariance Surrogate Matrix
    *   Find Eigenvectors and Eigenvalues
    *   Derive the final Eigenfaces
1.  **Component Explorer**: Inspect individual Eigenfaces and view a Scree Plot (Explained Variance) to understand data compression.
2.  **Step-by-Step Reconstruction**: A detailed mathematical walkthrough showing how a face is built layer-by-layer:
    *   Centering ($\text{Face} - \text{Mean}$)
    *   Projection (Calculating Weights/DNA)
    *   Weighted Sum (Mixing the ingredients)
    *   Final Reconstruction
3.  **Live Recognition**: Pick a random "Unknown" face and see the two-path algorithm (Face Detection + Identity Check) with threshold explanations.
4.  **Webcam Recognition**: Take a photo of yourself to see how the algorithm processes real-world images.
    *   **Auto-Face Detection**: Uses OpenCV/Haar Cascades to automatically find and crop your face.
    *   **Real-time Processing**: Preprocesses your image to match the training data format (64x64 grayscale).
5.  **History & Context**: Learn about the 1991 revolution in Computer Vision.
6.  **Modern AI**: A comparison between Eigenfaces (Holistic) vs. Modern Deep Learning/CNNs (Feature-based).

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/DavidHavoc/eigenFaces.git
    cd eigenFaces
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run main.py
    ```

## Technologies Used

*   **Python 3.x**
*   **Streamlit**: For the interactive web interface.
*   **Scikit-Learn**: For the PCA implementation and Olivetti Faces dataset.
*   **OpenCV**: For real-time face detection and image processing.
*   **Pillow (PIL)**: For handling image uploads and formatting.
*   **NumPy**: For linear algebra calculations (Dot products, Norms).
*   **Matplotlib**: For plotting face images and graphs.

## Educational Value

This project helps answer:
*   How can math describe a face?
*   What is a "Weight Vector" or "Face DNA"?
*   How does dimensionality reduction (PCA) work on images?
*   Why did we switch from Eigenfaces to Deep Learning?

## References

*   **Turk, M., & Pentland, A. (1991).** "Eigenfaces for Recognition." *Journal of Cognitive Neuroscience*, 3(1), 71-86.  
    [IEEE Xplore](https://ieeexplore.ieee.org/document/139758/authors#authors)

## Common Issues & Troubleshooting

### 1. Images Not Loading / Broken Images
**Issue:** Sometimes the face images or plots appear as broken icons or do not render immediately.
**Fix:** This is often a Streamlit caching or rendering hiccup.
*   **Refresh the page** (Cmd+R or F5).
*   If running locally, try clearing the Streamlit cache by clicking the "Hamburger menu" (top right) -> "Clear cache" and then rerun.

### 2. "ModuleNotFoundError"
**Issue:** Passages like `ModuleNotFoundError: No module named 'cv2'` or `sklearn`.
**Fix:** Ensure you have installed the requirements in your active environment:
```bash
pip install -r requirements.txt
```

### 3. Webcam Not Starting
**Issue:** The webcam component stays blank or shows an error.
**Fix:**
*   Ensure your browser has **permission** to access the camera.
*   Make sure no other application (like Zoom or Teams) is currently using the camera.
*   If on macOS, check System Settings -> Privacy & Security -> Camera to ensure your terminal/browser has access.

### 4. Port Already in Use
**Issue:** Error "Port 8501 is already in use" or the app doesn't launch.
**Fix:**
*   **Option A:** Close the *other* terminal window where Streamlit is currently running (Press `Ctrl+C`).
*   **Option B:** Run on a different port manually:
    ```bash
    streamlit run main.py --server.port 8502
    ```

## License

This project is open-source and available under the MIT License.
