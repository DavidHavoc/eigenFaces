# Eigenfaces Demo: Educational Facial Recognition App 🎭

An interactive Streamlit application that breaks down and visualizes the **Eigenfaces** facial recognition algorithm (Turk & Pentland, 1991).

This project is designed for students and enthusiasts to understand the mathematics behind early computer vision, specifically how **Principal Component Analysis (PCA)** is used to reconstruct and recognize human faces.

## Features

The app is divided into 6 educational tabs:

1.  **The Face Space**: Visualizes the "Average Face" and the top Eigenfaces (Ghostly features).
2.  **Step-by-Step Reconstruction**: A detailed mathematical walkthrough showing how a face is built layer-by-layer:
    *   Centering ($\text{Face} - \text{Mean}$)
    *   Projection (Calculating Weights/DNA)
    *   Weighted Sum (Mixing the ingredients)
    *   Final Reconstruction
3.  **Live Recognition**: Pick a random "Unknown" face and see the algorithm find the best match using **Euclidean Distance**. Includes vector visualizations.
4.  **Component Explorer**: Inspect individual Eigenfaces and view a Scree Plot (Explained Variance) to understand data compression.
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
*   **NumPy**: For linear algebra calculations (Dot products, Norms).
*   **Matplotlib**: For plotting face images and graphs.

## Educational Value

This project helps answer:
*   How can math describe a face?
*   What is a "Weight Vector" or "Face DNA"?
*   How does dimensionality reduction (PCA) work on images?
*   Why did we switch from Eigenfaces to Deep Learning?

## License

This project is open-source and available under the MIT License.
