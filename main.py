import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

# --- Page Configuration ---
st.set_page_config(
    page_title="Eigenfaces Demo (Turk & Pentland)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data
def load_and_prepare_data():
    """
    Loads the Olivetti faces dataset.
    Splits into Training and Testing sets.
    Returns flattened vectors and their original shape info.
    """
    # Download/Load dataset (Cache enabled)
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data  # Flattened images (400 samples, 4096 features)
    y = data.target # Person IDs (0-39)
    
    # Split data: 75% Train, 25% Test
    # We stratify to ensure each person is in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

@st.cache_data
def compute_full_pca(X_train, max_components):
    """
    Computes PCA on the training set up to the maximum possible components.
    We cache this so we don't recompute the expensive eigendecomposition 
    every time the slider moves. We will slice the results dynamically.
    """
    # n_components=min(n_samples, n_features) = min(300, 4096) = 300
    pca = PCA(n_components=max_components, whiten=True, random_state=42)
    pca.fit(X_train)
    return pca

def plot_face(img, title, ax, cmap='gray'):
    """Utility to plot a single face on a given axis."""
    ax.imshow(img.reshape(64, 64), cmap=cmap)
    ax.axis('off')
    ax.set_title(title, fontsize=10)

# --- Main Application Logic ---

def main():
    st.title("Eigenfaces: Face Recognition using PCA (Principal Component Analysis)")
    st.markdown("""
    Based on the paper *'Face Recognition Using Eigenfaces'* by Turk & Pentland (1991).
    
    This demo reconstructs faces by projecting them onto a 'Face Space' defined by the 
    principal components (Eigenfaces) of the training set.
    """)

    # 1. Data Setup
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    n_samples, n_features = X_train.shape
    
    # The maximum number of components we can calculate is limited by the number of training samples
    max_n_components = n_samples 
    
    # Fit PCA once (cached)
    # Note: Theoretically, the paper centers the data first. sklearn.PCA does this automatically.
    full_pca = compute_full_pca(X_train, max_n_components)

    # 2. Sidebar Controls
    st.sidebar.header("Dimensionality Controls")
    
    # Slider for Number of Eigenfaces
    # We map 1..150 (as requested) though we technically calculated up to 300.
    n_components = st.sidebar.slider(
        "Number of Eigenfaces (K)",
        min_value=1,
        max_value=150, # As per requirement
        value=10,
        step=1,
        help="Controls the dimensionality of the Face Space. Increasing K adds more detail."
    )

    # Explanation of what happens when slider moves
    with st.sidebar.expander("What changes when I slide?"):
        st.write(f"1. We take the top **{n_components}** Eigenfaces.")
        st.write("2. We project a face onto these {n_components} vectors.")
        st.write("3. Reconstruction accuracy improves as K increases.")

    # --- Update Logic ---
    # Instead of re-fitting PCA (slow), we slice the already fitted model.
    # This simulates training with fewer components for visualization speed.
    
    # Get the top K components
    eigenfaces = full_pca.components_[:n_components]
    
    # Get the Mean Face (Psi)
    mean_face = full_pca.mean_

    # Project training data to get Weights (Omega)
    # Omega_train shape: (n_samples, n_components)
    # Formula: Omega = (Face - Mean) dot Eigenfaces_T
    X_train_centered = X_train - mean_face
    weights_train = np.dot(X_train_centered, eigenfaces.T)

    # --- Main Area: Tabs ---
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["0. Training Deconstruction", "1. Explorer", "2. Reconstruction", "3. Live Recognition", "4. Webcam Recognition", "5. History & Context", "6. Future & Modern"])

    # =======================
    # TAB 0: TRAINING DECONSTRUCTION
    # =======================
    with tab0:
        st.header("How the Algorithm Learns: Training Deconstruction")
        st.markdown("""
        Before the Eigenfaces algorithm can recognize faces, it must **learn** from a set of training images.
        This tab shows the exact mathematical steps used to build the 'Face Space'.
        """)
        
        # --- STEP 1: Average Face ---
        st.markdown("---")
        st.subheader("Step 1: Calculate the Average Face (Psi)")
        st.markdown(r"""
        We calculate the arithmetic mean of all $M$ images in the training set.
        """)
        st.latex(r"\Psi = \frac{1}{M} \sum_{n=1}^{M} \Gamma_n")
        
        col_s1_1, col_s1_2 = st.columns([2, 1])
        with col_s1_1:
            st.markdown(r"""
            - $\Gamma_n$: The $n$-th training image vector (flattened to 4096 pixels).
            - $M$: The total number of training images (300 in our case).
            - $\Psi$: The resulting **Average Face**.
            """)
        with col_s1_2:
            fig_s1, ax_s1 = plt.subplots(figsize=(2.5, 2.5))
            plot_face(mean_face, "Average Face (Psi)", ax_s1)
            st.pyplot(fig_s1)
        
        st.info("The Average Face represents what an 'average' person looks like in our dataset. It becomes the **origin** of our Face Space.")
        
        # --- STEP 2: Difference Faces ---
        st.markdown("---")
        st.subheader("Step 2: Calculate the Difference Faces (Phi)")
        st.markdown(r"""
        We subtract the average face from each training image to isolate the **unique deviations**.
        """)
        st.latex(r"\Phi_i = \Gamma_i - \Psi")
        
        st.markdown(r"""
        - $\Phi_i$: The mean-centered face vector (the 'difference' face).
        - This removes common features and highlights what makes each face **unique**.
        """)
        
        # Show example difference faces
        st.write("**Example: First 5 training faces and their difference faces:**")
        fig_s2, axes_s2 = plt.subplots(3, 5, figsize=(12, 7))
        for i in range(5):
            # Original face
            plot_face(X_train[i], f"Original {i+1}", axes_s2[0, i])
            # Mean face (same for all)
            if i == 2:
                axes_s2[1, i].set_title("- Average =", fontsize=10)
            plot_face(mean_face, "", axes_s2[1, i])
            # Difference face
            diff_face = X_train[i] - mean_face
            vmax = np.max(np.abs(diff_face))
            axes_s2[2, i].imshow(diff_face.reshape(64, 64), cmap='seismic', vmin=-vmax, vmax=vmax)
            axes_s2[2, i].axis('off')
            axes_s2[2, i].set_title(f"Phi_{i+1}", fontsize=10)
        
        axes_s2[0, 0].set_ylabel("Original", fontsize=10)
        axes_s2[1, 0].set_ylabel("Mean", fontsize=10)
        axes_s2[2, 0].set_ylabel("Difference", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_s2)
        
        st.info("The difference faces look 'ghostly' because they only contain the deviations from average. Red = above average, Blue = below average.")
        
        # --- STEP 3: Covariance Surrogate Matrix ---
        st.markdown("---")
        st.subheader("Step 3: Calculate the Covariance Surrogate Matrix (L)")
        st.markdown(r"""
        The **Covariance Matrix** $C = A A^T$ would be huge ($4096 \times 4096$). 
        Instead, we use a clever trick and calculate the smaller **Surrogate Matrix** $L = A^T A$.
        """)
        st.latex(r"L_{mn} = \Phi_m^T \Phi_n")
        
        st.markdown(r"""
        - $L$: An $M \times M$ matrix (e.g., $300 \times 300$).
        - $\Phi_m^T \Phi_n$: The **dot product** of difference image $m$ and difference image $n$.
        - This captures how similar any two faces are to each other.
        """)
        
        st.warning("**Why this trick?** The eigenvectors of the small matrix $L$ can be mathematically transformed into the eigenvectors of the large matrix $C$. This saves enormous computation!")
        
        st.info(r"""
        **What does the Covariance Matrix actually tell us?**
        
        The Covariance Matrix is a massive grid that answers one question for every possible pair of pixels: 
        
        *"When Pixel A gets brighter, does Pixel B get brighter too?"*
        
        - **High Covariance:** Yes, they change together. (This implies **redundancy/structure**.)
        - **Zero Covariance:** No, they are random relative to each other. (This implies **noise**.)
        
        The Covariance Matrix captures the Shape of the data clogud.
        """)
        
        # --- STEP 4: Eigenvectors of L ---
        st.markdown("---")
        st.subheader("Step 4: Find the Eigenvectors of L (v)")
        st.markdown(r"""
        We solve the standard eigenvalue problem for the small matrix $L$.
        """)
        st.latex(r"L v_k = \lambda_k v_k")
        
        st.markdown(r"""
        - $v_k$: The $k$-th eigenvector of the small matrix $L$.
        - $\lambda_k$: The associated **eigenvalue** (a scalar indicating importance).
        - Larger eigenvalues = more important directions of variation.
        """)
        
        # Show eigenvalue spectrum
        eigenvalues = full_pca.explained_variance_
        fig_s4, ax_s4 = plt.subplots(figsize=(10, 4))
        ax_s4.bar(range(1, min(51, len(eigenvalues)+1)), eigenvalues[:50], color='steelblue')
        ax_s4.set_xlabel("Eigenvalue Index (k)")
        ax_s4.set_ylabel("Eigenvalue (lambda_k)")
        ax_s4.set_title("Eigenvalue Spectrum (First 50)")
        ax_s4.set_yscale('log')
        st.pyplot(fig_s4)
        
        st.info("Notice how eigenvalues drop rapidly. The first few eigenfaces capture most of the variation; later ones capture noise.")
        
        # --- STEP 5: Calculate Eigenfaces ---
        st.markdown("---")
        st.subheader("Step 5: Calculate the Eigenfaces (u)")
        st.markdown(r"""
        We define the **Eigenfaces** by mapping the small eigenvectors $v$ back into the original image space using the difference faces.
        """)
        st.latex(r"u_k = \sum_{n=1}^{M} v_{kn} \Phi_n")
        
        st.markdown(r"""
        - $u_k$: The final $k$-th **Eigenface**.
        - $v_{kn}$: The $n$-th component of the eigenvector $v_k$.
        - $\Phi_n$: The $n$-th difference face.
        
        Each Eigenface is a weighted combination of all the difference faces!
        """)
        
        # Show first 10 eigenfaces
        st.write("**The First 10 Eigenfaces:**")
        fig_s5, axes_s5 = plt.subplots(2, 5, figsize=(12, 5))
        for i in range(10):
            row, col = i // 5, i % 5
            plot_face(full_pca.components_[i], f"u_{i+1}", axes_s5[row, col])
        plt.tight_layout()
        st.pyplot(fig_s5)
        
        # --- STEP 6: Weight Vectors ---
        st.markdown("---")
        st.subheader("Step 6: Calculate & Save Known Weight Vectors (Ω_database)")
        st.markdown(r"""
        The computer takes the original $M$ training images, subtracts the **Average Face** ($\Psi$), 
        and projects them onto the **Eigenfaces** ($u$) we just created.
        """)
        st.latex(r"\Omega_i = ({\Gamma_i - \Psi}) \cdot U^T")
        
        st.markdown(r"""
        - $\Gamma_i$: The $i$-th training image
        - $\Psi$: The Average Face (from Step 1)
        - $U$: The matrix of Eigenfaces (from Step 5)
        - $\Omega_i$: The resulting **Weight Vector** (Face Print) for image $i$
        """)
        
        # Show example weight vectors
        st.write("**Example: Weight Vectors for first 5 training images:**")
        fig_s6, axes_s6 = plt.subplots(1, 5, figsize=(14, 3))
        for i in range(5):
            axes_s6[i].bar(range(min(10, n_components)), weights_train[i][:10], color='steelblue')
            axes_s6[i].set_title(f"Person {y_train[i]}", fontsize=10)
            axes_s6[i].set_xlabel("Component", fontsize=8)
            if i == 0:
                axes_s6[i].set_ylabel("Weight", fontsize=8)
            axes_s6[i].tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig_s6)
        
        st.info("""
        **Result:** A list of weight vectors (**Face Prints**) for every person (Alice, Bob, etc.) is saved.
        
        **Why Sixth?** We couldn't calculate these "coordinates" before we had the "map" (the Eigenfaces).
        """)
        
        # --- STEP 7: Thresholds ---
        st.markdown("---")
        st.subheader("Step 7: Calculate & Save Thresholds (θ_ε and θ_δ)")
        st.markdown(r"""
        The thresholds are **critical** parameters that determine how strict the system is. 
        They are learned from the training data by analyzing how the system behaves.
        
        The computer asks two critical questions:
        1. **"What is the max distance between a person's own photos?"** → This determines $\theta_\epsilon$ (identity threshold)
        2. **"What is the worst reconstruction error for known faces?"** → This determines $\theta_\delta$ (face detection threshold)
        """)
        
        # --- Path 1 Threshold (Face Detection) ---
        st.markdown("---")
        st.markdown("#### θ_δ (Face Detection Threshold)")
        st.error(r"""
        **Method:** Calculate the reconstruction error for all *known faces* in the training set, 
        then set the threshold to be slightly above the maximum observed error.
        
        To calculate the threshold $\theta_\delta$ (Step 7), we perform an internal **"sanity check"** during the training phase. 
        We reconstruct every face in our training set and find the one with the **worst error**. 
        We use that worst-case error to set the limit for the live system.
        """)
        
        # Calculate reconstruction errors for training faces
        train_phi = X_train - mean_face
        train_reconstructed = np.dot(np.dot(train_phi, eigenfaces.T), eigenfaces)
        train_errors = np.linalg.norm(train_phi - train_reconstructed, axis=1)
        
        st.markdown("**How is ε_train calculated?**")
        st.markdown(r"""
        For each training face, we measure how well Eigenfaces can reconstruct it:
        """)
        st.latex(r"\epsilon_i = ||\Phi_i - \Phi_{rec,i}^{(K)}||")
        st.markdown(r"""
        Where the reconstruction using $K$ Eigenfaces is:
        """)
        st.latex(r"\Phi_{rec,i}^{(K)} = \sum_{j=1}^{K} \omega_{ij} \cdot u_j")
        st.markdown(r"""
        - $\Gamma_i$ = Original training image
        - $\Psi$ = Average face  
        - $\Phi_i = \Gamma_i - \Psi$ = Centered face (difference from average)
        - $\Phi_{rec,i}^{(K)}$ = Reconstructed centered face using **K** Eigenfaces
        - $\omega_{ij}$ = Weight (projection) of face $i$ onto Eigenface $j$
        - $u_j$ = The $j$-th Eigenface
        - $||\cdot||$ = Euclidean distance (how different are the pixel values?)
        
        A **low ε** means the face reconstructs well → it's a valid face in our "Face Space".
        """)
        
        st.latex(r"\theta_\delta = \max(\epsilon_{train}) + \text{margin}")
        
        st.markdown(r"""
        **Breaking down the formula:**
        - $\max(\epsilon_{train})$ = The **worst** (highest) reconstruction error we observed among all training faces
        - $\text{margin}$ = A **safety buffer** (e.g., 10-20% extra) to account for natural variation in new images
        
        **In plain English:** *"If the worst-reconstructed known face had error X, then anything with error > X + buffer is probably NOT a face."*
        """)
        
        col_face_stats, col_face_hist = st.columns([1, 2])
        
        with col_face_stats:
            margin_example = np.max(train_errors) * 0.15  # 15% margin example
            st.markdown(f"""
            **From our training data:**
            - Max reconstruction error: **{np.max(train_errors):.3f}**
            - Example margin (15%): **{margin_example:.3f}**
            - **Suggested θ_δ:** {np.max(train_errors) + margin_example:.3f}
            """)
        
        with col_face_hist:
            # Show histogram
            fig_hist1, ax_hist1 = plt.subplots(figsize=(8, 3))
            ax_hist1.hist(train_errors, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
            ax_hist1.axvline(x=np.max(train_errors), color='red', linestyle='--', linewidth=2, label=f'Max Error = {np.max(train_errors):.3f}')
            ax_hist1.set_xlabel('Reconstruction Error')
            ax_hist1.set_ylabel('Count')
            ax_hist1.set_title('Distribution of Reconstruction Errors (Training Faces)')
            ax_hist1.legend()
            st.pyplot(fig_hist1)
        
        # --- Path 2 Threshold (Identity) ---
        st.markdown("---")
        st.markdown("#### θ_ε (Identity Threshold)")
        st.markdown(r"""
        **Method:** For each person, calculate the average distance between their own images. 
        The threshold should be larger than within-class distances but smaller than between-class distances.
        """)
        
        st.latex(r"\theta_\epsilon = \alpha \cdot \text{mean}(\text{within-class distances})")
        
        st.markdown(r"""
        **Breaking down the formula:**
        - $\text{mean}(\text{within-class distances})$ = The **average** distance between photos of the **same person**
        - $\alpha$ = A **multiplier** (typically 1.5 to 3.0) that sets how strict the system is
        
        **In plain English:** *"If photos of the same person are typically X units apart, then someone within α×X distance is probably that person."*
        
        - $\alpha = 1.5$ → **Strict** (fewer false positives, might reject valid matches)
        - $\alpha = 2.0$ → **Balanced** (good trade-off)
        - $\alpha = 3.0$ → **Lenient** (accepts more matches, risk of misidentification)
        """)
        
        # Calculate within-class distances
        within_class_dists_train = []
        for person_id in np.unique(y_train):
            person_indices = np.where(y_train == person_id)[0]
            if len(person_indices) > 1:
                person_weights = weights_train[person_indices]
                for i in range(len(person_weights)):
                    for j in range(i+1, len(person_weights)):
                        within_class_dists_train.append(np.linalg.norm(person_weights[i] - person_weights[j]))
        
        within_class_dists_train = np.array(within_class_dists_train)
        
        col_id_stats, col_id_hist = st.columns([1, 2])
        
        with col_id_stats:
            mean_dist = np.mean(within_class_dists_train)
            st.markdown(f"""
            **From our training data:**
            - Mean within-class distance: **{mean_dist:.2f}**
            
            **Example thresholds:**
            - α=1.5 (strict): **{1.5 * mean_dist:.2f}**
            - α=2.0 (balanced): **{2.0 * mean_dist:.2f}**
            - α=3.0 (lenient): **{3.0 * mean_dist:.2f}**
            """)
        
        with col_id_hist:
            # Show histogram of all pairwise distances
            fig_hist2, ax_hist2 = plt.subplots(figsize=(8, 3))
            ax_hist2.hist(within_class_dists_train, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
            ax_hist2.axvline(x=np.max(within_class_dists_train), color='red', linestyle='--', linewidth=2, label=f'Max = {np.max(within_class_dists_train):.2f}')
            ax_hist2.set_xlabel('Distance Between Face Weights')
            ax_hist2.set_ylabel('Count')
            ax_hist2.set_title('Distribution of Within-Class Distances (Same Person)')
            ax_hist2.legend()
            st.pyplot(fig_hist2)
        
        st.info("""
        **Key Insight:** If the threshold is too low, the system rejects everyone as 'Unknown'. 
        If too high, it misidentifies people. Finding the right balance requires experimentation 
        or techniques like ROC curve analysis.
        
        **Why Last?** You cannot calculate the "safe limits" of the system until you have generated the data and observed how the system behaves.
        """)
        
        # --- COMPRESSION METRIC DASHBOARD ---
        st.markdown("---")
        st.subheader("Dimensionality Reduction: The Power of PCA")
        
        # Calculate compression metrics
        original_size = 4096  # 64x64 pixels
        compressed_size = n_components
        compression_ratio = original_size / compressed_size
        
        # Create 3-column dashboard layout
        col_orig, col_comp, col_ratio = st.columns(3)
        
        with col_orig:
            st.metric(
                label="Original Input Space (N²)",
                value="4,096",
                help="Each 64×64 image has 4,096 pixel values"
            )
            st.caption("Pixels (Dimensions)")
        
        with col_comp:
            st.metric(
                label="Reduced Face Space (K)",
                value=f"{compressed_size}",
                help="Number of Eigenface components used"
            )
            st.caption("Weights (Dimensions)")
        
        with col_ratio:
            st.metric(
                label="Compression Ratio",
                value=f"{compression_ratio:.1f}x",
                delta="Efficient" if compression_ratio > 10 else "Low",
                delta_color="normal"
            )
            st.caption("Dimensionality Reduction")
        
        st.caption(r"""
        By projecting the high-dimensional image data ($N^2$) onto the low-dimensional Face Space ($K$), 
        we achieve a massive reduction in redundancy. According to **Information Theory**, this ratio represents 
        the efficiency of the **Karhunen-Loève expansion** in isolating the useful signal (Variance) from the noise.
        
        *Try moving the K slider in the sidebar to see how the compression ratio changes!*
        """)
        
        st.info(r"""
        **What is K / Face Space?**
        
        The **Face Space** is a $K$-dimensional subspace where every face can be represented as a single point (or vector). 
        Instead of storing 4,096 pixel values, we store just $K$ numbers — the **weights** that tell us "how much" of each 
        Eigenface to mix together. This compact representation is what makes recognition fast and efficient.
        """)
        
        st.latex(r"K = \{u_1, u_2, u_3, \ldots, u_K\}")
        st.caption("K represents the set of K Eigenfaces (basis vectors) that span the Face Space.")
        
        st.success("""
        **Training Complete!** 
        
        These Eigenfaces form the basis of our 'Face Space'. Any new face can now be:
        1. Centered (subtract the mean)
        2. Projected onto these Eigenfaces to get a compact 'weight vector'
        3. Reconstructed or compared to other faces
        """)

    # =======================
    # TAB 2: RECONSTRUCTION
    # =======================
    with tab2:
        st.header("Step-by-Step Reconstruction Math")
        st.write("Follow the exact journey of a face image as it gets compressed and reconstructed.")

        # Select Image
        img_idx = st.selectbox(
            "Select Training Image ID", 
            range(len(X_train)), 
            format_func=lambda x: f"Person {y_train[x]} (Image {x})"
        )
        
        # Calculations
        original_face = X_train[img_idx]
        mean_face = full_pca.mean_
        
        # STEP 1: CENTERING
        st.markdown("---")
        st.subheader("Step 1: Centering ($\Phi$)")
        st.markdown(r"First, we subtract the **Average Face** ($\Psi$) from the **Original Face** ($\Gamma$) to isolate the unique features.")
        st.latex(r"\Phi = \Gamma - \Psi")
        
        centered_face = original_face - mean_face
        
        col1, col2, col3, col4, col5 = st.columns([1,0.2,1,0.2,1])
        with col1:
            st.caption("Original ($\Gamma$)")
            fig, ax = plt.subplots(figsize=(2,2))
            plot_face(original_face, "Input", ax)
            st.pyplot(fig)
        with col2: st.markdown("## -")
        with col3:
            st.caption("Average ($\Psi$)")
            fig, ax = plt.subplots(figsize=(2,2))
            plot_face(mean_face, "Mean", ax)
            st.pyplot(fig)
        with col4: st.markdown("## =")
        with col5:
            st.caption("Centered ($\Phi$)")
            fig, ax = plt.subplots(figsize=(2,2))
            # Use diverging colormap for centered face (positive/negative values)
            vmax = np.max(np.abs(centered_face))
            ax.imshow(centered_face.reshape(64,64), cmap='seismic', vmin=-vmax, vmax=vmax)
            ax.axis('off')
            st.pyplot(fig)
            
        st.info("The 'Centered Face' looks ghost-like because it only contains the *differences* from the average.")

        # STEP 2: PROJECTION
        st.markdown("---")
        st.subheader("Step 2: Projection (Finding Weights $\omega$)")
        st.markdown(r"We calculate how much of each **Eigenface** ($u_i$) is needed to rebuild this specific face. This is done via dot product.")
        st.latex(r"\omega_i = \Phi \cdot u_i^T")
        
        # Calculate weights for just this face
        eigenfaces_k = full_pca.components_[:n_components]
        projection_weights = np.dot(centered_face, eigenfaces_k.T)
        
        st.write(f"Result: A list of {n_components} numbers (Weights).")
        st.bar_chart(projection_weights)
        st.caption(f"The unique recipe for this person. A high bar means this eigenface is 'very matching'.")

        # STEP 3: WEIGHTED SUM
        st.markdown("---")
        st.subheader("Step 3: Weighted Sum ($\Phi_{rec}$)")
        st.markdown(r"Now we rebuild the centered face by mixing the Eigenfaces according to our weights.")
        st.latex(r"\Phi_{rec} = \sum_{i=1}^{K} \omega_i u_i")
        
        reconstructed_centered = np.dot(projection_weights, eigenfaces_k)
        
        # Visual Sum (Simplified)
        st.write("Visualizing the mixture (First 3 components):")
        cols = st.columns(min(3, n_components))
        for i in range(len(cols)):
            with cols[i]:
                component_img = projection_weights[i] * eigenfaces_k[i]
                fig, ax = plt.subplots(figsize=(2,2))
                v_lim = np.max(np.abs(component_img))
                ax.imshow(component_img.reshape(64,64), cmap='seismic', vmin=-v_lim, vmax=v_lim)
                ax.axis('off')
                ax.set_title(f"w_{i+1} * u_{i+1}", fontsize=8)
                st.pyplot(fig)
        st.caption("(These layers are stacked together...)")

        # STEP 4: FINAL RECONSTRUCTION
        st.markdown("---")
        st.subheader("Step 4: Final Addition ($\Gamma_{rec}$)")
        st.markdown("Finally, we add the **Average Face** back to our weighted sum to get the recognizable human face.")
        st.latex(r"\Gamma_{rec} = \Phi_{rec} + \Psi")
        
        final_reconstruction = reconstructed_centered + mean_face
        error_map = original_face - final_reconstruction
        
        col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([1,0.2,1,0.2,1])
        with col_f1:
            st.caption("Weighted Sum ($\Phi_{rec}$)")
            fig, ax = plt.subplots(figsize=(2,2))
            vmax_r = np.max(np.abs(reconstructed_centered))
            ax.imshow(reconstructed_centered.reshape(64,64), cmap='seismic', vmin=-vmax_r, vmax=vmax_r)
            ax.axis('off')
            st.pyplot(fig)
        with col_f2: st.markdown("## +")
        with col_f3:
            st.caption("Average ($\Psi$)")
            fig, ax = plt.subplots(figsize=(2,2))
            plot_face(mean_face, "", ax)
            st.pyplot(fig)
        with col_f4: st.markdown("## =")
        with col_f5:
            st.caption(f"Result ($K={n_components}$)")
            fig, ax = plt.subplots(figsize=(2,2))
            plot_face(final_reconstruction, "Result", ax)
            st.pyplot(fig)
            
        st.warning("Note: If the result is blurry, it's because we only used a few components (Low K). Increase the slider to improve it!")

    # =======================
    # TAB 3: LIVE RECOGNITION
    # =======================
    with tab3:
        st.header("Live Face Recognition: The Two-Path Algorithm")
        st.markdown("""
        The Eigenfaces recognition system uses **two sequential checks** to classify an input image.
        This demonstrates the full algorithm as described by Turk & Pentland.
        """)
        
        # Info button explaining the data split
        with st.expander("info: How does Live Recognition work with the dataset?"):
            st.markdown("""
            #### Dataset Breakdown
            
            **Total Dataset:** 400 Images
            
            **Total People (Classes):** 40 People (labeled ID 0 to ID 39)
            
            **Images per Person:** 10 images each
            
            ---
            
            #### How the Code Splits the Data
            
            The code uses `train_test_split(test_size=0.25)`, which divides the data as follows:
            
            **The Training Set (M=300):**
            - 300 Images are used to calculate the Mean Face (Ψ), the Covariance Matrix, and the Eigenfaces.
            - The algorithm "studies" these 300 photos to learn what faces look like.
            
            **The Test Set (100 Images):**
            - 100 Images are hidden away.
            - These are used in the **"Live Recognition"** tab.
            - These represent "new" photos that the system has never seen before, but they belong to the same 40 people.
            
            ---
            
            #### Why Use a Test Set?
            
            The test set simulates **real-world recognition** scenarios:
            - The system has learned from the training images
            - When you click "Pick Random Test Face", you're testing if it can recognize a person from a photo it hasn't seen during training
            - This demonstrates the algorithm's ability to **generalize** to new images of known people
            """)
        
        
        col_btn, col_info = st.columns([1, 4])
        
        # Initialize session state for random index so it doesn't jump on slider change
        if 'random_test_idx' not in st.session_state:
            st.session_state.random_test_idx = 0
            
        def pick_random():
            st.session_state.random_test_idx = np.random.randint(0, len(X_test))
            
        if col_btn.button("Pick Random Test Face"):
            pick_random()
            
        test_idx = st.session_state.random_test_idx
        
        # --- Get input test face ---
        input_face = X_test[test_idx]
        actual_person = y_test[test_idx]
        
        # --- Step 1: Project into Face Space ---
        input_phi = input_face - mean_face
        omega_input = np.dot(input_phi, eigenfaces.T)
        
        # --- Step 2: Reconstruct the face ---
        phi_rec = np.dot(omega_input, eigenfaces)
        input_reconstructed = phi_rec + mean_face
        
        # Show the input face
        st.markdown("---")
        st.subheader("Input Image")
        col_in1, col_in2, col_in3 = st.columns([1, 1, 1])
        with col_in2:
            fig_input, ax_input = plt.subplots(figsize=(3, 3))
            plot_face(input_face, f"Test Image (Person {actual_person})", ax_input)
            st.pyplot(fig_input)
        
        # ========================================
        # PATH 1: SANITY CHECK (Face Detection)
        # ========================================
        st.markdown("---")
        st.subheader("Path 1: The 'Sanity Check' (Face Detection)")
        
        st.markdown(r"""
        **Goal:** Determine if the input image is a valid face.
        
        **The Logic:** Eigenfaces are trained *only* on faces. They cannot accurately reconstruct 
        a shoe, a wall, or a hand. If you feed them a non-face, the reconstruction will look like 
        a weird, blurry face-hybrid.
        """)
        
        st.latex(r"\epsilon = ||\Phi - \Phi_{rec}||")
        
        st.markdown(r"""
        - $\Phi$: The centered input image
        - $\Phi_{rec}$: The reconstructed centered image
        - $\epsilon$: The **reconstruction error** (how different they are)
        """)
        
        # Calculate reconstruction error
        reconstruction_error = np.linalg.norm(input_phi - phi_rec)
        
        # Threshold for face detection (tuned for Olivetti)
        face_threshold = 5.0
        is_face = reconstruction_error < face_threshold
        
        # Display Path 1 results
        col_p1_1, col_p1_2, col_p1_3 = st.columns(3)
        
        with col_p1_1:
            st.caption("Original (Centered)")
            fig_p1a, ax_p1a = plt.subplots(figsize=(2.5, 2.5))
            vmax = np.max(np.abs(input_phi))
            ax_p1a.imshow(input_phi.reshape(64, 64), cmap='seismic', vmin=-vmax, vmax=vmax)
            ax_p1a.axis('off')
            ax_p1a.set_title("Phi", fontsize=10)
            st.pyplot(fig_p1a)
            
        with col_p1_2:
            st.caption("Reconstructed (Centered)")
            fig_p1b, ax_p1b = plt.subplots(figsize=(2.5, 2.5))
            ax_p1b.imshow(phi_rec.reshape(64, 64), cmap='seismic', vmin=-vmax, vmax=vmax)
            ax_p1b.axis('off')
            ax_p1b.set_title("Phi_rec", fontsize=10)
            st.pyplot(fig_p1b)
            
        with col_p1_3:
            st.caption("Difference (Error)")
            fig_p1c, ax_p1c = plt.subplots(figsize=(2.5, 2.5))
            error_map = input_phi - phi_rec
            vmax_err = np.max(np.abs(error_map))
            ax_p1c.imshow(error_map.reshape(64, 64), cmap='seismic', vmin=-vmax_err, vmax=vmax_err)
            ax_p1c.axis('off')
            ax_p1c.set_title("Phi - Phi_rec", fontsize=10)
            st.pyplot(fig_p1c)
        
        # Decision
        col_metric1, col_decision1 = st.columns([1, 2])
        with col_metric1:
            st.metric("Reconstruction Error (epsilon)", f"{reconstruction_error:.3f}")
            st.caption(f"Threshold: {face_threshold}")
        with col_decision1:
            if is_face:
                st.success("**PASS:** This IS a face. Proceed to Path 2.")
            else:
                st.error("**FAIL:** This is NOT a valid face. Recognition aborted.")
        
        # ========================================
        # PATH 2: IDENTITY CHECK (Classification)
        # ========================================
        st.markdown("---")
        st.subheader("Path 2: The 'Identity Check' (Classification)")
        
        st.markdown(r"""
        **Goal:** Determine *who* the person is.
        
        **Data Used:** The input weights ($\Omega_{new}$) vs. the database weights ($\Omega_{known}$).
        
        **The Logic:** Every person in the database has a stored list of weights (their "Face Print"). 
        We find which stored weights are closest to the new input.
        """)
        
        st.latex(r"\epsilon_k = ||\Omega_{new} - \Omega_k||")
        
        st.markdown(r"""
        - $\Omega_{new}$: The weight vector of the input face
        - $\Omega_k$: The weight vector of person $k$ in the database
        - We find the person with the **smallest** $\epsilon_k$
        """)
        
        # Calculate distances to all training faces
        dists = np.linalg.norm(weights_train - omega_input, axis=1)
        best_match_idx = np.argmin(dists)
        min_distance = dists[best_match_idx]
        
        # Threshold for known vs unknown
        identity_threshold = 25.0
        is_known = min_distance < identity_threshold
        
        predicted_person = y_train[best_match_idx]
        
        # Display Path 2 results
        col_p2_1, col_p2_2 = st.columns(2)
        
        with col_p2_1:
            st.markdown("**Input Face**")
            fig_p2a, ax_p2a = plt.subplots(figsize=(3, 3))
            plot_face(input_face, f"Actual: Person {actual_person}", ax_p2a)
            st.pyplot(fig_p2a)
            
        with col_p2_2:
            st.markdown("**Closest Match in Database**")
            fig_p2b, ax_p2b = plt.subplots(figsize=(3, 3))
            plot_face(X_train[best_match_idx], f"Predicted: Person {predicted_person}", ax_p2b)
            st.pyplot(fig_p2b)
        
        # Decision
        col_metric2, col_decision2 = st.columns([1, 2])
        with col_metric2:
            st.metric("Identity Distance (epsilon_k)", f"{min_distance:.2f}")
            st.caption(f"Threshold: {identity_threshold}")
        with col_decision2:
            if is_known:
                if predicted_person == actual_person:
                    st.success(f"**IDENTIFIED:** Person {predicted_person} (Correct!)")
                else:
                    st.warning(f"**IDENTIFIED:** Person {predicted_person} (Wrong - actually Person {actual_person})")
            else:
                st.error("**UNKNOWN PERSON (Intruder):** Face detected, but no match in database.")
        
        # ========================================
        # SUMMARY
        # ========================================
        st.markdown("---")
        st.subheader("Algorithm Summary")
        
        # Create summary table
        summary_data = {
            "Path": ["Path 1: Face Detection", "Path 2: Identity Check"],
            "What it Checks": ["Is this a face?", "Who is it?"],
            "Data Compared": ["Phi vs Phi_rec (pixels)", "Omega_new vs Omega_k (weights)"],
            "Metric": [f"epsilon = {reconstruction_error:.3f}", f"epsilon_k = {min_distance:.2f}"],
            "Result": ["PASS (Face)" if is_face else "FAIL (Not a face)", 
                      f"Person {predicted_person}" if is_known else "Unknown (Intruder)"]
        }
        st.table(summary_data)
        
        # Weight vector comparison
        with st.expander("See the Weight Vectors (Face Prints)"):
            st.write(f"**Input Face Weights (first 10):** {omega_input[:10].round(2)}")
            st.write(f"**Best Match Weights (first 10):** {weights_train[best_match_idx][:10].round(2)}")
            
            st.write("**Visual Comparison:**")
            chart_data = {"Index": range(len(omega_input)), "Input (Omega_new)": omega_input, "Match (Omega_k)": weights_train[best_match_idx]}
            st.line_chart(chart_data, x="Index", y=["Input (Omega_new)", "Match (Omega_k)"])
            st.caption("If the two lines overlap closely, the faces are similar.")
        


    # =======================
    # TAB 4: WEBCAM RECOGNITION
    # =======================
    with tab4:
        st.header("Webcam Face Recognition")
        st.markdown("""
        **Try it yourself!** Take a photo with your webcam and see the Eigenfaces two-path algorithm in action.
        
        The system will:
        1. **Detect** if the image contains a valid face (Path 1: Sanity Check)
        2. **Identify** who it is or flag as "New Face" (Path 2: Identity Check)
        """)
        
        # Threshold controls in sidebar or main area
        st.markdown("### Threshold Settings")
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            face_threshold_webcam = st.slider(
                "Face Detection Threshold (epsilon)",
                min_value=1.0,
                max_value=20.0,
                value=8.0,  # Recommended for webcam
                step=0.5,
                help="Lower = stricter face detection. Webcam images typically have error 4-10."
            )
            st.caption("Recommended: 8.0 | Webcam typical: 4-10")
            
        with col_t2:
            identity_threshold_webcam = st.slider(
                "Identity Threshold (epsilon_k)",
                min_value=5.0,
                max_value=50.0,
                value=25.0,  # Recommended
                step=1.0,
                help="Lower = stricter identity matching. Recommended: 25.0"
            )
            st.caption("Recommended: 25.0 | Range: [5, 50]")
        
        st.markdown("---")
        
        # Camera input
        camera_photo = st.camera_input("Take a photo of your face")
        
        if camera_photo is not None:
            # Process the captured image
            st.subheader("Processing Pipeline")
            
            # Load image from camera
            pil_image = Image.open(camera_photo)
            img_array = np.array(pil_image)
            
            # === PREPROCESSING PIPELINE ===
            # Matching Olivetti: detect -> crop -> resize -> grayscale -> histogram equalization -> normalize
            
            # Step 1: Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Step 2: Detect face using Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            haar_face_detected = len(faces) > 0
            
            if haar_face_detected:
                # Use the largest detected face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                
                # Add padding (20%)
                padding = int(0.2 * max(w, h))
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(gray.shape[1], x + w + padding)
                y_end = min(gray.shape[0], y + h + padding)
                
                # Crop the face region
                face_crop = gray[y_start:y_end, x_start:x_end]
                
                # Make it square
                fh, fw = face_crop.shape
                if fh > fw:
                    diff = (fh - fw) // 2
                    face_crop = face_crop[diff:diff+fw, :]
                elif fw > fh:
                    diff = (fw - fh) // 2
                    face_crop = face_crop[:, diff:diff+fh]
                
                cropped = face_crop
            else:
                # Fallback: center crop
                h_img, w_img = gray.shape
                min_dim = min(h_img, w_img)
                start_h = (h_img - min_dim) // 2
                start_w = (w_img - min_dim) // 2
                cropped = gray[start_h:start_h+min_dim, start_w:start_w+min_dim]
            
            # Step 3: Resize to 64x64 (Olivetti face size)
            resized = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_AREA)
            
            # Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Better than regular histogram equalization for matching Olivetti
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_applied = clahe.apply(resized)
            
            # Step 5: Normalize to [0, 1] range
            normalized_raw = clahe_applied.astype(np.float64) / 255.0
            
            # Step 6: Match Olivetti statistics
            # Olivetti faces have specific mean/std characteristics
            olivetti_mean = np.mean(X_train)  # ~0.55 typically
            olivetti_std = np.std(X_train)    # ~0.23 typically
            
            webcam_mean = np.mean(normalized_raw)
            webcam_std = np.std(normalized_raw)
            
            # Standardize to match Olivetti distribution
            if webcam_std > 0:
                normalized = (normalized_raw - webcam_mean) / webcam_std * olivetti_std + olivetti_mean
            else:
                normalized = normalized_raw
            
            # Clip to valid range [0, 1]
            normalized = np.clip(normalized, 0, 1)
            
            # Flatten to vector
            webcam_face = normalized.flatten()
            
            # Show preprocessing pipeline
            with st.expander("View Preprocessing Pipeline", expanded=False):
                cols_pipe = st.columns(6)
                
                with cols_pipe[0]:
                    st.caption("1. Original")
                    st.image(pil_image, use_container_width=True)
                    
                with cols_pipe[1]:
                    st.caption("2. Grayscale")
                    st.image(gray, use_container_width=True)
                    
                with cols_pipe[2]:
                    st.caption("3. Face Crop")
                    st.image(cropped, use_container_width=True)
                    
                with cols_pipe[3]:
                    st.caption("4. Resize 64x64")
                    st.image(resized, use_container_width=True)
                    
                with cols_pipe[4]:
                    st.caption("5. CLAHE")
                    st.image(clahe_applied, use_container_width=True)
                    
                with cols_pipe[5]:
                    st.caption("6. Normalized")
                    # Show as grayscale image
                    fig_norm, ax_norm = plt.subplots(figsize=(2, 2))
                    ax_norm.imshow(normalized.reshape(64, 64), cmap='gray', vmin=0, vmax=1)
                    ax_norm.axis('off')
                    st.pyplot(fig_norm)
                
                if haar_face_detected:
                    st.success(f"Haar Cascade detected a face: {w}x{h} pixels")
                else:
                    st.warning("No face detected by Haar Cascade. Using center crop.")
                
                st.info(f"**Stats:** Olivetti mean={olivetti_mean:.3f}, std={olivetti_std:.3f} | Your image after matching: mean={np.mean(normalized):.3f}, std={np.std(normalized):.3f}")
            
            # ========================================
            # PATH 1: FACE DETECTION (Reconstruction Error)
            # ========================================
            st.markdown("---")
            st.subheader("Path 1: Face Detection (Sanity Check)")
            
            # Project and reconstruct
            webcam_phi = webcam_face - mean_face
            omega_webcam = np.dot(webcam_phi, eigenfaces.T)
            phi_rec_webcam = np.dot(omega_webcam, eigenfaces)
            reconstructed_webcam = phi_rec_webcam + mean_face
            
            # Calculate reconstruction error
            reconstruction_error_webcam = np.linalg.norm(webcam_phi - phi_rec_webcam)
            
            is_valid_face = reconstruction_error_webcam < face_threshold_webcam
            
            col_face1, col_face2, col_face3 = st.columns(3)
            
            with col_face1:
                st.caption("Your Image (Phi)")
                fig_f1, ax_f1 = plt.subplots(figsize=(2.5, 2.5))
                plot_face(webcam_face, "", ax_f1)
                st.pyplot(fig_f1)
                
            with col_face2:
                st.caption("Reconstructed (Phi_rec)")
                fig_f2, ax_f2 = plt.subplots(figsize=(2.5, 2.5))
                plot_face(reconstructed_webcam, "", ax_f2)
                st.pyplot(fig_f2)
                
            with col_face3:
                st.caption("Difference")
                fig_f3, ax_f3 = plt.subplots(figsize=(2.5, 2.5))
                diff_img = webcam_face - reconstructed_webcam
                vmax = np.max(np.abs(diff_img))
                ax_f3.imshow(diff_img.reshape(64, 64), cmap='seismic', vmin=-vmax, vmax=vmax)
                ax_f3.axis('off')
                st.pyplot(fig_f3)
            
            col_metric_face, col_decision_face = st.columns([1, 2])
            with col_metric_face:
                st.metric("Reconstruction Error (epsilon)", f"{reconstruction_error_webcam:.3f}")
                st.caption(f"Threshold: {face_threshold_webcam}")
            
            with col_decision_face:
                if is_valid_face:
                    st.success("**PASS:** This IS a valid face. Proceeding to identification...")
                else:
                    st.error("**FAIL:** This does NOT appear to be a valid face.")
                    st.markdown("""
                    The reconstruction error is too high. This means:
                    - The Eigenfaces cannot reconstruct your input well
                    - Your input may not be a proper face image
                    - Try: face the camera directly, improve lighting, or remove obstructions
                    """)
            
            # Only proceed to Path 2 if it's a valid face
            if is_valid_face:
                # ========================================
                # PATH 2: IDENTITY CHECK
                # ========================================
                st.markdown("---")
                st.subheader("Path 2: Identity Check")
                
                # Find Nearest Neighbor
                dists_webcam = np.linalg.norm(weights_train - omega_webcam, axis=1)
                best_match_idx_webcam = np.argmin(dists_webcam)
                min_distance_webcam = dists_webcam[best_match_idx_webcam]
                
                is_known_webcam = min_distance_webcam < identity_threshold_webcam
                predicted_person_webcam = y_train[best_match_idx_webcam]
                
                # Always show the closest match
                col_you, col_match_webcam = st.columns(2)
                
                with col_you:
                    st.markdown("**Your Face**")
                    fig_you, ax_you = plt.subplots(figsize=(3, 3))
                    plot_face(webcam_face, "You", ax_you)
                    st.pyplot(fig_you)
                    
                with col_match_webcam:
                    st.markdown("**Closest Match in Database**")
                    fig_match, ax_match = plt.subplots(figsize=(3, 3))
                    plot_face(X_train[best_match_idx_webcam], f"Person {predicted_person_webcam}", ax_match)
                    st.pyplot(fig_match)
                
                # Metrics and decision
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Identity Distance (epsilon_k)", f"{min_distance_webcam:.2f}")
                with col_m2:
                    st.metric("Closest Person ID", f"{predicted_person_webcam}")
                
                st.caption(f"Identity Threshold: {identity_threshold_webcam}")
                
                if is_known_webcam:
                    st.success(f"**IDENTIFIED:** You look like Person {predicted_person_webcam} in the database!")
                    st.markdown(f"Distance ({min_distance_webcam:.2f}) is below threshold ({identity_threshold_webcam}).")
                else:
                    st.warning("**NEW FACE DETECTED - Not in Database**")
                    st.markdown(f"""
                    Distance ({min_distance_webcam:.2f}) exceeds threshold ({identity_threshold_webcam}).
                    
                    This means your face is **not** one of the 40 people in the Olivetti dataset.
                    The closest match (Person {predicted_person_webcam}) is shown above for reference.
                    """)
            else:
                st.info("Identity check skipped because the input was not recognized as a valid face.")
        else:
            st.info("Click the camera button above to take a photo and test face recognition!")

    # =======================
    # TAB 1: COMPONENT EXPLORER
    # =======================
    with tab1:
        st.header("Component Explorer")
        st.write("Understand the 'building blocks' (Eigenfaces) and why we chose them.")
        
        # 1. Variance Explained (Scree Plot)
        st.subheader("1. Why these components?")
        st.markdown("""
        PCA orders components by how much information (Variance) they capture. 
        The first few eigenfaces are the most important; they look like generic faces. 
        Later ones describe specific details (noise).
        """)
        
        explained_variance = full_pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Plotting
        fig_var, ax_var = plt.subplots(figsize=(10, 4))
        ax_var.plot(range(1, len(explained_variance)+1), cumulative_variance, marker='.', linestyle='--')
        ax_var.axvline(x=n_components, color='r', linestyle='-', label=f'Current K={n_components}')
        ax_var.set_xlabel('Number of Components')
        ax_var.set_ylabel('Cumulative Explained Variance')
        ax_var.set_title('Explained Variance vs. Number of Components')
        ax_var.legend()
        ax_var.grid(True)
        st.pyplot(fig_var)
        
        st.info(f"With K={n_components}, you are capturing **{cumulative_variance[n_components-1]:.1%}** of the total face information.")
        
        # 2. Gallery of Eigenfaces
        st.subheader("2. Gallery of Eigenfaces")
        st.write("Browse the top components to see what features they represent.")
        
        # Pagination for gallery
        page_size = 12
        total_pages = max(1, len(full_pca.components_) // page_size)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Display grid
        grid_cols = st.columns(4)
        for i, idx in enumerate(range(start_idx, min(end_idx, len(full_pca.components_)))):
            with grid_cols[i % 4]:
                fig_g, ax_g = plt.subplots(figsize=(2,2))
                plot_face(full_pca.components_[idx], f"Eigenface {idx+1}", ax_g)
                st.pyplot(fig_g)

    # =======================
    # TAB 5: HISTORY & CONTEXT
    # =======================
    with tab5:
        st.header("History & Context")
        
        st.subheader("1. The Origin Story (1991)")
        st.markdown("""
        Before **Eigenfaces** (Turk & Pentland, 1991), computer vision mostly tried to measure geometry:
        "How far is the eye from the nose?", "How wide is the mouth?".
        
        **The Problem:** This was fragile. If you turned your head slightly, the measurements failed.
        
        **The Revolution:** Eigenfaces took a different approach. Instead of measuring specific parts, it treated the **whole face image as a single pattern**.
        It asked: *"What are the main ingredients that make a face a face?"*
        """)
        
        st.subheader("2. Where was it used?")
        st.markdown("""
        In the 1990s and early 2000s, this was the state-of-the-art for:
        - **Security Access:** Controlled lighting entry points.
        - **CCTV Analysis:** (Though it struggled with lighting changes).
        - **Organization:** Sorting digital photo albums.
        """)
        
        st.subheader("3. Why Linear Algebra?")
        st.markdown("""
        Computers back then were slow. 
        - Eigenfaces uses **Linear Algebra** (Matrix Multiplication), which is extremely fast.
        - It compressed huge images (4096 pixels) into tiny vectors (e.g., just 20 numbers).
        - This made it possible to search databases of thousands of people in milliseconds.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("**Pros**\n\n- Very Fast\n- Simple Math\n- No 'Black Box' (Interpretable)")
        with col2:
            st.error("**Cons**\n\n- Fails with Lighting Changes\n- Fails with Rotations\n- Needs Centered Faces")

    # =======================
    # TAB 6: FUTURE & MODERN
    # =======================
    with tab6:
        st.header("Modern AI & The Future")
        
        st.subheader("What do we use now? (Deep Learning)")
        st.write("Eigenfaces is a 'Holistic' method (looks at the whole face). Modern AI is 'Feature-based'.")
        
        st.markdown("""
        **Convolutional Neural Networks (CNNs)** changed everything around 2012-2014 (e.g., FaceNet, ArcFace, DeepFace).
        
        | Feature | Eigenfaces (PCA) | Deep Learning (CNNs) |
        | :--- | :--- | :--- |
        | **Logic** | "Ghosts" & Shadows | Edges, Textures, Shapes |
        | **Lighting** | Fails easily | Robust to lighting |
        | **Pose** | Must be frontal | Works from angles |
        | **Data** | Small (100s) | Massive (Millions) |
        """)
        
        st.info("Today, when you unlock your phone with FaceID, it uses a 3D version of Deep Learning, not Eigenfaces.")
        
        st.subheader("How does Deep Learning actually work?")
        st.markdown("""
        Unlike Eigenfaces, which tries to match the **whole face at once**, Deep Learning builds understanding in **layers**, like Lego bricks.
        
        1.  **Layer 1 (The Edges):** The network looks at pixels and finds simple lines, curves, and edges ($|$ $-$ $/$).
        2.  **Layer 2 (The Parts):** It combines those edges to find simple shapes: an eye, a nose, a curve of a lip.
        3.  **Layer 3 (The Face):** It combines parts to form a full face structure.
        4.  **Layer 4 (The Identity):** It decides "This is David".
        
        *Analogy:* Eigenfaces is like matching a blurry photo to another blurry photo. Deep Learning is like reading a description: "Has a scar on left cheek, wide nose, green eyes."
        """)
        



if __name__ == "__main__":
    main()