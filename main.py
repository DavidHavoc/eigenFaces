import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    st.title("Eigenfaces: Face Recognition using PCA")
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1. The Face Space", "2. Reconstruction", "3. Live Recognition", "4. Explorer", "5. History & Context", "6. Future & Modern"])

    # =======================
    # TAB 1: FACE SPACE
    # =======================
    with tab1:
        st.header("The 'Face Space'")
        st.markdown(f"**Average Face ($\Psi$):** The center of our coordinate system.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_mean, ax_mean = plt.subplots(figsize=(3, 3))
            plot_face(mean_face, "Average Face", ax_mean)
            st.pyplot(fig_mean)
            
        with col2:
            st.markdown(f"**Top {n_components} Eigenfaces:** The directions of maximum variance.")
            st.info("These 'ghostly' faces capture features like lighting, nose length, and eye position.")
            
            # Plot top 5 eigenfaces (or fewer if n_components < 5)
            num_to_plot = min(5, n_components)
            fig_eigen, axes_eigen = plt.subplots(1, num_to_plot, figsize=(10, 3))
            if num_to_plot == 1:
                axes_eigen = [axes_eigen] # Make iterable
                
            for i in range(num_to_plot):
                plot_face(eigenfaces[i], f"Eigenface {i+1}", axes_eigen[i])
            
            st.pyplot(fig_eigen)

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
        st.header("Live Face Recognition")
        st.write("We use the current Face Space (defined by the slider) to identify a new face from the Test Set.")
        
        col_btn, col_info = st.columns([1, 4])
        
        # Initialize session state for random index so it doesn't jump on slider change
        if 'random_test_idx' not in st.session_state:
            st.session_state.random_test_idx = 0
            
        def pick_random():
            st.session_state.random_test_idx = np.random.randint(0, len(X_test))
            
        if col_btn.button("Pick Random Test Face"):
            pick_random()
            
        test_idx = st.session_state.random_test_idx
        
        # --- Recognition Algorithm (Nearest Neighbor in Face Space) ---
        
        # 1. Get input test face
        input_face = X_test[test_idx]
        
        # 2. Project Input Face into Face Space
        # Calculate weights Omega_input
        input_phi = input_face - mean_face
        omega_input = np.dot(input_phi, eigenfaces.T)
        
        # 3. Find Nearest Neighbor in Training Set
        # Calculate Euclidean distance between Omega_input and all Omega_train
        # dists = sqrt(sum((Omega_input - Omega_train)^2))
        dists = np.linalg.norm(weights_train - omega_input, axis=1)
        
        best_match_idx = np.argmin(dists)
        min_distance = dists[best_match_idx]
        
        # 4. Thresholding (Unknown vs Known)
        # Heuristic: Threshold is dynamic based on the max distance seen in training
        # Or a fixed value suitable for Olivetti (approx 20-30 for whitened, higher for raw)
        # We'll use a relative threshold: 1.5 * average training distance
        threshold = 25.0 # Tuned for whitened PCA on Olivetti
        
        is_known = min_distance < threshold
        
        predicted_person = y_train[best_match_idx]
        actual_person = y_test[test_idx]
        
        # --- Display Results ---
        match_color = "green" if (predicted_person == actual_person) else "red"
        
        col_input, col_match = st.columns(2)
        
        with col_input:
            st.subheader("Input Face (Unknown)")
            fig_in, ax_in = plt.subplots()
            plot_face(input_face, f"Person ID: {actual_person}", ax_in)
            st.pyplot(fig_in)
            
        with col_match:
            st.subheader("Best Match")
            match_status = "✅ KNOWN" if is_known else "❌ UNKNOWN (Too far)"
            
            fig_out, ax_out = plt.subplots()
            plot_face(X_train[best_match_idx], f"Predicted: {predicted_person}", ax_out)
            st.pyplot(fig_out)
            
            st.metric("Euclidean Distance", f"{min_distance:.2f}")
            
            if is_known:
                st.success(f"Status: {match_status}")
                if predicted_person == actual_person:
                    st.markdown(f"<span style='color:green'>Correctly Identified!</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:red'>Incorrect Match.</span>", unsafe_allow_html=True)
            else:
                st.error(f"Status: {match_status}")
                st.caption("The distance is too high. The face is considered 'Unknown'.")

        st.info(f"""
        **Algorithm Details:**
        1. Projected Test Face onto {n_components} Eigenfaces.
        2. Compared against {len(X_train)} training faces using Euclidean distance.
        3. Threshold for 'Unknown': {threshold}.
        
        *Tip: Lower the slider to reduce components. Recognition accuracy drops as distinct features are lost.*
        """)
        
        # --- Educational: Euclidean Distance ---
        st.markdown("---") 
        st.subheader("How the Match was Found (The Distance Metric)")
        st.markdown(r"""
        We found the match by calculating the **Euclidean Distance** between the weights of the input face and the training faces.
        
        $$ d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^n (q_i - p_i)^2} $$
        
        Where $\mathbf{p}$ is the weight vector of the Input Face and $\mathbf{q}$ is the weight vector of a Training Face.
        """)
        
        with st.expander("See the actual numbers (Vectors)"):
            st.write(f"**Input Face Weights (Top 5):** {omega_input[:5].round(2)}")
            st.write(f"**Best Match Weights (Top 5):** {weights_train[best_match_idx][:5].round(2)}")
            st.caption("The computer compares these lists of numbers. A smaller total difference means the faces are more similar.")
            
            # Simple vector plot
            st.write("**Visualizing the Distance:**")
            chart_data = {"Index": range(len(omega_input)), "Input": omega_input, "Match": weights_train[best_match_idx]}
            st.line_chart(chart_data, x="Index", y=["Input", "Match"])
            st.caption("If the two lines overlap closely, the distance is small (Good Match).")

    # =======================
    # TAB 4: COMPONENT EXPLORER
    # =======================
    with tab4:
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