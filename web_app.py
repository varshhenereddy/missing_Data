"""Streamlit interface for running the inpainting comparison locally."""

from __future__ import annotations

import time

import cv2
import numpy as np
import streamlit as st

from src.classical_baseline import inpaint_method
from src.metrics import evaluate
from src.probabilistic_inpainter import ProbabilisticInpainter
from src.synthetic_corruptor import apply_mask, generate_mask


st.set_page_config(
    page_title="Probabilistic Image Inpainting",
    layout="wide",
)
st.title("Missing Data Image Inpainting")
st.caption(
    "Compare OpenCV baselines with probabilistic patch-GMM reconstruction. "
    "Uploaded images are processed locally by this app."
)

with st.sidebar:
    st.header("Inpainting settings")
    mask_type = st.selectbox("Missing-pixel mask", ("block", "noise", "irregular"))
    amount = st.slider("Mask amount", min_value=0.01, max_value=0.8, value=0.25)
    seed = st.number_input("Random seed", min_value=0, value=0, step=1)
    components = st.slider("GMM components", min_value=1, max_value=8, value=4)
    patch_size = st.selectbox("Patch size", (3, 5), index=0)
    iterations = st.slider("Inference iterations", min_value=1, max_value=20, value=8)
    radius = st.slider("OpenCV radius", min_value=1.0, max_value=10.0, value=3.0)

uploaded_file = st.file_uploader(
    "Choose an image", type=("png", "jpg", "jpeg", "bmp", "webp")
)

if uploaded_file is not None:
    if uploaded_file.size > 20 * 1024 * 1024:
        st.error("Please upload an image smaller than 20 MB.")
    else:
        encoded_image = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("The selected file could not be decoded as an image.")
        elif min(image_bgr.shape[:2]) < 3:
            st.error("The image must be at least 3 pixels tall and wide.")
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_rgb /= 255.0
            st.image(image_rgb, caption=f"Original: {uploaded_file.name}", width=450)

            if st.button("Run inpainting comparison", type="primary"):
                with st.spinner("Generating masks and reconstructing the image..."):
                    mask = generate_mask(
                        image_rgb.shape,
                        kind=mask_type,
                        amount=amount,
                        seed=int(seed),
                    )
                    corrupted = apply_mask(image_rgb, mask).astype(np.float32)
                    outputs: dict[str, np.ndarray] = {}
                    timings: dict[str, float] = {}

                    for method_name in ("Telea", "Navier-Stokes"):
                        start = time.perf_counter()
                        outputs[method_name] = inpaint_method(
                            corrupted, mask, method_name, radius=radius
                        )
                        timings[method_name] = time.perf_counter() - start

                    inpainter = ProbabilisticInpainter(
                        patch_size=patch_size,
                        components=components,
                        inference_iterations=iterations,
                        seed=int(seed),
                    )
                    start = time.perf_counter()
                    outputs["Probabilistic GMM"] = inpainter.inpaint(corrupted, mask)
                    timings["Probabilistic GMM"] = time.perf_counter() - start

                st.subheader("Visual comparison")
                columns = st.columns(5)
                comparison = [
                    ("Original", image_rgb),
                    ("Corrupted", corrupted),
                    ("Telea", outputs["Telea"]),
                    ("Navier-Stokes", outputs["Navier-Stokes"]),
                    ("Probabilistic GMM", outputs["Probabilistic GMM"]),
                ]
                for column, (name, image) in zip(columns, comparison):
                    if name == "Corrupted":
                        image = image.copy()
                        image[mask] = (1.0, 0.0, 0.0)
                    column.image(image, caption=name)

                st.subheader("Performance report")
                rows = []
                for name, result in outputs.items():
                    scores = evaluate(image_rgb, result)
                    rows.append(
                        {
                            "Method": name,
                            "PSNR (dB)": f"{scores['PSNR']:.3f}",
                            "SSIM": f"{scores['SSIM']:.4f}",
                            "MAE": f"{scores['MAE']:.5f}",
                            "Time (s)": f"{timings[name]:.3f}",
                        }
                    )
                st.dataframe(rows, hide_index=True)
                st.caption("Metrics are calculated over the complete image.")
else:
    st.info("Upload a local image to configure and run the inpainting comparison.")
