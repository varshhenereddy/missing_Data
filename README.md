# Missing Data Image Inpainting and Reconstruction using a Probabilistic Approach to Missing Pixels

This project compares OpenCV's Telea and Navier-Stokes inpainting algorithms
with a probabilistic patch-prior method. The latter fits a full-covariance
Gaussian mixture model (GMM) to complete local patches using expectation
maximization (EM), then approximates the posterior over missing pixels with
damped mean-field updates. Image values are normalized to `[0, 1]` during
probabilistic inference.

## Setup

From the repository root, create and activate a virtual environment, then
install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.10 or newer is required.

In VS Code, select the `.venv` Python interpreter. `argparse` is included with
Python and does not need a separate package.

## Run

Pass an image path to `main.py`. The default creates a rectangular mask and
opens a side-by-side comparison:

```powershell
python main.py --image .\path\to\image.jpg
```

Available mask types are `block`, `noise`, and `irregular`. Save a visualization
without opening a window with:

```powershell
python main.py --image .\path\to\image.jpg --mask irregular --amount 0.25 --seed 7 --save-figure .\comparison.png --no-display
```

The CLI prints whole-image PSNR, SSIM, MAE, and execution time for each
inpainting method. Use `--components`, `--patch-size`, `--iterations`, and
`--radius` to configure the probabilistic model and OpenCV baselines. Images
must be readable by OpenCV; they are interpreted as RGB after loading and
normalized to `[0, 1]`.

## Local browser app

Launch the browser interface from the repository root:

```powershell
python -m streamlit run web_app.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501), upload an image, adjust
the mask and model settings in the sidebar, and select **Run inpainting
comparison**. The uploaded image is processed by the local Python process and
is not sent to an external service. Stop the server with `Ctrl+C` in the
terminal.

## Tests

Run the focused checks from the repository root:

```powershell
python -m unittest discover -s tests
```

## Optional CIFAR-10 sample

The helper downloads CIFAR-10 using `torchvision` and saves one image for the
CLI:

```powershell
python .\datasets\download_sample.py
python main.py --image .\datasets\sample.png --mask noise --amount 0.2
```
