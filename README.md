# Image Alignment Using ORB Feature Matching and Homography

## Aim
To align a query image with a reference image by detecting and matching ORB features, estimating a homography matrix using RANSAC, and applying a perspective transformation.

## Overview
This notebook implements an image alignment workflow using OpenCV.

The notebook performs these steps:

1. Loads a query image and a reference (target) image.
2. Checks that both images were loaded successfully.
3. Converts the images to RGB for display.
4. Converts both images to grayscale for feature detection.
5. Detects keypoints and descriptors using the ORB feature detector.
6. Matches ORB descriptors using a Brute-Force Matcher with Hamming distance.
7. Sorts the matches and selects the best 15%.
8. Displays the selected raw feature matches before RANSAC.
9. Extracts the corresponding keypoint coordinates from both images.
10. Estimates a homography matrix using RANSAC with a 5.0 reprojection threshold.
11. Prints the number of selected matches, RANSAC inliers, and the homography matrix.
12. Warps the query image to align it with the reference image and displays both images.

## Technologies Used
- Python
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab

## Project Structure
```text
CV-Lab-12/
├── CVLAB12.ipynb
├── images013.jpg       # Query image (add this file)
├── images016.jpg       # Reference image (add this file)
└── README.md
```

## Requirements
Install the required Python packages:

```bash
pip install opencv-python numpy matplotlib
```

## How to Run

### Option 1: Jupyter Notebook

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Install the dependencies:
   ```bash
   pip install opencv-python numpy matplotlib
   ```
4. Place `images013.jpg` and `images016.jpg` in the project folder.
5. Open the notebook:
   ```bash
   jupyter notebook CVLAB12.ipynb
   ```
6. Update the image paths in the first code cell as shown below.
7. Run all cells.

### Option 2: Google Colab

1. Upload `CVLAB12.ipynb` to Google Colab.
2. Upload both images (`images013.jpg` and `images016.jpg`) to your Google Drive under the `COMPUTER VISION` folder.
3. Mount Google Drive if it is not already mounted:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Run all cells.

## Image Paths

The notebook currently reads the images from these Google Drive paths:

```python
img1 = cv2.imread('/content/drive/MyDrive/COMPUTER VISION/images013.jpg')
img2 = cv2.imread('/content/drive/MyDrive/COMPUTER VISION/images016.jpg')
```

If both images are stored in the same folder as the notebook, change the paths to:

```python
img1 = cv2.imread('images013.jpg')
img2 = cv2.imread('images016.jpg')
```

- `img1`: Query image to be aligned.
- `img2`: Reference or target image.

## Main Techniques

### 1. ORB Feature Detection

```python
orb = cv2.ORB_create(nfeatures=2000)
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
```

ORB detects image keypoints and computes binary descriptors. The notebook sets the maximum number of features to 2,000.

### 2. Brute-Force Feature Matching

```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
```

The Brute-Force Matcher compares ORB descriptors using Hamming distance. Matches are sorted by distance, and the notebook retains the best 15%.

### 3. Homography Estimation with RANSAC

```python
h_matrix, mask = cv2.findHomography(
    points1,
    points2,
    cv2.RANSAC,
    5.0
)
```

Homography estimates the projective transformation between corresponding points in the two images. RANSAC helps identify inlier matches and reduce the effect of incorrect matches.

### 4. Image Alignment

```python
height, width = img2.shape[:2]
aligned_img = cv2.warpPerspective(img1, h_matrix, (width, height))
```

The estimated homography is applied to the query image. The output is generated using the reference image's width and height.

## Expected Output
The notebook produces:

1. **Feature Matching Visualization** – the selected top feature matches between the query and reference images before RANSAC.
2. **Console Results** – total selected matches, RANSAC inlier count, and the estimated homography matrix.
3. **Image Alignment Comparison** – the reference image and the aligned query image displayed side by side.

## Troubleshooting

- **Image not found:** Check the image filenames and paths. Both images must be available.
- **Not enough features detected:** Try images with more visible texture or distinctive features.
- **Homography calculation failed:** Use images with sufficient overlapping content and reliable feature matches.
- **At least 4 feature matches are required:** Homography estimation requires at least four corresponding point pairs.

## Notes
- The notebook uses ORB with `nfeatures=2000`.
- It retains the best 15% of sorted feature matches, with a minimum of four.
- RANSAC uses a reprojection threshold of `5.0`.
- The input images are not embedded in the notebook. Add both images to the repository so others can reproduce the experiment.

## Author
Velaga Manikanta Sai.

241FA23016.

III-Data Science.

