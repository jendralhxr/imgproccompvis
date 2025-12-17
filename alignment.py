import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise IOError("Image not found")
    return img

# 2. Preprocess: grayscale + edge detection
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return gray, edges

# 3. Detect lines using Hough Transform
def detect_hough_lines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return []
    return lines[:, 0, :]

# 4. Separate lines into vertical and horizontal
def split_lines(lines):
    horizontal, vertical = [], []
    for rho, theta in lines:
        if abs(theta) < np.pi / 4 or abs(theta - np.pi) < np.pi / 4:
            vertical.append((rho, theta))
        elif abs(theta - np.pi / 2) < np.pi / 4:
            horizontal.append((rho, theta))
    return horizontal, vertical

# 5. Harris Corner Detection
def detect_harris_corners(gray):
    gray_f = np.float32(gray)
    harris = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    return harris

# 6. Extract corner points near line intersections
def get_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    return int(x0), int(y0)

# 7. Estimate four document corners
def estimate_corners(horizontal, vertical):
    corners = []
    for h in horizontal:
        for v in vertical:
            corners.append(get_intersection(h, v))
    return np.array(corners)

# 8. Order corners: top-left, top-right, bottom-right, bottom-left
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# 9. Perspective transform
def warp_perspective(img, rect):
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

# ------------------- Main pipeline -------------------
if __name__ == "__main__":
    img = load_image("taper2.jpg")
    gray, edges = preprocess(img)

    lines = detect_hough_lines(edges)
    horizontal, vertical = split_lines(lines)

    harris = detect_harris_corners(gray)

    corners = estimate_corners(horizontal, vertical)
    rect = order_corners(corners)

    warped = warp_perspective(img, rect)

    import matplotlib.pyplot as plt

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Edges")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Warped (Recovered)")
    plt.imshow(warped_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
