import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# load the image
# ----------------------------
img = plt.imread("apple.jpg")  # change to your image file

if img.dtype == np.float32 or img.dtype == np.float64:
    img = (img * 255).astype(np.uint8)

# Drop alpha channel if present
if img.ndim == 3 and img.shape[2] == 4:
    img = img[:, :, :3]

h, w = img.shape[:2]

def affine_transform_forward(img, A, b):
    """
    Perform affine transformation (forward mapping) using NumPy.
    - img: input image as NumPy array (H×W or H×W×C)
    - A: 2×2 affine matrix
    - b: 2×1 translation vector
    Returns transformed image with automatically determined size.
    """
    h, w = img.shape[:2]

    # transformed canvas to determine output size
    corners = np.array([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])
    corners_prime = (corners @ A.T) + b

    x_min, y_min = np.floor(corners_prime.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(corners_prime.max(axis=0)).astype(int)

    out_w = x_max - x_min
    out_h = y_max - y_min

    out_shape = (out_h, out_w, img.shape[2]) if img.ndim == 3 else (out_h, out_w)
    out = np.zeros(out_shape, dtype=np.uint8)

    # loop over input pixels
    for y_in in range(h):
        for x_in in range(w):
            x_prime, y_prime = A @ np.array([x_in, y_in]) + b
            x_prime_i = int(round(x_prime - x_min))
            y_prime_i = int(round(y_prime - y_min))

            if 0 <= x_prime_i < out_w and 0 <= y_prime_i < out_h:
                out[y_prime_i, x_prime_i] = img[y_in, x_in]
            
    for y_prime_i in range(out_h):
        for x_prime_i in range(out_w):
            if out[y_prime_i, x_prime_i]== 0: # means it is empty
                # here we do some interpolation
                ... (complete this)
                ...
                
        

    return out

# ----------------------------
#  >> Define affine parameters
# ----------------------------
angle = np.deg2rad(45)
scale = 1
tx, ty = 20, 0
shear_coeff_x= 0
shear_coeff_y= 0
flip_x = 1 # eiter 1 or -1
flip_y = 1 # eiter 1 or -1

A = scale * np.array([
    [np.cos(angle)*flip_x, -np.sin(angle)+shear_coeff_x],
    [np.sin(angle)+shear_coeff_y,  np.cos(angle)*flip_y]
])
b = np.array([tx, ty])

out = affine_transform_forward(img, A, b)

# ----------------------------
# plot the output side by side
# ----------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img)
ax[0].set_title("Original image")
ax[1].imshow(out)
ax[1].set_title("Affine-transformed")


