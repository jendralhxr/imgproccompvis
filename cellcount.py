from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import os

def save_binary_image(binary, path):
    img = (binary * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)

# Load image
img = Image.open("generated_images.jpg").convert("RGB")
w, h = img.size

# ---- circular mask ----
cx, cy = w // 2-20, h // 2-70
radius = min(cx, cy) - 50   # adjust offset here

mask = Image.new("L", (w, h), 0)
draw = ImageDraw.Draw(mask)
draw.ellipse(
    (cx - radius, cy - radius, cx + radius, cy + radius),
    fill=255
)

# ---- green overlay ----
img_np = np.array(img)
mask_np = np.array(mask) > 0   # boolean mask
masked_img = img_np * mask_np[..., None]

# --- Override green channel inside mask ---
img_np[mask_np, 1] = 150   # G channel only

# Convert back to PIL
overlay = Image.fromarray(img_np, mode="RGB")

# ---- Display ----
plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title("Mask Verification (PIL, Green Overlay)")
plt.axis("off")
plt.show()

label_x0, label_y0 = 380, 300      # top-left
label_x1, label_y1 = 650, 440   # bottom-right

mask_label = np.ones((h, w), dtype=np.uint8)
mask_label[label_y0:label_y1, label_x0:label_x1] = 0

masked_img = masked_img * mask_label[..., None]
plt.imshow(masked_img)

##--- thresholding
gray = masked_img[:, :, 0]   # R channel, biological cells usually have tinge of red/yellow for animal-like cells and blue/green for plant-like cells
plt.imshow(gray, cmap='gray')


# otsu
hist, _ = np.histogram(
    gray, bins=256, range=(1, 255) # 0 bears no information
)

hist = hist.astype(float)
hist /= hist.sum()

omega = np.cumsum(hist)
mu = np.cumsum(hist * np.arange(256))
mu_t = mu[-1]

sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1 - omega) + 1e-12)
otsu_thresh = np.argmax(sigma_b2)

otsu = np.zeros_like(gray, dtype=np.uint8)
otsu [(gray >= otsu_thresh)] = 1
plt.imshow(otsu, cmap='gray')
plt.title(f"Otsu threshold, T = {otsu_thresh}")

# triangle threshold
peak = np.argmax(hist)

# Find last non-zero bin (end of tail)
end = np.max(np.nonzero(hist))

# Construct line between peak and end
x = np.arange(peak, end + 1)
y = hist[peak:end + 1]

line = np.linspace(hist[peak], hist[end], len(x))

# Maximum distance from line
triangle_thresh = x[np.argmax(np.abs(y - line))]

tri = np.zeros_like(gray, dtype=np.uint8)
tri [(gray >= triangle_thresh)] = 1
plt.imshow(tri, cmap='gray')
plt.title(f"Triangle threshold, T = {triangle_thresh}")

# difference
diff = otsu ^ tri 
h, w = otsu.shape
vis = np.zeros((h, w, 3), dtype=np.uint8)

# Optional: show union in gray/white
union = (otsu | tri)
vis[union == 1] = [180, 180, 180]

# Highlight differences in green
vis[diff == 1] = [0, 255, 0]
plt.imshow(vis)
plt.title("Difference: Otsu vs Triangle")
plt.axis("off")

##--- erosion, dilation
def pad_binary(img):
    return np.pad(img, 1, mode="constant", constant_values=0)

def binary_erosion(img):
    padded = pad_binary(img)
    h, w = img.shape

    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbors.append(
                padded[1+dy : 1+dy+h, 1+dx : 1+dx+w]
            )

    # AND over all neighbors → min
    eroded = np.logical_and.reduce(neighbors)
    return eroded.astype(np.uint8)

def binary_dilation(img):
    padded = pad_binary(img)
    h, w = img.shape

    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbors.append(
                padded[1+dy : 1+dy+h, 1+dx : 1+dx+w]
            )

    # OR over neighbors → max
    dilated = np.logical_or.reduce(neighbors)
    return dilated.astype(np.uint8)

def opening_numpy(binary, erosion_iter=1, dilation_iter=1):
    out = binary.copy()

    # Erosion phase
    for _ in range(erosion_iter):
        out = binary_erosion(out)

    # Dilation phase
    for _ in range(dilation_iter):
        out = binary_dilation(out)

    return out

#----
output_dir = "/shm/"
os.makedirs(output_dir, exist_ok=True)

for erosion_iter in range(1, 9):
    for dilation_iter in range(1, 9):
        if erosion_iter>dilation_iter:
            opened = opening_numpy(
                otsu,
                erosion_iter=erosion_iter,
                dilation_iter=dilation_iter,
            )
    
            filename = (
                f"opening_erosion{erosion_iter}"
                f"_dilation{dilation_iter}.png"
            )
    
            save_binary_image(
                opened,
                os.path.join(output_dir, filename)
            )



#------ flood fill, connected component
coba= opening_numpy(otsu,4,2)
h, w = coba.shape
visited = np.zeros_like(coba, dtype=bool)

def flood_fill(y, x):
    stack = [(y, x)]
    visited[y, x] = True
    area = 0

    while stack:
        cy, cx = stack.pop()
        area += 1

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < h and 0 <= nx < w and
                    not visited[ny, nx] and coba[ny, nx]):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return area

count = 0
areas = []

# connected counting

for y in range(h):
    for x in range(w):
        if coba[y, x] and not visited[y, x]:
            area = flood_fill(y, x)
            areas.append(area)
            count += 1
            
# result
print("Blob count:", count)
print("Mean area:", np.mean(areas))
print("Median area:", np.median(areas))


