import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# load the image
image_path = "C:\\cartoon.jpg"
img = mpimg.imread(image_path)
plt.imshow(img)

# scaled value is also okay
img_scaled = img/255.0
plt.imshow(img_scaled)

red= img[:, :, 0] # red channel
green= img[:, :, 1] # green channel
blue= img[:, :, 2] # blue channel

from matplotlib.colors import LinearSegmentedColormap

# red 
redscale = LinearSegmentedColormap.from_list("redscale", [(0, "black"), (1, "red")])
plt.imshow(red, cmap=redscale)
cbar = plt.colorbar()

# green
greenscale = LinearSegmentedColormap.from_list("redscale", [(0, "black"), (1, "green")])
plt.imshow(green, cmap=greenscale)
cbar = plt.colorbar()

# blue
bluescale = LinearSegmentedColormap.from_list("redscale", [(0, "black"), (1, "blue")])
plt.imshow(blue, cmap=bluescale)
cbar = plt.colorbar()

# converting to grayscale
grayscale = 0.3 * red + 0.6 * green * 0.1 * blue
plt.imshow(grayscale, cmap='gray')



# make image brighter or darker
# brightess
# contrast
import numpy as np
brightness = -0.2   # positive = brighter, negative = darker
contrast = 1.5     # >1 = higher contrast, <1 = lower contrast
adjusted = contrast * (img_scaled - 0.5) + 0.5 + brightness
adjusted = np.clip(adjusted, 0, 1)
plt.imshow(adjusted)

# convert to HSV
from matplotlib.colors import rgb_to_hsv

img= img_scaled
img = mpimg.imread(image_path)
img = np.asarray(img, dtype=np.float32)/255.0
hsv = rgb_to_hsv(img)

h = hsv[..., 0]
s = hsv[..., 1]
v = hsv[..., 2]
cyan_mask = (h>0.45) & (h<0.55) & (s > 0.3) & (v > 0.2)

# positive mask on the cyan part
isolated = img.copy()
isolated[~cyan_mask] = 0  # zero out non-green pixels
plt.imshow(isolated)

# highlight
highlight = img.copy()
highlight[~cyan_mask] = highlight[~cyan_mask] / 5 # dim background
plt.imshow(highlight)

