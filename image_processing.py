import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
# STEP 2: Load and Display an Image
# Commit: "Added image loading functionality"
# ─────────────────────────────────────────────

image = cv2.imread('images/sample.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB for display

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.savefig("outputs/original.png", bbox_inches='tight')
plt.show()
print("✅ Step 2: Image loaded and displayed.")