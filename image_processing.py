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


# ─────────────────────────────────────────────
# STEP 3a: Rotate the Image
# Commit: "Added image rotation"
# ─────────────────────────────────────────────

(h, w) = image.shape[:2]
center = (w // 2, h // 2)
angle = 45
scale = 1.0

rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image (45°)")
plt.axis("off")
plt.savefig("outputs/rotated.png", bbox_inches='tight')
plt.show()
print("✅ Step 3a: Image rotated 45°.")



# ─────────────────────────────────────────────
# STEP 3b: Scale the Image
# Commit: "Added image scaling"
# ─────────────────────────────────────────────

scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title("Scaled Image (1.5x)")
plt.axis("off")
plt.savefig("outputs/scaled.png", bbox_inches='tight')
plt.show()
print("✅ Step 3b: Image scaled 1.5x.")




# ─────────────────────────────────────────────
# STEP 4: Simulate Camera Focal Lengths (3D Vision)
# Commit: "Added camera focal length simulation"
# ─────────────────────────────────────────────

focal_lengths = [50, 100, 200]

plt.figure(figsize=(12, 4))
for i, f in enumerate(focal_lengths):
    f_matrix = np.array([[f, 0, w // 2],
                         [0, f, h // 2],
                         [0,  0,     1]], dtype=np.float32)
    warped = cv2.warpPerspective(image, f_matrix, (w, h))
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title(f"Focal Length: {f}")
    plt.axis("off")

plt.suptitle("Camera Focal Length Simulation", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/focal_lengths.png", bbox_inches='tight')
plt.show()
print("✅ Step 4: Focal length simulation complete.")
