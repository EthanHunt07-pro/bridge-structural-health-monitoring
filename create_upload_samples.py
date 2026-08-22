import numpy as np
import os
import glob

# ==============================
# Load test dataset
# ==============================

X_test = np.load("data/processed/X_test.npy")

print(f"Total test samples: {len(X_test)}")


# ==============================
# Create upload_samples folder
# ==============================

os.makedirs("upload_samples", exist_ok=True)


# ==============================
# Remove previously generated samples
# ==============================

old_samples = glob.glob("upload_samples/test_sample_*.npy")

for file in old_samples:
    os.remove(file)


# ==============================
# Save all test samples
# ==============================

for i, sample in enumerate(X_test):
    np.save(
        f"upload_samples/test_sample_{i:04d}.npy",
        sample
    )


print(f"Generated {len(X_test)} test samples.")
print("All test samples are now available in upload_samples/")