import numpy as np
import os

class ArrayNormalizer:
    def __init__(self, directory, min = None, max = None):
        self.directory = directory
        self.global_min = min
        self.global_max = max

    def read_np_arrays(self):
        arrays = []
        filenames = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.npy'):
                filepath = os.path.join(self.directory, filename)
                arrays.append(np.load(filepath))
                filenames.append(filepath)
        return arrays, filenames

    def find_global_min_max(self, arrays):
        self.global_min = min(array.min() for array in arrays)
        self.global_max = max(array.max() for array in arrays)

    def normalize_arrays(self, arrays):
        normalized_arrays = []
        for array in arrays:
            normalized = (array - self.global_min) / (self.global_max - self.global_min)
            normalized_arrays.append(normalized)
        return normalized_arrays

    def process_and_save(self):
        arrays, filenames = self.read_np_arrays()
        self.find_global_min_max(arrays)
        normalized_arrays = self.normalize_arrays(arrays)

        # Overwrite each file with its normalized array
        for filepath, normalized_array in zip(filenames, normalized_arrays):
            np.save(filepath, normalized_array)

# Usage
directory = "Audio_Processing//latents"
normalizer = ArrayNormalizer(directory)
arrays, filenames = normalizer.read_np_arrays()
normalizer.find_global_min_max(arrays)
# normalizer.process_and_save()
print(f"Global Min: {normalizer.global_min}, Global Max: {normalizer.global_max}")


# Example of denormalizing the first array (if it exists)
# if normalized_arrays:
#     original_data = normalizer.denormalize_array(normalized_arrays[0])
#     print("First denormalized array:\n", original_data)
