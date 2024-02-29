import numpy as np
import os

class ArrayNormalizer:
    def __init__(self, directory):
        self.directory = directory

    def read_np_arrays(self):
        arrays = []
        filenames = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.npy'):
                filepath = os.path.join(self.directory, filename)
                arrays.append(np.load(filepath))
                filenames.append(filepath)
        return arrays, filenames

    def normalize_arrays(self, arrays):
        normalized_arrays = []
        for array in arrays:
            normalized = array * 0.18215
            normalized_arrays.append(normalized)
        return normalized_arrays

    def process_and_save(self):
        arrays, filenames = self.read_np_arrays()
        normalized_arrays = self.normalize_arrays(arrays)

        # Overwrite each file with its normalized array
        for filepath, normalized_array in zip(filenames, normalized_arrays):
            np.save(filepath, normalized_array)
    
    def denormalize_array(self, normalized_array):
        original_data = normalized_array * (1 / 0.18215)
        return original_data

<<<<<<< HEAD
=======
# # Usage
# directory = "Audio_Processing//latents"
# normalizer = ArrayNormalizer(directory)
# normalizer.process_and_save()


>>>>>>> 97c148c9d96a3152729a07516e916549d42571e5
# # Example of denormalizing the first array (if it exists)
# # if normalized_arrays:
# #     original_data = normalizer.denormalize_array(normalized_arrays[0])
# #     print("First denormalized array:\n", original_data)
