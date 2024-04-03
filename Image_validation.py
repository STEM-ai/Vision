from datasets import load_dataset
import PIL

# Load the dataset
dataset = load_dataset("STEM-AI-mtl/City_map", split="train")

# Print the total number of images in the dataset
total_images = len(dataset)
print(f"Total number of images in the dataset: {total_images}")

# Initialize a counter for skipped images
skipped_images = 2 #On a déjà identifié 113 et 227

# Iterate over the dataset for the first interval: from 0 to 112
for i in range(113):
    try:
        example = dataset[i]
        image = example['image']  # Accessing the image data
        print(f"Image {i} type: {type(image)}")
    except PIL.UnidentifiedImageError:
        # Print an error message and continue to the next iteration if an error occurs
        print(f"Error processing image {i}: UnidentifiedImageError")
        skipped_images += 1
        continue
    except Exception as e:
        # Print any other unexpected errors and continue to the next iteration
        print(f"Error processing image {i}: {e}")
        skipped_images += 1
        continue

# Iterate over the dataset for the second interval: from 114 to 226
for i in range(114, 227):
    try:
        example = dataset[i]
        image = example['image']  # Accessing the image data
        print(f"Image {i} type: {type(image)}")
    except PIL.UnidentifiedImageError:
        # Print an error message and continue to the next iteration if an error occurs
        print(f"Error processing image {i}: UnidentifiedImageError")
        skipped_images += 1
        continue
    except Exception as e:
        # Print any other unexpected errors and continue to the next iteration
        print(f"Error processing image {i}: {e}")
        skipped_images += 1
        continue

# Iterate over the dataset for the third interval: from 228 to the end
for i in range(228, total_images):
    try:
        example = dataset[i]
        image = example['image']  # Accessing the image data
        print(f"Image {i} type: {type(image)}")
    except PIL.UnidentifiedImageError:
        # Print an error message and continue to the next iteration if an error occurs
        print(f"Error processing image {i}: UnidentifiedImageError")
        skipped_images += 1
        continue
    except Exception as e:
        # Print any other unexpected errors and continue to the next iteration
        print(f"Error processing image {i}: {e}")
        skipped_images += 1
        continue

# Print the total number of skipped images
print(f"Total skipped images due to errors: {skipped_images}")
