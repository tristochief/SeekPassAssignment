import os
import random
import shutil

def split_dataset(source_dir, train_dir, test_dir, test_ratio=0.1):
    """
    Splits images from each class in the source directory into training and testing sets.
    Moves the images into corresponding subdirectories in the train and test directories.

    Parameters:
        source_dir (str): Path to the source directory containing class subdirectories.
        train_dir (str): Path to the training directory.
        test_dir (str): Path to the testing directory.
        test_ratio (float): Proportion of images to be used for testing (default is 0.1).
    """
    # Ensure the train and test directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over each class (subdirectory) in the source directory
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            # Create corresponding subdirectories in train and test directories
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # List all image files in the class directory
            image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            # Shuffle the list to ensure random selection
            random.shuffle(image_files)

            # Determine the number of test images
            num_test_images = int(len(image_files) * test_ratio)
            # Split the images into test and train sets
            test_images = image_files[:num_test_images]
            train_images = image_files[num_test_images:]

            # Move the images to the respective directories
            for image in test_images:
                shutil.move(os.path.join(class_path, image), os.path.join(test_dir, class_name, image))
            for image in train_images:
                shutil.move(os.path.join(class_path, image), os.path.join(train_dir, class_name, image))

if __name__ == "__main__":
    # Define the paths
    source_directory = "images"  # Replace with your source directory path
    train_directory = "images/train"  # Replace with your training directory path
    test_directory = "images/test"  # Replace with your testing directory path

    # Split the dataset
    split_dataset(source_directory, train_directory, test_directory)
