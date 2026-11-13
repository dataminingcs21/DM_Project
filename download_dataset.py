import kagglehub
import os

# Folder where this Python file is located
project_folder = os.path.dirname(os.path.abspath(__file__))

# Create an empty dataset folder
dataset_folder = os.path.join(project_folder, "dataset")
os.makedirs(dataset_folder, exist_ok=True)

# Download
path = kagglehub.dataset_download(
    "chethuhn/network-intrusion-dataset",
    output_dir=dataset_folder
)

print("Path to dataset files:", path)