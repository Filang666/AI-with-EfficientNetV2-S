import kagglehub
import os
import shutil

def setup_dataset():
    # 1. Download dataset to system cache
    print("Downloading dataset via kagglehub...")
    raw_path = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")
    
    # 2. Define project structure
    target_dir = "dataset"
    # Kaggle structure: casting_data/casting_data/test and casting_data/casting_data/train
    source_base = os.path.join(raw_path, 'casting_data', 'casting_data')
    
    # Classes to process
    classes = ['def_front', 'ok_front']

    print(f"Merging 'train' and 'test' folders into {target_dir}...")

    for class_name in classes:
        target_class_path = os.path.join(target_dir, class_name)
        
        # Merge both 'train' and 'test' sources into one project folder
        for split in ['train', 'test']:
            source_path = os.path.join(source_base, split, class_name)
            
            if os.path.exists(source_path):
                # dirs_exist_ok=True allows merging files into the same folder
                shutil.copytree(source_path, target_class_path, dirs_exist_ok=True)
                print(f"✅ Synced {split}/{class_name} -> {target_dir}/{class_name}")

    print("\n🚀 Dataset prepared successfully! Your train.py will now auto-split it.")

if __name__ == "__main__":
    setup_dataset()