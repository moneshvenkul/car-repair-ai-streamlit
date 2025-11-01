# Final Fixed Fine-Tuning Script for Vision Agent
# Handles Kaggle's train/test/valid folder structure

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import TrainingArguments, Trainer
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import shutil
import zipfile
from collections import defaultdict

class CarPartsDataset(Dataset):
    """Custom dataset for car parts classification"""
    
    def __init__(self, image_paths, labels, processor, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Process image for model
        inputs = self.processor(image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class VisionAgentTrainer:
    """Final fixed fine-tuning trainer for Vision Agent"""
    
    def __init__(self, model_name="microsoft/resnet-50"):
        self.model_name = model_name
        self.dataset_path = "data/car_parts_dataset"
        self.output_dir = "models/vision_agent_finetuned"
        
    def download_kaggle_dataset(self):
        """Download and prepare Kaggle dataset"""
        print("📥 Downloading Kaggle car parts dataset...")
        
        try:
            # Try using kagglehub first
            try:
                import kagglehub
                path = kagglehub.dataset_download('gpiosenka/car-parts-40-classes')
                print(f"✅ Dataset downloaded to: {path}")
                
                if os.path.exists(path):
                    os.makedirs(self.dataset_path, exist_ok=True)
                    
                    # Copy dataset
                    for root, dirs, files in os.walk(path):
                        if dirs:
                            for item in dirs:
                                src = os.path.join(root, item)
                                dst = os.path.join(self.dataset_path, item)
                                if os.path.isdir(src) and not os.path.exists(dst):
                                    shutil.copytree(src, dst)
                                    print(f"Copied: {item}")
                            break
                
                return True
                
            except ImportError:
                print("kagglehub not available, trying kaggle API...")
            
            # Fallback to kaggle API
            import kaggle
            
            os.makedirs("data", exist_ok=True)
            
            kaggle.api.dataset_download_files(
                'gpiosenka/car-parts-40-classes',
                path='data/',
                unzip=True
            )
            
            print(f"Checking downloaded files in data/...")
            for item in os.listdir('data/'):
                print(f"  - {item}")
            
            # Find dataset location
            possible_paths = [
                'data/car-parts-40-classes',
                'data/Car Parts',
                'data/',
            ]
            
            for possible_path in possible_paths:
                if os.path.exists(possible_path):
                    subdirs = [d for d in os.listdir(possible_path) 
                              if os.path.isdir(os.path.join(possible_path, d))]
                    
                    if subdirs:
                        print(f"✅ Found dataset in: {possible_path}")
                        print(f"   Classes found: {len(subdirs)}")
                        
                        if possible_path != self.dataset_path:
                            os.makedirs(self.dataset_path, exist_ok=True)
                            for subdir in subdirs:
                                src = os.path.join(possible_path, subdir)
                                dst = os.path.join(self.dataset_path, subdir)
                                if not os.path.exists(dst):
                                    shutil.copytree(src, dst)
                        
                        return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False
    
    def prepare_dataset_from_nested_structure(self, max_samples_per_class=200):
        """
        Prepare dataset from Kaggle's train/test/valid structure
        
        Structure:
        data/car_parts_dataset/
        ├── train/
        │   ├── BATTERY/
        │   ├── TIRE/
        │   └── ... (40 classes)
        ├── test/
        │   └── ... (40 classes)
        └── valid/
            └── ... (40 classes)
        """
        print("📊 Preparing training dataset from train/test/valid structure...")
        
        dataset_root = Path(self.dataset_path)
        
        if not dataset_root.exists():
            print(f"❌ Dataset path does not exist: {dataset_root}")
            return None, None, None
        
        # Check for train/test/valid folders
        train_dir = dataset_root / "train"
        test_dir = dataset_root / "test"
        valid_dir = dataset_root / "valid"
        
        if not train_dir.exists():
            print(f"❌ Train directory not found: {train_dir}")
            return None, None, None
        
        print(f"✅ Found nested structure with train/test/valid folders")
        
        # Collect class names from train directory
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            print(f"❌ No class directories found in: {train_dir}")
            return None, None, None
        
        class_names = sorted([d.name for d in class_dirs])
        
        print(f"\n🎯 Found {len(class_names)} car part classes:")
        for i, name in enumerate(class_names[:10]):
            print(f"  {i+1}. {name}")
        if len(class_names) > 10:
            print(f"  ... and {len(class_names) - 10} more")
        
        # Collect images from all splits (train, test, valid)
        all_image_paths = []
        all_labels = []
        
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Process each split
        for split_name, split_dir in [("train", train_dir), ("test", test_dir), ("valid", valid_dir)]:
            if not split_dir.exists():
                print(f"⚠️ {split_name} directory not found, skipping...")
                continue
            
            print(f"\n📂 Processing {split_name} split...")
            split_count = 0
            
            for class_name in class_names:
                class_dir = split_dir / class_name
                
                if not class_dir.exists():
                    continue
                
                # Find all image files
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(list(class_dir.glob(ext)))
                
                if not image_files:
                    continue
                
                # Limit samples per class per split
                if len(image_files) > max_samples_per_class:
                    image_files = image_files[:max_samples_per_class]
                
                for img_file in image_files:
                    all_image_paths.append(str(img_file))
                    all_labels.append(class_to_idx[class_name])
                    split_count += 1
            
            print(f"   Collected {split_count} images from {split_name}")
        
        total_images = len(all_image_paths)
        print(f"\n✅ Total samples collected: {total_images}")
        
        if total_images == 0:
            print("❌ No images found!")
            return None, None, None
        
        # Show distribution
        print(f"\n📊 Class distribution (first 5 classes):")
        from collections import Counter
        label_counts = Counter(all_labels)
        for idx, count in list(label_counts.items())[:5]:
            print(f"   {class_names[idx]}: {count} images")
        
        return all_image_paths, all_labels, class_names
    
    def create_model(self, num_classes):
        """Create model for fine-tuning"""
        print(f"🔥 Creating model for {num_classes} classes...")
        
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        return model, processor
    
    def train_model(self, epochs=3, batch_size=16, learning_rate=2e-5):
        """Fine-tune the vision model"""
        print("🚀 Starting fine-tuning process...")
        
        # Prepare dataset using nested structure
        image_paths, labels, class_names = self.prepare_dataset_from_nested_structure()
        
        if not image_paths or not labels or not class_names:
            print("❌ Failed to prepare dataset")
            return False
        
        # Split dataset (using stratified split to maintain class distribution)
        print(f"\n📊 Creating train/validation split...")
        try:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels, test_size=0.15, random_state=42, stratify=labels
            )
        except ValueError:
            # If stratify fails, do regular split
            print("⚠️ Some classes have too few samples for stratified split, using regular split")
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels, test_size=0.15, random_state=42
            )
        
        print(f"Training samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")
        
        # Create model
        print(f"\n🤖 Loading base model: {self.model_name}")
        model, processor = self.create_model(len(class_names))
        
        # Create datasets
        print("📦 Creating PyTorch datasets...")
        train_dataset = CarPartsDataset(train_paths, train_labels, processor)
        val_dataset = CarPartsDataset(val_paths, val_labels, processor)
        
        # Training arguments
        print("\n⚙️ Configuring training parameters...")
        training_args = TrainingArguments(
    output_dir=self.output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="steps",  # Changed from evaluation_strategy
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none"
)
        
        # Create trainer
        print("🎓 Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train model
        print("\n🔥 Starting training...")
        print("=" * 60)
        print(f"📊 Training Configuration:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch Size: {batch_size}")
        print(f"   - Learning Rate: {learning_rate}")
        print(f"   - Total Training Steps: {len(train_paths) // batch_size * epochs}")
        print("=" * 60)
        
        try:
            trainer.train()
            
            # Save final model
            print(f"\n💾 Saving model to {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
            trainer.save_model()
            processor.save_pretrained(self.output_dir)
            
            # Save class names
            class_names_path = os.path.join(self.output_dir, "class_names.json")
            with open(class_names_path, 'w') as f:
                json.dump(class_names, f, indent=2)
            print(f"💾 Saved class names to {class_names_path}")
            
            # Save training metadata
            metadata = {
                "model_name": self.model_name,
                "num_classes": len(class_names),
                "total_training_samples": len(train_paths),
                "total_validation_samples": len(val_paths),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
            
            metadata_path = os.path.join(self.output_dir, "training_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"💾 Saved training metadata to {metadata_path}")
            
            print("\n✅ Vision Agent fine-tuning completed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_model(self):
        """Evaluate the fine-tuned model"""
        print("\n📊 Evaluating fine-tuned model...")
        
        try:
            if not os.path.exists(self.output_dir):
                print(f"❌ Model directory not found: {self.output_dir}")
                return False
            
            # Load model
            print("📥 Loading fine-tuned model...")
            model = AutoModelForImageClassification.from_pretrained(self.output_dir)
            processor = AutoImageProcessor.from_pretrained(self.output_dir)
            
            # Load class names
            class_names_path = os.path.join(self.output_dir, "class_names.json")
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    class_names = json.load(f)
            else:
                class_names = [f"Class_{i}" for i in range(model.config.num_labels)]
            
            print(f"✅ Model loaded with {len(class_names)} classes")
            
            # Test on sample images from test/valid sets
            test_dir = Path(self.dataset_path) / "test"
            
            if not test_dir.exists():
                test_dir = Path(self.dataset_path) / "valid"
            
            if not test_dir.exists():
                print("⚠️ No test/valid directory found for evaluation")
                return True
            
            # Get sample images from different classes
            sample_images = []
            for class_name in class_names[:5]:  # Test first 5 classes
                class_dir = test_dir / class_name
                if class_dir.exists():
                    images = list(class_dir.glob("*.jpg"))
                    if images:
                        sample_images.append(images[0])
            
            if not sample_images:
                print("⚠️ No sample images found")
                return True
            
            print(f"\n🧪 Testing on {len(sample_images)} sample images:")
            print("=" * 60)
            
            correct = 0
            total = 0
            
            for img_path in sample_images:
                try:
                    image = Image.open(img_path).convert('RGB')
                    inputs = processor(image, return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        predicted_class_id = predictions.argmax().item()
                        confidence = predictions[0][predicted_class_id].item()
                    
                    predicted_class = class_names[predicted_class_id]
                    actual_class = img_path.parent.name
                    
                    is_correct = predicted_class.upper() == actual_class.upper()
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    match = "✅" if is_correct else "❌"
                    
                    print(f"\n{match} Image: {img_path.name}")
                    print(f"   Actual: {actual_class}")
                    print(f"   Predicted: {predicted_class}")
                    print(f"   Confidence: {confidence:.1%}")
                    
                except Exception as e:
                    print(f"❌ Error testing {img_path.name}: {e}")
            
            accuracy = (correct / total * 100) if total > 0 else 0
            print("\n" + "=" * 60)
            print(f"📊 Sample Accuracy: {correct}/{total} ({accuracy:.1f}%)")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main fine-tuning script"""
    print("🚗 Car Parts Vision Agent Fine-Tuning")
    print("=" * 60)
    
    trainer = VisionAgentTrainer()
    
    # Step 1: Verify dataset exists
    print("\nStep 1: Checking for dataset...")
    dataset_path = Path(trainer.dataset_path)
    
    if dataset_path.exists():
        train_dir = dataset_path / "train"
        if train_dir.exists():
            print(f"✅ Dataset found at: {dataset_path.absolute()}")
        else:
            print("❌ Dataset structure incorrect (no train folder)")
            print("💡 Run download step or check dataset manually")
            return
    else:
        print("❌ Dataset not found")
        response = input("Download dataset now? (y/n): ")
        if response.lower() == 'y':
            if not trainer.download_kaggle_dataset():
                print("Dataset download failed. Exiting.")
                return
    
    # Step 2: Verify dataset
    print("\nStep 2: Verifying dataset structure...")
    image_paths, labels, class_names = trainer.prepare_dataset_from_nested_structure()
    
    if not image_paths:
        print("\n❌ Dataset verification failed")
        return
    
    print(f"\n✅ Dataset ready: {len(image_paths)} images, {len(class_names)} classes")
    
    # Step 3: Train
    print("\n" + "=" * 60)
    print("Step 3: Training Configuration")
    print("=" * 60)
    print(f"This will:")
    print(f"  - Train for 5 epochs")
    print(f"  - Use batch size of 8")
    print(f"  - Take approximately 30-60 minutes")
    print(f"  - Require ~4GB RAM")
    
    response = input("\nStart training? (y/n): ")
    
    if response.lower() == 'y':
        if trainer.train_model(epochs=5, batch_size=8):
            print("\n🎉 Training completed successfully!")
            
            # Evaluate
            trainer.evaluate_model()
            
            print(f"\n📁 Model saved to: {os.path.abspath(trainer.output_dir)}")
            print("🚀 Ready to use in your car repair app!")
        else:
            print("\n❌ Training failed")
    else:
        print("Training cancelled")

if __name__ == "__main__":
    main()