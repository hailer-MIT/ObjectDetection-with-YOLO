"""
YOLO Training Script
"""
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil


class YOLOTrainer:
       
    def __init__(self, config_path="config/training_config.yaml",
                 dataset_config_path="config/dataset.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(dataset_config_path, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        self.dataset_config_path = dataset_config_path
        self.results_dir = Path(self.config.get('save_dir', 'results'))
        self.results_dir.mkdir(exist_ok=True)
    
    def setup_data_augmentation(self):
        return {
            'hsv_h': self.config.get('hsv_h', 0.015),
            'hsv_s': self.config.get('hsv_s', 0.7),
            'hsv_v': self.config.get('hsv_v', 0.4),
            'degrees': self.config.get('degrees', 10.0),
            'translate': self.config.get('translate', 0.1),
            'scale': self.config.get('scale', 0.5),
            'flipud': self.config.get('flipud', 0.0),
            'fliplr': self.config.get('fliplr', 0.5),
            'mosaic': self.config.get('mosaic', 1.0),
            'mixup': self.config.get('mixup', 0.0),
        }
    
    def train(self):
        print("=" * 60)
        print("YOLO Training Configuration")
        print("=" * 60)
        print(f"Model: {self.config['model']}")
        print(f"Image Size: {self.config['image_size']}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Learning Rate: {self.config['learning_rate']}")
        print(f"Dataset: {self.dataset_config['path']}")
        print(f"Classes: {self.dataset_config['nc']}")
        print("=" * 60)
        
        # Initialize model
        model = YOLO(self.config['model'])
        
        # Get augmentation config
        augment_config = self.setup_data_augmentation()
        
        # Train model
        results = model.train(
            data=self.dataset_config_path,
            epochs=self.config['epochs'],
            imgsz=self.config['image_size'],
            batch=self.config['batch_size'],
            lr0=self.config['learning_rate'],
            momentum=self.config.get('momentum', 0.937),
            weight_decay=self.config.get('weight_decay', 0.0005),
            hsv_h=augment_config['hsv_h'],
            hsv_s=augment_config['hsv_s'],
            hsv_v=augment_config['hsv_v'],
            degrees=augment_config['degrees'],
            translate=augment_config['translate'],
            scale=augment_config['scale'],
            flipud=augment_config['flipud'],
            fliplr=augment_config['fliplr'],
            mosaic=augment_config['mosaic'],
            mixup=augment_config['mixup'],
            project=str(self.results_dir),
            name='train',
            save=True,
            save_period=self.config.get('save_period', 10),
            val=True,
            plots=True,
            patience=self.config.get('early_stopping_patience', 5),
        )
        
        # Save best model
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            shutil.copy(best_model_path, models_dir / 'best_model.pt')
            print(f"
✅ Best model saved to: {models_dir / 'best_model.pt'}")
        
        print("
🎉 Training completed!")
        return results


if __name__ == "__main__":
    trainer = YOLOTrainer()
    trainer.train()
