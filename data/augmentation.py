"""
Data augmentation module for X-ray images with security-specific transformations.
"""
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import Dict, List, Tuple, Optional

class XrayAugmentation:
    """X-ray specific augmentation pipeline for contraband detection."""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (640, 640),
                 severity: str = 'medium'):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_size: Target image size (width, height)
            severity: Augmentation intensity ('light', 'medium', 'heavy')
        """
        self.image_size = image_size
        self.severity = severity
        
        # Define augmentation pipelines by severity
        self.pipelines = {
            'light': self._create_light_pipeline(),
            'medium': self._create_medium_pipeline(), 
            'heavy': self._create_heavy_pipeline()
        }
        
        # Validation pipeline (no augmentation)
        self.val_pipeline = A.Compose([
            A.Resize(height=image_size[1], width=image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    
    def _create_light_pipeline(self) -> A.Compose:
        """Light augmentation for validation/testing."""
        return A.Compose([
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            
            # Geometric transformations (minimal)
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=5, p=0.3),
            
            # Photometric (X-ray specific)
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.3
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=0.2),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    
    def _create_medium_pipeline(self) -> A.Compose:
        """Medium augmentation for training."""
        return A.Compose([
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=10, 
                p=0.4
            ),
            
            # X-ray specific photometric
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Noise and artifacts (X-ray machines can introduce these)
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            
            # Simulated occlusions (baggage overlap)
            A.CoarseDropout(
                max_holes=3, 
                max_height=32, 
                max_width=32, 
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    
    def _create_heavy_pipeline(self) -> A.Compose:
        """Heavy augmentation for robust training."""
        return A.Compose([
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            
            # Aggressive geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),  # Less common for X-rays
            A.Rotate(limit=25, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.15, 
                scale_limit=0.15, 
                rotate_limit=15, 
                p=0.5
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Advanced photometric
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=0.6
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            
            # Noise and quality degradation
            A.OneOf([
                A.GaussNoise(var_limit=(10, 80), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8), p=1.0),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=1.0)
            ], p=0.4),
            
            # Simulated baggage complexity
            A.CoarseDropout(
                max_holes=5, 
                max_height=64, 
                max_width=64, 
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=0,
                p=0.4
            ),
            
            # Blur (motion/focus issues)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0)
            ], p=0.2),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    
    def __call__(self, image: np.ndarray, bboxes: List, class_labels: List, 
                 is_training: bool = True) -> Dict:
        """
        Apply augmentation pipeline.
        
        Args:
            image: Input image as numpy array
            bboxes: List of bounding boxes in COCO format [x, y, width, height]
            class_labels: List of class labels corresponding to bboxes
            is_training: Whether to apply training augmentations
            
        Returns:
            Dictionary with augmented image and transformed bboxes
        """
        if not is_training:
            pipeline = self.val_pipeline
        else:
            pipeline = self.pipelines[self.severity]
            
        # Apply augmentation
        transformed = pipeline(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        return {
            'image': transformed['image'],
            'bboxes': transformed['bboxes'],
            'class_labels': transformed['class_labels']
        }
    
    def simulate_xray_artifacts(self, image: np.ndarray, 
                               intensity: float = 0.3) -> np.ndarray:
        """
        Simulate X-ray machine specific artifacts.
        
        Args:
            image: Input image
            intensity: Artifact intensity (0-1)
            
        Returns:
            Image with simulated X-ray artifacts
        """
        h, w = image.shape[:2]
        
        # Simulate beam hardening (darker edges)
        if random.random() < intensity:
            center_x, center_y = w // 2, h // 2
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Create vignette effect
            vignette = 1 - (dist / max_dist) * 0.3
            vignette = np.clip(vignette, 0.7, 1.0)
            
            if len(image.shape) == 3:
                vignette = vignette[:, :, np.newaxis]
            
            image = image * vignette
        
        # Simulate scatter radiation (random bright spots)
        if random.random() < intensity * 0.5:
            num_spots = random.randint(1, 5)
            for _ in range(num_spots):
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                radius = random.randint(5, 20)
                
                cv2.circle(image, (x, y), radius, 
                          (255, 255, 255) if len(image.shape) == 3 else 255, 
                          -1)
        
        # Simulate detector noise (grid pattern)
        if random.random() < intensity * 0.3:
            grid_size = random.choice([8, 16, 32])
            noise_pattern = np.random.normal(0, 5, (h//grid_size, w//grid_size))
            noise_pattern = cv2.resize(noise_pattern, (w, h), 
                                     interpolation=cv2.INTER_LINEAR)
            
            if len(image.shape) == 3:
                noise_pattern = noise_pattern[:, :, np.newaxis]
                
            image = np.clip(image + noise_pattern, 0, 255)
        
        return image.astype(np.uint8)


class MixUp:
    """MixUp augmentation for X-ray images."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, image1: np.ndarray, image2: np.ndarray,
                 labels1: List, labels2: List) -> Tuple[np.ndarray, List, float]:
        """
        Apply MixUp between two images.
        
        Returns:
            Mixed image, combined labels, and mixing ratio
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # Combine labels (both sets are valid with different weights)
        combined_labels = labels1 + labels2
        
        return mixed_image.astype(np.uint8), combined_labels, lam


def create_augmentation_config(severity: str = 'medium') -> Dict:
    """Create augmentation configuration for different training phases."""
    config = {
        'image_size': (640, 640),
        'severity': severity,
        'mixup_alpha': 0.2 if severity in ['medium', 'heavy'] else 0.0,
        'mosaic_prob': 0.3 if severity == 'heavy' else 0.0,
        'copy_paste_prob': 0.2 if severity == 'heavy' else 0.0
    }
    return config


if __name__ == "__main__":
    # Example usage
    aug = XrayAugmentation(severity='medium')
    
    # Create dummy data
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bboxes = [[100, 100, 50, 50], [200, 200, 80, 60]]
    dummy_labels = [1, 2]  # firearm, knife
    
    # Apply augmentation
    result = aug(dummy_image, dummy_bboxes, dummy_labels, is_training=True)
    
    print(f"Original image shape: {dummy_image.shape}")
    print(f"Augmented image shape: {result['image'].shape}")
    print(f"Original bboxes: {dummy_bboxes}")
    print(f"Augmented bboxes: {result['bboxes']}")