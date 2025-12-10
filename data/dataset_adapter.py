"""
X-ray dataset adapter for COCO format ingestion and validation.
"""
import json
import os
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class XrayDatasetAdapter:
    """Adapter for ingesting X-ray images and COCO annotations."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Standard contraband classes
    CLASS_MAPPING = {
        1: 'firearm',
        2: 'knife', 
        3: 'explosive',
        4: 'drug_package',
        5: 'contraband_electronics',
        6: 'liquid_battery',
        7: 'prohibited_item_misc'
    }
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def ingest_single_image(self, image_path: str, annotation_path: Optional[str] = None) -> Dict:
        """Ingest a single X-ray image with optional COCO annotation."""
        image_path = Path(image_path)
        
        if not self._validate_image(image_path):
            raise ValueError(f"Invalid image: {image_path}")
            
        # Load image metadata
        img = Image.open(image_path)
        width, height = img.size
        
        result = {
            'image_id': image_path.stem,
            'file_name': image_path.name,
            'width': width,
            'height': height,
            'annotations': []
        }
        
        # Load annotations if provided
        if annotation_path and Path(annotation_path).exists():
            with open(annotation_path, 'r') as f:
                coco_data = json.load(f)
                result['annotations'] = self._extract_annotations(coco_data, image_path.stem)
                
        return result
    
    def ingest_batch_archive(self, archive_path: str) -> List[Dict]:
        """Ingest batch of images from zip/tar archive."""
        archive_path = Path(archive_path)
        extract_dir = self.data_dir / f"batch_{archive_path.stem}"
        
        # Extract archive
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix in ['.tar', '.tar.gz']:
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
            
        # Process extracted files
        results = []
        for img_file in extract_dir.rglob('*'):
            if img_file.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    # Look for corresponding annotation file
                    ann_file = img_file.with_suffix('.json')
                    result = self.ingest_single_image(
                        str(img_file), 
                        str(ann_file) if ann_file.exists() else None
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to process {img_file}: {e}")
                    
        return results
    
    def _validate_image(self, image_path: Path) -> bool:
        """Validate image integrity and format."""
        try:
            if not image_path.exists():
                return False
                
            if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return False
                
            # Check if image can be opened
            img = Image.open(image_path)
            width, height = img.size
            
            # Basic sanity checks
            if width < 64 or height < 64:
                logger.warning(f"Image too small: {width}x{height}")
                return False
                
            if width > 4096 or height > 4096:
                logger.warning(f"Image too large: {width}x{height}")
                return False
                
            # Check aspect ratio (X-ray images should be reasonable)
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 5.0:
                logger.warning(f"Unusual aspect ratio: {aspect_ratio}")
                
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def _extract_annotations(self, coco_data: Dict, image_id: str) -> List[Dict]:
        """Extract annotations for specific image from COCO format."""
        annotations = []
        
        # Find image info
        image_info = None
        for img in coco_data.get('images', []):
            if str(img.get('id')) == image_id or img.get('file_name', '').startswith(image_id):
                image_info = img
                break
                
        if not image_info:
            return annotations
            
        # Extract annotations for this image
        for ann in coco_data.get('annotations', []):
            if ann.get('image_id') == image_info.get('id'):
                category_id = ann.get('category_id')
                bbox = ann.get('bbox', [])  # [x, y, width, height]
                
                if len(bbox) == 4 and category_id in self.CLASS_MAPPING:
                    annotations.append({
                        'category_id': category_id,
                        'category_name': self.CLASS_MAPPING[category_id],
                        'bbox': bbox,
                        'area': ann.get('area', bbox[2] * bbox[3]),
                        'iscrowd': ann.get('iscrowd', 0)
                    })
                    
        return annotations
    
    def validate_dataset(self, dataset_path: str) -> Dict:
        """Validate entire dataset and return statistics."""
        dataset_path = Path(dataset_path)
        stats = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'total_annotations': 0,
            'class_distribution': {name: 0 for name in self.CLASS_MAPPING.values()},
            'errors': []
        }
        
        # Process all images
        for img_file in dataset_path.rglob('*'):
            if img_file.suffix.lower() in self.SUPPORTED_FORMATS:
                stats['total_images'] += 1
                
                try:
                    if self._validate_image(img_file):
                        stats['valid_images'] += 1
                        
                        # Check for annotations
                        ann_file = img_file.with_suffix('.json')
                        if ann_file.exists():
                            with open(ann_file, 'r') as f:
                                coco_data = json.load(f)
                                annotations = self._extract_annotations(coco_data, img_file.stem)
                                stats['total_annotations'] += len(annotations)
                                
                                for ann in annotations:
                                    class_name = ann['category_name']
                                    stats['class_distribution'][class_name] += 1
                    else:
                        stats['invalid_images'] += 1
                        
                except Exception as e:
                    stats['invalid_images'] += 1
                    stats['errors'].append(f"{img_file}: {str(e)}")
                    
        return stats


def create_sample_coco_annotation():
    """Create sample COCO annotation format for reference."""
    sample = {
        "info": {
            "description": "X-ray Contraband Detection Dataset",
            "version": "1.0",
            "year": 2024
        },
        "categories": [
            {"id": cat_id, "name": name, "supercategory": "contraband"}
            for cat_id, name in XrayDatasetAdapter.CLASS_MAPPING.items()
        ],
        "images": [
            {
                "id": 1,
                "file_name": "xray_001.jpg",
                "width": 640,
                "height": 480
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,  # firearm
                "bbox": [100, 150, 200, 100],  # [x, y, width, height]
                "area": 20000,
                "iscrowd": 0
            }
        ]
    }
    return sample


if __name__ == "__main__":
    # Example usage
    adapter = XrayDatasetAdapter("./sample_data")
    
    # Create sample annotation
    sample_coco = create_sample_coco_annotation()
    with open("sample_annotation.json", "w") as f:
        json.dump(sample_coco, f, indent=2)
        
    print("Sample COCO annotation created: sample_annotation.json")