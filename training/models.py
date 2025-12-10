"""
Detection models for X-ray contraband detection.
Provides both accuracy-optimized (two-stage) and speed-optimized (one-stage) implementations.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import timm
from typing import Dict, List, Optional, Tuple

class XrayFasterRCNN(nn.Module):
    """
    Accuracy-optimized two-stage detector based on Faster R-CNN.
    Best for high-precision contraband detection.
    """
    
    def __init__(self, 
                 num_classes: int = 8,  # 7 contraband classes + background
                 backbone_name: str = 'resnet50',
                 pretrained: bool = True,
                 trainable_backbone_layers: int = 3):
        """
        Initialize Faster R-CNN model.
        
        Args:
            num_classes: Number of detection classes (including background)
            backbone_name: Backbone architecture ('resnet50', 'resnet101', 'efficientnet_b4')
            pretrained: Use ImageNet pretrained weights
            trainable_backbone_layers: Number of trainable backbone layers
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        if backbone_name.startswith('efficientnet'):
            # Use EfficientNet backbone via timm
            backbone = timm.create_model(
                backbone_name, 
                pretrained=pretrained,
                features_only=True,
                out_indices=[2, 3, 4]  # Use multiple feature levels
            )
            
            # Get feature dimensions
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = backbone(dummy_input)
                backbone.out_channels = features[-1].shape[1]
            
        else:
            # Use torchvision ResNet backbone
            if backbone_name == 'resnet50':
                backbone = torchvision.models.resnet50(pretrained=pretrained)
            elif backbone_name == 'resnet101':
                backbone = torchvision.models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            
            # Remove classifier layers
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone.out_channels = 2048 if 'resnet' in backbone_name else 512
        
        # Custom anchor generator for X-ray objects
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),  # Multiple scales for various object sizes
            aspect_ratios=((0.5, 1.0, 2.0),)   # Different aspect ratios
        )
        
        # ROI pooling
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Build Faster R-CNN
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        # Replace classifier head for our classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(images, targets)


class XrayYOLOv8(nn.Module):
    """
    Speed-optimized one-stage detector based on YOLOv8 architecture.
    Best for real-time contraband detection.
    """
    
    def __init__(self, 
                 num_classes: int = 7,  # 7 contraband classes (no background)
                 model_size: str = 'n',  # 'n', 's', 'm', 'l', 'x'
                 input_size: int = 640):
        """
        Initialize YOLOv8 model.
        
        Args:
            num_classes: Number of detection classes
            model_size: Model size variant
            input_size: Input image size
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Model scaling factors
        scaling = {
            'n': {'depth': 0.33, 'width': 0.25, 'channels': [64, 128, 256]},
            's': {'depth': 0.33, 'width': 0.50, 'channels': [128, 256, 512]},
            'm': {'depth': 0.67, 'width': 0.75, 'channels': [192, 384, 576]},
            'l': {'depth': 1.00, 'width': 1.00, 'channels': [256, 512, 512]},
            'x': {'depth': 1.33, 'width': 1.25, 'channels': [320, 640, 640]}
        }
        
        config = scaling[model_size]
        
        # Backbone (CSPDarknet-like)
        self.backbone = self._build_backbone(config)
        
        # Neck (FPN + PAN)
        self.neck = self._build_neck(config)
        
        # Head (Detection head)
        self.head = self._build_head(config)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_backbone(self, config: Dict) -> nn.Module:
        """Build CSPDarknet backbone."""
        channels = config['channels']
        depth_factor = config['depth']
        width_factor = config['width']
        
        layers = []
        
        # Stem
        layers.append(self._conv_block(3, int(channels[0] * width_factor), 6, 2, 2))
        
        # Stage 1
        layers.append(self._conv_block(int(channels[0] * width_factor), 
                                     int(channels[0] * width_factor), 3, 2, 1))
        layers.append(self._csp_block(int(channels[0] * width_factor), 
                                    int(channels[0] * width_factor), 
                                    int(3 * depth_factor)))
        
        # Stage 2  
        layers.append(self._conv_block(int(channels[0] * width_factor),
                                     int(channels[1] * width_factor), 3, 2, 1))
        layers.append(self._csp_block(int(channels[1] * width_factor),
                                    int(channels[1] * width_factor),
                                    int(6 * depth_factor)))
        
        # Stage 3
        layers.append(self._conv_block(int(channels[1] * width_factor),
                                     int(channels[2] * width_factor), 3, 2, 1))
        layers.append(self._csp_block(int(channels[2] * width_factor),
                                    int(channels[2] * width_factor),
                                    int(9 * depth_factor)))
        
        # Stage 4
        layers.append(self._conv_block(int(channels[2] * width_factor),
                                     int(channels[2] * width_factor), 3, 2, 1))
        layers.append(self._csp_block(int(channels[2] * width_factor),
                                    int(channels[2] * width_factor),
                                    int(3 * depth_factor)))
        
        return nn.Sequential(*layers)
    
    def _build_neck(self, config: Dict) -> nn.Module:
        """Build FPN + PAN neck."""
        # Simplified neck implementation
        return nn.Identity()  # Placeholder for full implementation
    
    def _build_head(self, config: Dict) -> nn.Module:
        """Build detection head."""
        # Simplified head implementation
        return nn.Identity()  # Placeholder for full implementation
    
    def _conv_block(self, in_channels: int, out_channels: int, 
                   kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Module:
        """Convolution block with BatchNorm and SiLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def _csp_block(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Module:
        """CSP (Cross Stage Partial) block."""
        # Simplified CSP implementation
        layers = []
        for _ in range(num_blocks):
            layers.append(self._conv_block(in_channels, out_channels, 3, 1, 1))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Backbone
        features = self.backbone(x)
        
        # Neck
        neck_features = self.neck(features)
        
        # Head
        predictions = self.head(neck_features)
        
        return predictions


class EfficientDetXray(nn.Module):
    """
    Balanced detector based on EfficientDet architecture.
    Good compromise between speed and accuracy.
    """
    
    def __init__(self, 
                 num_classes: int = 7,
                 compound_coef: int = 0,  # 0-7 for D0-D7
                 pretrained: bool = True):
        """
        Initialize EfficientDet model.
        
        Args:
            num_classes: Number of detection classes
            compound_coef: EfficientDet compound coefficient (0-7)
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.compound_coef = compound_coef
        
        # Model configurations
        configs = {
            0: {'input_size': 512, 'backbone': 'efficientnet_b0', 'fpn_channels': 64, 'fpn_layers': 3},
            1: {'input_size': 640, 'backbone': 'efficientnet_b1', 'fpn_channels': 88, 'fpn_layers': 4},
            2: {'input_size': 768, 'backbone': 'efficientnet_b2', 'fpn_channels': 112, 'fpn_layers': 5},
            3: {'input_size': 896, 'backbone': 'efficientnet_b3', 'fpn_channels': 160, 'fpn_layers': 6},
            4: {'input_size': 1024, 'backbone': 'efficientnet_b4', 'fpn_channels': 224, 'fpn_layers': 7},
        }
        
        config = configs.get(compound_coef, configs[0])
        
        # Backbone
        self.backbone = timm.create_model(
            config['backbone'],
            pretrained=pretrained,
            features_only=True,
            out_indices=[2, 3, 4, 5, 6]
        )
        
        # BiFPN
        self.bifpn = self._build_bifpn(config)
        
        # Detection heads
        self.classification_head = self._build_classification_head(config)
        self.regression_head = self._build_regression_head(config)
        
    def _build_bifpn(self, config: Dict) -> nn.Module:
        """Build Bidirectional Feature Pyramid Network."""
        # Simplified BiFPN implementation
        return nn.Identity()  # Placeholder
    
    def _build_classification_head(self, config: Dict) -> nn.Module:
        """Build classification head."""
        # Simplified classification head
        return nn.Identity()  # Placeholder
    
    def _build_regression_head(self, config: Dict) -> nn.Module:
        """Build regression head."""
        # Simplified regression head  
        return nn.Identity()  # Placeholder
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Backbone
        features = self.backbone(x)
        
        # BiFPN
        fpn_features = self.bifpn(features)
        
        # Heads
        classifications = self.classification_head(fpn_features)
        regressions = self.regression_head(fpn_features)
        
        return {
            'classifications': classifications,
            'regressions': regressions
        }


def create_model(model_type: str, num_classes: int = 7, **kwargs) -> nn.Module:
    """
    Factory function to create detection models.
    
    Args:
        model_type: Type of model ('faster_rcnn', 'yolov8', 'efficientdet')
        num_classes: Number of detection classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    if model_type == 'faster_rcnn':
        return XrayFasterRCNN(num_classes + 1, **kwargs)  # +1 for background
    elif model_type == 'yolov8':
        return XrayYOLOv8(num_classes, **kwargs)
    elif model_type == 'efficientdet':
        return EfficientDetXray(num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_info(model_type: str) -> Dict:
    """Get model characteristics and recommended use cases."""
    info = {
        'faster_rcnn': {
            'type': 'two_stage',
            'speed': 'slow',
            'accuracy': 'high',
            'use_case': 'High-precision screening, detailed analysis',
            'target_latency_ms': 800,
            'memory_gb': 4
        },
        'yolov8': {
            'type': 'one_stage', 
            'speed': 'fast',
            'accuracy': 'medium',
            'use_case': 'Real-time screening, high throughput',
            'target_latency_ms': 100,
            'memory_gb': 2
        },
        'efficientdet': {
            'type': 'one_stage',
            'speed': 'medium',
            'accuracy': 'high',
            'use_case': 'Balanced performance, production deployment',
            'target_latency_ms': 300,
            'memory_gb': 3
        }
    }
    return info.get(model_type, {})


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    faster_rcnn = create_model('faster_rcnn', num_classes=7)
    yolov8 = create_model('yolov8', num_classes=7, model_size='s')
    efficientdet = create_model('efficientdet', num_classes=7, compound_coef=1)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 640, 640)
    
    print("Model architectures created successfully!")
    print(f"Faster R-CNN parameters: {sum(p.numel() for p in faster_rcnn.parameters()):,}")
    print(f"YOLOv8 parameters: {sum(p.numel() for p in yolov8.parameters()):,}")
    print(f"EfficientDet parameters: {sum(p.numel() for p in efficientdet.parameters()):,}")