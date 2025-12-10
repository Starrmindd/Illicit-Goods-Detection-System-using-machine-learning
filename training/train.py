"""
Training script for X-ray contraband detection models.
Supports multiple architectures with mixed precision, checkpointing, and comprehensive logging.
"""
import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torchmetrics
from tqdm import tqdm
import wandb

from models import create_model, get_model_info
from dataset import XrayDataset
from losses import FocalLoss, IoULoss
from metrics import DetectionMetrics
from utils import setup_logging, save_checkpoint, load_checkpoint

class XrayTrainer:
    """Main trainer class for X-ray contraband detection."""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = setup_logging(config.get('log_level', 'INFO'))
        
        # Initialize model
        self.model = create_model(
            config['model']['type'],
            num_classes=config['model']['num_classes'],
            **config['model'].get('params', {})
        ).to(self.device)
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss functions
        self.criterion = self._setup_loss_functions()
        
        # Setup metrics
        self.metrics = DetectionMetrics(
            num_classes=config['model']['num_classes'],
            iou_threshold=config['training'].get('iou_threshold', 0.5)
        )
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging and checkpointing
        self.writer = SummaryWriter(config['logging']['tensorboard_dir'])
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_map = 0.0
        
        # Initialize wandb if configured
        if config['logging'].get('use_wandb', False):
            wandb.init(
                project=config['logging']['wandb_project'],
                config=config,
                name=config['logging'].get('experiment_name', 'xray_detection')
            )
    
    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup training and validation data loaders."""
        data_config = self.config['data']
        
        # Training dataset
        train_dataset = XrayDataset(
            data_dir=data_config['train_dir'],
            annotation_file=data_config['train_annotations'],
            image_size=tuple(data_config['image_size']),
            augmentation_severity=data_config.get('augmentation_severity', 'medium'),
            is_training=True
        )
        
        # Validation dataset
        val_dataset = XrayDataset(
            data_dir=data_config['val_dir'],
            annotation_file=data_config['val_annotations'],
            image_size=tuple(data_config['image_size']),
            is_training=False
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=train_dataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.get('val_batch_size', data_config['batch_size']),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=val_dataset.collate_fn
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        elif opt_config['type'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        elif opt_config['type'] == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        sched_config = self.config.get('scheduler')
        if not sched_config:
            return None
        
        if sched_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_config['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_config['type'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 5),
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_config['type']}")
        
        return scheduler
    
    def _setup_loss_functions(self) -> nn.Module:
        """Setup loss functions."""
        loss_config = self.config['loss']
        
        if loss_config['type'] == 'focal':
            criterion = FocalLoss(
                alpha=loss_config.get('alpha', 0.25),
                gamma=loss_config.get('gamma', 2.0),
                num_classes=self.config['model']['num_classes']
            )
        elif loss_config['type'] == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(loss_config.get('class_weights', [1.0] * self.config['model']['num_classes']))
            )
        else:
            # Default to model's built-in loss
            criterion = None
        
        return criterion
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    if self.config['model']['type'] == 'faster_rcnn':
                        loss_dict = self.model(images, targets)
                        losses =