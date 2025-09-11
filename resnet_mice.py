import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

from mice_dataset import MiceClassificationDataset

class ResNetVideoClassifier(nn.Module):
    """
    video classifier using ResNet
    """
    def __init__(self, num_classes=2, num_frames=16, 
                 backbone='resnet50', freeze_layers=4, 
                 temporal_fusion='mean', dropout=0.5):
        super().__init__()
        
        self.num_frames = num_frames
        self.temporal_fusion = temporal_fusion
        
        # Load pretrained ResNet
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            feat_dim = 2048
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            feat_dim = 512
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final FC layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers
        if freeze_layers > 0:
            # ResNet has 4 main blocks after initial conv
            for i, child in enumerate(self.feature_extractor.children()):
                if i < freeze_layers + 3:  # +3 for conv1, bn1, relu, pool
                    for param in child.parameters():
                        param.requires_grad = False
        
        # Temporal fusion options
        if temporal_fusion == 'lstm':
            self.temporal = nn.LSTM(feat_dim, 512, batch_first=True, num_layers=2, dropout=dropout)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
        elif temporal_fusion == 'attention':
            self.temporal_attention = nn.MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
        else:  # mean pooling
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Process each frame through ResNet
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(B * T, C, H, W)
        
        # Extract features
        features = self.feature_extractor(x)  # [B*T, feat_dim, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B*T, feat_dim]
        features = features.reshape(B, T, -1)  # [B, T, feat_dim]
        
        # Temporal fusion
        if self.temporal_fusion == 'lstm':
            lstm_out, _ = self.temporal(features)
            # Use last timestep
            output = self.classifier(lstm_out[:, -1])
        elif self.temporal_fusion == 'attention':
            attn_out, _ = self.temporal_attention(features, features, features)
            # Global average pooling
            pooled = attn_out.mean(dim=1)
            output = self.classifier(pooled)
        else:  # mean pooling
            pooled = features.mean(dim=1)  # [B, feat_dim]
            output = self.classifier(pooled)
        
        return output
    
    def get_trainable_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (videos, labels, _, _) in enumerate(pbar):
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device, desc='Val'):
    """Validation/Test"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=desc)
    #for videos, labels, _, _ in pbar:
        #videos = videos.to(device)
        #labels = labels.to(device)
    for batch in pbar:
        if len(batch) == 5:
            videos, labels, _, _, _ = batch
        else:
            videos, labels, _, _ = batch
        videos = videos.to(device)
        labels = labels.to(device)
        
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser('ResNet Video Classifier')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--nb_classes', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=224)
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--temporal_fusion', type=str, default='mean',
                       choices=['mean', 'lstm', 'attention'])
    parser.add_argument('--freeze_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # DLC ROI parameters
    parser.add_argument('--use_dlc_roi', action='store_true')
    parser.add_argument('--dlc_dir', type=str, default=None)
    parser.add_argument('--dlc_likelihood_threshold', type=float, default=0.6)
    parser.add_argument('--roi_padding', type=float, default=0.25)
    parser.add_argument('--roi_min_size', type=int, default=96)
    parser.add_argument('--roi_prob', type=float, default=1.0)
    
    # Other parameters
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = MiceClassificationDataset(
        csv_path=f"{args.data_path}/train.csv",
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        mode='train',
        use_dlc_roi=args.use_dlc_roi,
        dlc_dir=args.dlc_dir,
        dlc_likelihood_threshold=args.dlc_likelihood_threshold,
        roi_padding=args.roi_padding,
        roi_min_size=args.roi_min_size,
        roi_prob=args.roi_prob
    )
    
    test_dataset = MiceClassificationDataset(
        csv_path=f"{args.data_path}/test.csv",
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        mode='test',
        use_dlc_roi=args.use_dlc_roi,
        dlc_dir=args.dlc_dir,
        dlc_likelihood_threshold=args.dlc_likelihood_threshold,
        roi_padding=args.roi_padding,
        roi_min_size=args.roi_min_size,
        roi_prob=1.0  # Always use ROI in test
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = ResNetVideoClassifier(
        num_classes=args.nb_classes,
        num_frames=args.num_frames,
        backbone=args.backbone,
        freeze_layers=args.freeze_layers,
        temporal_fusion=args.temporal_fusion,
        dropout=args.dropout
    )
    model = model.to(device)
    
    print(f"Model: {args.backbone} with {args.temporal_fusion} fusion")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_acc = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluation only
    if args.eval_only:
        test_loss, test_acc, preds, labels = validate(
            model, test_loader, criterion, device, 'Test'
        )
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save predictions
        results = {
            'predictions': preds,
            'labels': labels,
            'accuracy': test_acc
        }
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(results, f)
        return
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(start_epoch, args.epochs):
        # Warmup
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        test_loss, test_acc, _, _ = validate(
            model, test_loader, criterion, device, 'Test'
        )
        
        # Step scheduler (after warmup)
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        # Log
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'args': args
        }
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        # Save latest model
        torch.save(checkpoint, output_dir / 'latest_model.pth')
        
        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f)
    
    print(f"Training completed. Best test accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
