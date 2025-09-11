#!/usr/bin/env python
"""
SVM-based video classifier using pretrained CNN features
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from mice_dataset import MiceClassificationDataset


class FeatureExtractor:
    """Extract features using pretrained CNN models"""
    
    def __init__(self, model_name='resnet18', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load pretrained model
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feat_dim = 512
        elif model_name == 'resnet34':
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.feat_dim = 512
        elif model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feat_dim = 2048
        elif model_name == 'mobilenet':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.feat_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove the classification head
        if 'resnet' in model_name:
            self.model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'mobilenet':
            self.model = nn.Sequential(*list(model.children())[:-1])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Feature extractor: {model_name} (output dim: {self.feat_dim})")
    
    @torch.no_grad()
    def extract_features(self, dataloader, augment=False):
        """
        Extract features from videos
        
        Args:
            dataloader: PyTorch dataloader
            augment: If True, extract features from multiple temporal windows
        
        Returns:
            features: numpy array of shape (n_samples, feat_dim)
            labels: numpy array of shape (n_samples,)
        """
        all_features = []
        all_labels = []
        
        print(f"Extracting features from {len(dataloader.dataset)} videos...")
        
        for batch in tqdm(dataloader):
            # Handle different batch formats
            if len(batch) == 5:
                videos, labels, _, _, _ = batch
            else:
                videos, labels, _, _ = batch
            
            videos = videos.to(self.device)
            B, C, T, H, W = videos.shape
            
            if augment:
                # Extract features from multiple temporal windows
                features_list = []
                
                # Full sequence
                frames = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                feat = self.model(frames)
                feat = feat.reshape(B, T, -1).mean(dim=1)
                features_list.append(feat)
                
                # First half
                frames_first = videos[:, :, :T//2].permute(0, 2, 1, 3, 4).reshape(B * (T//2), C, H, W)
                feat_first = self.model(frames_first)
                feat_first = feat_first.reshape(B, T//2, -1).mean(dim=1)
                features_list.append(feat_first)
                
                # Second half
                frames_second = videos[:, :, T//2:].permute(0, 2, 1, 3, 4).reshape(B * (T//2), C, H, W)
                feat_second = self.model(frames_second)
                feat_second = feat_second.reshape(B, T//2, -1).mean(dim=1)
                features_list.append(feat_second)
                
                # Concatenate all features
                features = torch.cat(features_list, dim=1)
                
            else:
                # Standard feature extraction
                # Process all frames
                frames = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                features = self.model(frames)
                
                # Average pool over time
                features = features.reshape(B, T, -1)
                
                # Try different pooling strategies
                mean_pool = features.mean(dim=1)
                max_pool, _ = features.max(dim=1)
                std_pool = features.std(dim=1)
                
                # Concatenate different pooling methods
                features = torch.cat([mean_pool, max_pool, std_pool], dim=1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
        
        features = np.vstack(all_features)
        labels = np.hstack(all_labels)
        
        print(f"Extracted features shape: {features.shape}")
        return features, labels


def train_svm_classifier(X_train, y_train, X_test, y_test, cv_folds=5):
    """
    Train SVM with cross-validation and grid search
    """
    print("\n" + "="*50)
    print("Training SVM Classifier")
    print("="*50)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid search for best hyperparameters
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
    
    svm = SVC(probability=True, random_state=42)
    
    # Use StratifiedKFold for small datasets
    cv = StratifiedKFold(n_splits=min(cv_folds, np.min(np.bincount(y_train))), 
                         shuffle=True, random_state=42)
    
    print(f"Running grid search with {cv.n_splits}-fold cross-validation...")
    grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Cross-validation scores on training data
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Test set performance
    test_pred = best_model.predict(X_test_scaled)
    test_acc = (test_pred == y_test).mean()
    
    print(f"Test accuracy: {test_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))
    
    return best_model, scaler, test_acc


def train_ensemble(X_train, y_train, X_test, y_test):
    """
    Train ensemble of different classifiers
    """
    print("\n" + "="*50)
    print("Training Ensemble Classifiers")
    print("="*50)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = {
        'SVM (Linear)': SVC(kernel='linear', C=1.0, probability=True),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    predictions = []
    
    for name, clf in classifiers.items():
        # Train
        clf.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y_train))), 
                            shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv)
        
        # Test
        test_pred = clf.predict(X_test_scaled)
        test_proba = clf.predict_proba(X_test_scaled)
        test_acc = (test_pred == y_test).mean()
        
        predictions.append(test_proba[:, 1])
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_acc': test_acc
        }
        
        print(f"\n{name}:")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  Test Accuracy: {test_acc:.3f}")
    
    # Ensemble prediction (average)
    ensemble_proba = np.mean(predictions, axis=0)
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    ensemble_acc = (ensemble_pred == y_test).mean()
    
    print(f"\nEnsemble (Average):")
    print(f"  Test Accuracy: {ensemble_acc:.3f}")
    
    return results, ensemble_acc


def main():
    parser = argparse.ArgumentParser('SVM Video Classifier with Pretrained Features')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--nb_classes', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=224)
    
    # Feature extraction
    parser.add_argument('--feature_model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet'])
    parser.add_argument('--augment_features', action='store_true',
                       help='Extract features from multiple temporal windows')
    
    # DLC ROI parameters
    parser.add_argument('--use_dlc_roi', action='store_true')
    parser.add_argument('--dlc_dir', type=str, default=None)
    parser.add_argument('--dlc_likelihood_threshold', type=float, default=0.6)
    parser.add_argument('--roi_padding', type=float, default=0.25)
    parser.add_argument('--roi_min_size', type=int, default=128)
    
    # Other parameters
    parser.add_argument('--output_dir', type=str, default='svm_output')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_ensemble', action='store_true')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
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
        roi_prob=1.0
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
        roi_prob=1.0
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    # Extract features
    extractor = FeatureExtractor(args.feature_model, args.device)
    
    X_train, y_train = extractor.extract_features(train_loader, augment=args.augment_features)
    X_test, y_test = extractor.extract_features(test_loader, augment=False)
    
    # Train classifiers
    if args.use_ensemble:
        results, ensemble_acc = train_ensemble(X_train, y_train, X_test, y_test)
        
        # Save results
        with open(output_dir / 'ensemble_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nFinal ensemble accuracy: {ensemble_acc:.3f}")
        
    else:
        model, scaler, test_acc = train_svm_classifier(X_train, y_train, X_test, y_test)
        
        # Save model
        joblib.dump(model, output_dir / 'svm_model.pkl')
        joblib.dump(scaler, output_dir / 'scaler.pkl')
        
        print(f"\nModel saved to {output_dir}")
        print(f"Final test accuracy: {test_acc:.3f}")
    
    # Save feature extractor info
    config = {
        'feature_model': args.feature_model,
        'num_frames': args.num_frames,
        'input_size': args.input_size,
        'use_dlc_roi': args.use_dlc_roi,
        'feature_dim': X_train.shape[1],
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test)
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
