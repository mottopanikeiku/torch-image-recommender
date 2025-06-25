import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import pickle
from config import Config

class ImagePreprocessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)

class ImageFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', feature_dim=512):
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim)
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        
        features = self.feature_head(features)
        return features

class ImageDataset(Dataset):
    def __init__(self, image_paths, item_ids, preprocessor):
        self.image_paths = image_paths
        self.item_ids = item_ids
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        item_id = self.item_ids[idx]
        
        image = self.preprocessor.preprocess_image(image_path)
        
        return {
            'image': image,
            'item_id': item_id,
            'image_path': image_path
        }

class ImageFeatureManager:
    def __init__(self, model_name='resnet50', feature_dim=512, batch_size=32):
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        
        self.device = Config.DEVICE
        self.preprocessor = ImagePreprocessor()
        self.extractor = ImageFeatureExtractor(model_name, feature_dim).to(self.device)
        self.extractor.eval()
        
        self.cache_dir = f"{Config.DATA_DIR}/features"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_cache_path(self):
        return f"{self.cache_dir}/{self.model_name}_features.pkl"
    
    def save_features(self, features):
        cache_path = self.get_cache_path()
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features cached to {cache_path}")
    
    def load_features(self):
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                features = pickle.load(f)
            print(f"Features loaded from cache: {cache_path}")
            return features
        return None
    
    def extract_features_from_dict(self, images_dict):
        cached_features = self.load_features()
        if cached_features is not None:
            return cached_features
        
        print(f"Extracting features using {self.model_name}...")
        
        item_ids = list(images_dict.keys())
        image_paths = list(images_dict.values())
        
        dataset = ImageDataset(image_paths, item_ids, self.preprocessor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        features_dict = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image'].to(self.device)
                batch_item_ids = batch['item_id']
                
                batch_features = self.extractor(images)
                batch_features = batch_features.cpu().numpy()
                
                for i, item_id in enumerate(batch_item_ids):
                    features_dict[item_id] = batch_features[i]
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {(batch_idx + 1) * self.batch_size} images...")
        
        print(f"Feature extraction completed. Extracted features for {len(features_dict)} items.")
        
        self.save_features(features_dict)
        return features_dict
    
    def get_feature_statistics(self, features_dict):
        all_features = np.array(list(features_dict.values()))
        
        stats = {
            'num_items': len(features_dict),
            'feature_dim': all_features.shape[1],
            'mean': np.mean(all_features, axis=0).mean(),
            'std': np.std(all_features, axis=0).mean(),
            'min': np.min(all_features),
            'max': np.max(all_features)
        }
        
        return stats 