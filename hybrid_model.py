import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import Config
from collaborative_filtering import RatingsDataset
from image_features import ImagePreprocessor

class HybridDataset(Dataset):
    def __init__(self, ratings_df, images_dict, image_features, preprocessor=None):
        self.ratings_df = ratings_df.reset_index(drop=True)
        self.images_dict = images_dict
        self.image_features = image_features
        self.preprocessor = preprocessor
        
        from sklearn.preprocessing import LabelEncoder
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        self.ratings_df['user_encoded'] = self.user_encoder.fit_transform(self.ratings_df['user_id'])
        self.ratings_df['item_encoded'] = self.item_encoder.fit_transform(self.ratings_df['item_id'])
        
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)
        
    def __len__(self):
        return len(self.ratings_df)
    
    def __getitem__(self, idx):
        row = self.ratings_df.iloc[idx]
        
        user_id = row['user_encoded']
        item_id = row['item_encoded']
        rating = row['rating']
        item_id_str = row['item_id']
        
        image_features = self.image_features.get(item_id_str, np.zeros(Config.EMBEDDING_DIM))
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'image_features': torch.tensor(image_features, dtype=torch.float),
            'rating': torch.tensor(rating, dtype=torch.float)
        }

class HybridRecommender(nn.Module):
    def __init__(self, num_users, num_items, cf_embedding_dim=64, image_feature_dim=512, hidden_dim=256):
        super(HybridRecommender, self).__init__()
        
        # Collaborative filtering components
        self.user_embedding = nn.Embedding(num_users, cf_embedding_dim)
        self.item_embedding = nn.Embedding(num_items, cf_embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Content-based components
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion components
        self.cf_projection = nn.Linear(cf_embedding_dim * 2, hidden_dim)
        
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learnable weight for combining CF and content predictions
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
    
    def forward(self, user_ids, item_ids, image_features):
        # Collaborative filtering prediction
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        user_bias = self.user_bias(user_ids).squeeze()
        item_bias = self.item_bias(item_ids).squeeze()
        
        cf_features = torch.cat([user_emb, item_emb], dim=1)
        cf_projected = self.cf_projection(cf_features)
        
        cf_prediction = (user_emb * item_emb).sum(dim=1) + user_bias + item_bias + self.global_bias
        
        # Content-based prediction
        content_features = self.image_projection(image_features)
        
        # Hybrid prediction
        fused_features = torch.cat([cf_projected, content_features], dim=1)
        hybrid_prediction = self.fusion_network(fused_features).squeeze()
        
        # Weighted combination
        alpha = torch.sigmoid(self.alpha)
        final_prediction = alpha * cf_prediction + (1 - alpha) * hybrid_prediction
        
        return {
            'prediction': final_prediction,
            'cf_prediction': cf_prediction,
            'hybrid_prediction': hybrid_prediction,
            'alpha': alpha
        }

class HybridTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            user_ids = batch['user_id'].to(Config.DEVICE)
            item_ids = batch['item_id'].to(Config.DEVICE)
            image_features = batch['image_features'].to(Config.DEVICE)
            ratings = batch['rating'].to(Config.DEVICE)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(user_ids, item_ids, image_features)
            loss = self.criterion(outputs['prediction'], ratings)
            
            # Add L2 regularization
            l2_reg = sum(p.pow(2).sum() for p in self.model.parameters())
            loss = loss + 1e-6 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                user_ids = batch['user_id'].to(Config.DEVICE)
                item_ids = batch['item_id'].to(Config.DEVICE)
                image_features = batch['image_features'].to(Config.DEVICE)
                ratings = batch['rating'].to(Config.DEVICE)
                
                outputs = self.model(user_ids, item_ids, image_features)
                loss = self.criterion(outputs['prediction'], ratings)
                
                total_loss += loss.item()
                all_predictions.extend(outputs['prediction'].cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2))
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        
        return avg_loss, rmse, mae
    
    def train(self, num_epochs=30):
        print(f"Training hybrid model for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, rmse, mae = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{Config.MODEL_DIR}/hybrid_model_best.pth")
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                
                # Print alpha value to see CF vs content balance
                with torch.no_grad():
                    alpha = torch.sigmoid(self.model.alpha).item()
                    print(f"Alpha (CF weight): {alpha:.3f}, Content weight: {1-alpha:.3f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(f"{Config.MODEL_DIR}/hybrid_model_best.pth"))
        print("Training completed! Best model loaded.")
        
        return self.model 