import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import Config

class RatingsDataset(Dataset):
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df.reset_index(drop=True)
        
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
        return {
            'user_id': torch.tensor(row['user_encoded'], dtype=torch.long),
            'item_id': torch.tensor(row['item_encoded'], dtype=torch.long),
            'rating': torch.tensor(row['rating'], dtype=torch.float)
        }

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=Config.NUM_FACTORS):
        super(MatrixFactorization, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        user_bias = self.user_bias(user_ids).squeeze()
        item_bias = self.item_bias(item_ids).squeeze()
        
        dot_product = (user_emb * item_emb).sum(dim=1)
        prediction = dot_product + user_bias + item_bias + self.global_bias
        
        return prediction

class CollaborativeFilteringTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            user_ids = batch['user_id'].to(Config.DEVICE)
            item_ids = batch['item_id'].to(Config.DEVICE)
            ratings = batch['rating'].to(Config.DEVICE)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                user_ids = batch['user_id'].to(Config.DEVICE)
                item_ids = batch['item_id'].to(Config.DEVICE)
                ratings = batch['rating'].to(Config.DEVICE)
                
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs=20):
        print(f"Training collaborative filtering model for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return self.model 