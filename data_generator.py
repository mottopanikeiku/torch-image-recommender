import numpy as np
import pandas as pd
from PIL import Image
import os
import json
from sklearn.model_selection import train_test_split
from config import Config

class SyntheticDataGenerator:
    def __init__(self, num_users=1000, num_items=500, num_ratings=10000):
        self.num_users = num_users
        self.num_items = num_items
        self.num_ratings = num_ratings
        self.num_features = 5
        
    def generate_user_preferences(self):
        return np.random.rand(self.num_users, self.num_features)
    
    def generate_item_characteristics(self):
        return np.random.rand(self.num_items, self.num_features)
    
    def create_ratings_matrix(self, user_preferences, item_characteristics):
        ratings_data = []
        
        for _ in range(self.num_ratings):
            user_id = np.random.randint(0, self.num_users)
            item_id = np.random.randint(0, self.num_items)
            
            preference_score = np.dot(user_preferences[user_id], item_characteristics[item_id])
            base_rating = preference_score * 2 + 2
            
            noise = np.random.normal(0, 0.5)
            final_rating = np.clip(base_rating + noise, 1, 5)
            
            ratings_data.append({
                'user_id': f'user_{user_id}',
                'item_id': f'item_{item_id}',
                'rating': round(final_rating, 1)
            })
        
        ratings_df = pd.DataFrame(ratings_data)
        return ratings_df.drop_duplicates(subset=['user_id', 'item_id'])
    
    def generate_synthetic_images(self, item_characteristics):
        os.makedirs(Config.IMAGES_DIR, exist_ok=True)
        images_dict = {}
        
        for item_idx in range(self.num_items):
            item_id = f'item_{item_idx}'
            char = item_characteristics[item_idx]
            
            img_array = np.random.rand(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)
            
            if char[0] > 0.5:
                center = Config.IMAGE_SIZE // 2
                y, x = np.ogrid[:Config.IMAGE_SIZE, :Config.IMAGE_SIZE]
                mask = (x - center) ** 2 + (y - center) ** 2 <= (Config.IMAGE_SIZE // 4) ** 2
                img_array[mask] = [char[1], char[2], char[3]]
            
            if char[4] > 0.5:
                start_x, start_y = Config.IMAGE_SIZE // 4, Config.IMAGE_SIZE // 4
                end_x, end_y = 3 * Config.IMAGE_SIZE // 4, 3 * Config.IMAGE_SIZE // 4
                img_array[start_y:end_y, start_x:end_x] = [char[2], char[3], char[1]]
            
            img_array = (img_array * 255).astype(np.uint8)
            image = Image.fromarray(img_array)
            
            image_path = f"{Config.IMAGES_DIR}/{item_id}.jpg"
            image.save(image_path)
            images_dict[item_id] = image_path
        
        return images_dict
    
    def split_data(self, ratings_df):
        train_df, temp_df = train_test_split(ratings_df, test_size=0.4, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        return train_df, val_df, test_df
    
    def generate_complete_dataset(self):
        print("Generating synthetic dataset...")
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        
        user_preferences = self.generate_user_preferences()
        item_characteristics = self.generate_item_characteristics()
        ratings_df = self.create_ratings_matrix(user_preferences, item_characteristics)
        images_dict = self.generate_synthetic_images(item_characteristics)
        
        ratings_df.to_csv(f"{Config.DATA_DIR}/ratings.csv", index=False)
        
        with open(f"{Config.DATA_DIR}/images_dict.json", 'w') as f:
            json.dump(images_dict, f)
        
        np.save(f"{Config.DATA_DIR}/user_preferences.npy", user_preferences)
        np.save(f"{Config.DATA_DIR}/item_characteristics.npy", item_characteristics)
        
        print(f"Dataset created: {len(ratings_df)} ratings, {self.num_users} users, {self.num_items} items")
        
        return ratings_df, images_dict, user_preferences, item_characteristics 