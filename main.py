import os
import torch
from data_generator import SyntheticDataGenerator
from collaborative_filtering import RatingsDataset, MatrixFactorization, CollaborativeFilteringTrainer
from torch.utils.data import DataLoader
from config import Config

def main():
    print("Image Recommender System - Commit 1: Basic Collaborative Filtering")
    print("=" * 60)
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(num_users=500, num_items=200, num_ratings=5000)
    ratings_df, images_dict, user_prefs, item_chars = generator.generate_complete_dataset()
    
    # Split data
    train_df, val_df, test_df = generator.split_data(ratings_df)
    print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = RatingsDataset(train_df)
    val_dataset = RatingsDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    num_users = train_dataset.num_users
    num_items = train_dataset.num_items
    model = MatrixFactorization(num_users, num_items)
    
    print(f"Model initialized - Users: {num_users}, Items: {num_items}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = CollaborativeFilteringTrainer(model, train_loader, val_loader)
    trained_model = trainer.train(num_epochs=20)
    
    # Save model
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    torch.save(trained_model.state_dict(), f"{Config.MODEL_DIR}/collaborative_filtering.pth")
    
    print(f"\nTraining completed! Final validation loss: {trainer.val_losses[-1]:.4f}")
    print("Model saved to models/collaborative_filtering.pth")
    
    # Quick evaluation on test set
    test_dataset = RatingsDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    trained_model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            user_ids = batch['user_id'].to(Config.DEVICE)
            item_ids = batch['item_id'].to(Config.DEVICE)
            ratings = batch['rating'].to(Config.DEVICE)
            
            predictions = trained_model(user_ids, item_ids)
            loss = torch.nn.MSELoss()(predictions, ratings)
            test_loss += loss.item()
    
    test_rmse = (test_loss / len(test_loader)) ** 0.5
    print(f"Test RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    main() 