import os
import torch
from data_generator import SyntheticDataGenerator
from collaborative_filtering import RatingsDataset, MatrixFactorization, CollaborativeFilteringTrainer
from image_features import ImageFeatureManager
from hybrid_model import HybridDataset, HybridRecommender, HybridTrainer
from torch.utils.data import DataLoader
from config import Config
import numpy as np

def main():
    print("Image Recommender System - Commit 2: Hybrid Model with Image Features")
    print("=" * 70)
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(num_users=300, num_items=150, num_ratings=3000)
    ratings_df, images_dict, user_prefs, item_chars = generator.generate_complete_dataset()
    
    # Split data
    train_df, val_df, test_df = generator.split_data(ratings_df)
    print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Extract image features
    print("\nExtracting image features...")
    feature_manager = ImageFeatureManager(model_name='resnet50', feature_dim=Config.EMBEDDING_DIM)
    image_features = feature_manager.extract_features_from_dict(images_dict)
    
    # Print feature statistics
    stats = feature_manager.get_feature_statistics(image_features)
    print(f"Feature statistics: {stats}")
    
    # Create hybrid datasets
    train_dataset = HybridDataset(train_df, images_dict, image_features)
    val_dataset = HybridDataset(val_df, images_dict, image_features)
    test_dataset = HybridDataset(test_df, images_dict, image_features)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize hybrid model
    num_users = train_dataset.num_users
    num_items = train_dataset.num_items
    
    hybrid_model = HybridRecommender(
        num_users=num_users,
        num_items=num_items,
        cf_embedding_dim=Config.NUM_FACTORS,
        image_feature_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM
    )
    
    print(f"\nHybrid model initialized - Users: {num_users}, Items: {num_items}")
    print(f"Model parameters: {sum(p.numel() for p in hybrid_model.parameters()):,}")
    
    # Train hybrid model
    trainer = HybridTrainer(hybrid_model, train_loader, val_loader)
    trained_model = trainer.train(num_epochs=25)
    
    # Save model
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    torch.save(trained_model.state_dict(), f"{Config.MODEL_DIR}/hybrid_model.pth")
    
    print(f"\nTraining completed! Best validation loss: {trainer.best_val_loss:.4f}")
    print("Model saved to models/hybrid_model.pth")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    trained_model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []
    cf_predictions = []
    hybrid_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            user_ids = batch['user_id'].to(Config.DEVICE)
            item_ids = batch['item_id'].to(Config.DEVICE)
            image_features = batch['image_features'].to(Config.DEVICE)
            ratings = batch['rating'].to(Config.DEVICE)
            
            outputs = trained_model(user_ids, item_ids, image_features)
            loss = torch.nn.MSELoss()(outputs['prediction'], ratings)
            test_loss += loss.item()
            
            all_predictions.extend(outputs['prediction'].cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
            cf_predictions.extend(outputs['cf_prediction'].cpu().numpy())
            hybrid_predictions.extend(outputs['hybrid_prediction'].cpu().numpy())
    
    # Calculate metrics
    test_rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2))
    test_mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
    correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
    
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Correlation: {correlation:.4f}")
    
    # Analyze component contributions
    cf_rmse = np.sqrt(np.mean((np.array(cf_predictions) - np.array(all_targets)) ** 2))
    hybrid_rmse = np.sqrt(np.mean((np.array(hybrid_predictions) - np.array(all_targets)) ** 2))
    
    print(f"\nComponent Analysis:")
    print(f"CF-only RMSE: {cf_rmse:.4f}")
    print(f"Hybrid-only RMSE: {hybrid_rmse:.4f}")
    print(f"Final ensemble RMSE: {test_rmse:.4f}")
    
    with torch.no_grad():
        final_alpha = torch.sigmoid(trained_model.alpha).item()
        print(f"Final alpha (CF weight): {final_alpha:.3f}")
        print(f"Content weight: {1-final_alpha:.3f}")

if __name__ == "__main__":
    main() 