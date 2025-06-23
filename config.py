import torch

class Config:
    # Data parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Model parameters
    EMBEDDING_DIM = 512
    HIDDEN_DIM = 256
    NUM_FACTORS = 64
    DROPOUT_RATE = 0.3
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    DATA_DIR = 'data'
    MODEL_DIR = 'models'
    IMAGES_DIR = 'data/images' 