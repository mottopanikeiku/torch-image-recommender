# Advanced Image Recommender System

A comprehensive PyTorch implementation of a sophisticated hybrid image recommendation system that combines content-based filtering, collaborative filtering, and advanced deep learning techniques.

## Features

### Core Components
- **Hybrid Architecture**: Combines content-based and collaborative filtering
- **Deep Visual Features**: Pre-trained CNN models (ResNet50, VGG16) for image feature extraction
- **Attention Mechanisms**: Multi-head attention for feature fusion
- **Neural Matrix Factorization**: Advanced collaborative filtering with neural networks
- **Cross-Modal Learning**: Fusion of visual and user interaction data

### Advanced Techniques
- **Graph Neural Networks**: User-item interaction modeling
- **Contrastive Learning**: Better representation learning with InfoNCE loss
- **Multi-Task Learning**: Simultaneous rating and purchase prediction
- **Ensemble Methods**: Multiple model fusion strategies
- **Explainable AI**: Gradient-based feature importance analysis

### Evaluation & Metrics
- Comprehensive evaluation suite (RMSE, MAE, Correlation)
- Training visualization and monitoring
- Feature importance analysis
- Recommendation diversity metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd torch-image-recommender

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run the main recommender system
python main.py
```

## Architecture Overview

```
Input: User ID + Item ID + Image
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Architecture                      │
├─────────────────────────┬───────────────────────────────────┤
│    Content-Based        │    Collaborative Filtering       │
│                         │                                   │
│  ┌─────────────────┐   │  ┌─────────────────────────────┐  │
│  │ Image Feature   │   │  │ User-Item Embeddings       │  │
│  │ Extractor       │   │  │                             │  │
│  │ - ResNet50      │   │  │ - Neural Matrix            │  │
│  │ - VGG16         │   │  │   Factorization            │  │
│  │ - Attention     │   │  │ - Bias Terms               │  │
│  └─────────────────┘   │  └─────────────────────────────┘  │
│           │             │             │                     │
│           └─────────────┼─────────────┘                     │
│                         │                                   │
│              ┌─────────────────────┐                       │
│              │ Cross-Modal         │                       │
│              │ Attention Fusion    │                       │
│              └─────────────────────┘                       │
│                         │                                   │
│              ┌─────────────────────┐                       │
│              │ Final Prediction    │                       │
│              │ Network             │                       │
│              └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
                 Final Rating Prediction
```

## Configuration

Key parameters can be configured in the `Config` class:

```python
class Config:
    # Data parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    
    # Model parameters
    EMBEDDING_DIM = 512
    HIDDEN_DIM = 256
    NUM_FACTORS = 64
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
```

## Training Process

The training pipeline includes:

1. **Data Generation**: Synthetic dataset creation with realistic patterns
2. **Feature Extraction**: Pre-trained CNN feature extraction
3. **Model Training**: Hybrid model training with early stopping
4. **Evaluation**: Comprehensive metrics calculation
5. **Visualization**: Training progress and results visualization

### Training Metrics

- **RMSE**: Root Mean Square Error for rating prediction
- **MAE**: Mean Absolute Error
- **Correlation**: Pearson correlation between predicted and actual ratings

## Advanced Components

### Graph Neural Networks
```python
from advanced_components import GraphNeuralRecommender

gnn_model = GraphNeuralRecommender(num_users=1000, num_items=500)
user_emb, item_emb = gnn_model(user_ids, item_ids)
```

### Contrastive Learning
```python
from advanced_components import ContrastiveLearningModule

contrastive = ContrastiveLearningModule(feature_dim=512)
loss = contrastive.compute_contrastive_loss(features, labels)
```

### Multi-Task Learning
```python
from advanced_components import MultiTaskRecommender

multi_task_model = MultiTaskRecommender(base_model, num_categories=10)
outputs = multi_task_model(user_ids, item_ids, images)
```

## Evaluation Results

Example performance metrics:

| Metric | Value |
|--------|-------|
| RMSE | 0.892 |
| MAE | 0.671 |
| Correlation | 0.734 |

## Visualization

The tutorial includes comprehensive visualization:

- Training progress plots
- Feature importance heatmaps
- Attention weight visualization
- Recommendation explanation plots

## Explainable AI

The system provides explanations for recommendations:

```python
from advanced_components import ExplainableRecommendations

explainer = ExplainableRecommendations(model)
importance = explainer.compute_gradient_importance(user_ids, item_ids, images)
explainer.visualize_explanations(user_id, item_id, image, importance)
```

## Project Structure

1. **Core System** (`main.py`):
   - Complete hybrid recommender implementation
   - Synthetic data generation
   - Training and evaluation pipeline
   - Model persistence and loading

2. **Configuration** (`config.py`):
   - Centralized configuration management
   - Model hyperparameters
   - Training settings

3. **Data Generation** (`data_generator.py`):
   - Synthetic dataset creation
   - Realistic user-item interaction patterns
   - Image generation based on item characteristics

4. **Models** (`collaborative_filtering.py`, `image_features.py`):
   - Matrix factorization implementation
   - CNN feature extraction
   - Neural network architectures

## Use Cases

This tutorial is perfect for:

- **Learning**: Understanding modern recommendation systems
- **Research**: Experimenting with hybrid architectures
- **Production**: Adapting components for real-world systems
- **Education**: Teaching advanced deep learning concepts

## Customization

### Adding New Models
```python
class CustomRecommender(nn.Module):
    def __init__(self, ...):
        # Your custom architecture
        pass
    
    def forward(self, user_ids, item_ids, images):
        # Your custom forward pass
        pass
```

### Custom Loss Functions
```python
def custom_loss(predictions, targets, regularization_term):
    base_loss = F.mse_loss(predictions, targets)
    return base_loss + regularization_term
```

### Custom Evaluation Metrics
```python
def custom_metric(predictions, targets):
    # Your custom evaluation logic
    return metric_value
```

## Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Batch Size**: Adjust based on available memory
3. **Data Loading**: Use multiple workers for faster data loading
4. **Mixed Precision**: Consider using automatic mixed precision for efficiency

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image resolution
2. **Slow Training**: Enable GPU acceleration and optimize data loading
3. **Poor Performance**: Tune hyperparameters or increase model capacity

### Debug Mode
```python
# Enable debugging
import torch
torch.autograd.set_detect_anomaly(True)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the tutorial documentation

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Research papers that inspired the architecture
- Open source community for valuable feedback

---

**Advanced Image Recommender System**