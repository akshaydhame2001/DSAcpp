# https://medium.com/@fractal.ai/guide-to-build-faster-rcnn-in-pytorch-42d47cb0ecd3
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_faster_rcnn_pytorch(num_classes):
    # Load pretrained ResNet-50
    backbone = torchvision.models.resnet50(pretrained=True)
    
    # Modify the backbone for Faster R-CNN
    # Remove the classification head and avg pool
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048
    
    # Define anchor generator for RPN
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # ROI Pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,  # Background + your classes
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model

# Training loop
def train_model(model, train_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}")

# Usage
num_classes = 21  # Example: 20 classes + background
model = create_faster_rcnn_pytorch(num_classes)

# Example data format for training
# images = [torch.rand(3, 600, 800)]
# targets = [{
#     'boxes': torch.tensor([[100, 100, 200, 200]]),
#     'labels': torch.tensor([1])
# }]