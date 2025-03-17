# Define and train NN in TensorFlow
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
    
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

x = torch.randn(5, 10)
y = torch.randn(5, 1)

for epochs in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# TensorFlow backpropogation
import tensorflow as tf
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())  # Output: 6.0

# PyTorch Backpropogation
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()  # Computes dy/dx
print(x.grad)  # Output: tensor(4.0)

# Save and load the model
torch.save(model.state_dict(), "model.pth")
torch.load_state_dict(torch.load("model.pth"))
model.eval()

# Disable grad to speed up inference
with torch.no_grad():
    y = model(x)

# Mixed precision (FP16 where possible)
torch.cuda.amp
tf.keras.mixed_precision

# Batch Normalization (stabilize training, faster convergence)
nn.BatchNorm1d
tf.keras.layers.BatchNormalization

# Optimizations (normalization, dropout, regularization)
# learning rate scheduling
torch.optim.lr_scheduler.ReduceLROnPlateau
tf.keras.callbacks.ReduceLROnPlateau
# gradient clipping
torch.nn.utils.clip_grad_norm_()
tf.clip_by_global_norm()

# Transfer Learning
import torchvision.models as models
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze layers
model.fc = nn.Linear(2048, 10)  # Modify classifier

base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layers
model = tf.keras.Sequential([base_model, tf.keras.layers.Dense(10, activation='softmax')])
