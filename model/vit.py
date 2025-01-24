from transformers import ViTImageProcessor, ViTForImageClassification

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 获取最后的FC层的输入特征维度
last_layer_in_features = model.classifier.in_features

# 修改最后的FC层为多任务的三个FC层
model.classifier = nn.Sequential(
    nn.Linear(last_layer_in_features, num_classes1),  # 第一个任务的类别数量
    nn.Linear(num_classes1, num_classes2),  # 第二个任务的类别数量
    nn.Linear(num_classes2, num_classes3)  # 第三个任务的类别数量
)