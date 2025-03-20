# import os
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import io

# # 自定义 Dataset 类
# class MultiParquetDataset(Dataset):
#     def __init__(self, parquet_files, transform=None):
#         self.parquet_files = parquet_files
#         self.transform = transform
#         self.data = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image_bytes = self.data.iloc[0]['image']['bytes']
#         label = self.data.iloc[idx]['label']
#         image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

from scipy.io import loadmat
path = '/lpai/open_clip-main/src/classes/cars_annos.mat'
annotations = loadmat(path)
class_name = tuple([annotations['class_names'][0][i][0] for i in range(196)])
print(class_name[1])
print(annotations['class_names'][0][1][0])
print(annotations['annotations'][0][1][5])

# import os
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image

# # 自定义 Dataset 类
# class CustomImageDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.classes = sorted(os.listdir(data_dir))
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
#         self.image_paths = []
#         self.labels = []
#         for cls_name in self.classes:
#             cls_dir = os.path.join(data_dir, cls_name)
#             for img_name in os.listdir(cls_dir):
#                 self.image_paths.append(os.path.join(cls_dir, img_name))
#                 self.labels.append(self.class_to_idx[cls_name])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         label = self.labels[idx]
#         return image, label

# # 定义预处理
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # 加载数据集
# train_dataset = CustomImageDataset(data_dir='/lpai/volumes/so-volume-bd-ga/lhp/datasets/stanford_car/train', transform=preprocess)
# test_dataset = CustomImageDataset(data_dir='/lpai/volumes/so-volume-bd-ga/lhp/datasets/stanford_car/test', transform=preprocess)


# # 创建 DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# # 测试 DataLoader
# for images, labels in train_loader:
#     print(f"Train batch - Images shape: {images.shape}, Labels: {labels}")
#     break

# for images, labels in test_loader:
#     print(f"Test batch - Images shape: {images.shape}, Labels: {labels}")
#     break