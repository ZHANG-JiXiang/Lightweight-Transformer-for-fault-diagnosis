import torch
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split


def dataloader(batch_size, workers=2):
    # 使用原始字符串来处理文件路径
    file_path1 = r"D:\1硕士资料\CNN -Tranformer1.0\附件一（训练集）.xlsx"
    data1 = pd.read_excel(file_path1, header=0)  # 从第一行开始读取并将其视为标题
    file_path2 = r"D:\1硕士资料\CNN -Tranformer1.0\附件二（测试集）.xlsx"
    data2 = pd.read_excel(file_path2, header=0)  # 同上

    # 获取特征列（第五列到1029列）和标签列（第四列）
    X = data1.iloc[:, 4:1028].values  # 第五列到1029列作为特征列
    y = data1.iloc[:, 3].values  # 第四列作为标签列
    X1 = data2.iloc[:, 4:1028].values  # 从测试集中获取特征列
    print(X.shape)
    print(y.shape)
    print(X1.shape)
    # 按比例划分训练集和测试集，test_size表示测试集所占比例
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将 numpy 数组转换为 PyTorch 张量 (Tensor)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 使用 long 类型
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    X1_tensor = torch.tensor(X1, dtype=torch.float32)

    # 创建一个标签数组，全部赋值为 -1
    y_test_tensor_negative = torch.full((X1_tensor.size(0),), -1, dtype=torch.long)

    # 创建 DataLoader
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(X_train_tensor, y_train_tensor),
                                   batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(X_test_tensor, y_test_tensor),
                                 batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(X1_tensor, y_test_tensor_negative),
                                  batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False)
    return train_loader, val_loader, test_loader


    # 参数



batch_size = 64
# 加载数据
train_loader, val_loader, test_loader = dataloader(batch_size)
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))