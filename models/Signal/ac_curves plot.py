# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 设置CSV文件所在的文件夹路径
# folder_path = r"D:\1Deeplearning\实验数据\logs\XJTU-10\CVS"
#
# # 获取文件夹内所有CSV文件的路径
# csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
#
# # 手动设置每个CSV文件的图例标签
# labels = [
#     'ViT',
#     'Unirormer',
#     'NAT',
#     'MSACT',
#     'MCSwin-T',
#     'Cross vit',
#     'Conformer',
#     # ...为每个CSV文件继续添加标签
# ]
#
# # 确保标签列表的长度与CSV文件列表的长度相同
# assert len(labels) == len(csv_files), "Labels list and CSV files list must have the same length."
#
# # 绘制每个CSV文件的收敛曲线
# for csv_file, label in zip(csv_files, labels):
#     # 读取CSV文件
#     data = pd.read_csv(csv_file)
#     # 假设第二列是迭代次数，第三列是收敛值
#     plt.plot(data.iloc[:, 1], data.iloc[:, 2], label=label)
#
# # 添加图例
# plt.legend()
#
# # 添加坐标轴标签和标题
# plt.xlabel('Iteration')
# plt.ylabel('Value')
# plt.title('Convergence Curves')
#
# # 显示网格
# plt.grid(True)
#
# # 显示图形
# plt.show()
#
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def exponential_moving_average(y, alpha):
    s = [y[0]]  # 初始值
    for i in range(1, len(y)):
        s.append(alpha * y[i] + (1 - alpha) * s[i-1])
    return np.array(s)

# 设置CSV文件所在的文件夹路径
folder_path = r"D:\1Deeplearning\论文\logs\XJTU-100\CVS"

# 获取文件夹内所有CSV文件的路径
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# 手动设置每个CSV文件的图例标签
labels = [
    'ViT',
    'Unirormer',
    'NAT',
    'MCSAT',
    'MCSwin-T',
    'Cross vit',
    'Conformer',
    # ...为每个CSV文件继续添加标签
]

# 确保标签列表的长度与CSV文件列表的长度相同
assert len(labels) == len(csv_files), "Labels list and CSV files list must have the same length."

# 绘制每个CSV文件的收敛曲线
for csv_file, label in zip(csv_files, labels):
    # 读取CSV文件
    data = pd.read_csv(csv_file)
    # 获取第三列数据并进行平滑处理
    y_values = data.iloc[:, 2].values
    y_smooth = exponential_moving_average(y_values, alpha=0.6)  # 设置alpha值为0.6进行平滑

    # 假设第二列是迭代次数
    x_values = data.iloc[:, 1].values
    plt.plot(x_values, y_smooth, label=label)

# 添加图例
plt.legend()

# 添加坐标轴标签和标题
plt.xlabel('Iteration')
plt.ylabel('Accuracy / %')
plt.title('training sample size: 100')

# 显示网格
plt.grid(True)

# 显示图形
plt.show()