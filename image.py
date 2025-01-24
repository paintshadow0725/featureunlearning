import matplotlib.pyplot as plt

# 数据
categories = ['mine', 'only shared layers', 'task-specific layers']
values = [0.42, 0.77, ]

# 绘制柱状图
plt.bar(categories, values)

# 设置图表标题和坐标轴标签
plt.title('comparison')
plt.xlabel('类别')
plt.ylabel('值')

# 显示图形
plt.show()