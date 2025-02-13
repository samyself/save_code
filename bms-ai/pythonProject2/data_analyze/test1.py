import matplotlib.pyplot as plt


# 数据
data = [
    (10.0, 20.0, 75861),
    (20.0, 30.0, 191125),
    (30.0, 40.0, 8714904),
    (40.0, 50.0, 183364),
    (50.0, 60.0, 9779901),
    (60.0, 70.0, 38464),
    (70.0, 80.0, 3909),
    (80.0, 90.0, 4226),
    (90.0, 100.0, 61875),
    (100.0, 110.0, 73),
    (110.0, 120.0, 12),
    (120.0, 130.0, 0),
    (130.0, 140.0, 1),
    (140.0, 150.0, 1),
]

# data = [
#     (0.0, 1.0, 50207965),
#     (1.0, 2.0, 7182650),
#     (2.0, 3.0, 977852),
#     (3.0, 4.0, 144471),
#     (4.0, 5.0, 27605),
#     (5.0, 6.0, 7294),
#     (6.0, 7.0, 1868),
#     (7.0, 8.0, 429),
#     (8.0, 9.0, 303),
#     (9.0, 10.0, 30),
#
# ]

# 提取边界和频率
boundaries, frequencies = zip(*[(b[0], b[2]) for b in data])

# 计算每个区间的左边界
left_edges = [b[0] for b in data]

# 设置宽度（区间宽度）
widths = [b[1] - b[0] for b in data]

# 绘制直方图
plt.bar(left_edges, frequencies, width=widths, align='edge', edgecolor='black')

# 设置图表标题和坐标轴标签
plt.title('Frequency Distribution Histogram')
plt.xlabel('Value Range')
plt.ylabel('Frequency')

# 显示图形
plt.show()
