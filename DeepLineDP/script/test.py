import numpy as np

# 示例的 hit_count 数组
hit_count = np.array([10, 5, 8, 3, 12, 6, 9])

# 示例的 line_threshold 值
line_threshold = 0.5

# 执行排序和截取操作
sorted_index = np.argsort(hit_count).tolist()[::-1][:int(len(hit_count) * line_threshold)]

# 输出结果
print(sorted_index)