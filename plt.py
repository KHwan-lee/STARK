import matplotlib.pyplot as plt
from matplotlib import rc

# 한글 폰트 설정
rc('font', family='Malgun Gothic')  # Windows
# rc('font', family='AppleGothic')  # MacOS

# 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# Data for performance
# layer_nums = [2, 5, 10, 20]
# mrrs_base = [0.222, 0.223, 0.222, 0.223]
# mrrs_change = [0.223, 0.222, 0.223, 0.224]
# mrrs_cen =[0.221, 0.216, 0.214, 0.215]
# mrrs_dis =[0.223, 0.223, 0.222, 0.224]

layer_nums = [2, 5, 10, 20]
mrrs_base = [0.159, 0.159, 0.159, 0.159]
mrrs_change = [0.162, 0.161, 0.161, 0.161]
mrrs_cen =[0.159, 0.148, 0.136, 0.144]
mrrs_dis =[0.161, 0.161, 0.161, 0.161]

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(layer_nums, mrrs_base, marker='o', color='blue', linestyle='-', linewidth=2, markersize=8, label='거리-중심성')
plt.plot(layer_nums, mrrs_change, marker='s', color='green', linestyle='-', linewidth=2, markersize=8, label='중심성-거리')
plt.plot(layer_nums, mrrs_cen, marker='^', color='red', linestyle='-', linewidth=2, markersize=8, label='중심성')
plt.plot(layer_nums, mrrs_dis, marker='D', color='purple', linestyle='-', linewidth=2, markersize=8, label='거리')

# Graph settings

# plt.title("FB-CKGE", fontsize=16, fontweight='bold')
# plt.yticks([0.214, 0.218, 0.222])
# plt.ylim(0.210, 0.226)

plt.title("WN-CKGE", fontsize=16, fontweight='bold')
plt.yticks([0.14, 0.15, 0.16])
plt.ylim(0.13, 0.17)

plt.xlabel("레이어 개수", fontsize=14)
plt.ylabel("평균 역순위 (MRR)", fontsize=14)
plt.xticks(layer_nums)
plt.grid(axis='y')  

plt.legend(loc='upper center', fontsize=12, title='레이어링 전략', title_fontsize='13', bbox_to_anchor=(0.5, -0.2), ncol=4)

# Save and show the plot with tight layout
plt.tight_layout()


# Save and show the plot
plt.show()
