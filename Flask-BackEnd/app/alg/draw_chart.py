import os.path
import matplotlib.pyplot as plt
import numpy as np

data_name = '2024_04_17#18_33_20_0_all'
#data_name2 = '2024_02_29#20_20_43'

plt.figure()
plt.title('Epoch_ave = 60', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Value', fontsize=14)

y1 = np.loadtxt(os.path.join('chart_data', data_name + '.txt'))
#y2 = np.loadtxt(os.path.join('chart_data', data_name2 + '.txt'))
x = np.arange(y1.shape[1])

y1_loss, y1_acc, y1_auc = y1[0], y1[1], y1[2]
#y2_loss, y2_acc, y2_auc = y2[0], y2[1], y2[2]

plt.plot(x, y1_loss, label='Loss')
plt.plot(x, y1_acc, label='Acc')
plt.plot(x, y1_auc, label='AUC')

plt.legend(loc='upper left', fontsize=8, handlelength=0.5, handleheight=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig(os.path.join('chart_img', 'a' + '.jpg'))
plt.show()
