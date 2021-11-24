import matplotlib.pyplot  as plt
import numpy as np

D_MAE = np.array([0.7238373329786405, 0.5339490494460141, 0.8019125830382109, 1.3048780798912047, 1.3806452107429505, 0,
                  3.0158731738726297, 0, 0, 0, 0, 0, 2.424242377281189])
D_OBO = np.array([0.0136986301369863, 0.0963302752293578, 0.125, 0.1, 0.24, 0, 0.0, 0, 0, 0, 0, 0, 0.2222222222222222])
x = []
for i in range(13):
    s = i * 5
    e = s + 5
    if i != 12:
        st = str(s) + '-' + str(e)
    else:
        st = "60+"
    x.append(st)
fig = plt.figure()
ax = fig.add_subplot(121)
plt.barh(x, D_MAE)
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
plt.xlabel('MAE')
plt.title('MAE in different duration')

ax = fig.add_subplot(122)
plt.barh(x, D_OBO)
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
plt.xlabel('OBO')
plt.title('OBO in different duration')

# plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.show()
plt.savefig('test_transRAC_dataB without training of features.png')
# print(np.mean(total_MAE))
# print(np.mean(total_OBO))
