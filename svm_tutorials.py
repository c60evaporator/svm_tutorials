# %% NBA、NFL選手(LB)の身長体重
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df_athelete = pd.read_csv(f'./nba_nfl_1.csv')  # データ読込
sns.scatterplot(x='height', y='weight', data=df_athelete, hue='league')  # 説明変数と目的変数のデータ点の散布図をプロット
plt.xlabel('height [cm]')
plt.ylabel('weight [kg]')

# %% 線形SVM
import numpy as np
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
def label_str_to_int(y):  # 目的変数をstr型→int型に変換(plot_decision_regions用)
    label_names = list(dict.fromkeys(y[:, 0]))
    label_dict = dict(zip(label_names, range(len(label_names))))
    y_int=np.vectorize(lambda x: label_dict[x])(y)
    return y_int
def legend_int_to_str(ax, y):  # 凡例をint型→str型に変更(plot_decision_regions用)
    hans, labs = ax.get_legend_handles_labels()
    ax.legend(handles=hans, labels=list(dict.fromkeys(y[:, 0])))

X = df_athelete[['height','weight']].values  # 説明変数(身長、体重)
y = df_athelete[['league']].values  # 目的変数(種目)
stdsc = StandardScaler()  # 標準化用インスタンス
X = stdsc.fit_transform(X)  # 説明変数を標準化
y_int = label_str_to_int(y)  # 目的変数をint型に変換
model = SVC(kernel='linear', C=1000)  # 線形SVMを定義
model.fit(X, y_int)  # SVM学習を実行

ax = plot_decision_regions(X, y_int[:, 0], clf=model, zoom_factor=2) #決定境界を可視化
plt.xlabel('height [normalized]')  # x軸のラベル
plt.ylabel('weight [normalized]')  # y軸のラベル
legend_int_to_str(ax, y)  # 凡例をint型→str型に変更

# %% B) NBA、NFL選手(OF)の身長体重
df_athelete = pd.read_csv(f'./nba_nfl_2.csv')
sns.scatterplot(x='height', y='weight', data=df_athelete, hue='league')  # 説明変数と目的変数のデータ点の散布図をプロット
plt.xlabel('height')  # x軸のラベル
plt.ylabel('weight')  # y軸のラベル
# %% B) カーネルトリックのイメージ
from mpl_toolkits.mplot3d import Axes3D
def make_donut(diameter, sigma, n):  # ドーナツ作成
    r = np.vectorize(lambda x: x * sigma + diameter)(np.random.randn(n))  # 動径方向
    theta = np.vectorize(lambda x: x * 2 * np.pi)(np.random.rand(n))  # 角度方向
    x = r * np.cos(theta)  # x座標
    y = r * np.sin(theta)  # y座標
    return np.stack([x, y], 1)
plt.figure(figsize=(5,5))
d1 = make_donut(1, 1, 100)  # 内側ドーナツ
plt.scatter(d1[:, 0], d1[:, 1], color = 'blue')
d2 = make_donut(10, 1, 100)  # 外側ドーナツ
plt.scatter(d2[:, 0], d2[:, 1], color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
def add_x2_plus_y2(data): #x^2+y^2軸を追加
    x2_plus_y2 = data[:, 0] ** 2 + data[:, 1] ** 2
    return np.insert(data, data.shape[1], x2_plus_y2, axis=1)
d1_conv = add_x2_plus_y2(d1)  # 座標変換（内側ドーナツ）
d2_conv = add_x2_plus_y2(d2)  # 座標変換（外側ドーナツ）
fig = plt.figure()
ax = Axes3D(fig)  # 3次元プロット用
ax.scatter3D(d1_conv[:, 0], d1_conv[:, 1], d1_conv[:, 2], color = 'blue')
ax.scatter3D(d2_conv[:, 0], d2_conv[:, 1], d2_conv[:, 2], color = 'red')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('x^2+y^2')
ax.view_init(elev=0, azim=30)  # 3次元グラフの向き調整

# %% B) RBFカーネルのgammaを変更
X = df_athelete[['height','weight']].values  # 説明変数(身長、体重)
y = df_athelete[['league']].values  # 目的変数(種目)
stdsc = StandardScaler()  # 標準化用インスタンス
X = stdsc.fit_transform(X)  # 説明変数を標準化
y_int = label_str_to_int(y)
for gamma in [10, 1, 0.1, 0.01]:  # gammaを変えてループ
    model = SVC(kernel='rbf', gamma=gamma)  # RBFカーネルのSVMをgammaを変えて定義
    model.fit(X, y_int)  # SVM学習を実行
    ax = plot_decision_regions(X, y_int[:, 0], clf=model, zoom_factor=2)
    plt.xlabel('height [normalized]')
    plt.ylabel('weight [normalized]')
    legend_int_to_str(ax, y)
    plt.text(np.amax(X[:, 0]), np.amin(X[:, 1]), f'gamma={model.gamma}, C={model.C}', verticalalignment='bottom', horizontalalignment='right')  # gammaとCを表示
    plt.show()
# %% C) Cを変更
for C in [10, 1, 0.1]:  # Cを変えてループ
    model = SVC(kernel='rbf', gamma=1, C=C)  # RBFカーネルのSVMをCを変えて定義
    model.fit(X, y_int)  # SVM学習を実行
    ax = plot_decision_regions(X, y_int[:, 0], clf=model, zoom_factor=2) 
    plt.xlabel('height [normalized]')
    plt.ylabel('weight [normalized]')
    legend_int_to_str(ax, y)
    plt.text(np.amax(X[:, 0]), np.amin(X[:, 1]), f'gamma={model.gamma}, C={model.C}', verticalalignment='bottom', horizontalalignment='right')  # gammaとCを表示
    plt.show()
# %% 修正) 標準化の例
stdsc = StandardScaler()  # 標準化用インスタンス
norm = stdsc.fit_transform(df_athelete[['height', 'weight']].values)  # 標準化
df_norm = df_athelete.copy()
df_norm['height [normalized]'] = pd.Series(norm[:, 0])
df_norm['weight [normalized]'] = pd.Series(norm[:, 1])
sns.scatterplot(x='height [normalized]', y='weight [normalized]', data=df_norm, hue='league')
# %%
