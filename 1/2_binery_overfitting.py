

# -----------------------------
# 1. 极小数据集：4个样本，2维输入，2分类
# -----------------------------
import numpy as np
np.random.seed(0)

N_train = 10
N_test = 100
useful_dim = 2
noise_dim = 98  # 引入大量毫无意义的噪音特征
in_dim = useful_dim + noise_dim
num_classes = 2

# 1. 构造训练集
# 先生成有用的特征和标签
X_train_useful = np.random.randn(N_train, useful_dim)
y_train = (X_train_useful[:, 0] + X_train_useful[:, 1] > 0).astype(np.int64) 
# 加入纯随机的噪音特征
X_train_noise = np.random.randn(N_train, noise_dim)
X_train = np.hstack([X_train_useful, X_train_noise]) # 拼接起来，维度变成 (10, 100)

Y_train = np.zeros((N_train, num_classes), dtype=np.float64)
Y_train[np.arange(N_train), y_train] = 1.0

# 2. 构造测试集 (同理，但样本量更大)
X_test_useful = np.random.randn(N_test, useful_dim)
y_test = (X_test_useful[:, 0] + X_test_useful[:, 1] > 0).astype(np.int64)
X_test_noise = np.random.randn(N_test, noise_dim)
X_test = np.hstack([X_test_useful, X_test_noise])

# -----------------------------
# 2. 初始化参数
#    线性分类器: logits = X @ W + b
# -----------------------------
W = 0.01 * np.random.randn(in_dim, num_classes)
b = np.zeros((1, num_classes), dtype=np.float64)

lr = 0.5
epochs = 1000


# -----------------------------
# 3. 基础函数
# -----------------------------
def softmax(logits):
    # 数值稳定版 softmax
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy(probs, Y_true):
    # 平均交叉熵
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(Y_true * np.log(probs)) / Y_true.shape[0]


def accuracy(probs, y_true):
    pred = np.argmax(probs, axis=1)
    return np.mean(pred == y_true)


# -----------------------------
# 4. 训练
# -----------------------------
for epoch in range(epochs):
    # --- 训练集前向传播 ---
    logits_train = X_train @ W + b
    probs_train = softmax(logits_train)
    loss_train = cross_entropy(probs_train, Y_train)
    acc_train = accuracy(probs_train, y_train)

    # --- 反向传播 (仅用训练集更新参数) ---
    dlogits = (probs_train - Y_train) / N_train
    dW = X_train.T @ dlogits
    db = np.sum(dlogits, axis=0, keepdims=True)
    
    W -= lr * dW
    b -= lr * db

    # --- 测试集前向传播 (不参与反向传播，只做观察) ---
    if epoch % 100 == 0 or epoch == epochs - 1:
        logits_test = X_test @ W + b
        probs_test = softmax(logits_test)
        acc_test = accuracy(probs_test, y_test)
        
        print(f"epoch={epoch:4d} | Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.2f} | Test Acc: {acc_test:.2f}")

# -----------------------------
# 5. 训练后查看结果
# -----------------------------
# logits = X_test @ W + b
# probs = softmax(logits)
# pred = np.argmax(probs, axis=1)

# print("\nW =\n", W)
# print("\nb =\n", b)
# print("\nfinal logits =\n", logits)
# print("\nfinal probs =\n", probs)
# print("\npred =", pred)
# print("true =", y)