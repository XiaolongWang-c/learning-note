import numpy as np

# 固定随机种子，方便复现
np.random.seed(0)

# -----------------------------
# 1. 极小数据集：4个样本，2维输入，2分类
# -----------------------------
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=np.float64)

y = np.array([0, 0, 1, 1], dtype=np.int64)

N = X.shape[0]
in_dim = X.shape[1]
num_classes = 2

# one-hot 标签
Y = np.zeros((N, num_classes), dtype=np.float64)
Y[np.arange(N), y] = 1.0

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
    # forward
    logits = X @ W + b              # (N, 2)
    probs = softmax(logits)         # (N, 2)
    loss = cross_entropy(probs, Y)
    acc = accuracy(probs, y)

    # backward
    # softmax + cross entropy 的梯度:
    # dL/dlogits = (probs - Y) / N
    dlogits = (probs - Y) / N

    # 链式法则
    dW = X.T @ dlogits              # (2, N) @ (N, 2) -> (2, 2)
    db = np.sum(dlogits, axis=0, keepdims=True)

    # 参数更新
    W -= lr * dW
    b -= lr * db

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"epoch={epoch:4d}, loss={loss:.6f}, acc={acc:.3f}")

# -----------------------------
# 5. 训练后查看结果
# -----------------------------
logits = X @ W + b
probs = softmax(logits)
pred = np.argmax(probs, axis=1)

print("\nW =\n", W)
print("\nb =\n", b)
print("\nfinal logits =\n", logits)
print("\nfinal probs =\n", probs)
print("\npred =", pred)
print("true =", y)