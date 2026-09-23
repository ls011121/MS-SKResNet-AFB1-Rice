import os
import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy.signal import savgol_filter
import pywt
from tqdm import tqdm

from MS_SKResNet import SpectralModel4ClassWithSKLite_V3


# =======================
# Matplotlib font
# =======================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


# =======================
# Device
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📟 使用设备: {device}")


# =======================
# Seed
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# =======================
# Label mapping
# =======================
def value_to_category(value):
    """将连续值转换为分类标签"""
    if 1 <= value < 5:
        return 0  # 1~5
    elif 5 <= value < 10:
        return 1  # 5~10
    elif 10 <= value < 20:
        return 2  # 10~20
    elif 20 <= value <= 40:
        return 3  # 20~40
    else:
        return -1


# =======================
# Data augmentation
# =======================
def spectral_scaling(X, y, num_augments=5, scale_range=(0.9, 1.1)):

    augmented_X, augmented_y = [], []
    for i in range(len(X)):
        for _ in range(num_augments):
            scale = np.random.uniform(*scale_range)
            augmented_X.append(X[i] * scale)
            augmented_y.append(y[i])
    return np.asarray(augmented_X, dtype=X.dtype), np.asarray(augmented_y, dtype=y.dtype)


# =======================
# Preprocessor
# =======================
class SpectralPreprocessor:
    def __init__(self, method=None, wavelength=None, use_standardize=False, **kwargs):
        self.method = method.lower() if method else None
        self.wavelength = wavelength
        self.params = kwargs
        self.use_standardize = use_standardize
        self.scaler = None
        self.msc_reference = None

    def fit(self, data):

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = np.asarray(data)
        if self.method == 'msc':
            self.msc_reference = np.mean(data, axis=0)
        self.scaler = None
        if self.use_standardize:
            self.scaler = StandardScaler().fit(self.apply(data))
        return self

    def apply(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if self.method is not None:
            if self.method == "sg":
                data = savgol_filter(data, self.params.get("w", 11), self.params.get("p", 3), axis=1)

            elif self.method == "msc":
                if self.msc_reference is None:
                    raise RuntimeError('MSC必须先用真实训练集调用fit。')
                ref = self.msc_reference
                result = []
                for x in data:
                    a, b = np.polyfit(ref, x, 1)
                    if not np.isfinite(a) or abs(a) < 1e-12:
                        raise ValueError('MSC斜率接近零，无法校正该光谱。')
                    corrected = (x - b) / a
                    result.append(corrected)
                data = np.array(result)

            elif self.method == "snv":
                mean = np.mean(data, axis=1, keepdims=True)
                std = np.std(data, axis=1, keepdims=True)
                data = (data - mean) / (std + 1e-8)

            elif self.method == "normalize":
                data = (data - np.min(data, axis=1, keepdims=True)) / (np.ptp(data, axis=1, keepdims=True) + 1e-8)

            elif self.method == "standardize":
                data = preprocessing.scale(data, axis=1)

            elif self.method == "d1":
                if self.wavelength is None:
                    raise ValueError("d1 方法需要提供 wavelength 参数")
                data = np.gradient(data, self.wavelength, axis=1)

            elif self.method == "d2":
                if self.wavelength is None:
                    raise ValueError("d2 方法需要提供 wavelength 参数")
                data = np.gradient(np.gradient(data, self.wavelength, axis=1), self.wavelength, axis=1)

            elif self.method == "minmax":
                data = preprocessing.MinMaxScaler().fit_transform(data.T).T

            elif self.method == "wavelet":
                def wave_one(x):
                    w = pywt.Wavelet('db8')
                    maxlev = pywt.dwt_max_level(len(x), w.dec_len)
                    coeffs = pywt.wavedec(x, 'db8', level=maxlev)
                    threshold = 0.04
                    for i in range(1, len(coeffs)):
                        coeffs[i] = pywt.threshold(coeffs[i], threshold * np.max(np.abs(coeffs[i])))
                    x_rec = pywt.waverec(coeffs, 'db8')
                    return x_rec[:len(x)]
                data = np.array([wave_one(row) for row in data])

            else:
                raise ValueError(f"未知预处理方法: {self.method}")

        if self.use_standardize and self.scaler is not None:
            data = self.scaler.transform(data)

        return data


# =======================
# Dataset
# =======================
class SpectralDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file, header=None, dtype=float)

        wavelengths = self.df.iloc[0, 1:].values
        data = self.df.iloc[1:, :].values

        print("数据形状（去除波长行后）:", data.shape)

        continuous_labels = data[:, 0]
        labels = np.array([value_to_category(val) for val in continuous_labels], dtype=int)
        features = data[:, 1:].astype(np.float32)

        valid_mask = labels != -1
        if valid_mask.sum() != len(labels):
            raise ValueError("发现无效标签，请先检查CSV；不自动删除独立样本。")
        labels = labels[valid_mask]
        features = features[valid_mask]

        # 保留CSV第一列的真实浓度，与features、labels逐行对应。
        self.continuous_labels = continuous_labels[valid_mask].copy()
        self.sample_ids = np.arange(1, len(continuous_labels) + 1)[valid_mask]
        # CSV第1行为波长，首个样本在第2行；这里的行号从1开始。
        self.csv_row_numbers = self.sample_ids + 1
        self.labels = torch.LongTensor(labels)
        self.features = torch.FloatTensor(features)
        self.wavelengths = wavelengths
        self.transform = transform

        unique, counts = np.unique(labels, return_counts=True)
        print("原始类别分布:")
        for cls, cnt in zip(unique, counts):
            print(f"  类别 {cls}: {cnt} 个样本")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.transform:
            x = self.transform.apply(x.numpy().reshape(1, -1))[0]
            x = torch.FloatTensor(x)
        y = self.labels[idx]
        return x, y


# =======================
# 先划分真实样本，再仅增强训练集（单组训练）
# =======================
def split_real_indices(y, test_size=80, val_size=0.2, random_state=42):
    if len(y) != 264:
        raise ValueError(f'应有264个真实独立样本，实际为{len(y)}个，请检查CSV。')
    indices = np.arange(len(y))
    train_val_ids, test_ids = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size, stratify=y[train_val_ids],
        random_state=random_state + 1
    )
    parts = [np.sort(ids) for ids in (train_ids, val_ids, test_ids)]
    assert len(np.unique(np.concatenate(parts))) == len(y)
    for ids in parts:
        if not np.array_equal(np.unique(y[ids]), np.arange(4)):
            raise ValueError('训练、验证、测试集都必须包含全部四类样本。')
    return tuple(parts)


def get_data_loaders(csv_file, batch_size=16, test_size=80,
                     val_size=0.2, random_state=42, transform=None):
    base_dataset = SpectralDataset(csv_file, transform=None)
    all_X = base_dataset.features.numpy()
    all_y = base_dataset.labels.numpy()
    print(f"\n📦 原始数据量: {all_X.shape[0]} 样本, {all_X.shape[1]} 波段")
    print(f"📦 类别数量: {len(np.unique(all_y))}")

    train_ids, val_ids, test_ids = split_real_indices(all_y, test_size, val_size, random_state)
    train_X, train_y = all_X[train_ids], all_y[train_ids]
    val_X, val_y = all_X[val_ids], all_y[val_ids]
    test_X, test_y = all_X[test_ids], all_y[test_ids]
    # 使用与测试光谱完全相同的索引；后续测试DataLoader始终shuffle=False。
    test_metadata = pd.DataFrame({
        'test_order': np.arange(1, len(test_ids) + 1),
        'sample_id': base_dataset.sample_ids[test_ids],
        'csv_row_number': base_dataset.csv_row_numbers[test_ids],
        'AFB1_concentration': base_dataset.continuous_labels[test_ids],
        'true_class': test_y,
    })
    print('\n' + '=' * 60)
    print('📊 使用分层随机划分（真实独立样本）')
    for name, labels in [('训练集', train_y), ('验证集', val_y), ('测试集', test_y)]:
        print(f'{name}样本数: {len(labels)}')
        print(f'{name}类别分布:', dict(zip(*np.unique(labels, return_counts=True))))
    print('=' * 60)

    # 保留原附件的处理顺序：原始光谱缩放增强 → 光谱预处理 → StandardScaler。
    # 仅训练集增强；参考谱及Scaler只由真实、未增强训练样本拟合。
    aug_X, aug_y = spectral_scaling(train_X, train_y, num_augments=5, scale_range=(0.9, 1.1))
    train_full_X = np.concatenate([train_X, aug_X], axis=0)
    train_full_y = np.concatenate([train_y, aug_y], axis=0)
    print(f"✅ 训练集增强后样本数: {len(train_full_X)}（不添加高斯噪声）")
    if transform:
        transform.fit(train_X)
        train_full_X = transform.apply(train_full_X)
        train_X = transform.apply(train_X)
        val_X = transform.apply(val_X)
        test_X = transform.apply(test_X)

    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.FloatTensor(X).unsqueeze(1), torch.LongTensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (make_loader(train_full_X, train_full_y, True), make_loader(val_X, val_y),
            make_loader(test_X, test_y), make_loader(train_X, train_y), test_metadata)


# =======================
# StandardScaler只在未增强的真实训练集上拟合
# =======================
def standardize_features(train_loader, val_loader, test_loader, original_train_loader):
    original_X = original_train_loader.dataset.tensors[0].numpy().squeeze(1)
    scaler = StandardScaler().fit(original_X)

    def transform_loader(loader, shuffle=False):
        X, y = loader.dataset.tensors
        Xs = scaler.transform(X.numpy().squeeze(1))
        ds = TensorDataset(torch.FloatTensor(Xs).unsqueeze(1), y.clone())
        return DataLoader(ds, batch_size=loader.batch_size, shuffle=shuffle)

    return (transform_loader(train_loader, True), transform_loader(val_loader),
            transform_loader(test_loader), transform_loader(original_train_loader), scaler)


# =======================
# Evaluate (macro + per-class)
# =======================
def macro_classification_report(*args, **kwargs):
    """沿用sklearn报告的每类指标，仅移除weighted avg行。"""
    report = classification_report(*args, **kwargs)
    if isinstance(report, dict):
        report.pop('weighted avg', None)
        return report
    return '\n'.join(line for line in report.split('\n')
                     if not line.strip().startswith('weighted avg'))


def evaluate_any(model, loader, device='cpu'):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)

    p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)


    report = macro_classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return (acc,
            p_macro, r_macro, f1_macro,
            report, cm,
            y_pred, y_true)


# =======================
# Train (record train/validation acc per epoch)
# =======================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=200, device='cpu', patience=80, verbose=True):
    model = model.to(device)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val = -1.0
    best_state = None
    wait = 0

    epoch_bar = tqdm(total=num_epochs, desc="🚀 Training", colour="green", dynamic_ncols=True)
    start_time = time.time()

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # ---- val ----
        model.eval()
        running_vloss = 0.0
        v_correct, v_total = 0, 0
        all_preds, all_trues = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vloss = criterion(logits, y)
                running_vloss += vloss.item() * x.size(0)

                pred = torch.argmax(logits, dim=1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_trues.extend(y.cpu().numpy())

        val_loss = running_vloss / len(val_loader.dataset)
        val_acc = v_correct / v_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # extra metrics (macro, for logging only)
        p_log_macro = precision_score(all_trues, all_preds, average='macro', zero_division=0)
        r_log_macro = recall_score(all_trues, all_preds, average='macro', zero_division=0)
        f1_log_macro = f1_score(all_trues, all_preds, average='macro', zero_division=0)

        scheduler.step()

        # early stop by val_acc (保持与你原逻辑一致)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        # ETA
        elapsed = time.time() - start_time
        avg_ep = elapsed / (epoch + 1)
        remain = avg_ep * (num_epochs - epoch - 1)
        eta_min, eta_sec = int(remain // 60), int(remain % 60)

        epoch_bar.update(1)

        if verbose and (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f"\n📌 Epoch [{epoch+1}/{num_epochs}] | "
                f"TrainLoss: {train_loss:.6f} | ValLoss: {val_loss:.6f} | "
                f"TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | "
                f"P(macro): {p_log_macro:.4f} R(macro): {r_log_macro:.4f} F1(macro): {f1_log_macro:.4f} | "
                f"ETA: {eta_min:02d}:{eta_sec:02d} | "
                f"Patience: {wait}/{patience} | LR: {lr:.2e}"
            )

        if wait >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
            break

    epoch_bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("⚠️ best_state 为空：将使用最后一个epoch的参数（一般不应该发生）")

    return model, train_losses, val_losses, train_accs, val_accs


# =======================
# Plots
# =======================
def plot_training_process(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='训练损失', linewidth=2)
    ax1.plot(val_losses, label='验证损失', linewidth=2)
    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title('训练 / 验证 损失曲线', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='训练准确率', linewidth=2)
    ax2.plot(val_accs, label='验证准确率', linewidth=2)
    ax2.set_xlabel('训练轮次', fontsize=12)
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.set_title('训练 / 验证 准确率曲线', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_acc_curve_only(train_accs, val_accs):
    """你要求新增的：训练集/验证集准确率 随epoch变化（单独一张图）"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='训练准确率', linewidth=2)
    plt.plot(val_accs, label='验证准确率', linewidth=2)
    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('训练集 / 验证集 Accuracy 随训练轮次变化', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confusion_mat, class_names=None, save_path="confusion_matrix.png"):
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_mat))]

    # Word 目标尺寸：宽3.4 cm，高2.75 cm
    fig_width_inch = 3.4 / 2.54
    fig_height_inch = 2.75 / 2.54

    # 最浅色偏白，但不全白
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_blue_gradient",
        ["#F2F7FA", "#0073B1"]
    )

    plt.figure(figsize=(fig_width_inch, fig_height_inch))
    ax = sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap=custom_cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        annot_kws={"size": 4.5, "fontfamily": "Times New Roman"}
    )

    ax.set_xlabel('Predicted class', fontsize=4.5, fontname='Times New Roman', labelpad=2)
    ax.set_ylabel('True class', fontsize=4.5, fontname='Times New Roman', labelpad=2)
    ax.set_title('Confusion Matrix', fontsize=4.5, fontname='Times New Roman', pad=3)

    ax.tick_params(axis='x', labelsize=2.8, pad=1, direction='in', length=0.6, width=0.2)
    ax.tick_params(axis='y', labelsize=2.8, pad=1, direction='in', length=0.6, width=0.2)

    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_rotation(0)

    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_rotation(0)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontname('Times New Roman')
    cbar.ax.yaxis.label.set_size(4.5)
    cbar.ax.tick_params(labelsize=2.8, pad=1, length=0.8, width=0.2, direction='in')
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname('Times New Roman')

    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.show()


def plot_classification_report_bar(report_dict, title="分类指标（Macro Avg）"):
    """保留你原先的：只画 macro avg 的柱状图"""
    macro = report_dict.get('macro avg', None)
    if macro is None:
        print("⚠️ report_dict 中没有 macro avg")
        return

    metrics = ['precision', 'recall', 'f1-score']
    values = [macro[m] for m in metrics]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, edgecolor='black', linewidth=1.5)
    plt.ylim(0, 1)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()


def print_metrics_block(name, acc, p_macro, r_macro, f1_macro, report, class_names):
    print("\n" + "=" * 80)
    print(f"📊 {name} 性能:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro    - P: {p_macro:.4f} | R: {r_macro:.4f} | F1: {f1_macro:.4f}")

    print("\nPer-class（每类指标）:")
    for i, cname in enumerate(class_names):
        cls = report.get(str(i), None)
        if cls is None:
            continue
        print(f"  {cname:<8s} | P: {cls['precision']:.4f} | R: {cls['recall']:.4f} | F1: {cls['f1-score']:.4f} | support: {int(cls['support'])}")
    print("=" * 80 + "\n")


# =======================
# Predict single sample (same as yours)
# =======================
def predict_new_data(model, new_data, feature_scaler, device='cpu'):
    model.eval()

    if isinstance(new_data, np.ndarray):
        features_np = new_data
    else:
        features_np = new_data.numpy()

    if features_np.ndim == 1:
        features_reshaped = features_np.reshape(1, -1)
    elif features_np.ndim == 2 and features_np.shape[1] == 1:
        features_reshaped = features_np.reshape(1, features_np.shape[0])
    else:
        features_reshaped = features_np

    features_standardized = feature_scaler.transform(features_reshaped)
    features_tensor = torch.FloatTensor(features_standardized).reshape(features_standardized.shape[0], 1, -1).to(device)

    with torch.no_grad():
        logits = model(features_tensor)
        pred = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

    return pred.cpu().numpy(), probs.cpu().numpy()


# =======================
# 单次流程：固定原有超参数；由main独立重复运行10次
# =======================
def run_single_split(random_state=42, output_dir=None):
    set_seed(random_state)
    CSV = "data-1050-(0.7~0.9).csv"

    df = pd.read_csv(CSV, header=None)
    wavelength = df.iloc[0, 1:].values.astype(float)

    preprocessor = SpectralPreprocessor(method='d2', wavelength=wavelength)
    print(f'所使用光谱预处理方法: {preprocessor.method}')

    train_loader, val_loader, test_loader, original_train_loader, test_metadata = get_data_loaders(
        csv_file=CSV,
        batch_size=16,
        test_size=80,
        val_size=0.2,
        random_state=random_state,
        transform=preprocessor
    )

    # 无需等待训练结束，即可查看/导出测试集全部80组真实浓度。
    # 浓度单位沿用输入CSV；若输入为μg/kg，此列也为μg/kg，不做单位换算。
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(CSV)), 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, 'test_80_AFB1_labels.csv')
    test_metadata.to_csv(labels_path, index=False, encoding='utf-8-sig')
    print("\n测试集80组样本的真实AFB1浓度及类别（类别编号0–3）：")
    print(test_metadata.to_string(index=False))
    print(f"\n浓度标签已保存：{labels_path}")
    test_concentrations = test_metadata['AFB1_concentration'].to_numpy(copy=True)
    print(f"测试集浓度标签数组形状：{test_concentrations.shape}")

    for orig_features, orig_labels in test_loader:
        raw_sample_features = orig_features[0].numpy()
        if raw_sample_features.ndim == 2 and raw_sample_features.shape[-1] == 1:
            raw_sample_features = raw_sample_features.squeeze(-1)
        raw_sample_label = int(orig_labels[0].item())
        break

    train_loader, val_loader, test_loader, original_train_loader, feature_scaler = standardize_features(
        train_loader, val_loader, test_loader, original_train_loader
    )

    model = SpectralModel4ClassWithSKLite_V3(num_classes=4)

    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    unique, counts = np.unique(all_labels, return_counts=True)
    class_weights = torch.FloatTensor([len(all_labels) / (len(unique) * c) for c in counts])
    class_weights = class_weights / class_weights.sum() * len(unique)
    class_weights = class_weights.to(device)

    print(f"\n类别权重: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80 + "\n")

    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=200, device=device, patience=80, verbose=True
    )

    plot_training_process(train_losses, val_losses, train_accs, val_accs)
    plot_acc_curve_only(train_accs, val_accs)

    class_names = ['ClassⅠ', 'ClassⅡ', 'Class Ⅲ', 'Class Ⅳ']

    (train_acc,
     train_p_macro, train_r_macro, train_f1_macro,
     train_report, train_cm,
     train_pred, train_true) = evaluate_any(model, train_loader, device=device)

    (val_acc,
     val_p_macro, val_r_macro, val_f1_macro,
     val_report, val_cm,
     val_pred, val_true) = evaluate_any(model, val_loader, device=device)

    (test_acc,
     test_p_macro, test_r_macro, test_f1_macro,
     test_report, test_cm,
     test_pred, test_true) = evaluate_any(model, test_loader, device=device)

    # evaluate_any按测试DataLoader顺序返回结果，与test_metadata逐行对齐。
    if not np.array_equal(test_true, test_metadata['true_class'].to_numpy()):
        raise ValueError('测试集真实类别顺序与浓度表不一致，请检查DataLoader顺序。')
    test_results = test_metadata.copy()
    test_results['predicted_class'] = test_pred
    test_results['correct'] = test_pred == test_true
    predictions_path = os.path.join(output_dir, 'test_80_AFB1_predictions.csv')
    test_results.to_csv(predictions_path, index=False, encoding='utf-8-sig')
    print(f"\n测试集浓度与预测结果已保存：{predictions_path}")

    print_metrics_block("训练集", train_acc, train_p_macro, train_r_macro, train_f1_macro,
                        train_report, class_names)

    print_metrics_block("验证集", val_acc, val_p_macro, val_r_macro, val_f1_macro,
                        val_report, class_names)

    print_metrics_block("测试集", test_acc, test_p_macro, test_r_macro, test_f1_macro,
                        test_report, class_names)

    plot_confusion_matrix(test_cm, class_names=class_names, save_path=os.path.join(output_dir, "confusion_matrix.png"))
    plot_classification_report_bar(test_report, title="分类指标（Macro Avg）")

    print(f"\n预测样本：原始类别 = {class_names[raw_sample_label]}")
    sample_pred, sample_probs = predict_new_data(model, raw_sample_features, feature_scaler, device=device)
    predicted_class = class_names[int(sample_pred[0])]
    print(f"预测类别 = {predicted_class}")
    print(f"各类别概率: {dict(zip(class_names, sample_probs[0]))}")

    print("\n详细分类报告(测试集):")
    print(f"Accuracy: {test_acc:.4f}")
    print(macro_classification_report(test_true, test_pred, target_names=class_names, zero_division=0))

    return {
        'seed': random_state,
        'Accuracy': test_acc,
        'Precision_macro': test_p_macro,
        'Recall_macro': test_r_macro,
        'F1_macro': test_f1_macro,
    }


def summarize_repeated_results(results):
    """10次运行间的均值和样本标准差，不是标准误或置信区间。"""
    columns = ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro']
    values = pd.DataFrame(results)
    if len(values) != 10 or not np.isfinite(values[columns].to_numpy(dtype=float)).all():
        raise ValueError('必须完成10次有效运行后才汇总均值和标准差。')
    summary = pd.DataFrame({
        'metric': columns,
        'mean': values[columns].mean().to_numpy(),
        'std_ddof1': values[columns].std(ddof=1).to_numpy(),
    })
    summary['mean_plus_minus_std'] = [
        f'{mean:.4f} ± {std:.4f}' for mean, std in zip(summary['mean'], summary['std_ddof1'])
    ]
    return values, summary


def main():
    # 10次重复分层随机划分，不是10折交叉验证。
    seeds = list(range(42, 52))
    output_root = os.path.abspath(os.path.join(
        'ms_skresnet_repeat10_results', f'run_{time.time_ns()}'))
    os.makedirs(output_root, exist_ok=False)
    results = []
    for repeat_index, seed in enumerate(seeds, start=1):
        print(f"\n{'=' * 80}")
        print(f"第 {repeat_index}/10 次分层随机划分，seed={seed}")
        print(f"{'=' * 80}\n")
        output_dir = os.path.join(output_root, f'repeat_{repeat_index:02d}_seed_{seed}')
        # 完整重跑原单次流程：重新读原始样本、划分、增强、拟合预处理/Scaler并初始化模型。
        result = run_single_split(random_state=seed, output_dir=output_dir)
        result = {'repeat': repeat_index, **result}
        results.append(result)
        pd.DataFrame(results).to_csv(os.path.join(output_root, 'ten_splits_metrics.csv'),
                                     index=False, encoding='utf-8-sig')
    _, summary = summarize_repeated_results(results)
    summary.to_csv(os.path.join(output_root, 'ten_splits_mean_std.csv'),
                   index=False, encoding='utf-8-sig')
    print("\n10次分层随机划分：四项测试指标均值 ± 样本标准差（ddof=1）")
    print(summary[['metric', 'mean_plus_minus_std']].to_string(index=False))
    print(f"\n每次标签、预测、混淆矩阵以及总体汇总保存于：{output_root}")



if __name__ == "__main__":
    main()
