
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CSV_FILE = "data-1050-(0.7~0.9).csv"
OUTPUT_DIRECTORY = "ms_skresnet_repeat10_results"
SPLIT_ONLY = False
CATEGORY_RULE = "original"
PREPROCESS_METHOD = "d2"
EXPECTED_SAMPLES = 264
N_REPEATS = 10
BASE_SEED = 42
PREDICTION_SIZE = 80
VALIDATION_FRACTION = 0.2  # 剩余184条中的20%：37条；原始训练集147条。
BATCH_SIZE = 16
NUM_AUGMENTS = 5          # 每条训练光谱新增5条；147×6=882条。
SCALE_RANGE = (0.9, 1.1)
MAX_EPOCHS = 200
PATIENCE = 80
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
ETA_MIN = 1e-5
SAVE_PLOTS = True
SHOW_PLOTS = False
SAVE_EXCEL = True
CLASS_LABELS = np.arange(4, dtype=np.int64)
CLASS_NAMES = ("Class I", "Class II", "Class III", "Class IV")

# 延迟导入训练库；只导出浓度时不需要torch或模型文件。
torch = None


@dataclass(frozen=True)
class Config:
    csv_file: str
    output_directory: str
    split_only: bool = SPLIT_ONLY
    category_rule: str = CATEGORY_RULE
    method: Optional[str] = PREPROCESS_METHOD
    batch_size: int = BATCH_SIZE
    num_augments: int = NUM_AUGMENTS
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    eta_min: float = ETA_MIN
    save_plots: bool = SAVE_PLOTS
    show_plots: bool = SHOW_PLOTS
    save_excel: bool = SAVE_EXCEL
    device: str = "auto"


@dataclass
class SpectralData:
    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    concentrations: np.ndarray
    concentration_text: np.ndarray
    sample_ids: np.ndarray
    csv_row_numbers: np.ndarray
    source_path: str
    source_sha256: str


@dataclass(frozen=True)
class SplitPlan:
    repeat: int
    seed: int
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray


def require_torch():
    global torch
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(

            ) from exc
    return torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    t = require_torch()
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False



def concentrations_to_categories(values, rule: str = CATEGORY_RULE) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("浓度必须是一维数组。")
    if rule == "original":
        valid = np.isfinite(values) & (values >= 1) & (values <= 40)
        labels = np.searchsorted([5., 10., 20.], values, side="right")
    elif rule == "upper":
        valid = np.isfinite(values) & (values >= 0) & (values <= 40)
        labels = np.searchsorted([5., 10., 20.], values, side="left")
    else:
        raise ValueError("category_rule必须是original或upper。")
    return np.where(valid, labels, -1).astype(np.int64)


def value_to_category(value, rule: str = CATEGORY_RULE) -> int:
    return int(concentrations_to_categories([value], rule)[0])


def category_ranges(rule: str):
    if rule == "original":
        return ("[1,5)", "[5,10)", "[10,20)", "[20,40]")
    if rule == "upper":
        return ("[0,5]", "(5,10]", "(10,20]", "(20,40]")
    raise ValueError("未知类别边界规则。")


def load_spectral_data(csv_file: str, category_rule: str) -> SpectralData:
    path = Path(csv_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到原始CSV：{path}\n请修改CSV_FILE，或使用 --csv 指定正确文件。"
        )
    # 按字符串读取，保留原文件浓度文本；不要求左上角单元格为数字。
    frame = pd.read_csv(path, header=None, dtype=str, encoding="utf-8-sig")
    if frame.shape[0] != EXPECTED_SAMPLES + 1 or frame.shape[1] < 4:
        raise ValueError(
            f"CSV应包含1行波长和{EXPECTED_SAMPLES}条样本、至少3个波段；"
            f"实际形状为{frame.shape}。不自动删除样本。"
        )
    try:
        wavelengths = pd.to_numeric(frame.iloc[0, 1:], errors="raise").to_numpy(dtype=np.float64)
        concentration_text = frame.iloc[1:, 0].fillna("").str.strip().to_numpy()
        concentrations = pd.to_numeric(pd.Series(concentration_text), errors="raise").to_numpy(dtype=np.float64)
        X = frame.iloc[1:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("波长、浓度或光谱含非数字内容；请检查CSV。") from exc
    if not np.isfinite(wavelengths).all():
        raise ValueError("波长含NaN或无穷值。")
    differences = np.diff(wavelengths)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("波长必须严格递增或递减，不能重复；否则导数预处理不可靠。")
    bad_rows = np.flatnonzero(~np.isfinite(X).all(axis=1))
    if bad_rows.size:
        raise ValueError(f"光谱有缺失或非有限值，CSV行号：{(bad_rows + 2).tolist()}")
    y = concentrations_to_categories(concentrations, category_rule)
    invalid = np.flatnonzero(y < 0)
    if invalid.size:
        details = [(int(i + 2), concentration_text[i]) for i in invalid]
        raise ValueError(f"浓度超出类别范围或为空（CSV行号，浓度）：{details}")
    if not np.array_equal(np.unique(y), CLASS_LABELS):
        raise ValueError("原始数据必须包含0、1、2、3全部四类。")
    sample_ids = np.arange(1, len(y) + 1, dtype=np.int64)
    data = SpectralData(
        X=X, y=y, wavelengths=wavelengths, concentrations=concentrations,
        concentration_text=concentration_text, sample_ids=sample_ids,
        csv_row_numbers=sample_ids + 1, source_path=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    print(f"原始数据：{len(y)}条样本，{X.shape[1]}个波段。")
    classes, counts = np.unique(y, return_counts=True)
    print("原始类别分布：", {int(c): int(n) for c, n in zip(classes, counts)})
    print("类别边界：", dict(zip(CLASS_NAMES, category_ranges(category_rule))))
    return data


def split_real_indices(y, test_size: int = PREDICTION_SIZE,
                       val_size: float = VALIDATION_FRACTION,
                       random_state: int = BASE_SEED):
    y = np.asarray(y, dtype=np.int64)
    if y.shape != (EXPECTED_SAMPLES,):
        raise ValueError(f"应有{EXPECTED_SAMPLES}条原始样本，实际形状为{y.shape}。")
    if not np.array_equal(np.unique(y), CLASS_LABELS):
        raise ValueError("划分标签必须包含0–3全部四类。")
    indices = np.arange(len(y), dtype=np.int64)
    train_val_ids, test_ids = train_test_split(
        indices, test_size=test_size, stratify=y,
        random_state=random_state, shuffle=True,
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size, stratify=y[train_val_ids],
        random_state=random_state + 1, shuffle=True,
    )
    parts = tuple(np.sort(ids) for ids in (train_ids, val_ids, test_ids))
    joined = np.concatenate(parts)
    # 显式检查，不使用可能被python -O关闭的assert。
    if len(joined) != len(y) or not np.array_equal(np.sort(joined), indices):
        raise RuntimeError("训练、验证、预测集存在重复索引或遗漏样本。")
    if len(parts[2]) != test_size:
        raise RuntimeError("预测集数量不正确。")
    for ids in parts:
        if not np.array_equal(np.unique(y[ids]), CLASS_LABELS):
            raise ValueError("训练、验证、预测集均须包含四类；请检查每类数量。")
    return parts


def build_split_plans(y) -> list[SplitPlan]:
    plans = []
    seen_prediction_sets = set()
    for repeat, seed in enumerate(range(BASE_SEED, BASE_SEED + N_REPEATS), start=1):
        train_ids, val_ids, test_ids = split_real_indices(y, random_state=seed)
        key = tuple(test_ids.tolist())
        if key in seen_prediction_sets:
            raise RuntimeError(
                f"第{repeat}次预测集与已有划分完全相同；停止运行，避免汇报重复划分。"
            )
        seen_prediction_sets.add(key)
        plans.append(SplitPlan(repeat, seed, train_ids, val_ids, test_ids))
    return plans


def sample_metadata(data: SpectralData, ids, config: Config) -> pd.DataFrame:
    ids = np.asarray(ids, dtype=np.int64)
    return pd.DataFrame({
        "sample_id": data.sample_ids[ids],
        "csv_row_number": data.csv_row_numbers[ids],
        "AFB1_concentration": data.concentrations[ids],
        "AFB1_concentration_text": data.concentration_text[ids],
        "true_class": data.y[ids],
        "true_grade": np.asarray(CLASS_NAMES)[data.y[ids]],
        "class_range": np.asarray(category_ranges(config.category_rule))[data.y[ids]],
    })


def prediction_metadata(data: SpectralData, plan: SplitPlan, config: Config) -> pd.DataFrame:
    table = sample_metadata(data, plan.test_ids, config)
    table.insert(0, "test_order", np.arange(1, len(table) + 1))
    table.insert(0, "seed", plan.seed)
    table.insert(0, "repeat", plan.repeat)
    return table


def save_csv(table: pd.DataFrame, path: Path):
    table.to_csv(path, index=False, encoding="utf-8-sig")


def save_excel_tables(path: Path, tables: dict[str, pd.DataFrame], enabled: bool):
    if not enabled:
        return
    try:

        importlib.import_module("openpyxl")
        get_column_letter = importlib.import_module("openpyxl.utils").get_column_letter
    except (ImportError, AttributeError) as exc:
        print(f"Excel导出依赖openpyxl不可用：{exc}")
        print("跳过Excel导出，继续输出CSV。需要Excel时，请在当前Python环境执行：")
        print("python -m pip install openpyxl")
        return
    temporary = path.with_name(f".{path.stem}.temporary.xlsx")
    try:
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            for name, table in tables.items():
                table.to_excel(writer, sheet_name=name, index=False, freeze_panes=(1, 0))
                ws = writer.sheets[name]
                ws.auto_filter.ref = ws.dimensions
                for col, header in enumerate(table.columns, start=1):
                    ws.column_dimensions[get_column_letter(col)].width = min(38, max(15, len(str(header)) + 2))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_all_split_tables(data: SpectralData, plans, config: Config, output_root: Path):
    metadata_by_repeat = {}
    assignments = []
    audit_rows = []
    index_arrays = {}
    wide = pd.DataFrame({"test_order": np.arange(1, PREDICTION_SIZE + 1)})
    for plan in plans:
        run_dir = output_root / f"repeat_{plan.repeat:02d}_seed_{plan.seed}"
        run_dir.mkdir(exist_ok=False)
        table = prediction_metadata(data, plan, config)
        metadata_by_repeat[plan.repeat] = table
        save_csv(table, run_dir / "test_80_AFB1_labels.csv")
        prefix = f"repeat_{plan.repeat:02d}_seed_{plan.seed}"
        wide[prefix + "_sample_id"] = table["sample_id"].to_numpy()
        wide[prefix + "_csv_row"] = table["csv_row_number"].to_numpy()
        wide[prefix + "_concentration"] = table["AFB1_concentration"].to_numpy()
        audit_row = {
            "repeat": plan.repeat, "seed": plan.seed,
            "n_train_original": len(plan.train_ids), "n_validation": len(plan.val_ids),
            "n_prediction": len(plan.test_ids),
            "prediction_ids_sha256": hashlib.sha256(
                np.asarray(plan.test_ids, dtype="<i8").tobytes()
            ).hexdigest(),
        }
        for partition, ids in (("train", plan.train_ids), ("validation", plan.val_ids), ("prediction", plan.test_ids)):
            index_arrays[f"repeat_{plan.repeat:02d}_{partition}_indices_0based"] = ids
            assignment = sample_metadata(data, ids, config)
            assignment.insert(0, "partition", partition)
            assignment.insert(0, "seed", plan.seed)
            assignment.insert(0, "repeat", plan.repeat)
            assignments.append(assignment)
            for cls in CLASS_LABELS:
                audit_row[f"{partition}_class_{cls}_count"] = int(np.sum(data.y[ids] == cls))
        audit_rows.append(audit_row)
        print(f"第{plan.repeat:02d}次 seed={plan.seed}：原始训练{len(plan.train_ids)}，验证{len(plan.val_ids)}，预测{len(plan.test_ids)}。")
        print(table[["test_order", "sample_id", "csv_row_number", "AFB1_concentration", "true_grade"]].to_string(index=False))

    all_concentrations = pd.concat(list(metadata_by_repeat.values()), ignore_index=True)
    if len(all_concentrations) != N_REPEATS * PREDICTION_SIZE:
        raise RuntimeError("10次预测集浓度总表应恰好包含800条记录。")
    all_assignments = pd.concat(assignments, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    overlap = pd.DataFrame({
        "repeat": [f"repeat_{p.repeat:02d}" for p in plans],
        **{f"repeat_{q.repeat:02d}": [
            len(np.intersect1d(p.test_ids, q.test_ids)) for p in plans
        ] for q in plans},
    })
    matrix = overlap.iloc[:, 1:].to_numpy()
    if not np.all(np.diag(matrix) == PREDICTION_SIZE):
        raise RuntimeError("预测集交集矩阵对角线应为80。")
    if np.any(matrix[~np.eye(N_REPEATS, dtype=bool)] >= PREDICTION_SIZE):
        raise RuntimeError("存在完全相同的预测集。")
    save_csv(all_concentrations, output_root / "ten_splits_concentrations.csv")
    save_csv(wide, output_root / "ten_splits_concentrations_wide.csv")
    save_csv(all_assignments, output_root / "ten_splits_assignments.csv")
    save_csv(audit, output_root / "split_audit.csv")
    save_csv(overlap, output_root / "prediction_set_overlap.csv")
    np.savez_compressed(output_root / "split_plan.npz", **index_arrays)
    config_record = {
        "config": asdict(config), "seeds": [p.seed for p in plans],
        "expected_samples": EXPECTED_SAMPLES, "prediction_size": PREDICTION_SIZE,
        "validation_fraction_of_remaining": VALIDATION_FRACTION,
        "scale_range": list(SCALE_RANGE), "category_ranges": category_ranges(config.category_rule),
        "source_csv_sha256": data.source_sha256,
        "sample_id_definition": "CSV第2行对应sample_id=1，以此类推；不是实验原始编号。",
        "concentration_unit": "与输入CSV相同，不进行单位换算。",
        "split_method": "10次重复分层随机划分；不同预测集允许部分样本重叠。",
    }
    (output_root / "run_configuration.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    workbook_tables = {
        "all_800_samples": all_concentrations, "80_rows_wide": wide,
        "split_audit": audit, "prediction_overlap": overlap,
        **{f"repeat_{p.repeat:02d}_seed_{p.seed}": metadata_by_repeat[p.repeat] for p in plans},
    }
    save_excel_tables(output_root / "ten_splits_concentrations.xlsx", workbook_tables, config.save_excel)
    print("已核验：10份预测集两两不同；每份80条，且各次训练/验证/预测集互不重叠。")
    return metadata_by_repeat, workbook_tables


def as_spectral_matrix(data) -> np.ndarray:
    if torch is not None and isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("光谱必须是非空二维数组：样本数×波段数。")
    if not np.isfinite(array).all():
        raise ValueError("光谱含NaN或无穷值。")
    return array


class SpectralPreprocessor:
    def __init__(self, method=None, wavelength=None, use_standardize=False, **kwargs):
        self.method = method.lower() if method else None
        self.wavelength = None if wavelength is None else np.asarray(wavelength, dtype=np.float64)
        self.params = kwargs
        self.use_standardize = use_standardize
        self.scaler = None
        self.msc_reference = None
        self.n_features = None
        self.fitted = False

    def fit(self, data):
        data = as_spectral_matrix(data)
        self.n_features = data.shape[1]
        if self.method == "msc":
            self.msc_reference = np.mean(data, axis=0)
            if np.ptp(self.msc_reference) < 1e-12:
                raise ValueError("MSC参考光谱近乎常数，不能进行可靠的线性校正。")
        self.scaler = None
        self.fitted = True
        if self.use_standardize:
            self.scaler = StandardScaler().fit(self._apply_basic(data))
        return self

    def _apply_basic(self, data):
        method = self.method
        if method is None:
            result = data.copy()
        elif method == "sg":
            from scipy.signal import savgol_filter
            window = self.params.get("w", 11)
            polyorder = self.params.get("p", 3)
            if window > data.shape[1] or window % 2 != 1 or polyorder < 0 or polyorder >= window:
                raise ValueError("SG窗口必须为不超过波段数的正奇数，且大于多项式阶数。")
            result = savgol_filter(data, window, polyorder, axis=1)
        elif method == "msc":
            if self.msc_reference is None:
                raise RuntimeError("MSC必须先由原始训练集fit。")
            rows = []
            for row in data:
                slope, intercept = np.polyfit(self.msc_reference, row, 1)
                if not np.isfinite(slope) or abs(slope) < 1e-12:
                    raise ValueError("MSC拟合斜率接近零，无法校正该条光谱。")
                rows.append((row - intercept) / slope)
            result = np.asarray(rows)
        elif method == "snv":
            result = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
        elif method in ("normalize", "minmax"):
            ranges = np.ptp(data, axis=1, keepdims=True)
            # normalize保留原脚本的epsilon；minmax对常数光谱使用分母1。
            denominator = ranges + 1e-8 if method == "normalize" else np.where(ranges == 0, 1., ranges)
            result = (data - data.min(axis=1, keepdims=True)) / denominator
        elif method == "standardize":
            from sklearn.preprocessing import scale
            result = scale(data, axis=1)
        elif method in ("d1", "d2"):
            if self.wavelength is None or self.wavelength.shape != (data.shape[1],):
                raise ValueError("导数预处理需要与波段数一一对应的波长。")
            delta = np.diff(self.wavelength)
            if not np.isfinite(self.wavelength).all() or not (np.all(delta > 0) or np.all(delta < 0)):
                raise ValueError("导数预处理的波长必须有限、无重复且严格单调。")
            result = np.gradient(data, self.wavelength, axis=1)
            if method == "d2":
                result = np.gradient(result, self.wavelength, axis=1)
        elif method == "wavelet":
            import pywt
            wavelet = pywt.Wavelet("db8")
            level = pywt.dwt_max_level(data.shape[1], wavelet.dec_len)
            rows = []
            for row in data:
                coeffs = pywt.wavedec(row, wavelet, level=level)
                for i in range(1, len(coeffs)):
                    coeffs[i] = pywt.threshold(coeffs[i], 0.04 * np.max(np.abs(coeffs[i])))
                rows.append(pywt.waverec(coeffs, wavelet)[:len(row)])
            result = np.asarray(rows)
        else:
            raise ValueError(f"未知预处理方法：{method}")
        if not np.isfinite(result).all():
            raise ValueError("预处理产生NaN或无穷值。")
        return np.asarray(result, dtype=np.float64)

    def apply(self, data):
        if not self.fitted:
            raise RuntimeError("预处理器须先用原始训练集fit。")
        data = as_spectral_matrix(data)
        if data.shape[1] != self.n_features:
            raise ValueError(f"波段数不一致：应为{self.n_features}，实际{data.shape[1]}。")
        result = self._apply_basic(data)
        if self.scaler is not None:
            result = self.scaler.transform(result)
        if not np.isfinite(result).all():
            raise ValueError("预处理输出含非有限值。")
        return result

    def state_dict(self):

        return {
            "method": self.method, "wavelength": self.wavelength,
            "use_standardize": self.use_standardize, "params": self.params,
            "scaler": self.scaler, "msc_reference": self.msc_reference,
            "n_features": self.n_features, "fitted": self.fitted,
        }

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state["method"], state["wavelength"], state["use_standardize"], **state["params"])
        for name in ("scaler", "msc_reference", "n_features", "fitted"):
            setattr(obj, name, state[name])
        return obj


def spectral_scaling(X, y, num_augments=NUM_AUGMENTS, scale_range=SCALE_RANGE, seed=BASE_SEED):
    X = as_spectral_matrix(X)
    y = np.asarray(y, dtype=np.int64)
    if y.shape != (len(X),) or num_augments < 0:
        raise ValueError("增强标签数量不匹配，或增强次数小于0。")
    low, high = scale_range
    if not (0 < low <= high):
        raise ValueError("缩放范围必须为正且下界不大于上界。")
    # 单独的随机数生成器使增强不受其他模块随机数调用次序影响。
    rng = np.random.RandomState(seed)
    scales = rng.uniform(low, high, size=(len(X), num_augments, 1))
    augmented_X = (X[:, None, :] * scales).reshape(len(X) * num_augments, X.shape[1])
    augmented_y = np.repeat(y, num_augments)
    return augmented_X, augmented_y


def prepare_split_arrays(data: SpectralData, plan: SplitPlan, config: Config):
    train_X, train_y = data.X[plan.train_ids], data.y[plan.train_ids]
    augmented_X, augmented_y = spectral_scaling(train_X, train_y, config.num_augments, seed=plan.seed)
    full_X = np.concatenate([train_X, augmented_X], axis=0)
    full_y = np.concatenate([train_y, augmented_y])
    full_ids = np.concatenate([
        data.sample_ids[plan.train_ids],
        np.repeat(data.sample_ids[plan.train_ids], config.num_augments),
    ])
    preprocessor = SpectralPreprocessor(config.method, data.wavelengths).fit(train_X)
    processed_real_train = preprocessor.apply(train_X)
    # 与原附件一致：参考谱和按波段StandardScaler都只拟合原始训练集。
    scaler = StandardScaler().fit(processed_real_train)
    arrays = {}
    for name, X, y, ids in (
        ("train_augmented", full_X, full_y, full_ids),
        ("train_original", train_X, train_y, data.sample_ids[plan.train_ids]),
        ("validation", data.X[plan.val_ids], data.y[plan.val_ids], data.sample_ids[plan.val_ids]),
        ("prediction", data.X[plan.test_ids], data.y[plan.test_ids], data.sample_ids[plan.test_ids]),
    ):
        scaled = scaler.transform(preprocessor.apply(X))
        scaled = scaled.astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError(f"{name}标准化结果含NaN、无穷值或float32溢出。")
        arrays[name] = (scaled, np.asarray(y, dtype=np.int64), np.asarray(ids, dtype=np.int64))
    return arrays, preprocessor, scaler


def get_data_loaders(arrays, config: Config, seed: int, device):
    t = require_torch()
    loaders = {}
    for offset, (name, (X, y, sample_ids)) in enumerate(arrays.items()):
        dataset = t.utils.data.TensorDataset(
            t.from_numpy(X).unsqueeze(1), t.from_numpy(y), t.from_numpy(sample_ids),
        )
        loaders[name] = t.utils.data.DataLoader(
            dataset, batch_size=config.batch_size,
            shuffle=(name == "train_augmented"), drop_last=False,
            num_workers=0, pin_memory=(device.type == "cuda"),
            generator=t.Generator().manual_seed(seed + offset),
        )
    return loaders


def logits_checked(model, x):
    t = require_torch()
    logits = model(x)
    if not isinstance(logits, t.Tensor) or tuple(logits.shape) != (len(x), len(CLASS_LABELS)):
        raise ValueError("模型必须返回[batch_size,4]的分类logits张量。")
    if not t.isfinite(logits).all().item():
        raise FloatingPointError("模型输出logits含NaN或无穷值。")
    return logits


def weighted_loss_mass(criterion, y) -> float:

    if criterion.reduction != "mean" or criterion.label_smoothing != 0:
        raise ValueError("本流程的损失汇总对应mean reduction且label_smoothing=0。")
    return float(len(y)) if criterion.weight is None else float(criterion.weight[y].sum().item())


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                config: Config, device):
    t = require_torch()
    best_val_acc = -1.
    best_state = None
    best_epoch = 0
    wait = 0
    rows = []
    start_time = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = train_loss_mass = 0.
        correct = total = 0
        learning_rate_used = optimizer.param_groups[0]["lr"]
        for x, y, _sample_ids in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = logits_checked(model, x)
            loss = criterion(logits, y)
            if not t.isfinite(loss).item():
                raise FloatingPointError("训练损失含NaN或无穷值。")
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            mass = weighted_loss_mass(criterion, y)
            train_loss_sum += float(loss.item()) * mass
            train_loss_mass += mass
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += len(y)

        model.eval()
        val_loss_sum = val_loss_mass = 0.
        val_correct = val_total = 0
        val_true, val_pred = [], []
        with t.no_grad():
            for x, y, _sample_ids in val_loader:
                x, y = x.to(device), y.to(device)
                logits = logits_checked(model, x)
                loss = criterion(logits, y)
                if not t.isfinite(loss).item():
                    raise FloatingPointError("验证损失含NaN或无穷值。")
                mass = weighted_loss_mass(criterion, y)
                val_loss_sum += float(loss.item()) * mass
                val_loss_mass += mass
                pred = logits.argmax(dim=1)
                val_correct += int((pred == y).sum().item())
                val_total += len(y)
                val_true.extend(y.cpu().tolist())
                val_pred.extend(pred.cpu().tolist())
        train_acc, val_acc = correct / total, val_correct / val_total
        train_loss, val_loss = train_loss_sum / train_loss_mass, val_loss_sum / val_loss_mass
        # 保留原脚本：仅根据验证准确率选模型并早停，不利用预测集选epoch。
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, wait = val_acc, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            wait += 1
        val_p = precision_score(val_true, val_pred, labels=CLASS_LABELS, average="macro", zero_division=0)
        val_r = recall_score(val_true, val_pred, labels=CLASS_LABELS, average="macro", zero_division=0)
        val_f1 = f1_score(val_true, val_pred, labels=CLASS_LABELS, average="macro", zero_division=0)
        rows.append({
            "epoch": epoch, "train_augmented_loss": train_loss, "validation_loss": val_loss,
            "train_augmented_accuracy": train_acc, "validation_accuracy": val_acc,
            "validation_precision_macro": val_p, "validation_recall_macro": val_r,
            "validation_f1_macro": val_f1, "learning_rate_used": learning_rate_used,
            "patience_count": wait,
        })
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.perf_counter() - start_time
            eta = elapsed / epoch * (config.max_epochs - epoch)
            print(
                f"Epoch {epoch}/{config.max_epochs} | "
                f"Loss {train_loss:.6f}/{val_loss:.6f} | "
                f"Acc {train_acc:.4f}/{val_acc:.4f} | Val Macro-F1 {val_f1:.4f} | "
                f"LR {learning_rate_used:.2e} | Patience {wait}/{config.patience} | ETA {eta / 60:.1f}min"
            )
        if wait >= config.patience:
            print(f"第{epoch}轮早停；最佳验证准确率位于第{best_epoch}轮。")
            break
    if best_state is None:
        raise RuntimeError("未获得有效模型权重。")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_epoch


def evaluate_any(model, loader, device):
    t = require_torch()
    model.eval()
    true_parts, pred_parts, prob_parts, id_parts = [], [], [], []
    with t.no_grad():
        for x, y, sample_ids in loader:
            logits = logits_checked(model, x.to(device))
            true_parts.append(y.numpy())
            pred_parts.append(logits.argmax(dim=1).cpu().numpy())
            prob_parts.append(t.softmax(logits, dim=1).cpu().numpy())
            id_parts.append(sample_ids.numpy())
    if not true_parts:
        raise ValueError("评估数据集为空。")
    y_true, y_pred = np.concatenate(true_parts), np.concatenate(pred_parts)
    report = classification_report(
        y_true, y_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0,
    )
    report.pop("weighted avg", None)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0),
        "report": report, "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS),
        "y_true": y_true, "y_pred": y_pred, "probabilities": np.concatenate(prob_parts),
        "sample_ids": np.concatenate(id_parts),
    }


def predict_new_data(model, new_raw_data, feature_scaler, preprocessor, device):
    """输入未经导数/MSC/SNV/StandardScaler处理的原始光谱。"""
    t = require_torch()
    if isinstance(new_raw_data, t.Tensor):
        new_raw_data = new_raw_data.detach().cpu().numpy()
    raw = np.asarray(new_raw_data, dtype=np.float64)
    if raw.ndim == 2 and raw.shape[1] == 1 and raw.shape[0] == feature_scaler.n_features_in_:
        raw = raw.T
    raw = as_spectral_matrix(raw)

    processed = preprocessor.apply(raw)
    scaled = feature_scaler.transform(processed).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("新样本标准化结果含非有限值。")
    model.eval()
    with t.no_grad():
        logits = logits_checked(model, t.from_numpy(scaled).unsqueeze(1).to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
        probs = t.softmax(logits, dim=1).cpu().numpy()
    return pred, probs


def diagnose_single_prediction(model, raw_sample, feature_scaler, preprocessor,
                               device, expected_input, bulk_pred, bulk_probs,
                               sample_metadata_row) -> pd.DataFrame:


    row = {
        "sample_id": int(sample_metadata_row["sample_id"]),
        "csv_row_number": int(sample_metadata_row["csv_row_number"]),
        "AFB1_concentration": float(sample_metadata_row["AFB1_concentration"]),
        "batch_predicted_class": int(bulk_pred),
        "single_predicted_class": None,
        "input_arrays_identical": None,
        "max_abs_input_difference": None,
        "max_abs_probability_difference": None,
        "probabilities_within_tolerance": None,
        "comparison_rtol": 1e-4,
        "comparison_atol": 1e-5,
        "predicted_classes_match": None,
        "batchnorm_using_batch_statistics_in_eval": "",
        "diagnostic_status": "not_checked",
        "diagnostic_message": "",
    }
    bulk_probs = np.asarray(bulk_probs, dtype=np.float64).reshape(-1)
    for cls, probability in enumerate(bulk_probs):
        row[f"batch_probability_class_{cls}"] = float(probability)
        row[f"single_probability_class_{cls}"] = None
    try:
        # 检查同一原始光谱的预处理输入；不把模型输出差异误当成样本错位。
        single_input = feature_scaler.transform(preprocessor.apply(raw_sample)).astype(np.float32)[0]
        expected_input = np.asarray(expected_input, dtype=np.float32).reshape(-1)
        if single_input.shape != expected_input.shape:
            raise ValueError("单样本与批量输入的波段数不同。")
        row["input_arrays_identical"] = bool(np.array_equal(single_input, expected_input))
        row["max_abs_input_difference"] = float(np.max(np.abs(single_input.astype(np.float64) - expected_input)))
        if torch is not None:
            batchnorm_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
            names = [
                name for name, layer in model.named_modules()
                if isinstance(layer, batchnorm_types) and (
                    not layer.track_running_stats or layer.running_mean is None or layer.running_var is None
                )
            ]
            row["batchnorm_using_batch_statistics_in_eval"] = "; ".join(names)
        sample_pred, sample_probs = predict_new_data(
            model, raw_sample, feature_scaler, preprocessor, device,
        )
        sample_probs = np.asarray(sample_probs, dtype=np.float64)
        sample_pred = np.asarray(sample_pred).reshape(-1)
        if sample_probs.shape != (1, len(CLASS_LABELS)) or sample_pred.size != 1:
            raise ValueError("单样本预测结果形状异常。")
        if bulk_probs.shape != (len(CLASS_LABELS),) or not np.isfinite(sample_probs).all():
            raise ValueError("诊断用概率数量异常或含非有限值。")
        row["single_predicted_class"] = int(sample_pred[0])
        row["predicted_classes_match"] = bool(sample_pred[0] == bulk_pred)
        row["max_abs_probability_difference"] = float(np.max(np.abs(sample_probs[0] - bulk_probs)))
        row["probabilities_within_tolerance"] = bool(
            np.allclose(sample_probs[0], bulk_probs, rtol=1e-4, atol=1e-5)
        )
        for cls in CLASS_LABELS:
            row[f"single_probability_class_{cls}"] = float(sample_probs[0, cls])
        consistent = (row["input_arrays_identical"] and row["predicted_classes_match"]
                      and row["probabilities_within_tolerance"])
        row["diagnostic_status"] = "consistent" if consistent else "difference_detected"
        if not consistent:
            row["diagnostic_message"] = (
                "检测到输入或预测差异；具体原因需结合输入差值、设备和网络实现排查。"
                "正式结果保持为批量预测，不替换概率或类别。"
            )
            print(
                "提示：单样本与批量预测存在差异，已记录诊断；继续下一次划分。"
                f"最大概率绝对差={row['max_abs_probability_difference']:.6g}"
            )
    except (RuntimeError, ValueError, TypeError, FloatingPointError) as exc:
        row["diagnostic_status"] = "single_prediction_failed"
        row["diagnostic_message"] = f"{type(exc).__name__}: {exc}"
        print(f"提示：单样本诊断未完成（{exc}）；正式批量预测已完成，继续后续划分。")
    return pd.DataFrame([row])


def save_training_plots(history, evaluation, run_dir: Path, config: Config):
    if not config.save_plots:
        return
    import matplotlib
    if not config.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    fonts = {item.name for item in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = "Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif"
    plt.rcParams["axes.unicode_minus"] = False

    def finish(fig, filename, dpi=300):
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=dpi)
        if config.show_plots:
            plt.show()
        plt.close(fig)

    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_augmented_loss"], label="Training (augmented)")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Weighted cross-entropy")
    axes[1].plot(epochs, history["train_augmented_accuracy"], label="Training (augmented)")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.3)
    finish(fig, "training_process.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_augmented_accuracy"], label="Training (augmented)")
    ax.plot(epochs, history["validation_accuracy"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    ax.grid(alpha=.3)
    ax.legend()
    finish(fig, "accuracy_curve.png")

    # 保存尺寸对应原脚本要求：3.4cm×2.75cm，不用bbox_inches改变尺寸。
    fig, ax = plt.subplots(figsize=(3.4 / 2.54, 2.75 / 2.54))
    cmap = LinearSegmentedColormap.from_list("custom_blue", ["#F2F7FA", "#0073B1"])
    cm = evaluation["confusion_matrix"]
    image = ax.imshow(cm, cmap=cmap, vmin=0, aspect="auto")
    ax.set_xticks(CLASS_LABELS, ("I", "II", "III", "IV"))
    ax.set_yticks(CLASS_LABELS, ("I", "II", "III", "IV"))
    ax.tick_params(labelsize=4, length=1, pad=1)
    ax.set_xlabel("Predicted class", fontsize=4.5, labelpad=2)
    ax.set_ylabel("True class", fontsize=4.5, labelpad=2)
    ax.set_title("Confusion matrix", fontsize=4.5, pad=3)
    for row in CLASS_LABELS:
        for col in CLASS_LABELS:
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=4.5,
                    color="white" if cm[row, col] > cm.max() / 2 else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=.05, pad=.04)
    cbar.ax.tick_params(labelsize=3, length=1)
    cbar.set_label("Count", fontsize=4)
    finish(fig, "confusion_matrix.png", dpi=1200)
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = ["Precision_macro", "Recall_macro", "F1_macro"]
    values = [evaluation[k] for k in keys]
    bars = ax.bar(["Macro-Precision", "Macro-Recall", "Macro-F1"], values)
    ax.set(ylabel="Score", ylim=(0, 1.08))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .01, f"{value:.4f}", ha="center")
    finish(fig, "macro_metrics.png")


def resolve_device(config: Config):
    t = require_torch()
    name = "cuda" if t.cuda.is_available() else "cpu"
    if config.device != "auto":
        name = config.device
    if name.startswith("cuda") and not t.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch环境没有可用CUDA设备。")
    return t.device(name)


def run_single_split(data, plan, config, metadata, output_root: Path, model_class):
    t = require_torch()
    set_seed(plan.seed)
    device = resolve_device(config)
    run_dir = output_root / f"repeat_{plan.repeat:02d}_seed_{plan.seed}"
    arrays, preprocessor, scaler = prepare_split_arrays(data, plan, config)
    loaders = get_data_loaders(arrays, config, plan.seed, device)
    print(f"训练使用设备：{device}；增强训练条数：{len(arrays['train_augmented'][1])}。")
    model = model_class(num_classes=4).to(device)
    counts = np.bincount(data.y[plan.train_ids], minlength=4)
    if np.any(counts == 0):
        raise ValueError("训练集存在缺失类别，无法计算类别权重。")
    weights = len(plan.train_ids) / (4. * counts)
    weights = weights / weights.sum() * 4.
    weights = t.tensor(weights, dtype=t.float32, device=device)
    print("类别权重：", weights.cpu().numpy())
    criterion = t.nn.CrossEntropyLoss(weight=weights, reduction="mean", label_smoothing=0.0)
    optimizer = t.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=config.eta_min)
    model, history, best_epoch = train_model(
        model, loaders["train_augmented"], loaders["validation"],
        criterion, optimizer, scheduler, config, device,
    )
    save_csv(history, run_dir / "training_history.csv")
    t.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, run_dir / "best_model_state.pt")
    import joblib
    joblib.dump({"preprocessor": preprocessor.state_dict(), "feature_scaler": scaler,
                 "category_rule": config.category_rule, "seed": plan.seed}, run_dir / "preprocessing_state.joblib")

    evaluations = {

        name: evaluate_any(model, loaders[name], device)
        for name in ("train_original", "validation", "prediction")
    }
    prediction = evaluations["prediction"]
    if not np.array_equal(prediction["sample_ids"], metadata["sample_id"].to_numpy()):
        raise RuntimeError("预测输出的样本编号与浓度表不一致。")
    if not np.array_equal(prediction["y_true"], metadata["true_class"].to_numpy()):
        raise RuntimeError("预测输出的真实类别与浓度表不一致。")
    result_table = metadata.copy()
    result_table["predicted_class"] = prediction["y_pred"]
    result_table["predicted_grade"] = np.asarray(CLASS_NAMES)[prediction["y_pred"]]
    result_table["correct"] = prediction["y_true"] == prediction["y_pred"]
    for cls in CLASS_LABELS:
        result_table[f"probability_class_{cls}"] = prediction["probabilities"][:, cls]
    save_csv(result_table, run_dir / "test_80_AFB1_predictions.csv")
    save_excel_tables(run_dir / "test_80_AFB1_predictions.xlsx", {"80_predictions": result_table}, config.save_excel)
    save_csv(pd.DataFrame(prediction["confusion_matrix"], columns=[f"predicted_{i}" for i in CLASS_LABELS])
             .rename_axis("true_class").reset_index(), run_dir / "confusion_matrix.csv")
    score_names = ("Accuracy", "Precision_macro", "Recall_macro", "F1_macro")
    scores = []
    for partition, evaluation in evaluations.items():
        row = {"partition": partition, "n_original_samples": len(evaluation["y_true"])}
        row.update({k: evaluation[k] for k in score_names})
        scores.append(row)
        print(f"{partition}：" + " | ".join(f"{k}={evaluation[k]:.4f}" for k in score_names))

        report_rows = [
            {"class_or_average": key, **value}
            for key, value in evaluation["report"].items()
            if isinstance(value, dict)
        ]
        report_rows.append({"class_or_average": "accuracy", "accuracy": evaluation["Accuracy"],
                            "support": len(evaluation["y_true"])})
        report_table = pd.DataFrame(report_rows)
        save_csv(report_table, run_dir / f"{partition}_classification_report.csv")
    save_csv(pd.DataFrame(scores), run_dir / "all_partitions_metrics.csv")
    save_training_plots(history, prediction, run_dir, config)

    # 用真正的原始光谱诊断单样本接口；该诊断不能阻断正式训练/评估流程。
    # 样本编号与真实标签的严格对应检查仍保留在上方。
    diagnostic = diagnose_single_prediction(
        model, data.X[plan.test_ids[0]], scaler, preprocessor, device,
        arrays["prediction"][0][0], prediction["y_pred"][0],
        prediction["probabilities"][0], metadata.iloc[0],
    )
    save_csv(diagnostic, run_dir / "single_prediction_diagnostic.csv")
    metrics = {
        "repeat": plan.repeat, "seed": plan.seed, "n_train_original": len(plan.train_ids),
        "n_validation": len(plan.val_ids), "n_prediction": len(plan.test_ids),
        "best_epoch": best_epoch, **{k: prediction[k] for k in score_names},
    }
    return metrics, result_table


def summarize_repeated_results(results):
    values = pd.DataFrame(results)
    columns = ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"]
    if len(values) != N_REPEATS or not np.isfinite(values[columns].to_numpy(dtype=float)).all():
        raise ValueError("必须完成全部10次有效训练后才能计算均值和标准差。")
    summary = pd.DataFrame({
        "metric": columns, "mean": values[columns].mean().to_numpy(),
        "std_ddof1": values[columns].std(ddof=1).to_numpy(),
    })
    summary["mean_percent"] = summary["mean"] * 100.
    summary["std_percent_ddof1"] = summary["std_ddof1"] * 100.
    summary["mean_plus_minus_std"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary["mean"], summary["std_ddof1"])]
    summary["mean_plus_minus_std_percent"] = [
        f"{m:.2f} ± {s:.2f}" for m, s in zip(summary["mean_percent"], summary["std_percent_ddof1"])
    ]
    return values, summary


def parse_config(argv=None):
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="10次重复分层随机划分：先导出每次80条浓度，再训练。")
    parser.add_argument("--csv", default=str(script_dir / CSV_FILE))
    parser.add_argument("--output-dir", default=str(script_dir / OUTPUT_DIRECTORY))
    parser.add_argument("--split-only", action="store_true", default=SPLIT_ONLY)
    parser.add_argument("--category-rule", choices=("original", "upper"), default=CATEGORY_RULE)
    parser.add_argument("--method", choices=("none", "d1", "d2", "sg", "msc", "snv", "normalize", "standardize", "minmax", "wavelet"), default=PREPROCESS_METHOD or "none")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", default="auto", help="auto、cpu或cuda")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true", default=SHOW_PLOTS)
    parser.add_argument("--no-excel", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or BATCH_SIZE < 1 or NUM_AUGMENTS < 0:
        parser.error("epochs、patience及batch_size必须为正，增强次数不能为负。")
    return Config(
        csv_file=str(Path(args.csv).expanduser().resolve()),
        output_directory=str(Path(args.output_dir).expanduser().resolve()),
        split_only=args.split_only, category_rule=args.category_rule,
        method=None if args.method == "none" else args.method,
        max_epochs=args.epochs, patience=args.patience, device=args.device,
        save_plots=SAVE_PLOTS and not args.no_plots, show_plots=args.show_plots,
        save_excel=SAVE_EXCEL and not args.no_excel,
    )


def main(argv=None):
    config = parse_config(argv)
    data = load_spectral_data(config.csv_file, config.category_rule)
    plans = build_split_plans(data.y)
    output_root = Path(config.output_directory) / f"run_{time.time_ns()}"
    output_root.mkdir(parents=True, exist_ok=False)
    print(f"输出目录：{output_root}")
    metadata_by_repeat, workbook_tables = export_all_split_tables(data, plans, config, output_root)
    print(f"全部10次、每次80条浓度已保存：{output_root / 'ten_splits_concentrations.csv'}")
    if config.split_only:
        print("只导出划分及浓度，已完成。将SPLIT_ONLY改成False即可执行完整训练。")
        return output_root


    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require_torch()
    try:
        module = importlib.import_module("MS_SKResNet")
        model_class = getattr(module, "SpectralModel4ClassWithSKLite_V3")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "全部浓度已导出。训练需要将原有MS_SKResNet.py放在本脚本同目录，"
            "其中应包含SpectralModel4ClassWithSKLite_V3；不应随意换成其他网络。"
        ) from exc
    results, prediction_tables = [], []
    for plan in plans:
        print(f"\n{'=' * 70}\n第{plan.repeat}/{N_REPEATS}次完整训练，seed={plan.seed}\n{'=' * 70}")
        metrics, table = run_single_split(
            data, plan, config, metadata_by_repeat[plan.repeat], output_root, model_class,
        )
        results.append(metrics)
        prediction_tables.append(table)

        save_csv(pd.DataFrame(results), output_root / "ten_splits_metrics.csv")
        save_csv(pd.concat(prediction_tables, ignore_index=True), output_root / "ten_splits_predictions.csv")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    values, summary = summarize_repeated_results(results)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    if len(predictions) != N_REPEATS * PREDICTION_SIZE:
        raise RuntimeError("全部预测结果应有800条。")
    save_csv(summary, output_root / "ten_splits_mean_std.csv")
    final_tables = {"metrics": values, "mean_std": summary, "all_800_predictions": predictions,
                    "80_rows_wide": workbook_tables["80_rows_wide"],
                    "prediction_overlap": workbook_tables["prediction_overlap"],
                    **{f"repeat_{p.repeat:02d}_seed_{p.seed}": table for p, table in zip(plans, prediction_tables)}}
    save_excel_tables(output_root / "ten_splits_results.xlsx", final_tables, config.save_excel)
    print("\n10次预测集指标（%）：均值 ± 样本标准差，ddof=1。")
    print(summary[["metric", "mean_plus_minus_std_percent"]].to_string(index=False))
    print(f"全部结果保存于：{output_root}")
    return output_root


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n运行失败：{exc}", file=sys.stderr)
        raise
