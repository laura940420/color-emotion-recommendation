"""
模型1：線性迴歸 — 顏色情緒預測
資料：DATA_binary.csv（二進制編碼版本）
方法：最小平方法（Least Square Method）
特徵：個人資訊（gender, age_group, fluentenglish,
              country_bin_1~6, lang_bin_1~5）+ 顏色 one-hot（12個）
目標：四大情緒分數（emotion_vitality, emotion_stability,
                   emotion_resonance, emotion_alert）
評估：MSE, RMSE, R²（訓練集 / 驗證集 / 測試集）
"""

import numpy as np
import pandas as pd
import os

# ============================================================
# 工具函式
# ============================================================

def train_val_test_split(X, y, val_ratio=0.1, test_ratio=0.1, seed=42):
    """80% train / 10% val / 10% test，隨機打亂"""
    np.random.seed(seed)
    idx    = np.random.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    n_val  = int(len(X) * val_ratio)
    return (X[idx[n_test+n_val:]], y[idx[n_test+n_val:]],
            X[idx[n_test:n_test+n_val]], y[idx[n_test:n_test+n_val]],
            X[idx[:n_test]],            y[idx[:n_test]])


def standardize(X_tr, X_val, X_te):
    """Z-score 標準化，以訓練集的均值與標準差為基準"""
    mean          = X_tr.mean(axis=0)
    std           = X_tr.std(axis=0)
    std[std == 0] = 1   # 常數欄位（std=0）不除以零
    return (X_tr-mean)/std, (X_val-mean)/std, (X_te-mean)/std, mean, std


def add_bias(X):
    """在最左側加一欄全 1（截距項 β₀）"""
    return np.hstack([np.ones((len(X), 1)), X])


def ols_fit(X, y):
    """
    最小平方法：
        β̂ = (XᵀX)⁻¹ Xᵀy
    使用 np.linalg.solve 求解。
    """
    return np.linalg.solve(X.T @ X, X.T @ y)


def predict(X, beta):
    return X @ beta


def mse(y_true, y_pred):
    """ Mean Squared Error"""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    """ Root Mean Squared Error"""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true, y_pred):
    """
    決定係數 R²
    = 1 - SS_res / SS_tot
    代表模型比「猜平均值」好多少比例
    """
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 0.0 if ss_tot == 0 else float(1 - np.sum((y_true-y_pred)**2) / ss_tot)


# ============================================================
# 主程式
# ============================================================

def main():
    # ----------------------------------------------------------
    # 1. 載入資料
    # ----------------------------------------------------------
    print("載入資料")
    df = pd.read_csv('DATA_binary.csv')
    print(f"資料筆數: {len(df)}，欄位數: {len(df.columns)}")

    target_cols   = ['emotion_vitality', 'emotion_stability',
                     'emotion_resonance', 'emotion_alert']
    personal_cols = [c for c in df.columns
                     if c not in target_cols + ['colour', 'colour_code']]

    print(f"\n個人資訊特徵（{len(personal_cols)} 個）: {personal_cols}")

    # ----------------------------------------------------------
    # 2. 建立顏色 one-hot，加入特徵
    # ----------------------------------------------------------
    colour_dummies   = pd.get_dummies(df['colour'], prefix='colour', dtype=int)
    colour_feat_cols = list(colour_dummies.columns)
    feature_cols     = personal_cols + colour_feat_cols

    print(f"顏色 one-hot（{len(colour_feat_cols)} 個）: {colour_feat_cols}")
    print(f"總特徵數: {len(feature_cols)}")

    df_model = pd.concat([df[personal_cols + target_cols], colour_dummies], axis=1)
    X_all    = df_model[feature_cols].values.astype(float)

    results      = []
    model_params = {}

    # ----------------------------------------------------------
    # 3. 每個情緒目標分別訓練一個模型
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("訓練")
    print("="*60)

    for target in target_cols:
        y_all = df_model[target].values.astype(float)

        # 切分資料
        X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(
            X_all, y_all, seed=42
        )

        # 標準化
        X_tr_s, X_val_s, X_te_s, feat_mean, feat_std = standardize(
            X_tr, X_val, X_te
        )

        # 加截距
        X_tr_b  = add_bias(X_tr_s)
        X_val_b = add_bias(X_val_s)
        X_te_b  = add_bias(X_te_s)

        # 訓練（OLS）
        beta = ols_fit(X_tr_b, y_tr)

        # 評估
        tr_mse  = mse(y_tr,  predict(X_tr_b,  beta))
        val_mse = mse(y_val, predict(X_val_b, beta))
        te_mse  = mse(y_te,  predict(X_te_b,  beta))
        tr_r2   = r2(y_tr,   predict(X_tr_b,  beta))
        val_r2  = r2(y_val,  predict(X_val_b, beta))
        te_r2   = r2(y_te,   predict(X_te_b,  beta))

        print(f"\n  [{target}]")
        print(f"    Train  MSE={tr_mse:.6f}  R²={tr_r2:.4f}")
        print(f"    Val    MSE={val_mse:.6f}  R²={val_r2:.4f}")
        print(f"    Test   MSE={te_mse:.6f}  R²={te_r2:.4f}")

        results.append({
            'target':    target,
            'train_MSE': round(tr_mse,  6),
            'val_MSE':   round(val_mse, 6),
            'test_MSE':  round(te_mse,  6),
            'train_R2':  round(tr_r2,   4),
            'val_R2':    round(val_r2,  4),
            'test_R2':   round(te_r2,   4),
        })

        model_params[target] = {
            'beta':             beta,
            'feat_mean':        feat_mean,
            'feat_std':         feat_std,
            'feature_cols':     feature_cols,
            'colour_feat_cols': colour_feat_cols,
        }

    # ----------------------------------------------------------
    # 4. 輸出結果表格
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("評估結果整理")
    print("="*60)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    df_results.to_csv('model1_results.csv', index=False)
    print("\n已儲存 model1_results.csv")

    # ----------------------------------------------------------
    # 5. 儲存模型參數（供模型3呼叫）
    # ----------------------------------------------------------
    os.makedirs('model1_params', exist_ok=True)
    for target, params in model_params.items():
        np.savez(
            f"model1_params/{target}.npz",
            beta             = params['beta'],
            feat_mean        = params['feat_mean'],
            feat_std         = params['feat_std'],
            feature_cols     = np.array(params['feature_cols']),
            colour_feat_cols = np.array(params['colour_feat_cols']),
        )
    print("已儲存模型參數至 model1_params/")

    return model_params, feature_cols, colour_feat_cols


# ============================================================
# 推論函式（供模型3呼叫）
# ============================================================

def load_model(target, params_dir='model1_params'):
    """載入指定情緒目標的模型參數"""
    data = np.load(f"{params_dir}/{target}.npz", allow_pickle=True)
    return {
        'beta':             data['beta'],
        'feat_mean':        data['feat_mean'],
        'feat_std':         data['feat_std'],
        'feature_cols':     list(data['feature_cols']),
        'colour_feat_cols': list(data['colour_feat_cols']),
    }


def inference(colour, personal_info: dict, params_dir='model1_params'):
    """
    給定顏色與個人資訊，預測四大情緒分數。

    personal_info 格式範例：
    {
        'gender':        1,   # 0=female, 1=male, 2=dnwta
        'age_group':     1,   # 0=minor, 1=young, 2=middle, 3=senior
        'fluentenglish': 7,   # 0-10
        'country_bin_1': 0,   # 二進制國家編碼（6個 bit）
        'country_bin_2': 1,
        'country_bin_3': 0,
        'country_bin_4': 1,
        'country_bin_5': 0,
        'country_bin_6': 0,
        'lang_bin_1':    1,   # 二進制語言編碼（5個 bit）
        'lang_bin_2':    0,
        'lang_bin_3':    0,
        'lang_bin_4':    0,
        'lang_bin_5':    0,
    }
    回傳：{'emotion_vitality': float, 'emotion_stability': float,
           'emotion_resonance': float, 'emotion_alert': float}
    """
    target_cols = ['emotion_vitality', 'emotion_stability',
                   'emotion_resonance', 'emotion_alert']
    predictions = {}

    for target in target_cols:
        params           = load_model(target, params_dir)
        feature_cols     = params['feature_cols']
        colour_feat_cols = params['colour_feat_cols']
        feat_mean        = params['feat_mean']
        feat_std         = params['feat_std']
        beta             = params['beta']

        # 建立輸入向量（顏色 one-hot + 個人資訊）
        x_raw = []
        for col in feature_cols:
            if col in colour_feat_cols:
                x_raw.append(1 if col == f'colour_{colour}' else 0)
            else:
                x_raw.append(personal_info.get(col, 0))
        x_raw = np.array(x_raw, dtype=float)

        # 標準化 → 加截距 → 預測 → clip 到 [0, 5]
        x_s = (x_raw - feat_mean) / feat_std
        x_b = np.concatenate([[1.0], x_s])
        predictions[target] = float(np.clip(x_b @ beta, 0, 5))

    return predictions


def predict_all_colours(personal_info: dict, params_dir='model1_params'):
    """
    給定個人資訊，一次預測所有 12 種顏色的情緒輪廓。
    回傳格式：{colour: {emotion: score, ...}, ...}
    供模型3使用。
    """
    colours = ['black', 'blue', 'brown', 'green', 'grey', 'orange',
               'pink', 'purple', 'red', 'turquoise', 'white', 'yellow']
    return {c: inference(c, personal_info, params_dir) for c in colours}


# ============================================================
# 執行
# ============================================================

if __name__ == '__main__':
    model_params, feature_cols, colour_feat_cols = main()

    # 推論示範
    print("\n" + "="*60)
    print("推論示範：yellow，年輕女性，英語流利")
    print("="*60)
    demo = {
        'gender': 0, 'age_group': 1, 'fluentenglish': 9,
        'country_bin_1': 0, 'country_bin_2': 1, 'country_bin_3': 0,
        'country_bin_4': 1, 'country_bin_5': 0, 'country_bin_6': 0,
        'lang_bin_1': 1,    'lang_bin_2': 0,    'lang_bin_3': 0,
        'lang_bin_4': 0,    'lang_bin_5': 0,
    }

    pred = inference('yellow', demo)
    print("  yellow 的情緒預測：")
    for k, v in pred.items():
        print(f"    {k}: {v:.4f}")

    print("\n  所有顏色情緒輪廓：")
    all_pred = predict_all_colours(demo)
    df_all   = pd.DataFrame(all_pred).T
    print(df_all.round(4).to_string())
