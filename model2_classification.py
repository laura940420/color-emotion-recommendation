import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 模型前資料整理與資料切分
# =============================================================================
# Step 1. 讀取資料
# =============================================================================
DATA_PATH = 'DATA_preprocessed.csv'

df = pd.read_csv(DATA_PATH) 
df.columns = df.columns.str.strip() # 把所有欄位名稱前後的空白去掉
country_mapping = pd.read_csv('country_binary_mapping.csv')
lang_mapping = pd.read_csv('lang_binary_mapping.csv')
print(df.shape)

# =============================================================================
# Step 2. 設定輸入特徵與目標
# =============================================================================
target_col = "colour"

emotion_features = [
    "emotion_vitality",
    "emotion_stability",
    "emotion_resonance",
    "emotion_alert"
]

basic_group_features = [
    "fluentenglish",
    "gender",
    "age_group"
]

country_features = [
    col for col in df.columns
    if col.startswith("country_bin_")
]

lang_features = [
    col for col in df.columns
    if col.startswith("lang_bin_")
]

# 把所有輸入特徵合併成一個 list
feature_cols = (
    emotion_features
    + basic_group_features
    + country_features
    + lang_features
)

print("Number of features:", len(feature_cols))

# =============================================================================
# Step 3. 整理模型資料
# =============================================================================
use_cols = feature_cols + [target_col]
df_model = df[use_cols].copy()

X = df_model[feature_cols].to_numpy(dtype=float)
y = df_model[target_col].to_numpy()

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nColour counts:")
print(df_model[target_col].value_counts().sort_index()) # .value_counts() 會統計每個顏色出現幾次；.sort_index() 則是按照顏色名稱排序。

# =============================================================================
# 做 7:2:1 stratified split
# =============================================================================
np.random.seed(42)

train_idx = []
val_idx = []
test_idx = []

classes = np.unique(y) # 取得所有顏色類別

for c in classes: # 每個顏色分開切成 7:2:1
    idx = np.where(y == c)[0] 
    np.random.shuffle(idx) # 把該顏色的資料順序打亂
    
    n = len(idx)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)
    
    train_idx.extend(idx[:n_train])
    val_idx.extend(idx[n_train:n_train + n_val])
    test_idx.extend(idx[n_train + n_val:])

# 把 list 轉成 NumPy array
train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

# 把 training、validation、test 裡面的 index 再各自打亂，因為資料是依照顏色一類一類加入的，打亂後更隨機
np.random.shuffle(train_idx)
np.random.shuffle(val_idx)
np.random.shuffle(test_idx)

X_train = X[train_idx] # X_train 是模型訓練用的輸入特徵
y_train = y[train_idx] # y_train 是模型訓練用的真實顏色標籤

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)
#%% 模型實作
# =============================================================================
# Step 5. 建立 Gaussian Bayes 訓練函式
# =============================================================================
def train_gaussian_bayes_params(X_train_input, y_train_input, reg):
    """
    訓練 Gaussian Bayesian Colour Classifier 的參數。

    Input:
        X_train_input: training features
        y_train_input: training labels
        reg: covariance regularization 的大小

    Output:
        model_params: 儲存模型參數的 dictionary
    """

    # 找出 training set 裡面所有顏色類別
    classes = np.unique(y_train_input)

    # 顏色類別數量
    n_classes = len(classes)

    # 輸入特徵數量，在本研究中包含情緒指標與族群資料
    n_features = X_train_input.shape[1]


    # -------------------------------------------------------------------------
    # 1. 計算每個顏色類別的 mean vector
    # -------------------------------------------------------------------------

    means = {}

    for c in classes:
        # 取出 training set 中屬於顏色 c 的所有資料
        X_c = X_train_input[y_train_input == c]

        # 計算該顏色類別的平均特徵向量
        # axis=0 是「沿著資料筆數方向取平均」，
        # 也就是每個特徵各自算平均
        means[c] = X_c.mean(axis=0)


    # -------------------------------------------------------------------------
    # 2. 計算 shared covariance
    # -------------------------------------------------------------------------

    centered_all = []

    for c in classes:
        # 取出顏色 c 的所有資料
        X_c = X_train_input[y_train_input == c]

        # 把該類別的資料扣掉自己的平均向量
        centered = X_c - means[c]

        # 把該類別中心化後的資料放進 centered_all
        centered_all.append(centered)

    # 把每個顏色的 centered data 上下堆疊成一個大矩陣
    centered_all = np.vstack(centered_all)

    # 計算 shared covariance matrix
    # rowvar=False 表示：每一欄是一個特徵，每一列是一筆資料
    shared_cov = np.cov(centered_all, rowvar=False)


    # -------------------------------------------------------------------------
    # 3. covariance regularization
    # -------------------------------------------------------------------------

    # 在 covariance matrix 的對角線加上一個很小的值
    shared_cov = shared_cov + reg * np.eye(n_features)


    # -------------------------------------------------------------------------
    # 4. uniform prior
    # -------------------------------------------------------------------------

    priors = {}

    for c in classes:
        # 設定每個顏色的 prior 都一樣
        # 由於 12 色資料量相當平均，因此使用 uniform prior
        # 讓每個顏色在模型判斷前有相同機會，最後主要由 likelihood 決定排序
        priors[c] = 1.0 / n_classes

    # -------------------------------------------------------------------------
    # 5. 預先計算 inverse covariance 與 log determinant
    # -------------------------------------------------------------------------

    # 計算 shared covariance matrix 的 pseudo-inverse
    # Gaussian likelihood 公式中會用到 covariance matrix 的反矩陣
    # 使用 pinv 比 inv 更穩定，可以避免矩陣接近 singular 時出錯
    inv_cov = np.linalg.pinv(shared_cov)

    # 計算 covariance matrix determinant 的 log 值
    sign, logdet = np.linalg.slogdet(shared_cov)

    # 檢查 covariance matrix 是否正常
    if sign <= 0:
        print(f"Warning: reg={reg} covariance matrix is not positive definite.")


    # -------------------------------------------------------------------------
    # 6. 將模型參數整理成 dictionary
    # -------------------------------------------------------------------------

    model_params = {
        "classes": classes,
        "n_classes": n_classes,
        "n_features": n_features,
        "means": means,
        "shared_cov": shared_cov,
        "inv_cov": inv_cov,
        "logdet": logdet,
        "priors": priors,
        "reg": reg
    }

    return model_params

# =============================================================================
# Step 6. 建立 posterior probability 計算函式
# =============================================================================
def compute_posterior_probability(X_input, model_params):
    """
    輸入一批資料 X_input，
    輸出每筆資料屬於 12 個顏色的 posterior probability。
    """

    # 從 model_params 中取出訓練好的模型參數
    classes = model_params["classes"]
    n_features = model_params["n_features"]
    means = model_params["means"]
    inv_cov = model_params["inv_cov"]
    logdet = model_params["logdet"]
    priors = model_params["priors"]

    # 建立空 list，用來存每個顏色類別的未正規化 posterior 分數
    all_log_values = []

    for c in classes:
        # 取出顏色 c 的 mean vector
        mean = means[c]

        # 計算輸入資料與該顏色平均向量之間的差距
        diff = X_input - mean

        # 計算 Gaussian likelihood 裡面的 quadratic term
        # 對應公式：(x - mu)^T Sigma^{-1} (x - mu)
        # 可以理解為「考慮 covariance 後的距離」
        #
        # np.einsum() 會對每一筆資料都算出一個 quadratic distance
        quadratic = np.einsum(
            "ij,jk,ik->i",
            diff,
            inv_cov,
            diff
        )

        # 計算 multivariate Gaussian 的 log likelihood
        log_likelihood = -0.5 * (
            n_features * np.log(2 * np.pi)
            + logdet
            + quadratic
        )

        # 計算該顏色 prior probability 的 log
        log_prior = np.log(priors[c])

        # 把 likelihood 和 prior 結合
        log_value = log_likelihood + log_prior

        # 把目前顏色類別的分數存起來
        all_log_values.append(log_value)

    # 轉置後，每一列是一筆資料，每一欄是一個顏色
    all_log_values = np.vstack(all_log_values).T

    # -------------------------------------------------------------------------
    # 轉成 posterior probability
    # -------------------------------------------------------------------------

    # 先減掉每筆資料中的最大值，確保數值穩定
    all_log_values = all_log_values - np.max(
        all_log_values,
        axis=1,
        keepdims=True
    )

    # 用 exponential 把 log value 轉回普通數值
    exp_values = np.exp(all_log_values)

    # 除以總和，轉成機率
    posterior_prob = exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )

    return posterior_prob

# =============================================================================
# Step 7. 用 validation set 選 covariance regularization
# =============================================================================
# 測試不同的 covariance regularization 大小
# validation set 會用來選出表現最好的 reg
reg_candidates = [1e-6, 1e-5, 1e-4, 1e-3]

validation_results = []

for reg in reg_candidates:

    # -------------------------------------------------------------------------
    # 1. 用 training set 訓練目前 reg 對應的 Gaussian Bayes 模型參數
    # -------------------------------------------------------------------------

    temp_model = train_gaussian_bayes_params(
        X_train_input=X_train,
        y_train_input=y_train,
        reg=reg
    )


    # -------------------------------------------------------------------------
    # 2. 在 validation set 上計算 posterior probability
    # -------------------------------------------------------------------------

    val_prob = compute_posterior_probability(
        X_input=X_val,
        model_params=temp_model
    )

    classes = temp_model["classes"]


    # -------------------------------------------------------------------------
    # 3. Validation Top-1 accuracy
    # -------------------------------------------------------------------------

    # 找出每筆資料 posterior probability 最大的顏色
    val_top1_idx = np.argmax(val_prob, axis=1)

    # 把顏色 index 轉成顏色名稱
    val_top1_pred = classes[val_top1_idx]

    # 計算 Top-1 accuracy
    val_top1_acc = np.mean(val_top1_pred == y_val)


    # -------------------------------------------------------------------------
    # 4. Validation Top-3 accuracy
    # -------------------------------------------------------------------------

    # 找出每筆資料 posterior probability 最高的前三個顏色
    val_top3_idx = np.argsort(val_prob, axis=1)[:, ::-1][:, :3]

    # 把 Top-3 的 index 轉成實際顏色名稱
    val_top3_pred = classes[val_top3_idx]

    # 檢查每筆資料的真實顏色是否有出現在 Top-3 裡
    val_top3_correct = []

    for i in range(len(y_val)):
        val_top3_correct.append(y_val[i] in val_top3_pred[i])

    # 計算 Top-3 accuracy
    val_top3_acc = np.mean(val_top3_correct)


    # -------------------------------------------------------------------------
    # 5. 儲存目前 reg 的 validation 結果
    # -------------------------------------------------------------------------

    validation_results.append({
        "reg": reg,
        "val_top1": val_top1_acc,
        "val_top3": val_top3_acc
    })
    
# =============================================================================
# Step 8. 選出最佳 reg
# =============================================================================
# 把 validation_results 轉成 DataFrame，方便查看結果
validation_results_df = pd.DataFrame(validation_results)
print("\nValidation tuning results:")
print(validation_results_df)

# 使用 validation Top-3 accuracy 作為選擇 reg 的標準
best_row = validation_results_df.loc[
    validation_results_df["val_top3"].idxmax()
]

# 取出 validation Top-3 accuracy 最好的 reg
best_reg = best_row["reg"]

print("\nBest regularization parameter:")
print(f"best_reg = {best_reg:.0e}")
print(f"Best validation Top-3 = {best_row['val_top3']:.4f}")

# =============================================================================
# Step 9. 用最佳 reg 訓練 final model
# =============================================================================
X_train_final = np.vstack([X_train, X_val])
y_train_final = np.concatenate([y_train, y_val])

final_model = train_gaussian_bayes_params(
    X_train_input=X_train_final,
    y_train_input=y_train_final,
    reg=best_reg
)

print("\nFinal model training finished.")
print("Number of classes:", final_model["n_classes"])
print("Number of features:", final_model["n_features"])
print("Best reg:", final_model["reg"])
print("Shared covariance shape:", final_model["shared_cov"].shape)
#%% 模型結果
# =============================================================================
# Step 10. 評估 final model on test set
# =============================================================================
# 使用 final model 計算 test set 的 posterior probability
test_prob = compute_posterior_probability(
    X_input=X_test,
    model_params=final_model
)

classes = final_model["classes"]

# Top-1 prediction
test_top1_idx = np.argmax(test_prob, axis=1)
test_top1_pred = classes[test_top1_idx]
test_top1_acc = np.mean(test_top1_pred == y_test)

# Top-3 prediction
test_top3_idx = np.argsort(test_prob, axis=1)[:, ::-1][:, :3]
test_top3_pred = classes[test_top3_idx]

# 檢查每筆資料的真實顏色是否有出現在 Top-3 裡
test_top3_correct = []

for i in range(len(y_test)):
    test_top3_correct.append(y_test[i] in test_top3_pred[i])

test_top3_acc = np.mean(test_top3_correct)

print("===== Final Test Result =====")
print(f"Top-1 accuracy = {test_top1_acc:.4f}")
print(f"Top-3 accuracy = {test_top3_acc:.4f}")

#%% 模型補充評估
# =============================================================================
# Step 11. 整理 posterior probability 排序
# =============================================================================
sorted_prob = np.sort(test_prob, axis=1)[:, ::-1] # sorted_prob 會把每筆資料對 12 個顏色的 posterior probability 由大到小排序

top1_prob = sorted_prob[:, 0] # 取出每一筆資料最高的 posterior probability
top2_prob = sorted_prob[:, 1]
top3_prob = sorted_prob[:, 2]

# Top-1 / Top-2 margin：
# 如果 margin 很小，代表模型在第一名和第二名顏色之間其實很猶豫
top1_top2_margin = top1_prob - top2_prob

# Top-3 probability sum：
# 表示模型把多少總機率集中在前三個推薦顏色上
top3_prob_sum = np.sum(sorted_prob[:, :3], axis=1)

print("\n===== Posterior Confidence / Uncertainty =====")
print(f"Mean Top-1 posterior probability = {np.mean(top1_prob):.4f}")
print(f"Median Top-1 posterior probability = {np.median(top1_prob):.4f}")
print(f"Mean Top-1 / Top-2 margin = {np.mean(top1_top2_margin):.4f}")
print(f"Median Top-1 / Top-2 margin = {np.median(top1_top2_margin):.4f}")
print(f"Mean Top-3 probability sum = {np.mean(top3_prob_sum):.4f}")
print(f"Median Top-3 probability sum = {np.median(top3_prob_sum):.4f}")

# =============================================================================
# Step 12. Reject / Doubt option analysis
# =============================================================================
# 這裡用 Top-1 posterior probability 當作信心指標。
# 如果 top1_prob >= threshold，模型才接受這筆推薦。
# 如果 top1_prob < threshold，視為 reject / doubt case。
# =============================================================================
# 放適合呈現的三個 threshold
confidence_thresholds = [0.30, 0.40, 0.50]

# 把原本的 Top-3 正確結果轉成 NumPy array
test_top1_correct_array = np.array(test_top1_pred == y_test)
test_top3_correct_array = np.array(test_top3_correct)

reject_records = []

for th in confidence_thresholds:

    # accept_mask=True 表示模型對這筆資料有足夠信心，可以給出推薦
    accept_mask = top1_prob >= th
    reject_mask = ~accept_mask # 沒被接受的樣本，就是 reject / doubt case

    coverage = np.mean(accept_mask) # 計算模型接受多少比例的樣本
    reject_ratio = np.mean(reject_mask) # 計算模型拒絕多少比例的樣本
 
    if np.sum(accept_mask) > 0: # 檢查有沒有任何樣本被接受
        accepted_top1_acc = np.mean(test_top1_correct_array[accept_mask]) # 只針對被接受的樣本，計算 Top-1 accuracy
        accepted_top3_acc = np.mean(test_top3_correct_array[accept_mask]) # 只針對被接受的樣本，計算 Top-3 accuracy
        accepted_mean_top1_prob = np.mean(top1_prob[accept_mask]) # 計算被接受樣本的平均 Top-1 posterior probability
        accepted_mean_margin = np.mean(top1_top2_margin[accept_mask]) # 計算被接受樣本的平均 Top-1 / Top-2 margin
    else: # 如果沒有任何樣本被接受，就把結果設成 NaN
        accepted_top1_acc = np.nan
        accepted_top3_acc = np.nan
        accepted_mean_top1_prob = np.nan
        accepted_mean_margin = np.nan

    reject_records.append({
        "accept_if_top1_prob_at_least": th,
        "coverage": coverage,
        "reject_ratio": reject_ratio,
        "accepted_top1_acc": accepted_top1_acc,
        "accepted_top3_acc": accepted_top3_acc,
        "accepted_mean_top1_prob": accepted_mean_top1_prob,
        "accepted_mean_top1_top2_margin": accepted_mean_margin
    })

reject_df = pd.DataFrame(reject_records)

print("\n===== Reject / Doubt Option Analysis =====")
print(reject_df.to_string(index=False))

# =============================================================================
# Step 13. Margin / Uncertainty analysis
# =============================================================================

low_margin_threshold = 0.05 # 設定 margin 門檻
low_margin_mask = top1_top2_margin < low_margin_threshold # 找出哪些樣本是低 margin
low_margin_ratio = np.mean(low_margin_mask) # 計算低 margin 樣本比例

print("\n===== Margin / Uncertainty Analysis =====")
print(f"Median Top-1 / Top-2 margin = {np.median(top1_top2_margin):.4f}") # 印出 Top-1 / Top-2 margin 的中位數
print(f"Low-margin ratio, margin < {low_margin_threshold:.2f} = {low_margin_ratio:.4f}")
print(f"Number of low-margin samples = {np.sum(low_margin_mask)}")

# =============================================================================
# Step 14. Row-normalized confusion matrix
# =============================================================================
# 混淆矩陣：
# row = true colour
# column = predicted colour
#
# 使用 row normalization，
# 因此每一列代表「真正屬於該顏色的樣本，被預測成各顏色的比例」。
# =============================================================================

# 使用模型原本的類別順序
plot_order = [
    "brown",
    "grey",
    "black",
    "white",
    "pink",
    "red",
    "yellow",
    "orange",
    "blue",
    "green",
    "purple",
    "turquoise",
]

# 建立 label 到 index 的對應
label_to_idx = {
    label: i
    for i, label in enumerate(plot_order)
}

n_classes = len(plot_order)

# 建立 confusion matrix counts
cm_counts = np.zeros(
    (n_classes, n_classes),
    dtype=int
)

for true_label, pred_label in zip(y_test, test_top1_pred):

    true_i = label_to_idx[true_label]
    pred_j = label_to_idx[pred_label]

    cm_counts[true_i, pred_j] += 1


# =============================================================================
# Row-normalized confusion matrix (%)
# =============================================================================

row_sums = cm_counts.sum(axis=1, keepdims=True)

# 避免某個類別在 test set 中沒有樣本時產生除以零錯誤
cm_percent = np.divide(
    cm_counts,
    row_sums,
    out=np.zeros_like(cm_counts, dtype=float),
    where=row_sums != 0
) * 100


# =============================================================================
# 儲存 confusion matrix CSV
# =============================================================================

cm_counts_df = pd.DataFrame(
    cm_counts,
    index=plot_order,
    columns=plot_order
)

cm_percent_df = pd.DataFrame(
    cm_percent,
    index=plot_order,
    columns=plot_order
)

cm_counts_df.to_csv(
    "model2_confusion_matrix_counts.csv",
    encoding="utf-8-sig"
)

cm_percent_df.to_csv(
    "model2_confusion_matrix_row_percent.csv",
    encoding="utf-8-sig"
)


# =============================================================================
# Plot confusion matrix
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 9))

max_value = np.max(cm_percent)

im = ax.imshow(
    cm_percent,
    cmap="Blues",
    vmin=0,
    vmax=max_value
)

# 座標軸設定
ax.set_xticks(np.arange(n_classes))
ax.set_yticks(np.arange(n_classes))

ax.set_xticklabels(
    plot_order,
    rotation=45,
    ha="right",
    fontsize=11
)

ax.set_yticklabels(
    plot_order,
    fontsize=11
)

ax.set_xlabel(
    "Predicted Colour",
    fontsize=14
)

ax.set_ylabel(
    "True Colour",
    fontsize=14
)

ax.set_title(
    "Row-normalized Confusion Matrix of Model 2 (Top-1 Prediction)",
    fontsize=16,
    fontweight="bold"
)

# Colorbar
cbar = plt.colorbar(im, ax=ax)

cbar.set_label(
    "Percentage within true colour (%)",
    fontsize=12
)

# 標註數值：
# 顯示對角線，以及比例大於等於 5% 的非對角線項目
for i in range(n_classes):

    for j in range(n_classes):

        value = cm_percent[i, j]

        if (i == j) or (value >= 5):

            text_color = (
                "white"
                if max_value > 0 and value > max_value * 0.55
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9
            )

plt.tight_layout()
plt.show()

print("\nSaved confusion matrix files:")
print("model2_confusion_matrix_counts.csv")
print("model2_confusion_matrix_row_percent.csv")


# =============================================================================
# Step 15. 儲存補充評估結果
# =============================================================================

reject_df.to_csv(
    "model2_reject_doubt_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved additional evaluation file:")
print("model2_reject_doubt_analysis.csv")

#%% 給整合層呼叫的 function
# =============================================================================
# Step 16. 建立 Model 2 推薦函式
# =============================================================================
def set_binary_category_input(user_input, mapping_df, category_col, binary_col, prefix, category_name):
    """
    把指定類別名稱轉成 binary encoding，並填入 user_input。

    user_input:
        原本的輸入 dictionary

    mapping_df:
        country_mapping 或 lang_mapping

    category_col:
        原始類別欄位名稱，例如 'residencecountry' 或 'mothertongue'

    binary_col:
        binary 編碼欄位名稱，例如 'country_binary' 或 'lang_binary'

    prefix:
        欄位前綴，例如 'country' 或 'lang'

    category_name:
        使用者指定的國家或語言，例如 'us'、'en'
    """

    # 如果輸入的類別不在 mapping 裡，就改成 other
    available_categories = mapping_df[category_col].tolist()

    if category_name not in available_categories:
        category_name = "other"

    # 找到該類別的 binary code
    binary_code = mapping_df.loc[
        mapping_df[category_col] == category_name,
        binary_col
    ].values[0]

    # 保險起見，轉成字串
    binary_code = str(binary_code)

    # 把 binary code 填入 user_input
    for i, bit in enumerate(binary_code, start=1):
        col_name = f"{prefix}_bin_{i}"

        if col_name in user_input:
            user_input[col_name] = int(bit)

    return user_input

def model2_predict_top3(input_dict, top_k=3):
    """
    Model 2 final prediction function.

    功能：
    給定一筆使用者輸入資料，輸出 Top-k 顏色推薦結果。

    input_dict 必須包含 feature_cols 裡面的所有欄位，
    並且欄位名稱要和訓練模型時使用的 feature_cols 一致。

    Output:
    [
        {"rank": 1, "colour": ..., "posterior_probability": ...},
        {"rank": 2, "colour": ..., "posterior_probability": ...},
        {"rank": 3, "colour": ..., "posterior_probability": ...}
    ]
    """

    # 1. 按照訓練時的 feature_cols 順序整理輸入
    x_input = np.array(
        [[input_dict[col] for col in feature_cols]],
        dtype=float
    )

    # 2. 計算這筆資料對 12 種顏色的 posterior probability
    prob = compute_posterior_probability(
        X_input=x_input,
        model_params=final_model
    )
    
    # 3. 從 final_model 裡取出顏色類別順序
    classes = final_model["classes"]
    
    # 4. 取 posterior probability 最高的 Top-k 顏色
    topk_idx = np.argsort(prob, axis=1)[:, ::-1][:, :top_k]

    topk_colours = classes[topk_idx][0]
    topk_probs = prob[0, topk_idx[0]]

    # 5. 輸出結果
    recommendations = []

    for rank, (colour, p) in enumerate(zip(topk_colours, topk_probs), start=1):
        recommendations.append({
            "rank": rank,
            "colour": colour,
            "posterior_probability": float(p)
        })

    return recommendations

# =============================================================================
# 設定的一筆使用者輸入測試
# =============================================================================
user_input = {col: 0 for col in feature_cols}

# 4 個情緒指標，範圍是 0 ~ 5
user_input["emotion_vitality"] = 4.5
user_input["emotion_stability"] = 3.5
user_input["emotion_resonance"] = 4.0
user_input["emotion_alert"] = 1.5

# 基本族群資料
user_input["fluentenglish"] = 8
user_input["gender"] = 0
user_input["age_group"] = 2

# country / language binary encoding
user_input = set_binary_category_input(
    user_input=user_input,
    mapping_df=country_mapping,
    category_col="residencecountry",
    binary_col="country_binary",
    prefix="country",
    category_name="us" # 這裡改國家
)

user_input = set_binary_category_input(
    user_input=user_input,
    mapping_df=lang_mapping,
    category_col="mothertongue",
    binary_col="lang_binary",
    prefix="lang",
    category_name="en" # 這裡改語言
)

recommendations = model2_predict_top3(
    input_dict=user_input,
    top_k=3
)

print("Custom input Top-3 recommendation:")

for rec in recommendations:
    print(
        f"{rec['rank']}. {rec['colour']} "
        f"posterior probability = {rec['posterior_probability']:.4f}"
    )
    