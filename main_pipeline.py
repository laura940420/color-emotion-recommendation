import numpy as np
import pandas as pd
import os

# =============================================================================
# 匯入 Model 1 與 Model 2
# =============================================================================
try:
    from model1_LSM import predict_all_colours
    from model2_classification import set_binary_category_input, model2_predict_top3, feature_cols
    HAS_TEAM_CODE = True
except Exception as e:     
    HAS_TEAM_CODE = False
    print(f"系統切換至模擬模式，因為匯入失敗！真實的錯誤原因是：{e}") 

# 載入二進位映射表
if os.path.exists("country_binary_mapping.csv") and os.path.exists("lang_binary_mapping.csv"):
    country_mapping = pd.read_csv("country_binary_mapping.csv")
    lang_mapping = pd.read_csv("lang_binary_mapping.csv")
else:
    country_mapping, lang_mapping = None, None

# =============================================================================
# 模型三：動態邊界演算法 
# =============================================================================
def model3_dynamic_adjustment(m1_color_scores, user_desired_scores):
    emotion_keys = ["emotion_vitality", "emotion_stability", "emotion_resonance", "emotion_alert"]
    adjusted_scores = {}
    
    for key in emotion_keys:
        m1_val = m1_color_scores.get(key, 0.0)
        u_val = user_desired_scores.get(key, 0.0)
        
        # 在保留真實數據尺度的前提下計算差距
        delta = abs(u_val - m1_val)
        
        # 動態 Alpha (使用者權重) 壓制邏輯
        if delta > 3.5:
            alpha = 0.1  # 嚴重衝突：壓制使用者，只聽使用者 10%，主色佔 90%
        elif delta > 2.0:
            alpha = 0.5  # 中等衝突：雙方各佔 50%
        else:
            alpha = 0.8  # 差距極小：高度信任使用者意圖，佔 80%
            
        # 採用加權平均法
        new_score = (u_val * alpha) + (m1_val * (1.0 - alpha))
        
        # 確保最後結果不低於0，也不超過5
        adjusted_scores[key] = max(0.0, min(5.0, new_score))
        
    return adjusted_scores

# =============================================================================
#  模型二：貝氏分類器
# =============================================================================
def get_model2_recommendations(user_input_m2):
    if not HAS_TEAM_CODE:
        return [{"colour": "yellow", "prob": 0.45}, {"colour": "red", "prob": 0.25}, {"colour": "orange", "prob": 0.15}]
    
    final_input = {col: 0 for col in feature_cols}
    final_input.update(user_input_m2)
    
    # 修正Bug：向模型索取 Top-5，確保過濾主色後依然有足夠的顏色
    raw_recommendations = model2_predict_top3(input_dict=final_input, top_k=5)
    
    recommendations = []
    for rec in raw_recommendations:
        recommendations.append({
            "colour": rec["colour"],
            "prob": float(rec["posterior_probability"])
        })
        
    return recommendations

# =============================================================================
# 系統運作程式（含退火機制）
# =============================================================================
def run_advanced_pipeline(user_profile, main_color, raw_user_emotions, confidence_threshold=0.3):
    if HAS_TEAM_CODE:
        all_m1 = predict_all_colours(user_profile, params_dir='model1_params')
        m1_scores = all_m1.get(main_color, {"emotion_vitality": 1.0, "emotion_stability": 1.0, "emotion_resonance": 1.0, "emotion_alert": 1.0})
    else:
        m1_scores = {"emotion_vitality": 0.036, "emotion_stability": 0.251, "emotion_resonance": 1.246, "emotion_alert": 1.205}
    
    current_emotions = raw_user_emotions.copy()
    max_attempts = 3
    recommendations = []
    
    for attempt in range(1, max_attempts + 1):
        adjusted_emotions = model3_dynamic_adjustment(m1_scores, current_emotions)
        user_input_m2 = adjusted_emotions.copy()
        user_input_m2.update({"fluentenglish": user_profile["fluentenglish"], "gender": user_profile["gender"], "age_group": user_profile["age_group"]})
        
        if HAS_TEAM_CODE and country_mapping is not None and lang_mapping is not None:
            user_input_m2 = set_binary_category_input(user_input_m2, country_mapping, "residencecountry", "country_binary", "country", user_profile.get("country_code", "us"))
            user_input_m2 = set_binary_category_input(user_input_m2, lang_mapping, "mothertongue", "lang_binary", "lang", user_profile.get("lang_code", "en"))
        
        recommendations = get_model2_recommendations(user_input_m2)
        top1_prob = recommendations[0]["prob"]
        
        # 退火迴圈
        if top1_prob >= confidence_threshold:
            break
        else:
            for k in current_emotions.keys():
                current_emotions[k] = current_emotions[k] * 0.9 + 2.5 * 0.1
            
    # 修正Bug：過濾掉與主色相同的顏色後，取前三名 (Top-3)
    final_recs = [r for r in recommendations if r["colour"] != main_color][:3]
    return final_recs

# =============================================================================
# 後台初步測試主程式運作正常（常態測試在app.py）
# =============================================================================
def get_float_input(prompt_text):
    while True:
        try:
            val = float(input(prompt_text))
            if 0.0 <= val <= 5.0: return val
            print(" 請輸入 0.0 到 5.0 之間的數字")
        except ValueError:
            print(" 請輸入有效的數字")

if __name__ == "__main__":
    print(" 進入後台測試")
    
    valid_colors = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'turquoise', 'white', 'yellow']
    print("\n可選的主色有：")
    print(" | ".join(valid_colors))
    
    while True:
        main_color = input("\n請輸入你的簡報主色: ").strip().lower()
        if main_color in valid_colors:
            break
        print(f" 找不到這個顏色！可選顏色: {', '.join(valid_colors)}")

    print("\n請設定你期望簡報給人的感受 (0.0 ~ 5.0)：")
    v = get_float_input("    活力感 (Vitality) : ")
    s = get_float_input("    穩定感 (Stability): ")
    r = get_float_input("    共鳴感 (Resonance): ")
    a = get_float_input("    警示感 (Alert)    : ")
    
    user_emotions = {
        "emotion_vitality": v,
        "emotion_stability": s,
        "emotion_resonance": r,
        "emotion_alert": a
    }
    
    print("\n(載入預設使用者背景: 女性, 青年, 英語流利度 8, 居住美國, 英文母語)")
    user_profile = {"gender": 0, "age_group": 1, "fluentenglish": 8, "country_code": "us", "lang_code": "en"}
    
    input("\n設定完成！請按 [Enter] 鍵開始計算")
    
    run_advanced_pipeline(user_profile, main_color, user_emotions)