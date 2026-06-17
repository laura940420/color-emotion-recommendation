import streamlit as st
import pandas as pd
import os
from main_pipeline import run_advanced_pipeline 

# ==========================================
# 1. 網頁基本設定與Caching
# ==========================================
st.set_page_config(page_title="簡報視覺配色模型", page_icon="🎨", layout="wide", initial_sidebar_state="expanded")

hide_streamlit_style = """
<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_data
def load_csv_mappings():
    cmap, lmap = None, None
    if os.path.exists("country_binary_mapping.csv"):
        cmap = pd.read_csv("country_binary_mapping.csv")
    if os.path.exists("lang_binary_mapping.csv"):
        lmap = pd.read_csv("lang_binary_mapping.csv")
    return cmap, lmap

country_mapping, lang_mapping = load_csv_mappings()

st.title("簡報視覺配色模型")
st.markdown("##### 基於機器學習，為您的商業提案與專案報告計算最具說服力的視覺搭配。")
st.divider()

# ==========================================
# 2. 國家語言對應字典 
# ==========================================
# 從 preprocessing_DATA.csv 取出的國家字典
COUNTRY_MAP = {
    'az': 'Azerbaijan', 'gr': 'Greece', 'no': 'Norway', 'de': 'Germany', 'ch': 'Switzerland', 
    'se': 'Sweden', 'eg': 'Egypt', 'es': 'Spain', 'pl': 'Poland', 'cn': 'China', 
    'gb': 'United Kingdom', 'fi': 'Finland', 'ee': 'Estonia', 'lt': 'Lithuania', 'ru': 'Russia', 
    'mx': 'Mexico', 'ge': 'Georgia', 'it': 'Italy', 'co': 'Colombia', 'us': 'United States', 
    'rs': 'Serbia', 'nl': 'Netherlands', 'fr': 'France', 'sa': 'Saudi Arabia', 'ir': 'Iran', 
    'be': 'Belgium', 'il': 'Israel', 'ng': 'Nigeria', 'nz': 'New Zealand', 'ua': 'Ukraine', 
    'at': 'Austria', 'ph': 'Philippines', 'jp': 'Japan', 'cy': 'Cyprus', 'lv': 'Latvia', 
    'in': 'India', 'hr': 'Croatia', 'uy': 'Uruguay', 'pf': 'French Polynesia', 'tl': 'East Timor', 
    'za': 'South Africa', 'sg': 'Singapore', 'pt': 'Portugal', 'cl': 'Chile', 'my': 'Malaysia', 
    'mc': 'Monaco', 'ae': 'United Arab Emirates', 'dk': 'Denmark', 'pn': 'Pitcairn Islands', 
    'ke': 'Kenya', 'nu': 'Niue', 'dnwta': 'Do Not Want To Answer', 'om': 'Oman', 
    'do': 'Dominican Republic', 'ma': 'Morocco', 'sr': 'Suriname', 'pk': 'Pakistan', 
    'je': 'Jersey', 'sz': 'Eswatini', 'uz': 'Uzbekistan', 'md': 'Moldova', 'ki': 'Kiribati', 
    'ba': 'Bosnia and Herzegovina', 'br': 'Brazil', 'vn': 'Vietnam', 'pa': 'Panama', 
    'by': 'Belarus', 'ca': 'Canada', 'ar': 'Argentina', 'li': 'Liechtenstein', 'lu': 'Luxembourg', 
    'au': 'Australia', 'ad': 'Andorra', 'bg': 'Bulgaria', 'mt': 'Malta', 'so': 'Somalia', 
    'ug': 'Uganda', 'ie': 'Ireland', 'tr': 'Turkey', 'sk': 'Slovakia', 'id': 'Indonesia'
}

# 從 preprocessing_DATA.csv 取出的語言字典
LANG_MAP = {
    'az': 'Azerbaijani', 'gr': 'Greek', 'no': 'Norwegian', 'de': 'German', 
    'fr': 'French', 'se': 'Swedish', 'ar': 'Arabic', 'es': 'Spanish', 
    'pl': 'Polish', 'zh': 'Chinese', 'en': 'English', 'fi': 'Finnish', 
    'et': 'Estonian', 'lt': 'Lithuanian', 'ru': 'Russian', 'ka': 'Georgian', 
    'it': 'Italian', 'sr': 'Serbian', 'nl': 'Dutch', 'fa': 'Persian', 
    'he': 'Hebrew', 'uk': 'Ukrainian', 'ja': 'Japanese', 'lv': 'Latvian', 
    'hr': 'Croatian', 'dnwta': 'Do Not Want To Answer','el': 'Greek','ig': 'Igbo','sv': 'Swedish',
}

# 格式化函式：確保去除空白並轉小寫比對，失敗時輸出大寫
def format_country(code):
    clean_code = str(code).lower().strip()
    return COUNTRY_MAP.get(clean_code, str(code).upper().strip())

def format_lang(code):
    clean_code = str(code).lower().strip()
    return LANG_MAP.get(clean_code, str(code).upper().strip())

# ==========================================
# 3. 側邊欄 - 受眾設定
# ==========================================
with st.sidebar:
    st.header("👤 受眾設定")
    st.markdown("系統將依據您設定的受眾文化與背景，動態微調機率分佈。")
    
    gender_choice = st.radio("受眾主要性別？", ["女性", "男性", "不願透露"])
    age_choice = st.selectbox("受眾年齡層？", ["未成年 (<18)", "青年 (18-35)", "中年 (36-55)", "壯年 (56+)"], index=1)
    
    # 【防呆機制】：限制流利度區間 (6~9)，避免引發 OOD 導致模型崩潰
    fluency_level = st.selectbox("受眾英語流利度", ["基礎 (6)", "中等 (7)", "流利 (8)", "精通 (9)"], index=2)
    fluent_val = int(fluency_level[-2])
    
    st.markdown("---")
    st.markdown("#### 🌍 地緣與文化特徵")
    
    # 國家選單優化：自動整合二進位對應檔，並按照全名A-Z排序
    if country_mapping is not None:
        model_countries = country_mapping['residencecountry'].dropna().str.lower().str.strip().unique().tolist()
        country_options = sorted(model_countries, key=lambda x: COUNTRY_MAP.get(x, x.upper()))
        default_c = "us" if "us" in country_options else country_options[0]
        country_choice = st.selectbox("🌎 目標市場 (國家)", country_options, index=country_options.index(default_c), format_func=format_country)
    else:
        country_options = sorted(list(COUNTRY_MAP.keys()), key=lambda x: COUNTRY_MAP.get(x, x.upper()))
        country_choice = st.selectbox("🌎 目標市場 (國家)", country_options, format_func=format_country)
        
    # 語言選單優化：自動整合二進位對應檔，並按照A-Z排序
    if lang_mapping is not None:
        model_langs = lang_mapping['mothertongue'].dropna().str.lower().str.strip().unique().tolist()
        lang_options = sorted(model_langs, key=lambda x: LANG_MAP.get(x, x.upper()))
        default_l = "en" if "en" in lang_options else lang_options[0]
        lang_choice = st.selectbox("🗣️ 溝通母語", lang_options, index=lang_options.index(default_l), format_func=format_lang)
    else:
        lang_options = sorted(list(LANG_MAP.keys()), key=lambda x: LANG_MAP.get(x, x.upper()))
        lang_choice = st.selectbox("🗣️ 溝通母語", lang_options, format_func=format_lang)
    
    gender_map = {"女性": 0, "男性": 1, "不願透露": 2}
    age_map = {"未成年 (<18)": 0, "青年 (18-35)": 1, "中年 (36-55)": 2, "壯年 (56+)": 3}
    user_profile = {"gender": gender_map[gender_choice], "age_group": age_map[age_choice], "fluentenglish": fluent_val, "country_code": country_choice, "lang_code": lang_choice}

# ==========================================
# 4. 主畫面 - 情緒微調介面
# ==========================================
all_colors = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'turquoise', 'white', 'yellow']

st.subheader("1️⃣ 定義品牌/簡報主色 (Primary Color)")
main_color = st.selectbox("請選擇您這份提案選定的「主要底色」：", all_colors)

st.divider()

st.subheader("2️⃣ 策略性視覺感受 (Strategic Visual Impact)")
st.markdown("請調控以下情緒指標分數，將引導模型為您推薦最佳配色。")

# ==========================================
# 4-1. 指標說明函式
# ==========================================

st.markdown(
    """
    <style>
    .indicator-box {
        background-color: #F5F5F5;
        border: 1px solid #D9D9D9;
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 6px;
        margin-bottom: 18px;
        color: #333333;
        font-size: 0.92rem;
        line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def indicator_explanation(user_text, usage_text):
    st.markdown(
        f"""
        <div class="indicator-box">
            <b>指標說明：</b>{user_text}<br><br>
            <b>適合用途：</b>{usage_text}
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 4-2. 四個情緒指標輸入
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏃 活力感 (Vitality)")
    v = st.slider(
        "活力感分數",
        0.0, 5.0, 2.5, 0.1,
        label_visibility="collapsed"
    )
    indicator_explanation(
        user_text="開心、有趣、吸引人、想參與。",
        usage_text="社團活動、活動海報、IG 宣傳圖、餐飲娛樂、活動行銷。"
    )

    st.markdown("#### 💧 共鳴感 (Resonance)")
    r = st.slider(
        "共鳴感分數",
        0.0, 5.0, 2.5, 0.1,
        label_visibility="collapsed"
    )
    indicator_explanation(
        user_text="讓人有感、同理、反思、情緒沉澱。",
        usage_text="心理健康貼文、公益宣傳、議題型海報、紀念活動、社會議題宣傳。"
    )

with col2:
    st.markdown("#### 🧘 穩定感 (Stability)")
    s = st.slider(
        "穩定感分數",
        0.0, 5.0, 2.5, 0.1,
        label_visibility="collapsed"
    )
    indicator_explanation(
        user_text="可靠、安心、有完成度、看起來正式。",
        usage_text="課堂簡報、期末專題、成果發表、金融醫療、企業簡報、科技產品。"
    )

    st.markdown("#### ⚠️ 警示感 (Alert)")
    a = st.slider(
        "警示感分數",
        0.0, 5.0, 2.5, 0.1,
        label_visibility="collapsed"
    )
    indicator_explanation(
        user_text="緊張、強烈、提醒、危機感。",
        usage_text="警示海報、反詐騙宣傳、安全提醒、危機宣導、強烈議題廣告。"
    )


user_emotions = {
    "emotion_vitality": v,
    "emotion_stability": s,
    "emotion_resonance": r,
    "emotion_alert": a
}

st.divider()

# ==========================================
# 5. 色碼與顏色解釋字典
# ==========================================
UI_COLOR_MAP = {
    "black": "#1E293B", "blue": "#3B82F6", "brown": "#92400E", "green": "#10B981", 
    "grey": "#64748B", "orange": "#F97316", "pink": "#EC4899", "purple": "#8B5CF6", 
    "red": "#EF4444", "turquoise": "#06B6D4", "white": "#F8FAFC", "yellow": "#EAB308"
}

XAI_EXPLANATION = {
    "black": "深邃且具備絕對權威感，在色彩心理學中能最大化『穩定感』與高端商業價值，適合奠定不可動搖的專業基調。",
    "blue": "全球百大企業最愛的信任色，具備極高的『穩定』與『理性』特質，能有效降低受眾防備心並提升說服力。",
    "brown": "帶有大地般的沉穩與務實感，能建立溫暖且可靠的『共鳴感』，適合強調永續、傳統或穩健成長的提案。",
    "green": "象徵成長、和平與生機，能帶來極佳的『共鳴感』與『穩定感』，是推動 ESG、環保或友善創新的首選視覺。",
    "grey": "中立且不喧賓奪主的高級背景色，能完美襯托數據與重點，提供無壓力的閱讀體驗，展現極致的『專業穩定』。",
    "orange": "充滿熱情與平易近人的擴張色，能激發高度的『活力動能』，且比紅色更具親和力，適合促銷或激勵型提案。",
    "pink": "具備柔和與創新的顛覆性特質，能創造強烈的『情感共鳴』與記憶點，適合打破常規、強調溫柔堅定的破壞式創新。",
    "purple": "融合了紅色的活力與藍色的穩定，帶有神秘與尊貴感，適合強調『獨特性』、『奢華』或『前瞻性科技』的場景。",
    "red": "最具視覺衝擊力的高能色彩，具備頂尖的『警示驅動』與『活力動能』，能瞬間聚焦受眾視線並促成行動。",
    "turquoise": "結合藍色科技感與綠色生機，給人清晰、靈動的『活力感』，非常適合數位轉型、醫療科技或年輕品牌。",
    "white": "極簡、純粹且包容萬物，能提供最大的呼吸空間，讓內容本身成為主角，是現代極簡商業設計的終極武器。",
    "yellow": "自帶光源的極亮色，能爆發出最強的『活力動能』與樂觀情緒，適合激發創意、傳遞希望或作為強烈的重點提示。"
}

def get_text_color(bg_color_name):
    dark_colors = ['black', 'blue', 'brown', 'green', 'purple', 'red', 'grey']
    return "white" if bg_color_name.lower() in dark_colors else "#1E293B"

# ==========================================
# 6. 執行推論與結果展示
# ==========================================
if st.button("✨ 啟動模型計算最佳配色", use_container_width=True, type="primary"):
    with st.spinner("模型運算中..."):
        
        recommendations = run_advanced_pipeline(user_profile, main_color, user_emotions)
        st.success("✅ 分析完成！")
        
        c1, c2, c3 = recommendations[0]['colour'], recommendations[1]['colour'], recommendations[2]['colour']
        hex_main = UI_COLOR_MAP.get(main_color, main_color)
        hex_c1, hex_c2, hex_c3 = UI_COLOR_MAP.get(c1, c1), UI_COLOR_MAP.get(c2, c2), UI_COLOR_MAP.get(c3, c3)
        
        st.subheader(f"🎨 針對主色「{main_color.upper()}」，模型推薦的最佳搭配：")
        
        res_cols = st.columns(3)
        for idx, col in enumerate(res_cols):
            if idx < len(recommendations):
                rec = recommendations[idx]
                real_prob = rec['prob']
                
                if real_prob >= 0.30: badge = "🔥 Excellent"
                elif real_prob >= 0.15: badge = "⭐ good"
                else: badge = "💡 considerable"
                    
                col.metric(
                    label=f"🏆 第 {idx+1} 名", 
                    value=rec['colour'].capitalize(), 
                    delta=f"機率: {real_prob:.1%} | {badge}",
                    delta_color="off"
                )
                
        # 實體配色預覽
        st.write("")
        st.markdown("#### 👀 參考配色 (Color Palette)")
        st.markdown("本配色僅作為參考示意，由於模型以英文色彩詞彙進行判斷，實際結果著重於色彩意圖，而非完全對應圖中色碼。")
        
        main_border = "border: 1px solid #E2E8F0;" if main_color == "white" else ""
        html_code = f"""
        <div style="display: flex; border-radius: 12px; overflow: hidden; height: 120px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); margin-top: 10px;">
            <div style="{main_border} flex: 4; background-color: {hex_main}; color: {get_text_color(main_color)}; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: bold; border-right: 2px solid rgba(255,255,255,0.3);">主色 {main_color.capitalize()}</div>
            <div style="flex: 2; background-color: {hex_c1}; color: {get_text_color(c1)}; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; font-size: 16px;">
                <span style="font-size: 12px; opacity: 0.8;">Top 1</span><span>{c1.capitalize()}</span>
            </div>
            <div style="flex: 2; background-color: {hex_c2}; color: {get_text_color(c2)}; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; font-size: 16px;">
                <span style="font-size: 12px; opacity: 0.8;">Top 2</span><span>{c2.capitalize()}</span>
            </div>
            <div style="flex: 2; background-color: {hex_c3}; color: {get_text_color(c3)}; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; font-size: 16px;">
                <span style="font-size: 12px; opacity: 0.8;">Top 3</span><span>{c3.capitalize()}</span>
            </div>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)
        st.write("")

        # 一鍵複製色碼
        st.markdown("##### ✂️ 一鍵複製色碼 (Hex Codes)")
        code_main, code_col1, code_col2, code_col3 = st.columns(4)
        with code_main: 
            st.caption("主色 (Primary)")
            st.code(hex_main, language=None)
        with code_col1: 
            st.caption("Top 1")
            st.code(hex_c1, language=None)
        with code_col2: 
            st.caption("Top 2")
            st.code(hex_c2, language=None)
        with code_col3: 
            st.caption("Top 3")
            st.code(hex_c3, language=None)

        # 可解釋性
        st.write("")
        with st.expander("💡 為什麼 AI 推薦這些顏色？ (點擊查看模型可解釋性)"):
            st.markdown(f"**🥇 Top 1 ({c1.capitalize()}):** {XAI_EXPLANATION.get(c1, '')}")
            st.markdown(f"**🥈 Top 2 ({c2.capitalize()}):** {XAI_EXPLANATION.get(c2, '')}")
            st.markdown(f"**🥉 Top 3 ({c3.capitalize()}):** {XAI_EXPLANATION.get(c3, '')}")