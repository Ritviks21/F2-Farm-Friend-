# ============================================================
# Farm Friend — brain.py  (FIXED v2)
# Fix 1: TreeExplainer built ONCE globally (not per call) → fast
# Fix 2: SHAP values grouped by parent feature → meaningful insights
# ============================================================

import joblib
import pandas as pd
import numpy as np
import shap

# ── Load model and feature columns once globally ──────────
model           = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ── FIX 1: Build explainer ONCE at startup, not per call ──
#    This saves 5-10 seconds on every prediction click
explainer = shap.TreeExplainer(model)

# ── MSP Dictionary (₹ per quintal, 2023-24 govt rates) ───
MSP = {
    "Wheat": 2275, "Rice": 2183, "Maize": 2090, "Bajra": 2500,
    "Jowar": 3180, "Ragi": 3846, "Barley": 1735,
    "Gram": 5440, "Arhar/Tur": 7000, "Moong(Green Gram)": 8558,
    "Urad": 6950, "Masoor": 6425, "Groundnut": 6377,
    "Sunflower": 6760, "Soyabean": 4600, "Sesamum": 8635,
    "Rapeseed &Mustard": 5650, "Safflower": 5800,
    "Sugarcane": 315, "Cotton(lint)": 6620, "Jute": 5050,
    "Coconut": 3375, "Potato": 400, "Onion": 800,
    "Turmeric": 7000, "Ginger": 2200, "Garlic": 3200,
}
DEFAULT_MSP = 2000

# ── Recommendation Rules ───────────────────────────────────
RECOMMENDATIONS = {
    "low_rain": {
        "English": "⚠️ Rainfall is low — consider increasing irrigation.",
        "Hindi":   "⚠️ वर्षा कम है — सिंचाई बढ़ाने पर विचार करें।",
        "Odia":    "⚠️ ବର୍ଷା କମ୍ — ସିଚାଇ ବୃଦ୍ଧି କରନ୍ତୁ।"
    },
    "high_rain": {
        "English": "💧 Excess rainfall — ensure proper drainage.",
        "Hindi":   "💧 अत्यधिक वर्षा — उचित निकासी सुनिश्चित करें।",
        "Odia":    "💧 ଅଧିକ ବର୍ଷା — ଠିକ ଭାବରେ ପାଣି ନିଷ୍କାଶନ କରନ୍ତୁ।"
    },
    "low_fert": {
        "English": "🌱 Fertilizer per hectare is low — consider soil enrichment.",
        "Hindi":   "🌱 प्रति हेक्टेयर उर्वरक कम है — मिट्टी की उर्वरता बढ़ाएं।",
        "Odia":    "🌱 ପ୍ରତି ହେକ୍ଟରରେ ସର କମ୍ — ମାଟି ସମୃଦ୍ଧି କରନ୍ତୁ।"
    },
    "high_fert": {
        "English": "🧪 Fertilizer per hectare is very high — check for nutrient overload.",
        "Hindi":   "🧪 प्रति हेक्टेयर उर्वरक बहुत अधिक — पोषक तत्वों की जांच करें।",
        "Odia":    "🧪 ପ୍ରତି ହେକ୍ଟରରେ ସର ଅଧିକ — ପୋଷକତତ୍ୱ ଯାଞ୍ଚ କରନ୍ତୁ।"
    },
    "high_pest": {
        "English": "🐞 High pesticide usage — monitor for resistance or overuse.",
        "Hindi":   "🐞 कीटनाशक अधिक है — प्रतिरोध या अति-उपयोग पर नजर रखें।",
        "Odia":    "🐞 କୀଟନାଶକ ଅଧିକ — ପ୍ରତିରୋଧ କିମ୍ବା ଅଧିକ ବ୍ୟବହାର ପରୀକ୍ଷା କରନ୍ତୁ।"
    },
    "balanced": {
        "English": "✅ Inputs look balanced — maintain current practices.",
        "Hindi":   "✅ इनपुट संतुलित हैं — वर्तमान प्रथाओं को बनाए रखें।",
        "Odia":    "✅ ଇନପୁଟ୍ ସମତୁଳିତ — ବର୍ତ୍ତମାନ ପ୍ରକ୍ରିୟା ରଖନ୍ତୁ।"
    }
}


def predict_crop_yield(language, crop, season, state,
                       area, rainfall, fertilizer, pesticide):

    # ── 1. Build input DataFrame ───────────────────────────
    input_dict = {
        'Area': area,
        'Annual_Rainfall': rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide,
        'Crop': crop,
        'Season': season,
        'State': state
    }
    df = pd.DataFrame([input_dict])

    # ── 2. One-hot encode ──────────────────────────────────
    df_encoded = pd.get_dummies(df, columns=['Crop', 'Season', 'State'])

    # ── 3. Align columns with training features ────────────
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]

    # ── 4. Predict yield ───────────────────────────────────
    prediction = round(float(model.predict(df_encoded)[0]), 4)

    # ── 5. MSP Revenue ────────────────────────────────────
    msp_price = MSP.get(crop, DEFAULT_MSP)
    revenue = round(prediction * area * 10 * msp_price)
    revenue_formatted = f"₹{revenue:,}"

    # ── 6. SHAP Explanation (FIXED) ───────────────────────
    # Use the globally built explainer — no rebuild cost
    shap_values = explainer.shap_values(df_encoded, check_additivity=False)
    shap_array  = shap_values[0]

    # FIX 2: Group one-hot columns back to parent feature
    # Raw features: Area, Annual_Rainfall, Fertilizer, Pesticide
    # One-hot groups: Crop_*, Season_*, State_*
    # This gives 7 meaningful groups instead of 95 raw columns
    grouped = {
        "Area":             0.0,
        "Annual Rainfall":  0.0,
        "Fertilizer":       0.0,
        "Pesticide":        0.0,
        f"Crop ({crop})":   0.0,
        f"Season ({season})": 0.0,
        f"State ({state})": 0.0,
    }

    for i, col in enumerate(feature_columns):
        val = shap_array[i]
        if col == "Area":
            grouped["Area"] += val
        elif col == "Annual_Rainfall":
            grouped["Annual Rainfall"] += val
        elif col == "Fertilizer":
            grouped["Fertilizer"] += val
        elif col == "Pesticide":
            grouped["Pesticide"] += val
        elif col.startswith("Crop_"):
            grouped[f"Crop ({crop})"] += val
        elif col.startswith("Season_"):
            grouped[f"Season ({season})"] += val
        elif col.startswith("State_"):
            grouped[f"State ({state})"] += val

    # Sort by absolute impact, show top 3
    sorted_groups = sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)
    explanation_lines = []
    for fname, impact in sorted_groups[:3]:
        direction = "▲ increased" if impact > 0 else "▼ decreased"
        explanation_lines.append(
            f"  • {fname} {direction} yield by {abs(round(impact, 3))} t/ha"
        )
    explanation = "🔍 Key factors:\n" + "\n".join(explanation_lines)

    # ── 7. Recommendations ────────────────────────────────
    fert_per_ha = fertilizer / area if area else 0
    pest_per_ha = pesticide  / area if area else 0
    recs = []

    if rainfall < 500:
        recs.append(RECOMMENDATIONS["low_rain"][language])
    elif rainfall > 3000:
        recs.append(RECOMMENDATIONS["high_rain"][language])
    if fert_per_ha < 100:
        recs.append(RECOMMENDATIONS["low_fert"][language])
    elif fert_per_ha > 1000:
        recs.append(RECOMMENDATIONS["high_fert"][language])
    if pest_per_ha > 50:
        recs.append(RECOMMENDATIONS["high_pest"][language])
    if not recs:
        recs.append(RECOMMENDATIONS["balanced"][language])

    recommendations = "\n".join(recs)

    return prediction, revenue_formatted, explanation, recommendations
