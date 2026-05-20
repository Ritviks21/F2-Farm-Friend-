"""
Farm Friend — Crop Yield Prediction
Team: Ritvik Singh, Chaitanya Nitai, Rishabh Srivastava
MUJ | ML CWS Sem 4 | Guide: Asst. Prof. Ashish Pandey
HuggingFace Space: srits21/farm-friend
"""

import gradio as gr
import brain

# ──────────────────────────────────────────────
# DATA LISTS
# ──────────────────────────────────────────────

CROPS = [
    "Arecanut", "Arhar/Tur", "Bajra", "Banana", "Barley",
    "Black pepper", "Cardamom", "Cashewnut", "Castor seed",
    "Coconut", "Coriander", "Cotton(lint)", "Cowpea(Lobia)",
    "Dry chillies", "Dry ginger", "Garlic", "Ginger",
    "Gram", "Grapes", "Groundnut", "Guar seed", "Horse-gram",
    "Jowar", "Jute", "Khesari", "Linseed", "Maize",
    "Masoor", "Mesta", "Moong(Green Gram)", "Moth", "Niger seed",
    "Onion", "Other  Rabi pulses", "Other Cereals",
    "Other Kharif pulses", "Other Summer Pulses",
    "Peas & beans (Pulses)", "Potato", "Rapeseed &Mustard",
    "Rice", "Rubber", "Safflower", "Sannhamp", "Sesamum",
    "Small millets", "Soyabean", "Sugarcane", "Sunflower",
    "Sweet potato", "Tapioca", "Tobacco", "Tomato",
    "Turmeric", "Urad", "Wheat"
]

SEASONS = ["Kharif", "Rabi", "Whole Year", "Summer", "Winter", "Autumn"]

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

# ──────────────────────────────────────────────
# MULTILINGUAL LABELS
# ──────────────────────────────────────────────

LABELS = {
    "English": {
        "crop":       "Crop",
        "season":     "Season",
        "state":      "State",
        "area":       "Area (hectares)",
        "area_info":  "Total land under this crop",
        "rainfall":   "Annual Rainfall (mm)",
        "rain_info":  "Average annual rainfall in your region",
        "fertilizer": "Fertilizer used (kg)",
        "fert_info":  "Total fertilizer applied this season",
        "pesticide":  "Pesticide used (kg)",
        "pest_info":  "Total pesticide applied this season",
        "btn":        "🔍 Predict Yield",
        "out_yield":  "Predicted Yield (t/ha)",
        "out_rev":    "Expected Revenue",
        "out_shap":   "Key Influencing Factors (SHAP)",
        "out_rec":    "Recommendations",
    },
    "Hindi": {
        "crop":       "फसल",
        "season":     "मौसम",
        "state":      "राज्य",
        "area":       "क्षेत्रफल (हेक्टेयर)",
        "area_info":  "इस फसल के अंतर्गत कुल भूमि",
        "rainfall":   "वार्षिक वर्षा (मिमी)",
        "rain_info":  "आपके क्षेत्र में औसत वार्षिक वर्षा",
        "fertilizer": "उपयोग किया गया उर्वरक (किग्रा)",
        "fert_info":  "इस मौसम में डाला गया कुल उर्वरक",
        "pesticide":  "उपयोग किया गया कीटनाशक (किग्रा)",
        "pest_info":  "इस मौसम में छिड़का गया कुल कीटनाशक",
        "btn":        "🔍 उत्पादन का अनुमान लगाएं",
        "out_yield":  "अनुमानित उत्पादन (टन/हेक्टेयर)",
        "out_rev":    "संभावित राजस्व",
        "out_shap":   "प्रमुख प्रभावकारी कारक (SHAP)",
        "out_rec":    "सिफारिशें",
    },
    "Odia": {
        "crop":       "ଫସଲ",
        "season":     "ଋତୁ",
        "state":      "ରାଜ୍ୟ",
        "area":       "କ୍ଷେତ୍ର (ହେକ୍ଟର)",
        "area_info":  "ଏହି ଫସଲ ଅଧୀନ ମୋଟ ଜମି",
        "rainfall":   "ବାର୍ଷିକ ବର୍ଷା (ମିମି)",
        "rain_info":  "ଆପଣଙ୍କ ଅଞ୍ଚଳରେ ହାରାହାରି ବାର୍ଷିକ ବର୍ଷା",
        "fertilizer": "ବ୍ୟବହୃତ ସାର (କିଗ୍ରା)",
        "fert_info":  "ଏହି ଋତୁରେ ପ୍ରୟୋଗ ହୋଇଥିବା ମୋଟ ସାର",
        "pesticide":  "ବ୍ୟବହୃତ କୀଟନାଶକ (କିଗ୍ରା)",
        "pest_info":  "ଏହି ଋତୁରେ ବ୍ୟବହୃତ ମୋଟ କୀଟନାଶକ",
        "btn":        "🔍 ଅମଳ ଆକଳନ କରନ୍ତୁ",
        "out_yield":  "ଆକଳିତ ଅମଳ (ଟନ/ହେକ୍ଟର)",
        "out_rev":    "ଆଶା କରା ଯାଉଥିବା ରାଜସ୍ୱ",
        "out_shap":   "ମୁଖ୍ୟ ପ୍ରଭାବକ କାରଣ (SHAP)",
        "out_rec":    "ପରାମର୍ଶ",
    },
}

# ──────────────────────────────────────────────
# PREDICTION WRAPPER
# ──────────────────────────────────────────────

def run_prediction(language, crop, season, state,
                   area, rainfall, fertilizer, pesticide):
    """Validate inputs, call brain.py, return 4 output strings."""

    # Basic validation
    if not all([crop, season, state]):
        return "⚠️ Please select Crop, Season, and State.", "", "", ""

    for val, name in [(area, "Area"), (rainfall, "Rainfall"),
                      (fertilizer, "Fertilizer"), (pesticide, "Pesticide")]:
        if val is None or val < 0:
            return f"⚠️ {name} must be a non-negative number.", "", "", ""

    try:
        # brain.predict_crop_yield returns a TUPLE:
        # (yield_float, revenue_str, explanation_str, recommendations_str)
        yield_val, revenue, explanation, recommendations = brain.predict_crop_yield(
            language   = language,
            crop       = crop,
            season     = season,
            state      = state,
            area       = float(area),
            rainfall   = float(rainfall),
            fertilizer = float(fertilizer),
            pesticide  = float(pesticide),
        )

        yield_out = f"{yield_val:.4f} t/ha"
        return yield_out, revenue, explanation, recommendations

    except Exception as e:
        return f"❌ Prediction error: {e}", "", "", ""


# ──────────────────────────────────────────────
# LABEL UPDATE — fires when language changes
# ──────────────────────────────────────────────

def update_labels(language):
    L = LABELS[language]
    return (
        gr.update(label=L["crop"]),
        gr.update(label=L["season"]),
        gr.update(label=L["state"]),
        gr.update(label=L["area"],       info=L["area_info"]),
        gr.update(label=L["rainfall"],   info=L["rain_info"]),
        gr.update(label=L["fertilizer"], info=L["fert_info"]),
        gr.update(label=L["pesticide"],  info=L["pest_info"]),
        gr.update(value=L["btn"]),
        gr.update(label=L["out_yield"]),
        gr.update(label=L["out_rev"]),
        gr.update(label=L["out_shap"]),
        gr.update(label=L["out_rec"]),
    )


# ──────────────────────────────────────────────
# UI LAYOUT
# ──────────────────────────────────────────────

with gr.Blocks(
    title="Farm Friend — Crop Yield Predictor",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
        #title-md { text-align: center; }
        #title-md h1 { font-size: 2rem; margin-bottom: 0.2rem; }
        #title-md p  { color: #555; font-size: 0.95rem; }
        footer       { display: none !important; }
    """,
) as demo:

    # ── Header ──────────────────────────────────────────
    gr.Markdown(
        """# 🌾 Farm Friend — Crop Yield Predictor
Predict your crop yield, estimate revenue, and get personalised farming recommendations powered by AI.""",
        elem_id="title-md",
    )

    gr.Markdown("---")

    # ── Language selector ────────────────────────────────
    with gr.Row():
        lang_dd = gr.Dropdown(
            choices=["English", "Hindi", "Odia"],
            value="English",
            label="🌐 Language / भाषा / ଭାଷା",
            scale=1,
        )

    # ── Input columns ────────────────────────────────────
    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            gr.Markdown("### 🌱 Crop Details")
            crop_dd   = gr.Dropdown(choices=CROPS,   label="Crop",   value="Wheat")
            season_dd = gr.Dropdown(choices=SEASONS, label="Season", value="Rabi")
            state_dd  = gr.Dropdown(choices=STATES,  label="State",  value="Punjab")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Field Parameters")
            area_num = gr.Number(label="Area (hectares)",
                                 value=2.0, minimum=0.01,
                                 info="Total land under this crop")
            rain_num = gr.Number(label="Annual Rainfall (mm)",
                                 value=600.0, minimum=0,
                                 info="Average annual rainfall in your region")
            fert_num = gr.Number(label="Fertilizer used (kg)",
                                 value=300.0, minimum=0,
                                 info="Total fertilizer applied this season")
            pest_num = gr.Number(label="Pesticide used (kg)",
                                 value=10.0, minimum=0,
                                 info="Total pesticide applied this season")

    # ── Predict button ───────────────────────────────────
    with gr.Row():
        predict_btn = gr.Button("🔍 Predict Yield", variant="primary", size="lg")

    gr.Markdown("---")

    # ── Output boxes ─────────────────────────────────────
    with gr.Row(equal_height=True):
        yield_out   = gr.Textbox(label="Predicted Yield (t/ha)",
                                 interactive=False, lines=1)
        revenue_out = gr.Textbox(label="Expected Revenue",
                                 interactive=False, lines=1)

    with gr.Row():
        shap_out = gr.Textbox(label="Key Influencing Factors (SHAP)",
                              interactive=False, lines=4)

    with gr.Row():
        rec_out  = gr.Textbox(label="Recommendations",
                              interactive=False, lines=4)

    # ── Footer ───────────────────────────────────────────
    gr.Markdown(
        "<center><sub>Farm Friend v1.0 · MUJ ML CWS Sem 4 · "
        "Model: Random Forest (R²=0.9558) · "
        "Revenue based on MSP prices</sub></center>"
    )

    # ──────────────────────────────────────────────────────
    # WIRING
    # ──────────────────────────────────────────────────────

    LANG_TARGETS = [
        crop_dd, season_dd, state_dd,
        area_num, rain_num, fert_num, pest_num,
        predict_btn,
        yield_out, revenue_out, shap_out, rec_out,
    ]

    lang_dd.change(
        fn=update_labels,
        inputs=[lang_dd],
        outputs=LANG_TARGETS,
    )

    predict_btn.click(
        fn=run_prediction,
        inputs=[lang_dd, crop_dd, season_dd, state_dd,
                area_num, rain_num, fert_num, pest_num],
        outputs=[yield_out, revenue_out, shap_out, rec_out],
    )

# ──────────────────────────────────────────────
# LAUNCH
# ──────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()
