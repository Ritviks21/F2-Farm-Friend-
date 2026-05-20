<p align="center">
  <img src="https://raw.githubusercontent.com/Ritviks21/F2-Farm-Friend-/main/docs/images/ChatGPT%20Image%20May%2020%2C%202026%2C%2006_45_28%20PM.png" alt="Farm Friend Project Banner" width="100%">
</p>

<h1 align="center">Farm Friend</h1>
<p align="center">
  <i>An end-to-end Explainable AI (XAI) and economic advisory platform for Indian agriculture.</i>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/Gradio-FFD21E?style=for-the-badge&logo=Gradio&logoColor=black" alt="Gradio UI">
    <img src="https://img.shields.io/badge/SHAP-black?style=for-the-badge" alt="SHAP XAI">
</p>

<p align="center">
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-project-overview">Project Overview</a> •
  <a href="#-our-journey">Our Journey</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-results">Results</a>
</p>

---

## 🚀 Live Demo

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/spaces/srits21/farm-friend-2)

**Click the badge above or follow this link to try the live application:** [https://huggingface.co/spaces/srits21/farm-friend-2](https://huggingface.co/spaces/srits21/farm-friend-2)

This system is hosted live on Hugging Face Spaces using a responsive, production-configured Gradio instance utilizing multi-threaded parallel execution protocols.

---

## 📖 Project Overview

**Farm Friend** is a localized, transparent "Glass Box" Machine Learning system designed to optimize macro-level agricultural choices across India. By integrating a high-accuracy regression engine with explicit Government of India **Minimum Support Price (MSP) benchmarks** and localized Explainable AI (XAI), the framework converts dense soil, climate, and input metrics into actionable, transparent, and multi-lingual visual insights.

<p align="center">
  <br>
  <img src="https://raw.githubusercontent.com/Ritviks21/F2-Farm-Friend-/main/docs/images/Picture1.png" alt="Farm Friend Architectural Pipeline Flowchart" width="550">
  <br>
  <i>Figure 1: Full Architecture Pipeline from user UI interaction layers down to granular XAI calculation engines.</i>
</p>

---

## 🚀 Our Journey: The Story of a Glass Box Model

Real-world AI cannot exist as an uninterpretable "Black Box" when dealing with human livelihoods like smallholder farming. Our platform evolved systematically from raw accuracy targeting to full, real-time explainability.

<details>
<summary><strong>V1: The High-Variance Black Box - A Trust Barrier</strong></summary>
<br>
Our first iteration focused entirely on high-dimensional model tuning. We achieved an excellent accuracy score, but the system couldn't explain *why* it made its decisions. Farmers and agricultural extension officers had no reason to trust its outputs.
<br><br>
💡 **Lesson Learned:** Predictive accuracy without structural explainability creates a fundamental trust gap in real-world deployment.
</details>

<details>
<summary><strong>V2: The Lagging Oracle - The Integration Bottleneck</strong></summary>
<br>
We integrated <strong>SHAP (Shapley Additive Explanations)</strong> to calculate localized individual feature impacts. However, calling the tree-explainer dynamically during runtime created massive lag, forcing users to wait over 8 seconds for a single form submission.
<br><br>
💡 **Lesson Learned:** Naive feature attribution calculations completely break UI/UX fluidity. Localized explainer pipelines must be mathematically cached or globally initialized.
</details>

<details>
<summary><strong>V3: The Unified Advisor - Real-Time Glass Box</strong></summary>
<br>
Our final system resolved all execution and usability bottlenecks to create a production-grade advisory engine:
<ul>
  <li><strong>Global SHAP Optimization:</strong> We optimized the system by pre-computing and initializing the <code>TreeExplainer</code> globally on app launch, cutting user response latency down from <strong>8,000ms to <200ms</strong>.</li>
  <li><strong>Trilingual Reverse Mapping:</strong> We implemented an implicit backend translation schema to support <strong>English, Hindi (हिंदी), and Odia (ଓଡ଼ିଆ)</strong> natively, without breaking categorical encoded inputs.</li>
  <li><strong>Robust Validation Strategy:</strong> Evaluated generalization using an 80/20 random-seeded train/test allocation cross-verified against an extensive 5-Fold grid validation layout.</li>
</ul>

<br>
<p align="center">
  <img src="https://raw.githubusercontent.com/Ritviks21/F2-Farm-Friend-/main/docs/images/Picture2.png" alt="Data split strategy map" width="100%">
</p>
<br>

✅ **The Result:** A reliable, split-second, transparent advisor that matches elite machine learning metrics with local community accessibility.
</details>

---

## ✨ Key Features

- **Explainable AI Engine**: Leverages SHAP localized attribution to explain the exact structural impact (+/- metric tonnes) of individual environmental inputs.
- **Economic Intelligence**: Integrates official Government of India MSP benchmarks (2023-24) to calculate localized total revenue directly via:
  $$\text{Income} = \text{Predicted Yield (t/ha)} \times \text{Area (ha)} \times 10 \times \text{MSP (₹/quintal)}$$
- **Trilingual Localization**: A fully interactive, reactive user interface adapting layout labels cleanly across English, Hindi, and Odia.
- **Data-Driven Feature Alignment**: Dynamically builds, fills, and drops category one-hot matrices via an exported template to avoid silent prediction crashes.

---

## 🛠️ Tech Stack

| Python | Scikit-Learn | Gradio | SHAP | Pandas | Joblib |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="https://github.com/Ritviks21/Silicon-Sentinel/raw/main/docs/images/Python.png" width="48"> | <img src="https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/brand/scikit-learn-logo-notext.png" width="48"> | <img src="https://raw.githubusercontent.com/gradio-app/gradio/main/guides/assets/logo.png" width="48"> | <img src="https://raw.githubusercontent.com/shap/shap/master/docs/artwork/shap_logo_v2.png" width="54"> | <img src="https://raw.githubusercontent.com/pandas-dev/pandas/main/web/pandas/static/img/pandas_secondary.svg" width="48"> | <img src="https://raw.githubusercontent.com/joblib/joblib/master/doc/images/joblib_logo.svg" width="48"> |

---

## 📊 Results & Evaluation

Our validation study benchmarked four leading ensemble architectures across **19,295 historical records**. The **Random Forest Regressor** demonstrated superior stability against climate anomalies:

| Model Architecture | $R^2$ Score | RMSE (t/ha) |
| :--- | :---: | :---: |
| 🌲 **Random Forest (Selected)** | **0.9558** | **2.1952** |
| 🐱 CatBoost Regressor | 0.9365 | 2.6334 |
| 🚀 XGBoost | 0.9249 | 2.8621 |
| ⚡ LightGBM | 0.9051 | 3.2174 |

<p align="center">
  <img src="https://raw.githubusercontent.com/Ritviks21/F2-Farm-Friend-/main/docs/images/Picture3.png" alt="Benchmarking bar plots metrics charts" width="100%">
</p>

### Balanced Metrics Multi-Axis Breakdown
To track trade-offs across speed, training resource efficiency, and residual variance constraints, a comparative multidimensional index map was constructed:

<p align="center">
  <img src="https://raw.githubusercontent.com/Ritviks21/F2-Farm-Friend-/main/docs/images/Picture4.png" alt="Radar multi-axis performance chart" width="500">
</p>

### Granular Feature Impact Calculations
Using the globally initialized interpretability engine, localized weight effects are mapped explicitly during validation checkpoints to isolate key performance indicators:

<p align="center">
  <img src="https://raw.githubusercontent.com/Ritviks21/F2-Farm-Friend-/main/docs/images/Picture5.png" alt="SHAP local feature waterfall chart" width="650">
</p>

---

### Application Interface Deployment Previews

<p align="center">
  <b>Trilingual User Interface Layout</b><br>
  <img src="https://raw.githubusercontent.com/Ritviks21/F2-Farm-Friend-/main/docs/images/Picture6.png" alt="Farm Friend Dashboard Attribute Fields" width="100%">
</p>

<p align="center">
  <b>Real-Time XAI Explanations & Financial Outputs</b><br>
  <img src="https://raw.githubusercontent.com/Ritviks21/F2-Farm-Friend-/main/docs/images/Picture7.png" alt="Inference Pipeline Output Performance View" width="100%">
</p>

---

## 🚀 Getting Started

<details>
<summary>Click here for instructions to run this project yourself locally.</summary>

1. **Clone the Repository**
    ```bash
    git clone [https://github.com/Ritviks21/F2-Farm-Friend-.git](https://github.com/Ritviks21/F2-Farm-Friend-.git)
    cd F2-Farm-Friend-
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the Production Model**
    ```bash
    python src/train.py
    ```

4. **Launch the Local Web App**
    ```bash
    python src/app.py
    ```

</details>

---

## 🤝 How to Contribute

Contributions are highly welcome! This platform is designed as an expanding tool for open-source digital agriculture systems. If you would like to expand its functionality, feel free to fork the repository and submit a Pull Request.

### Primary Areas for System Enhancement

* **Expand Regional Datasets**:
  * Integrate micro-level district data for Southern and Eastern coastal zones.
  * Add additional crop varieties (e.g., specific commercial spices, rubber, plantation crops) to the backend encoding script.

* **Incorporate Deep Regressors**:
  * Experiment with multi-layer neural networks or localized deep tabular models to further minimize residual RMSE variance thresholds below $2.19$.
  * Expand hyperparameter sweeps utilizing automated frameworks like Optuna.

* **Enhance the UI Framework**:
  * Introduce comparative visualizations (e.g., dynamic bar graphs or map charts) rendering right within the Gradio user interface layout.

---

## 🔗 Connect with Me

<p align="left">
<a href="https://github.com/Ritviks21" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" alt="Your Github" height="30" width="40" /></a>
<a href="https://x.com/gemdata21" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="Your Twitter" height="30" width="40" /></a>
<a href="https://huggingface.co/srits21" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/hugging-face.svg" alt="Your Hugging Face" height="30" width="40" /></a>
</p>
