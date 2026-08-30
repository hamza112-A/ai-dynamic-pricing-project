# 🎯 PRESENTATION CREATION PROMPT - 10 SLIDES VERSION

## Project Details
**Title:** Comparing Tree Ensembles and Deep Learning for Network Intrusion Detection on Imbalanced Data  
**Student:** Hamza Arshad  
**Level:** MSc Machine Learning Project  
**Duration:** 10-12 minutes (10 slides)

---

## 🎨 DESIGN SPECIFICATIONS

### Color Scheme
- **Primary:** Navy Blue (#1a237e) - for headers, key elements
- **Secondary:** White (#ffffff) - background
- **Accent 1:** Orange (#ff6f00) - for highlighting key results
- **Accent 2:** Teal (#00897b) - for data visualizations
- **Text:** Dark Gray (#424242) for body text, Black for headings

### Typography
- **Title Slide:** Arial Bold, 36-44pt
- **Slide Titles:** Arial Bold, 28-32pt, Navy Blue
- **Body Text:** Arial Regular, 18-22pt, Dark Gray
- **Bullet Points:** 20-24pt with 1.5 line spacing
- **Figure Captions:** Arial Italic, 14-16pt

### Layout Guidelines
- **Margins:** 0.5 inches on all sides
- **Slide Size:** 16:9 widescreen format
- **White Space:** Maximum 5-6 bullet points per slide
- **Figures:** 300 DPI, center-aligned

---

## 📊 10-SLIDE STRUCTURE

### SLIDE 1: Title Slide
**Layout:** Centered, professional

**Content:**
- **Title (Large, Bold):** Comparing Tree Ensembles and Deep Learning for Network Intrusion Detection on Imbalanced Data
- **Subtitle:** MSc Machine Learning Project
- **Student Name:** Hamza Arshad
- **Date:** August 2026

**Design:** Navy blue header or white background with navy elements

---

### SLIDE 2: Problem & Research Questions
**Layout:** Title + 2 sections (Problem + Questions)

**Title:** The Challenge & Our Questions

**Section 1 - The Problem:**
🚨 **Why This Matters:**
- Cybercrime: $10.5 trillion annually by 2025
- 99% accuracy is misleading with imbalanced data
- 85% benign traffic → model predicting "all benign" gets 99% accuracy but catches ZERO attacks

⚖️ **The Challenge:**
- CICIDS2017: 2.8M flows, 15 classes, extreme imbalance
- Smallest class: only 8 samples
- Need macro-F1, not accuracy

**Section 2 - Research Questions:**
**RQ1:** Can CNN match tree ensembles on rare attack classes (macro-F1)?  
**RQ2:** Which model offers best quality-deployment trade-off?


**Hypothesis:** Tree ensembles will outperform due to sample efficiency

---

### SLIDE 3: Methodology & Models
**Layout:** Title + flowchart + model comparison table

**Title:** Research Approach

**Methodology Flow (Top):**
📦 **Data:** CICIDS2017 → 2.23M flows (after cleaning)  
🔧 **Preprocessing:** Remove duplicates (21%), leak features, standardize  
📊 **Split:** 70% train / 15% val / 15% test (stratified)  
🤖 **Train:** 3 models under identical conditions  
📈 **Evaluate:** Quality + Deployment metrics

**Model Comparison Table (Bottom):**

| Model | Architecture | Size | Training | Key Feature |
|-------|-------------|------|----------|-------------|
| **Random Forest** | 500 trees | 294 MB | 15 min CPU | Robust, interpretable |
| **XGBoost** | 300 boosted trees | 50 MB | 8 min GPU | Fast, accurate |
| **1D-CNN** | 4 conv + 2 FC (159K params) | 0.64 MB | 12 min GPU | Compact, Focal Loss |

**Note:** CNN optimized extensively: 30% → 61% F1 (102% improvement)

---

### SLIDE 4: Main Results - Tree Ensembles Win
**Layout:** Title + results table + bar chart figure

**Title:** Results: Tree Ensembles Dominate by 24 Points

**Results Table:**

| Model | Accuracy | **Macro-F1** ⭐ | Precision | Recall | Status |
|-------|----------|------------|-----------|---------|---------|
| **XGBoost** 🏆 | 99.88% | **85.55%** | 91.98% | 83.46% | BEST |
| **Random Forest** | 99.55% | **83.34%** | 87.84% | **87.81%** | 2nd |
| **CNN (Improved)** | 97.01% | **61.13%** | 66.08% | 64.46% | 3rd |

**Key Findings:**
✅ XGBoost wins macro-F1 by 24 points over CNN  
✅ Random Forest has highest recall (catches most attacks)  
❌ CNN lags despite extensive optimization  
⚠️ Accuracy misleading: 99% vs 97% but 85% vs 61% F1

**Insert Figure (Bottom):**
📊 `paper_figures/figure1_model_comparison.png`

**Caption:** "Tree ensembles outperform deep learning by 24 percentage points on macro-F1."

---

### SLIDE 5: Per-Class Performance & Critical Threshold
**Layout:** Title + per-class highlights + threshold graph

**Title:** The 100-Sample Threshold Discovery

**Top Section - Per-Class Highlights:**

| Sample Size | RF F1 | XGB F1 | CNN F1 | Winner |
|-------------|-------|--------|--------|--------|
| **>1,000 samples** | >99% | >99% | >98% | All good ✅ |
| **100-1,000 samples** | 97-99% | 97-99% | 72-82% | Gap emerging ⚠️ |
| **<100 samples** | 60-70% | 60-80% | **0-40%** | CNN fails ❌ |

**Examples of Rare Class Performance:**
- **Botnet (3 samples):** RF: 67% | XGB: **80%** | CNN: **0%**
- **Heartbleed (5 samples):** RF: 29% | XGB: **33%** | CNN: **0%**
- **Web XSS (98 samples):** RF: **64%** | XGB: 28% | CNN: **0%**

**Bottom Half - Insert Figure:**
📊 `paper_figures/figure4_f1_vs_size.png` (F1 vs Training Samples)

**Key Insight Box:**
"100-sample threshold: Below this, tree ensembles maintain 60-70% F1 (graceful degradation), CNN drops to 0-40% (catastrophic failure). Tree ensembles are 5-10× more sample-efficient."


---

### SLIDE 6: Confusion Matrices - Prediction Quality
**Layout:** Title + 3 confusion matrices side-by-side

**Title:** Confusion Matrices: XGBoost Has Cleanest Predictions

**Insert Figure (Full Width):**
📊 `paper_figures/figure3_confusion_matrix.png`

**Annotations:**

**Random Forest:**
- Strong diagonal (good predictions)
- Some inter-attack confusion
- Moderate rare class handling

**XGBoost:** ⭐
- **Strongest diagonal** (best predictions)
- Minimal false positives
- Best rare class handling

**CNN:**
- Weaker diagonal on rare classes
- More scattered errors
- Struggles with minority classes

**Bottom Highlight:**
"XGBoost produces cleanest confusion matrix with most confident predictions on both common and rare classes."

---

### SLIDE 7: Deployment Analysis - XGBoost Wins
**Layout:** Title + deployment comparison table + use case recommendations

**Title:** Deployment Trade-offs: Production Readiness

**Deployment Metrics:**

| Metric | Random Forest | XGBoost ⭐ | CNN (GPU) | CNN (CPU) |
|--------|---------------|-----------|-----------|-----------|
| **Training Time** | 15 min | **8 min** ✅ | 12 min | 180 min ❌ |
| **Model Size** | 294 MB | **50 MB** ✅ | **0.64 MB** ✅ | 0.64 MB |
| **Inference/flow** | 50 ms | **30 ms** ✅ | 15 ms | 200 ms ❌ |
| **Hardware** | CPU only ✅ | CPU only ✅ | GPU required ❌ | CPU only ✅ |
| **Throughput** | 20 flows/s | **33 flows/s** ✅ | 67 flows/s | 5 flows/s ❌ |

**Recommendations by Use Case:**

🏢 **Enterprise Production NIDS:** → **XGBoost**  
- Best F1 (85.55%), fast CPU inference (30ms), no GPU needed

🔒 **High-Security / Critical Infrastructure:** → **Random Forest**  
- Highest recall (87.81%), catches most attacks, interpretable

📱 **Edge/IoT (extreme constraints):** → **CNN with caution**  
- Smallest (0.64 MB) BUT 24 points lower F1, needs GPU

**Bottom Box:**
"XGBoost offers best balance: top quality + fast training + CPU-only deployment = Production-ready"

---

### SLIDE 8: Research Question Answers
**Layout:** Title + 2 large answer boxes

**Title:** Answering Our Research Questions

**Box 1 (Left - Orange border):**
### RQ1: Can CNN Match Tree Ensembles?

**Answer: ❌ NO**

**Evidence:**
- CNN: 61.13% macro-F1
- XGBoost: 85.55% macro-F1
- **Gap: 24.42 percentage points**

**Why?**
- Tree ensembles more sample-efficient
- Below 100 samples: CNN → 0% F1, XGBoost → 60-80% F1
- Even with optimization (Focal Loss, 102% improvement), gap persists
- Fundamental limitation, not tuning issue

**Conclusion:** Tree ensembles win on imbalanced flow-based NIDS


**Box 2 (Right - Teal border):**
### RQ2: Best Quality-Deployment Trade-off?

**Answer: ✅ XGBoost**

**Evidence:**
- **Quality:** 85.55% macro-F1 (highest)
- **Speed:** 8 min training (fastest)
- **Size:** 50 MB (compact)
- **Inference:** 30 ms/flow on CPU
- **Hardware:** CPU-only (no GPU cost)

**vs Competitors:**
- vs RF: Better F1 (+2.21), faster, smaller
- vs CNN: Better F1 (+24.42), no GPU, comparable speed

**Conclusion:** XGBoost = production-ready for enterprise NIDS

---

### SLIDE 9: Contributions & Limitations
**Layout:** Title + 2 columns (Contributions | Limitations with defenses)

**Title:** What This Research Contributes & Limitations

**Left Column - Key Contributions:**

**1. 🔬 First Controlled Comparison**
- Identical preprocessing & splits
- Fair evaluation, no cherry-picking

**2. 📊 Comprehensive Evaluation**
- Quality (macro-F1) + Deployment (time, size, latency)
- Honest reporting (CNN 0% F1 on rare classes)

**3. 🎯 100-Sample Threshold Discovery**
- Critical finding for rare class detection
- Explains when to use which architecture

**4. 💡 Actionable Guidance**
- XGBoost for production
- Clear trade-off analysis
- Deployment-aware recommendations

**5. ✅ Challenges DL Assumption**
- CNN optimized 102% (30% → 61%)
- Still underperforms by 24 points
- "Deep learning not always better"

**Right Column - Limitations & Defenses:**

**❓ Single Dataset?**
✅ CICIDS2017 is standard benchmark (50+ papers)  
✅ Controlled comparison > multi-dataset survey

**❓ Flow-based only?**
✅ Necessary for high-throughput (>100 Gbps)  
✅ Matches production NIDS (Zeek, Suricata)

**❓ Only 1D-CNN?**
✅ Extensively optimized (102% improvement)  
✅ Sample efficiency is fundamental, not architectural

**❓ No adversarial robustness?**
✅ Separate research domain  
✅ Orthogonal to imbalanced learning focus

**Bottom Note:**
"Despite limitations, first rigorous comparison with quality + deployment analysis"

---

### SLIDE 10: Conclusion & Thank You
**Layout:** Title + key takeaways + closing

**Title:** Conclusion: Choose the Right Tool

**Main Conclusion (Large, centered):**
"For flow-based NIDS on imbalanced data, tree ensembles outperform deep learning by 24 points in macro-F1."

**Key Takeaways:**

**1️⃣ Tree Ensembles Win Quality**
- XGBoost: 85.55% | RF: 83.34% | CNN: 61.13%

**2️⃣ Sample Efficiency Matters**
- Below 100 samples: Trees 60-70% vs CNN 0-40%
- Trees need 5-10× fewer samples

**3️⃣ XGBoost = Production Winner**
- Best F1 + fastest + CPU-only + 30ms inference

**4️⃣ Deep Learning Isn't Always Best**
- Context matters: choose based on data characteristics
- Extensive optimization still left 24-point gap

**5️⃣ Use Right Metrics**
- Macro-F1 > Accuracy for imbalanced data
- Report all classes honestly

**Bottom Section:**
---
**Thank You**

**Questions?**

Hamza Arshad  
MSc Machine Learning

📊 Full Report Available | 💻 Code: [GitHub link if public]  
📈 Dataset: CICIDS2017

---


## 🎬 PRESENTATION DELIVERY TIPS (10-SLIDE VERSION)

### Timing Guide (10-12 minutes total)
- **Slide 1 (Title):** 30 sec - Quick intro
- **Slide 2 (Problem & RQs):** 1:30 min - Set context
- **Slide 3 (Method & Models):** 1:30 min - Explain approach
- **Slide 4 (Main Results):** 1:30 min - Key findings
- **Slide 5 (Threshold):** 1:30 min - Critical discovery
- **Slide 6 (Confusion):** 1:00 min - Visual validation
- **Slide 7 (Deployment):** 1:30 min - Practical analysis
- **Slide 8 (RQ Answers):** 1:30 min - Direct answers
- **Slide 9 (Contributions):** 1:30 min - What we contribute
- **Slide 10 (Conclusion):** 1:00 min - Wrap up

**Total: ~12 minutes** (perfect for 10-12 min requirement)

### What to Emphasize
- ⭐ **24-point gap** (repeat this number - it's your key finding)
- ⭐ **100-sample threshold** (your critical discovery)
- ⭐ **XGBoost wins** both quality AND deployment
- ⭐ **Macro-F1 vs accuracy** (why accuracy misleads)
- ⭐ **Honest reporting** (CNN 0% F1 on rare classes)
- ⭐ **102% improvement** but still 24 points behind (shows you tried hard)

### Presentation Best Practices
1. **Open strong:** "Today I'll show you that for network intrusion detection, traditional tree ensembles beat deep learning by 24 percentage points"
2. **Use pointer:** Highlight key numbers in tables and figures
3. **Pause for emphasis:** After key findings, pause 1-2 seconds
4. **Show enthusiasm:** This is interesting research - sound excited!
5. **Connect slides:** "This brings us to...", "Now let's see..."
6. **Close strong:** "The key message: choose the right tool for the job - sometimes it's not deep learning"

### Common Questions to Prepare
**Q: Why only one dataset?**
A: CICIDS2017 is the standard benchmark used in 50+ papers. Controlled comparison on one dataset is more valuable than superficial multi-dataset survey.

**Q: Would more data help CNN?**
A: We have 2.2M flows - substantial. Rare attacks are inherently rare in real networks. Can't magically get more Heartbleed or Botnet samples.

**Q: Did you try SMOTE or oversampling?**
A: Synthetic samples don't represent real attack diversity. Can lead to overfitting. We wanted to evaluate on real data.

**Q: Why not try Transformer or RNN?**
A: Sample efficiency is the fundamental issue, not architecture. 1D-CNN was extensively optimized (102% improvement). Other DL architectures face same challenge with <100 samples.

**Q: Production deployment?**
A: Academic project, but deployment metrics (8 min training, 50 MB, 30 ms inference, CPU-only) suggest XGBoost is production-ready.

---

## 📊 FIGURE INTEGRATION CHECKLIST

Figures to insert from `paper_figures/` folder:

- ✅ **Slide 4:** `figure1_model_comparison.png` - Overall metrics
- ✅ **Slide 5:** `figure4_f1_vs_size.png` - Sample size threshold
- ✅ **Slide 6:** `figure3_confusion_matrix.png` - Confusion matrices
- ✅ **Optional (if space):** `figure2_perclass_heatmap.png` - Can add to Slide 5

All figures are 300 DPI and professionally formatted.

---

## 🎨 DESIGN COMPARISON: 10 vs 19 Slides

### 10-Slide Version (THIS VERSION)
**Pros:**
- ✅ More focused and concise
- ✅ Easier to deliver in 10-12 minutes without rushing
- ✅ Less content per slide = clearer message
- ✅ Audience stays engaged (not overwhelmed)
- ✅ Easier to practice and memorize

**Cons:**
- ❌ Less detailed methodology explanation
- ❌ Fewer per-class examples
- ❌ Combined sections (e.g., contributions + limitations in one slide)

### 19-Slide Version (ORIGINAL)
**Pros:**
- ✅ More comprehensive and detailed
- ✅ Each concept gets its own slide
- ✅ More examples and evidence
- ✅ Thorough coverage of all aspects

**Cons:**
- ❌ Risk of going over time (14-15 minutes)
- ❌ More to memorize and practice
- ❌ Audience may feel overwhelmed
- ❌ Harder to maintain pace

### Recommendation
**Use 10-slide version** if:
- Time limit is strict (10-12 minutes)
- You prefer focused, punchy presentation
- Your audience prefers concise format
- You're less comfortable with long presentations

**Use 19-slide version** if:
- Time limit is flexible (12-15 minutes allowed)
- You want comprehensive coverage
- Your audience expects detailed technical depth
- You're comfortable with longer presentations

**My suggestion:** Start with **10-slide version** - it's cleaner, easier to deliver well, and still covers all essential points. You can always have the 19-slide version as backup for questions.

---

## 🎯 FINAL CHECKLIST

- ✅ **Total slides:** 10 (perfect for 10-12 min)
- ✅ **All key findings included:** 24-point gap, 100-sample threshold, XGBoost wins
- ✅ **Figures integrated:** 3-4 main figures from paper_figures/
- ✅ **Design specs:** Navy Blue theme, Arial fonts, 300 DPI
- ✅ **RQ answers:** Clear YES/NO with evidence
- ✅ **Contributions:** Listed with defenses for limitations
- ✅ **Practical value:** Deployment analysis and recommendations
- ✅ **Honest evaluation:** Reports CNN failures (0% F1)

---

## 💡 QUICK START GUIDE

1. **Choose presentation software:** PowerPoint, Google Slides, or Keynote
2. **Select template:** Modern, professional (e.g., "Simple Dark", "Ion", "Geometric")
3. **Customize colors:** Navy Blue (#1a237e) primary, white background
4. **Create 10 slides:** Follow structure above
5. **Insert figures:** From `paper_figures/` folder
6. **Practice delivery:** 2-3 times, aim for 10-12 minutes
7. **Record video:** Use Zoom, OBS, or built-in recording
8. **Review and submit:** Check quality, duration, file format

**You're ready to create an excellent presentation!** 🎓✨

---

END OF 10-SLIDE PRESENTATION PROMPT
