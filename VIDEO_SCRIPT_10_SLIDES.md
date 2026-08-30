# 🎬 VIDEO SCRIPT - 10 SLIDES VERSION

**Duration:** 10-12 minutes  
**Student:** Mohammed Abdul Razzak Baig (25207938)  
**Format:** Condensed, focused presentation

---

## 📜 COMPLETE NARRATION SCRIPT

### [0:00 - 0:30] SLIDE 1: Title
**[Pause 2 seconds]**

"Hello everyone. My name is Mohammed Abdul Razzak Baig, student ID 25207938. Today I'm presenting my MSc Machine Learning project: Comparing Tree Ensembles and Deep Learning for Network Intrusion Detection on Imbalanced Data."

**[Pause 1 second]**

"This research addresses a critical question: When dealing with severely imbalanced network traffic, which performs better—tree ensembles or deep learning? Let's find out."

**[ADVANCE TO SLIDE 2]**

---

### [0:30 - 2:00] SLIDE 2: Problem & Research Questions
**[Pause 1 second]**

"First, why does this problem matter? Cybercrime costs 10 point 5 trillion dollars annually. Many papers claim 99% accuracy, but this number is misleading."

**[Point to problem section]**

"In real network traffic, 85% is benign. A naive model predicting everything as benign gets 99% accuracy but catches zero attacks. This is the imbalance problem. We need macro-F1 score, which treats all classes equally."

**[Point to dataset info]**

"I used CICIDS 2017 with 2 point 8 million flows, 15 classes. The challenge: extreme imbalance. Six classes have fewer than 100 samples. The smallest class—Botnet—has only 8 examples."

**[Point to research questions]**

"This leads to our two research questions: First, can convolutional neural networks match tree ensemble performance on rare attack classes using macro-F1? Second, which model offers the best balance between detection quality and deployment constraints like training time and model size?"

**[Point to hypothesis]**

"Our hypothesis: tree ensembles will outperform due to better sample efficiency—they can learn from fewer examples."

**[ADVANCE TO SLIDE 3]**

---

### [2:00 - 3:30] SLIDE 3: Methodology & Models
**[Pause 1 second]**

"Let me explain our rigorous methodology."

**[Point to flow diagram]**

"I started with CICIDS 2017 and cleaned it—removing 21% duplicates and leak features like destination port. After preprocessing, 2 point 2 million clean flows. I used a stratified 70-15-15 split for training, validation, and testing."

**[Point to models table]**

"I trained three models under identical conditions. Random Forest: 500 decision trees, 294 megabytes, 15-minute CPU training. XGBoost: 300 gradient-boosted trees, 50 megabytes, 8-minute GPU training. And 1D-CNN: four convolutional layers, 159,000 parameters, only 0.64 megabytes, 12-minute GPU training."

**[Point to CNN note]**

"Important note: I extensively optimized the CNN. Initially it achieved only 30% macro-F1—completely broken. After applying Focal Loss, adjusting class weights, and deepening the architecture, it improved to 61% F1. That's a 102% improvement. This shows I put serious effort into making deep learning work."

**[Emphasize]**

"All three models trained on identical data for fair comparison."

**[ADVANCE TO SLIDE 4]**

---

### [3:30 - 5:00] SLIDE 4: Main Results
**[Pause 1 second]**

"Now, our main results."

**[Point to table, XGBoost row]**

"XGBoost is the clear winner. 99 point 88% accuracy and—more importantly—85 point 55% macro-F1. This is our key metric. XGBoost also achieves 91.98% precision and 83.46% recall."


**[Point to Random Forest row]**

"Random Forest comes second. 99 point 55% accuracy, 83 point 34% macro-F1. Interestingly, Random Forest has the highest recall at 87.81%—it catches the most attacks, with the trade-off of slightly more false positives."

**[Point to CNN row]**

"The CNN achieves 97% accuracy, which sounds good. But look at macro-F1: 61 point 13%. That's a 24-point gap behind XGBoost. This is our key finding."

**[Point to key observations]**

"Notice something critical: If we only looked at accuracy, CNN with 97% seems close to Random Forest with 99.5%. Only a 2-point gap. But macro-F1 tells the real story: 61% versus 83%—a 22-point gap. This is why accuracy is misleading on imbalanced data."

**[Point to figure]**

"This bar chart visualizes the comparison. XGBoost dominates. Random Forest is solid. CNN significantly trails on the metric that actually matters—rare attack detection."

**[Emphasize conclusion]**

"The conclusion: For imbalanced network intrusion detection, tree ensembles outperform deep learning by 24 percentage points when we evaluate fairly using macro-F1. Even with extensive CNN optimization—that 102% improvement I mentioned—the gap persists. This is a fundamental difference in sample efficiency."

**[ADVANCE TO SLIDE 5]**

---

### [5:00 - 6:30] SLIDE 5: The 100-Sample Threshold
**[Pause 1 second]**

"This slide reveals one of our most important discoveries."

**[Point to per-class table]**

"For large classes with more than 1,000 samples, all models perform excellently—above 98% F1. Even CNN has enough data. For medium classes with 100 to 1,000 samples, we see a small gap emerging. CNN drops to 72-82% while tree ensembles maintain 97-99%."

**[Point to rare class examples]**

"But for classes with fewer than 100 samples, we see catastrophic CNN failure. Botnet with 3 samples: XGBoost achieves 80% F1—impressive!—while CNN gets 0%. Zero. Heartbleed with 5 samples: tree ensembles get 29-33%, CNN gets 0%. Web Attack XSS with 98 samples: Random Forest gets 64%, CNN gets 0%."

**[Point to figure]**

"This graph plots F1 score versus training sample size. Notice the three lines: green for Random Forest, blue for XGBoost, red for CNN. Above 500 samples, all perform similarly. Between 100 and 500 samples, divergence emerges. Below 100 samples—our critical threshold—dramatic separation."

**[Point to threshold line]**

"Tree ensembles degrade gracefully to 60-70% F1. CNN crashes to 0-40% through catastrophic failure. This is the 100-sample threshold."

**[Point to highlight box]**

"This is a critical discovery: Below 100 training samples, tree ensembles demonstrate superior sample efficiency. CNN requires 5 to 10 times more samples for equivalent performance. This explains why tree ensembles win on network intrusion detection—attacks are inherently rare. Many have fewer than 100 examples. Tree ensembles simply learn more from less data."

**[ADVANCE TO SLIDE 6]**

---

### [6:30 - 7:30] SLIDE 6: Confusion Matrices
**[Pause 1 second]**

"Confusion matrices provide visual validation of our findings."

**[Point to Random Forest matrix]**

"Random Forest shows a strong diagonal—most predictions are correct. There's some confusion between attack types, but overall solid performance."

**[Point to XGBoost matrix]**

"XGBoost has the cleanest diagonal. Most predictions fall exactly on the correct class. Very few false positives or false negatives. Notice that even for rare classes at the bottom, XGBoost maintains strong diagonal presence. This is the best confusion matrix—the most reliable detector."

**[Point to CNN matrix]**

"CNN shows a weaker diagonal, especially for rare classes. More scattered errors. Many rare attacks are misclassified as benign. This visualizes the 0% F1 problem we saw earlier—the CNN literally cannot detect these rare attack classes."

**[Point to highlight]**

"XGBoost produces the cleanest confusion matrix with the most confident, accurate predictions on both common and rare classes."

**[ADVANCE TO SLIDE 7]**

---

### [7:30 - 9:00] SLIDE 7: Deployment Analysis
**[Pause 1 second]**

"Quality is important, but deployment constraints matter too for production systems."

**[Point to table]**

"Let's analyze practical metrics. Training time: XGBoost is fastest at 8 minutes. CNN takes 12 minutes on GPU but 180 minutes—3 hours—on CPU. If you don't have a GPU, CNN training is impractical."

"Model size: CNN is smallest at 0.64 megabytes. XGBoost is 50 megabytes. Random Forest is largest at 294 megabytes."

"Inference latency: XGBoost is fastest on CPU at 30 milliseconds per flow. CNN is very fast on GPU at 15 milliseconds but very slow on CPU at 200 milliseconds. For high-throughput networks processing thousands of flows per second, XGBoost offers the best CPU performance."

"Hardware requirements: Random Forest and XGBoost run fine on CPU only. CNN requires GPU for practical speed. This increases deployment cost."

**[Point to recommendations]**

"Let me summarize by use case: For enterprise production NIDS with high traffic, XGBoost is the winner—best F1 score, fast CPU inference, no GPU dependency, production-ready."

"For high-security environments with zero tolerance for missed attacks, Random Forest is best—highest recall catches the most attacks. Trade-off: slightly more false positives and larger model."

"For edge devices with extreme resource constraints, CNN might be considered because of its tiny 0.64 megabyte size. But the 24-point F1 gap is a serious concern. Only consider if size is absolutely critical and GPU is available."

**[Point to highlight]**

"Overall, XGBoost offers the best balance: top quality, fast training, fast CPU inference, reasonable size. It's the practical choice for deployment."

**[ADVANCE TO SLIDE 8]**

---

### [9:00 - 10:15] SLIDE 8: Research Question Answers
**[Pause 1 second]**

"Now let's formally answer our two research questions."

**[Point to RQ1 box]**

"Research Question 1: Can CNN match tree ensembles on rare attack classes? The answer is NO."

**[Point to evidence]**

"The evidence is clear: CNN achieves 61 point 13% macro-F1. XGBoost achieves 85 point 55% macro-F1. That's a 24 point 42 percentage point gap. Why not? Tree ensembles are more sample-efficient—they require 5 to 10 times fewer samples for equivalent performance. Below the 100-sample threshold, CNN fails catastrophically with 0% F1 on classes like Botnet and Heartbleed, while XGBoost achieves 33 to 80% F1."

**[Point to conclusion]**

"Even with extensive optimization—Focal Loss, class weight tuning, deeper architecture, 102% improvement—the gap persists. This is a fundamental limitation based on sample efficiency, not a tuning issue. For flow-based NIDS with severe imbalance, tree ensembles outperform deep learning by a significant margin."

**[Point to RQ2 box]**

"Research Question 2: Which model offers the best quality-deployment trade-off? The answer is XGBoost."

**[Point to evidence]**

"XGBoost delivers the highest quality at 85 point 55% macro-F1. It has the fastest training at 8 minutes. Compact size at 50 megabytes. Fast CPU inference at 30 milliseconds per flow. And requires only CPU—no expensive GPU dependency."

**[Point to comparison]**

"Compared to Random Forest, XGBoost has better F1 by 2 points, faster training, and smaller size. Compared to CNN, XGBoost has better F1 by 24 points, comparable speed, and doesn't require a GPU."

**[Point to conclusion]**

"XGBoost delivers the best detection quality with practical deployment constraints. It's production-ready for enterprise network intrusion detection."

**[ADVANCE TO SLIDE 9]**

---

### [10:15 - 11:30] SLIDE 9: Contributions & Limitations
**[Pause 1 second]**

"Let me highlight what this research contributes to the field."

**[Point to contributions column]**

"First, this is the first controlled comparison of tree ensembles and deep learning for NIDS under identical conditions—same preprocessing, same splits, fair evaluation, no cherry-picking."

"Second, comprehensive evaluation framework—quality metrics like macro-F1, deployment metrics like training time and model size, per-class analysis including rare classes, and honest reporting of failures."

"Third, we discovered the 100-sample threshold—below this, tree ensembles maintain 60 to 70% F1 while CNN drops to 0 to 40%. This explains when to use which architecture."

"Fourth, actionable guidance—we recommend XGBoost for production, explain the trade-offs for different scenarios, provide deployment-aware recommendations."

"Fifth, honest reporting of deep learning limitations—we optimized CNN extensively, achieved 102% improvement, yet it still underperforms by 24 points. This challenges the assumption that deep learning is always better."

**[Point to limitations column]**

"Now, limitations and defenses. Single dataset? Yes, but CICIDS 2017 is the standard benchmark used in 50-plus papers. Controlled comparison on one dataset is more valuable than a superficial multi-dataset survey."

"Flow-based only? Yes, but flow-based is necessary for high-throughput networks above 100 gigabits per second. Packet-level is infeasible at scale. Our approach matches real-world deployment constraints."

"Only 1D-CNN? Yes, but we extensively optimized it—102% improvement. Sample efficiency is the fundamental issue, not architecture. Other deep learning architectures face the same challenge with fewer than 100 samples."

"No adversarial robustness? Correct, but that's a separate research domain. Our focus is imbalanced learning and deployment trade-offs."

**[Point to highlight]**

"Despite these limitations, this is the first rigorous controlled comparison providing both quality and deployment analysis."

**[ADVANCE TO SLIDE 10]**

---

### [11:30 - 12:15] SLIDE 10: Conclusion
**[Pause 1 second]**

"Let me conclude with our key takeaways."

**[Point to main conclusion]**

"The main conclusion: For flow-based network intrusion detection on imbalanced data, tree ensembles outperform deep learning by 24 percentage points in macro-F1 score."

**[Point to takeaways]**

"Key Takeaway 1: Tree ensembles win on quality. XGBoost 85.55%, Random Forest 83.34%, CNN 61.13%—a 24-point gap."

"Key Takeaway 2: Sample efficiency matters. Below 100 samples, tree ensembles maintain 60 to 70% F1, CNN drops to 0 to 40%. Tree ensembles require 5 to 10 times fewer samples."

"Key Takeaway 3: XGBoost offers the best deployment trade-off—fastest training, compact size, fast CPU inference, no GPU dependency, production-ready."

"Key Takeaway 4: Deep learning isn't always the answer. We optimized CNN extensively, achieved 102% improvement, still underperforms by 24 points. This is fundamental—choose based on data characteristics."

"Key Takeaway 5: Use the right metrics. Accuracy is misleading. Macro-F1 reveals true rare-class performance. Report per-class results honestly."

**[Point to final highlight]**

"The overarching message: In machine learning, there is no one-size-fits-all solution. Understand your data, evaluate comprehensively, and choose the right tool for the job. Sometimes, the best tool is a traditional method, not deep learning."

**[Pause 2 seconds]**

"Thank you for your attention. I'm happy to answer any questions. Mohammed Abdul Razzak Baig, student ID 25207938."

**[Pause 2 seconds, wait before stopping recording]**

**[END]**

---

## 🎯 RECORDING TIPS FOR 10-SLIDE VERSION

### Timing Breakdown
- Slides 1-2: 2 minutes (intro + problem)
- Slide 3: 1.5 minutes (method)
- Slides 4-6: 3.5 minutes (results + threshold + confusion)
- Slide 7: 1.5 minutes (deployment)
- Slide 8: 1.25 minutes (RQ answers)
- Slide 9: 1.25 minutes (contributions)
- Slide 10: 0.75 minutes (conclusion)
**Total: ~12 minutes**

### Key Points to Emphasize
- **24-point gap** (say this clearly 3-4 times)
- **100-sample threshold** (your discovery)
- **102% improvement** but still 24 points behind (shows effort)
- **XGBoost wins** quality AND deployment
- **0% F1** on rare classes (honest reporting)

### Delivery Style
- **Confidence:** Speak with authority
- **Enthusiasm:** Show interest in findings
- **Clarity:** Explain technical terms simply
- **Pace:** Moderate, not rushed (aim for 11-12 min)
- **Pauses:** Brief pauses after key findings

---

END OF 10-SLIDE VIDEO SCRIPT
