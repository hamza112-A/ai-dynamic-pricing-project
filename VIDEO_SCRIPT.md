# 🎬 VIDEO PRESENTATION SCRIPT

## Project Information
**Title:** Comparing Tree Ensembles and Deep Learning for Network Intrusion Detection on Imbalanced Data  
**Student:** Hamza Arshad  
**Duration:** 10-13 minutes  
**Format:** Slide presentation with voiceover narration

---

## 🎯 VIDEO RECORDING SETUP

### Equipment Recommendations
**Minimum Setup:**
- Laptop/desktop with screen recording software
- Built-in or external microphone
- Quiet recording environment
- Stable internet (if recording via Zoom/Teams)

**Recommended Setup:**
- External USB microphone (Blue Yeti, Rode NT-USB, etc.)
- Headphones for monitoring
- Screen recording software (OBS Studio, Zoom, Camtasia, or PowerPoint recording)
- Good lighting if including webcam
- Backup recording (record locally AND to cloud)

### Recording Software Options
1. **PowerPoint/Keynote Built-in Recording** (Easiest)
   - Slideshow → Record Slideshow
   - Records slides + audio + timings automatically
   - Export as video file (MP4)

2. **Zoom** (Easy, reliable)
   - Start meeting, share screen with slides
   - Click "Record" (record locally, not cloud)
   - Speak through slides
   - Stop recording, Zoom converts to MP4

3. **OBS Studio** (Professional, free)
   - Add "Display Capture" source for slides
   - Add "Audio Input Capture" for microphone
   - Click "Start Recording"
   - More control but steeper learning curve

### Video Quality Settings
- **Resolution:** 1080p (1920×1080) minimum
- **Frame Rate:** 30 fps
- **Audio:** 48 kHz, stereo, MP3 or AAC
- **File Format:** MP4 (most compatible)
- **Bitrate:** 5-8 Mbps for video, 128-192 kbps for audio


---

## 📜 COMPLETE NARRATION SCRIPT WITH TIMING

### [0:00 - 0:35] SLIDE 1: Title Slide
**[Pause 2 seconds to let slide appear]**

"Hello everyone. My name is Hamza Arshad. Today, I'm presenting my Master of Science in Machine Learning project, titled 'Comparing Tree Ensembles and Deep Learning for Network Intrusion Detection on Imbalanced Data.'"

**[Pause 2 seconds]**

"This research addresses a critical question in cybersecurity: When dealing with severely imbalanced network traffic data, which machine learning approach performs better—traditional tree ensembles or modern deep learning? Let's find out."

**[ADVANCE TO SLIDE 2]**

---

### [0:35 - 1:30] SLIDE 2: The Problem
**[Pause 1 second for slide transition]**

"First, let's understand why this problem matters."

**[Point to cybercrime statistic]**

"Cybercrime is projected to cost the global economy 10 point 5 trillion dollars annually by 2025. Network intrusion detection systems, or NIDS, are our first line of defense. However, there's a critical problem that many research papers overlook."

**[Point to accuracy vs F1 comparison]**

"Many papers report 99% accuracy and claim their models are excellent. But this number is deeply misleading. Here's why: In real network traffic, about 85% of flows are benign—normal, legitimate traffic. Rare attacks like Heartbleed or Botnet might have fewer than 10 training examples."

**[Emphasize the key point]**

"A naive model that predicts everything is benign will achieve 99% accuracy—but it catches ZERO attacks. This is the imbalance problem. High accuracy doesn't mean good intrusion detection. We need to focus on rare attack detection, measured by macro-F1 score, which treats all classes equally."

**[ADVANCE TO SLIDE 3]**

---

### [1:30 - 2:20] SLIDE 3: Research Gap
**[Pause 1 second]**

"Current research has three major gaps that this project addresses."

**[Point to first column]**

"First, the Metric Gap. Most papers focus exclusively on accuracy, which we just saw is misleading. They ignore macro-F1 score, which reveals true performance on rare attack classes. This leads to inflated performance claims."


**[Point to second column]**

"Second, the Deployment Gap. Research papers rarely analyze training time, model size, or inference latency. They focus only on accuracy metrics. But in production systems, these deployment constraints matter enormously. A model that takes hours to train or requires expensive GPUs isn't practical for many organizations."

**[Point to third column]**

"Third, the Reporting Gap. Many papers exclude minority classes from their evaluation or cherry-pick results. They'll say 'we achieved 98% F1 on most classes' but quietly omit that they got 0% on rare attacks. This project reports results for ALL classes, including failures."

**[Point to bottom highlight]**

"This research addresses all three gaps with rigorous controlled comparison using identical data and evaluation methods."

**[ADVANCE TO SLIDE 4]**

---

### [2:20 - 3:10] SLIDE 4: Research Questions
**[Pause 1 second]**

"This brings us to our two research questions."

**[Point to RQ1]**

"Research Question 1: Can convolutional neural networks match tree ensemble performance on rare attack classes when evaluated using macro-F1 score? In other words, does deep learning really outperform traditional methods on imbalanced data? We're specifically focusing on rare attacks—the ones that are hardest to detect but potentially most dangerous."

**[Point to RQ2]**

"Research Question 2: Which model architecture offers the best balance between detection quality and deployment constraints like training time, model size, and inference latency? This is about practical deployment. It's not enough to have the most accurate model if it's too slow or too large for production use."

**[Point to hypothesis]**

"Our hypothesis is that tree ensembles will outperform deep learning on rare classes because they're more sample-efficient. They can learn from fewer examples. Let's test this hypothesis."

**[ADVANCE TO SLIDE 5]**

---

### [3:10 - 4:00] SLIDE 5: Dataset
**[Pause 1 second]**

"For this study, I used the CICIDS 2017 dataset—the most widely-used benchmark in network intrusion detection research."

**[Point to dataset statistics]**

"This dataset contains over 2 point 8 million network flow records, collected over 5 days in July 2017. Each flow has 77 features capturing network behavior—things like packet sizes, flow duration, protocol types. After preprocessing and removing duplicates, we have 2 point 2 million clean flows."


**[Point to class breakdown]**

"The data includes 15 classes: one benign class and 14 different attack types like DDoS, port scanning, botnet infections, and web attacks. But here's the challenge: The imbalance is severe."

**[Point to pie chart/visual]**

"84 point 9 percent of flows are benign. The largest attack class, DDoS, is only 9 point 6 percent. Many attacks have fewer than 1000 examples. And the smallest class—Botnet—has only 8 training samples. Eight! This extreme imbalance is exactly why we can't rely on accuracy alone."

**[Emphasize key point]**

"Six classes have fewer than 100 samples. This is where machine learning gets really hard. Can our models learn from such limited data? That's what we'll find out."

**[ADVANCE TO SLIDE 6]**

---

### [4:00 - 4:50] SLIDE 6: Methodology
**[Pause 1 second]**

"Let me walk you through our rigorous methodology."

**[Point to Step 1]**

"Step 1: Data Preprocessing. I removed 21% of the data that were exact duplicates. I handled missing values using median imputation. Critically, I removed leak features—things like destination port that directly reveal the attack type. This prevents the model from cheating. Finally, I standardized all features to zero mean and unit variance. This ensures fair comparison across models."

**[Point to Step 2]**

"Step 2: Train-Validation-Test Split. I used a 70-15-15 split, giving us 1 point 5 million training samples, 335,000 validation samples, and 335,000 test samples. The split is stratified, meaning class ratios are preserved in each set. This ensures every class, even the rare ones, appears in all three sets."

**[Point to Step 3]**

"Step 3: Model Training. I trained three model types: Random Forest with 500 trees, XGBoost with 300 boosting rounds, and a 1D Convolutional Neural Network with 159,000 parameters. All models were trained on identical data under identical conditions. No cherry-picking, no different preprocessing. Fair comparison."

**[Point to Step 4]**

"Step 4: Comprehensive Evaluation. I measured both quality metrics—accuracy, macro-F1, precision, recall—and deployment metrics—training time, model size, inference latency. I also analyzed per-class performance, especially for rare classes. This gives us a complete picture."

**[ADVANCE TO SLIDE 7]**

---

### [4:50 - 5:35] SLIDE 7: The Three Models
**[Pause 1 second]**

"Let me introduce our three competing models."


**[Point to Random Forest]**

"First, Random Forest. This is an ensemble of 500 decision trees, each trained on a random subset of data and features. It uses square root of 77 features, about 9, per split. Random Forest is known for being robust and requiring minimal hyperparameter tuning. The trained model is 294 megabytes. It's interpretable—we can see which features matter most."

**[Point to XGBoost]**

"Second, XGBoost—Extreme Gradient Boosting. This builds 300 trees sequentially, where each new tree corrects the errors of previous trees. It uses a learning rate of 0.1 and maximum depth of 6. XGBoost was trained with GPU acceleration for speed. The trained model is 50 megabytes. XGBoost is known for winning machine learning competitions and handling imbalanced data well."

**[Point to CNN]**

"Third, our deep learning model—a 1D Convolutional Neural Network. It has four convolutional layers with filter sizes increasing from 64 to 512. The architecture learns hierarchical features automatically. With 159,000 parameters, the model file is only 0.64 megabytes—the smallest by far. But here's the key: CNNs typically need lots of data. Will it handle our rare classes?"

**[Point to bottom note]**

"All three models trained on the exact same data. Let's see how they perform."

**[ADVANCE TO SLIDE 8]**

---

### [5:35 - 6:25] SLIDE 8: CNN Optimization Journey
**[Pause 1 second]**

"Before showing results, I want to share an important finding about deep learning optimization."

**[Point to initial CNN box]**

"Initially, I implemented a standard CNN with cross-entropy loss and balanced class weights. The result? 30 percent macro-F1. This is terrible—barely better than random guessing. On rare classes, it achieved 0% F1 score. The model was completely broken. It predicted everything as benign."

**[Point to arrow/optimizations]**

"So I applied several optimizations. First, I replaced cross-entropy with Focal Loss—a loss function that down-weights easy examples and focuses on hard-to-classify cases. Second, I adjusted the class weighting strategy to use square root of sample counts instead of inverse counts. This is less aggressive but more stable. Third, I increased the network depth to four convolutional layers. Fourth, I added dropout for regularization."

**[Point to improved CNN box]**

"After these optimizations, the CNN achieved 61 percent macro-F1. That's a 102% improvement! From broken to working. This shows I put serious effort into optimizing the deep learning model."

**[Point to highlight box]**

"But here's the crucial finding: Even with a 102% improvement, the CNN still trails tree ensembles by 24 percentage points. The gap is 61% versus 85% F1. This suggests the performance difference isn't due to poor tuning—it's fundamental. Tree ensembles are simply more sample-efficient on rare classes."

**[ADVANCE TO SLIDE 9]**


---

### [6:25 - 7:25] SLIDE 9: Main Results
**[Pause 1 second for slide transition]**

"Now, let's look at our main results."

**[Point to table, highlight XGBoost row]**

"XGBoost is the clear winner. It achieves 99 point 88 percent accuracy and—more importantly—85 point 55 percent macro-F1 score. That's our key metric. It also achieves 91.98% precision and 83.46% recall. A well-balanced model."

**[Point to Random Forest row]**

"Random Forest comes in second. 99 point 55 percent accuracy, 83 point 34 percent macro-F1. Interestingly, Random Forest has the highest recall at 87.81%, meaning it catches the most attacks. The trade-off is slightly more false positives."

**[Point to CNN row]**

"The CNN achieves 97 percent accuracy, which sounds good. But look at the macro-F1: 61 point 13 percent. That's a 24-point gap behind XGBoost. This is our key finding."

**[Point to key observations]**

"Notice something important: If we only looked at accuracy, CNN with 97% seems close to Random Forest with 99.5%. Only a 2-point gap. But macro-F1 tells the real story: 61% versus 83%—a 22-point gap. This is why accuracy is misleading. CNN does well on benign traffic but fails on rare attacks."

**[Point to figure]**

"This bar chart visualizes the comparison. Green bars for XGBoost—the winner. Orange for Random Forest—solid second place. Red for CNN—significantly behind on F1 score."

**[Emphasize key message]**

"The conclusion: For imbalanced network intrusion detection, tree ensembles outperform deep learning by a substantial margin when we evaluate fairly using macro-F1."

**[ADVANCE TO SLIDE 10]**

---

### [7:25 - 8:20] SLIDE 10: Per-Class Performance
**[Pause 1 second]**

"Let's dig deeper into per-class performance to understand where and why this gap occurs."

**[Point to large classes section]**

"For large classes with more than 10,000 samples—like Benign and DDoS—all three models perform excellently. F1 scores above 98%. At this scale, even CNN has enough data to learn effectively. No major differences."

**[Point to medium classes section]**

"For medium classes with 1,000 to 10,000 samples—like Bot and FTP-Patator—we start to see a small gap. XGBoost maintains 99% F1, while CNN drops to 93-98%. Still good, but the gap is emerging."


**[Point to small classes section]**

"For small classes with 100 to 1,000 samples—like SSH-Patator and Infiltration—the gap widens. Tree ensembles maintain 97-99% F1. CNN drops to 72-82%. This is where sample efficiency matters."

**[Point to tiny classes section]**

"For tiny classes with fewer than 100 samples, we see catastrophic failure of the CNN. Web Attack-XSS: Random Forest gets 63% F1, XGBoost gets 28%, CNN gets 0%. Zero. It literally cannot learn this class. Heartbleed with 5 samples: Tree ensembles get 28-33% F1, CNN gets 0%. Botnet with 3 samples: XGBoost gets 80% F1—impressive!—while CNN gets 0%."

**[Point to heatmap]**

"This heatmap visualizes the pattern. Dark green means high F1, red means failure. Notice how the CNN column turns red for rare classes. Tree ensemble columns stay green or yellow—they degrade gracefully, not catastrophically."

**[Emphasize key insight]**

"The key insight: Below 100 training samples, tree ensembles maintain 30-80% F1 through graceful degradation. CNN drops to 0% through catastrophic failure. This is the sample efficiency advantage of tree ensembles."

**[ADVANCE TO SLIDE 11]**

---

### [8:20 - 8:55] SLIDE 11: Confusion Matrices
**[Pause 1 second]**

"Confusion matrices provide another view of model behavior."

**[Point to Random Forest matrix]**

"Random Forest shows a strong diagonal—most predictions are correct. There's some confusion between attack types, but overall solid performance. Rare classes show moderate but not perfect prediction."

**[Point to XGBoost matrix]**

"XGBoost has the cleanest diagonal. Most predictions fall on the correct class. Very few false positives or false negatives. Notice that even for rare classes at the bottom, XGBoost maintains strong diagonal presence. This is the best confusion matrix."

**[Point to CNN matrix]**

"CNN shows a weaker diagonal, especially for rare classes. More scattered errors. Notice the bottom rows—rare attack classes—have more misclassifications. Many rare attacks are incorrectly predicted as benign. This visualizes the 0% F1 problem we saw earlier."

**[Point to highlight]**

"XGBoost produces the cleanest confusion matrix with fewest misclassifications. It's the most reliable detector."

**[ADVANCE TO SLIDE 12]**

---

### [8:55 - 9:40] SLIDE 12: Sample Size Threshold
**[Pause 1 second]**

"This slide shows one of our most important findings."

**[Point to graph]**

"This graph plots F1 score versus training sample size on a log scale. Three lines: green for Random Forest, blue for XGBoost, red for CNN."


**[Point to above 500 samples region]**

"Above 500 samples, all three models perform similarly—above 90% F1. At this scale, even CNN has enough data. The choice of algorithm doesn't matter much."

**[Point to 100-500 samples region]**

"Between 100 and 500 samples, we see divergence. Tree ensembles maintain 80-90% F1. CNN drops to 70-80% F1. A 10-point gap emerges."

**[Point to vertical line at 100 samples]**

"Below 100 samples—this is the critical threshold—we see dramatic divergence. Tree ensembles degrade gracefully to 60-70% F1. CNN drops to 40% F1 or crashes to 0%. A 24-point gap."

**[Point to highlight box]**

"This 100-sample threshold is a critical discovery. It tells us: If your rare classes have fewer than 100 training examples, tree ensembles will significantly outperform deep learning. CNN requires 5 to 10 times more samples to match tree ensemble performance."

**[Emphasize practical implication]**

"This explains why tree ensembles win on NIDS: Network attacks are inherently rare. Many attacks have fewer than 100 examples. Tree ensembles are simply more sample-efficient. They learn more from less data."

**[ADVANCE TO SLIDE 13]**

---

### [9:40 - 10:35] SLIDE 13: Deployment Analysis
**[Pause 1 second]**

"Quality is important, but deployment matters too. Let's analyze practical constraints."

**[Point to training time row]**

"Training time: XGBoost is fastest at 8 minutes. Random Forest takes 15 minutes. CNN takes 12 minutes on GPU but 180 minutes—3 hours—on CPU. If you don't have a GPU, CNN is impractical."

**[Point to model size row]**

"Model size: CNN is smallest at 0.64 megabytes. XGBoost is 50 megabytes. Random Forest is largest at 294 megabytes. For embedded devices with strict size constraints, CNN has an advantage here."

**[Point to hardware row]**

"Hardware requirements: Random Forest and XGBoost run fine on CPU. CNN needs a GPU for practical training and inference speed. This increases deployment cost."

**[Point to inference time row]**

"Inference latency per flow: XGBoost is fastest on CPU at 30 milliseconds. Random Forest takes 50 milliseconds. CNN is very fast on GPU at 15 milliseconds but very slow on CPU at 200 milliseconds. For high-throughput networks processing thousands of flows per second, XGBoost offers the best CPU performance."

**[Point to analysis section]**

"Let me summarize by use case:"


"For enterprise production NIDS with high traffic, XGBoost is the winner. Best F1 score, fast CPU inference, no GPU dependency. Production-ready."

"For high-security environments with zero tolerance for missed attacks, Random Forest is best. Highest recall means it catches the most attacks. Trade-off: slightly more false positives and larger model size."

"For edge devices with extreme resource constraints, CNN might be considered because of its tiny 0.64 MB size. But the 24-point F1 gap is a serious concern. Only consider this if size is absolutely critical and GPU is available."

**[Point to highlight]**

"Overall, XGBoost offers the best balance: top quality, fast training, fast CPU inference, reasonable size. It's the practical choice for deployment."

**[ADVANCE TO SLIDE 14]**

---

### [10:35 - 11:30] SLIDE 14: Research Question Answers
**[Pause 1 second]**

"Now let's formally answer our two research questions."

**[Point to RQ1 box]**

"Research Question 1: Can CNN match tree ensembles on rare attack classes? The answer is NO."

**[Point to evidence]**

"The evidence: CNN achieves 61.13% macro-F1. XGBoost achieves 85.55% macro-F1. That's a 24.42 percentage point gap. Why not? Tree ensembles are more sample-efficient. CNN requires 5 to 10 times more samples for equivalent performance. Below the 100-sample threshold, CNN fails catastrophically with 0% F1 on classes like Botnet, while XGBoost achieves 80% F1."

**[Point to conclusion]**

"Even with extensive optimization—Focal Loss, class weight tuning, deeper architecture—the gap persists. This is a fundamental limitation, not a tuning issue. For flow-based NIDS with severe imbalance, tree ensembles outperform deep learning by a significant margin."

**[Point to RQ2 box]**

"Research Question 2: Which model offers the best quality-deployment trade-off? The answer is XGBoost."

**[Point to evidence]**

"The evidence: XGBoost delivers the highest quality at 85.55% macro-F1. It has the fastest training at 8 minutes. It has a compact size at 50 megabytes. It has fast CPU inference at 30 milliseconds per flow. And it requires only CPU—no GPU dependency."

**[Point to comparison]**

"Compared to Random Forest, XGBoost has better F1 by 2.21 points, faster training, and smaller size. Compared to CNN, XGBoost has better F1 by 24.42 points, comparable speed, and doesn't require a GPU."

**[Point to conclusion]**

"XGBoost delivers best detection quality with practical deployment constraints. It's production-ready for enterprise network intrusion detection."

**[ADVANCE TO SLIDE 15]**

---

### [11:30 - 12:15] SLIDE 15: Research Contributions
**[Pause 1 second]**

"Let me highlight what this research contributes to the field."


**[Point to Contribution 1]**

"First, this is the first controlled comparison of tree ensembles and deep learning for NIDS under identical conditions. Same preprocessing, same data splits, same evaluation metrics. No cherry-picking, fair comparison."

**[Point to Contribution 2]**

"Second, I provide a comprehensive evaluation framework. I measure quality metrics like macro-F1, not just accuracy. I measure deployment metrics like training time, model size, inference latency. I analyze per-class performance, including rare classes. And I honestly report failures—like CNN's 0% F1 on rare classes."

**[Point to Contribution 3]**

"Third, I discovered the 100-sample threshold. Below 100 training samples, tree ensembles maintain 60-70% F1 through graceful degradation, while CNN drops to 0-40% through catastrophic failure. This threshold explains when to use which architecture."

**[Point to Contribution 4]**

"Fourth, I provide clear practical guidance. I recommend XGBoost for production NIDS, Random Forest for high-security environments, and explain the trade-offs. This isn't just 'which is best' but 'when to use which based on your constraints.'"

**[Point to Contribution 5]**

"Fifth, I honestly report deep learning limitations. I optimized CNN extensively—achieved 102% improvement—yet it still underperforms by 24 points. I report 0% F1 on rare classes. This challenges the assumption that deep learning is always better. Sometimes, traditional methods win."

**[Point to highlight]**

"These contributions provide actionable insights for practitioners deploying real-world intrusion detection systems."

**[ADVANCE TO SLIDE 16]**

---

### [12:15 - 13:00] SLIDE 16: Limitations
**[Pause 1 second]**

"Every research project has limitations. Let me address them transparently."

**[Point to Limitation 1]**

"First limitation: I only evaluated one dataset—CICIDS 2017. But here's my defense: CICIDS 2017 is the most widely-used benchmark in NIDS research. It contains realistic network traffic with 2.8 million flows. It represents the real-world class imbalance challenge. And results are comparable to 50-plus published papers using the same dataset. A controlled comparison on one dataset is more valuable than a superficial survey across multiple datasets."

**[Point to Limitation 2]**

"Second limitation: I only used flow-based features, not packet-level analysis. But here's why: Flow-based is necessary for high-throughput networks operating above 100 gigabits per second. Packet-level analysis is computationally infeasible at scale. Flow features are standard in production NIDS tools like Zeek and Suricata. My approach matches real-world deployment constraints. There's a trade-off: performance versus scalability. I chose scalability."


**[Point to Limitation 3]**

"Third limitation: I only tested one deep learning architecture—1D-CNN. Why not RNN or Transformer? Because I extensively optimized the CNN. I achieved 102% improvement from 30% to 61% F1 through Focal Loss, class weight tuning, and architecture changes. The 24-point gap isn't architectural—it's fundamental. It's about sample efficiency. Other deep learning architectures face the same challenge with fewer than 100 samples."

**[Point to Limitation 4]**

"Fourth limitation: I didn't evaluate adversarial robustness. But adversarial NIDS is a separate research domain. Tree ensembles are also vulnerable to adversarial perturbations. The focus of this work is imbalanced learning and deployment trade-offs. Adversarial robustness is orthogonal to sample efficiency."

**[Point to highlight]**

"Despite these limitations, this is the first rigorous controlled comparison providing both quality and deployment analysis. The findings are robust and actionable."

**[ADVANCE TO SLIDE 17]**

---

### [13:00 - 13:45] SLIDE 17: Recommendations
**[Pause 1 second]**

"Based on these findings, here are my practical recommendations for practitioners."

**[Point to Scenario 1]**

"For enterprise production NIDS with high traffic, use XGBoost. Why? Highest macro-F1 at 85.55%. Fast training at 8 minutes. CPU-only deployment—no GPU needed. 30-millisecond inference latency. Low memory footprint. Production-ready."

**[Point to Scenario 2]**

"For high-security or critical infrastructure environments, use Random Forest. Why? Highest recall at 87.81%—catches the most attacks. Robust and interpretable. No hyperparameter tuning needed. Tolerance for false positives is acceptable in high-security contexts where you'd rather investigate a false alarm than miss a real attack."

**[Point to Scenario 3]**

"For edge devices or IoT with extreme resource constraints, consider CNN with caution. Why? Smallest model at 0.64 megabytes—460 times smaller than Random Forest. Fits on embedded devices. But there's a big trade-off: 24 points lower F1 score. Requires GPU for real-time processing. I do not recommend this for production. It's acceptable only for research prototypes or when size is absolutely critical and you can tolerate lower detection rates."

**[Point to Scenario 4]**

"For research and feature learning exploration, explore CNN architectures. They're interesting for automatic feature learning and compact representations. But be honest about performance gaps in your publications."

**[Point to universal recommendation]**

"One universal recommendation: Always evaluate using macro-F1, not accuracy. Always report per-class results, especially for rare classes. Don't cherry-pick. Be honest about failures."

**[ADVANCE TO SLIDE 18]**

---

### [13:45 - 14:30] SLIDE 18: Conclusion
**[Pause 1 second]**

"Let me conclude with our key takeaways."


**[Point to main conclusion]**

"The main conclusion: For flow-based network intrusion detection on imbalanced data, tree ensembles outperform deep learning by 24 percentage points in macro-F1 score. This is a significant, robust finding."

**[Point to Takeaway 1]**

"Key Takeaway 1: Tree ensembles win on quality. XGBoost achieves 85.55% macro-F1. Random Forest achieves 83.34%. CNN achieves 61.13%—a 24-point gap. Tree ensembles are superior detectors of rare attacks."

**[Point to Takeaway 2]**

"Key Takeaway 2: Sample efficiency matters. Below 100 training samples, tree ensembles maintain 60-70% F1. Below 100 samples, CNN drops to 0-40% F1. Tree ensembles require 5 to 10 times fewer samples for equivalent performance. This is the fundamental advantage."

**[Point to Takeaway 3]**

"Key Takeaway 3: XGBoost offers the best deployment trade-off. Fastest training at 8 minutes. Compact size at 50 megabytes. Fast CPU inference at 30 milliseconds. No GPU dependency. Production-ready for enterprise NIDS."

**[Point to Takeaway 4]**

"Key Takeaway 4: Deep learning isn't always the answer. I optimized CNN extensively—achieved 102% improvement. It still underperforms by 24 points. This is a fundamental limitation based on sample efficiency, not tuning. Context matters. Choose based on data characteristics."

**[Point to Takeaway 5]**

"Key Takeaway 5: Use the right metrics. Accuracy is misleading. 99% accuracy versus 97% accuracy hides a 22-point macro-F1 gap. Macro-F1 reveals true rare-class performance. Report per-class results honestly. Don't cherry-pick."

**[Point to final highlight]**

"The overarching message: In machine learning, there is no one-size-fits-all solution. Understand your data characteristics. Evaluate comprehensively with both quality and deployment metrics. And choose the right tool for the job. Sometimes, the best tool is a traditional method, not deep learning."

**[Pause 2 seconds for emphasis]**

**[ADVANCE TO SLIDE 19]**

---

### [14:30 - 14:45] SLIDE 19: Thank You
**[Pause 1 second]**

"Thank you for your attention. I'm happy to answer any questions you may have."

**[Pause 3 seconds]**

"This has been Hamza Arshad, presenting my MSc Machine Learning project on comparing tree ensembles and deep learning for network intrusion detection."

**[Pause 2 seconds, smile if on camera]**

"Thank you."

**[END OF RECORDING - Wait 2 seconds before stopping]**

---

## 🎯 POST-RECORDING CHECKLIST

### Immediate After Recording
- ✅ Stop recording and save file
- ✅ Check file size (should be 200-500 MB for 12-15 min at 1080p)
- ✅ Verify video plays correctly
- ✅ Check audio quality (clear, no distortion)
- ✅ Watch first 30 seconds to ensure slide visibility


### Video Editing (Optional but Recommended)
If you want to polish the video:
1. **Trim silence:** Remove long pauses at start/end
2. **Add intro/outro:** 3-5 second title card with project name
3. **Add captions:** Auto-generate subtitles (YouTube can do this)
4. **Volume normalization:** Ensure consistent audio level
5. **Remove mistakes:** Cut out long pauses, stutters, or restarts

**Free Editing Software:**
- **Windows:** DaVinci Resolve (free, professional)
- **Mac:** iMovie (free, simple) or DaVinci Resolve
- **Online:** Kapwing, Clipchamp

### Quality Control Review
Watch the entire video and check:
- ✅ All slides visible and readable (check text size)
- ✅ Audio clear throughout (no background noise)
- ✅ No long awkward pauses (>5 seconds)
- ✅ Pacing is consistent (not too fast or slow)
- ✅ Duration is 10-15 minutes (check submission requirements)
- ✅ Figures are visible and clear
- ✅ Transitions between slides are smooth

### File Preparation
- **File Name:** Use format like: `Hamza_Arshad_MSc_ML_Project_Presentation.mp4`
- **File Format:** MP4 (H.264 codec, AAC audio) - most universally compatible
- **Resolution:** 1080p minimum (1920×1080)
- **File Size:** Check submission limits (usually 500 MB - 2 GB max)
- **Backup:** Save two copies—one on computer, one on cloud (Google Drive/OneDrive)

### Compression (If File Too Large)
If file exceeds submission limit:
1. **HandBrake** (free) - compress to smaller size without quality loss
   - Use "Fast 1080p30" preset
   - Adjust quality slider to achieve target size
2. **Online tools:** CloudConvert, FreeConvert
3. **Lower resolution:** 720p is acceptable if necessary (1280×720)

---

## 📝 COMMON MISTAKES TO AVOID

### Don't:
- ❌ Read slides word-for-word robotically
- ❌ Speak too fast (rushing through)
- ❌ Speak too quietly or mumble
- ❌ Use filler words excessively ("um," "uh," "like")
- ❌ Forget to advance slides (check timing)
- ❌ Skip explaining figures (point and describe)
- ❌ Go over time limit (>15 minutes)
- ❌ Sound bored or disinterested

### Do:
- ✅ Speak conversationally but professionally
- ✅ Show enthusiasm for your findings
- ✅ Use pointer/mouse to highlight key points
- ✅ Pause briefly between slides (1-2 seconds)
- ✅ Emphasize key numbers (85%, 24-point gap, 100-sample threshold)
- ✅ Explain why findings matter
- ✅ Practice 2-3 times before final recording
- ✅ Sound confident and knowledgeable

---

## 🎤 DELIVERY TIPS

### Voice and Speaking
- **Volume:** Speak clearly and loudly enough (test recording)
- **Pace:** Moderate speed—not too fast, not too slow (aim for 10-13 minutes)
- **Pauses:** Pause briefly after key points for emphasis
- **Inflection:** Vary your tone to maintain interest (don't be monotone)
- **Confidence:** Speak with authority—you're the expert on this project

### Presentation Style
- **Enthusiasm:** Show genuine interest in your findings
- **Clarity:** Explain technical terms simply
- **Emphasis:** Stress important numbers and conclusions
- **Transitions:** Use connecting phrases ("This brings us to...", "Now let's examine...")
- **Engagement:** Imagine you're teaching someone who wants to understand

### Technical Details
- **Numbers:** Pronounce clearly ("85 point 55 percent," not "eighty-five point five five")
- **Acronyms:** Say full term first, then acronym ("Network Intrusion Detection System, or NIDS")
- **Figures:** Always describe what you're showing ("This bar chart shows...")
- **Key Terms:** Define important concepts (macro-F1, sample efficiency)

---

## 🕐 TIMING BREAKDOWN

Total duration: 10-15 minutes (aim for 12-13 minutes)

| Slide Range | Topic | Time | Cumulative |
|-------------|-------|------|------------|
| 1 | Title/Intro | 0:35 | 0:35 |
| 2-4 | Problem & Questions | 2:35 | 3:10 |
| 5-8 | Dataset & Method | 3:15 | 6:25 |
| 9-13 | Results & Analysis | 4:10 | 10:35 |
| 14-16 | Answers & Limitations | 2:25 | 13:00 |
| 17-19 | Recommendations & Conclusion | 1:45 | 14:45 |

**Target:** 12-13 minutes allows buffer for natural variation

---

## 🎬 RECORDING ENVIRONMENT TIPS

### Setup Your Space
- **Quiet room:** Close windows, turn off AC/fans if noisy
- **No interruptions:** Put phone on silent, close door, inform others
- **Good lighting:** If using webcam, face window or light source
- **Neutral background:** If showing yourself, use plain wall
- **Test setup:** Do 30-second test recording first

### Audio Quality
- **Microphone distance:** 6-12 inches from mouth
- **Pop filter:** Use if you have one (reduces "p" and "b" sounds)
- **Room acoustics:** Record in carpeted room (less echo)
- **Background noise:** Eliminate hum from computers, AC, traffic
- **Headphones:** Wear headphones if monitoring (prevents feedback)

### Technical Setup
- **Close other apps:** Free up CPU/memory
- **Disable notifications:** Turn off pop-ups, email alerts
- **Full screen slides:** Ensure slides are in presentation mode
- **Resolution check:** Slides should be crisp and readable
- **Record test:** 30-second test to verify setup

---

## 💾 FINAL SUBMISSION CHECKLIST

Before submitting your video:

- ✅ **File plays correctly** (test on different device)
- ✅ **Duration is acceptable** (10-15 minutes)
- ✅ **Audio is clear** throughout entire video
- ✅ **All slides are visible** and text is readable
- ✅ **File name is correct** (follow submission guidelines)
- ✅ **File format is compatible** (MP4 recommended)
- ✅ **File size is within limits** (compress if needed)
- ✅ **You have backup copy** (don't lose the file!)
- ✅ **Submission deadline noted** (submit early if possible)

---

## 🎯 EXPECTED OUTCOMES

After watching your video, evaluators should:
- ✅ Understand the problem (imbalanced NIDS data)
- ✅ Know your research questions (can CNN match trees?)
- ✅ Understand your methodology (rigorous controlled comparison)
- ✅ Know your main finding (tree ensembles win by 24 points)
- ✅ Understand why (sample efficiency, 100-sample threshold)
- ✅ Know practical recommendations (use XGBoost for production)
- ✅ Appreciate the contributions (first controlled comparison)
- ✅ Recognize honest evaluation (reported CNN failures)

---

## 📞 TROUBLESHOOTING

**Problem: File too large (>500 MB)**
- Solution: Compress with HandBrake, use "Fast 1080p30" preset, or reduce to 720p

**Problem: Audio has background noise**
- Solution: Re-record in quieter environment, or use Audacity noise reduction

**Problem: Slides not visible clearly**
- Solution: Increase resolution to 1080p, check screen recording settings

**Problem: Video too long (>15 minutes)**
- Solution: Edit out pauses, speak slightly faster, trim intro/outro

**Problem: Went blank during recording**
- Solution: That's okay! Edit out the pause, or re-record that section

**Problem: Made a mistake**
- Solution: Pause, restart from beginning of that slide, edit out mistake later

---

## 🌟 FINAL ENCOURAGEMENT

You've done excellent work on this project:
- ✅ Trained three models rigorously
- ✅ Achieved strong results (XGBoost 85% F1)
- ✅ Discovered important findings (100-sample threshold)
- ✅ Provided practical recommendations
- ✅ Honest evaluation of deep learning

**Be confident!** You know this project better than anyone. Speak with authority. Show enthusiasm for your findings. The 24-point gap is a significant, interesting result. The 100-sample threshold is a valuable contribution.

**Good luck with your recording!** 🎬🎓

---

**END OF VIDEO SCRIPT**

**Total Script Length:** 14-15 minutes (allows natural pacing)  
**Total Slides:** 19  
**Format:** Professional academic presentation  
**Level:** MSc Machine Learning (appropriate technical depth)
