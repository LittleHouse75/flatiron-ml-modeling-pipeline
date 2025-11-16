![Banner](https://github.com/LittleHouse75/flatiron-resources/raw/main/NevitsBanner.png)
# Machine Learning Modeling and Pipeline: Synthentic Data for Modeling Fraud in Ethereum Transactions

### 1. Business Problem Scenario

### Business problem

Fraud on public blockchains like Ethereum erodes trust for everyone in the ecosystem.
Today, **wallet providers, exchanges, and blockchain-analytics companies** all try to keep internal “scam lists” of high-risk addresses, but there is no shared master scam list, labels are treated as proprietary and confidential, and simple rule-based systems struggle to keep up with evolving scam patterns.

⠀
For a wallet provider, this means: Users may unknowingly send funds to known scam addresses, providers face reputation and compliance risks for facilitating repeated fraud, and fraud‑operations teams often lose time to manual triage instead of higher‑value investigative work.

⠀
**Business goal:**
The aim is to flag high‑risk destinations before a transaction is sent, help internal risk teams prioritize the most suspicious addresses for review, and allow wallet providers to maintain an adaptive internal scam list without sharing sensitive labels.

⠀
### Why machine learning?
Ethereum produces massive, high‑dimensional, time‑dependent data that cannot be meaningfully reviewed by humans. Scam behavior changes rapidly and includes non‑linear patterns such as timing bursts, unusual gas‑price behavior, and characteristic transaction fan‑outs. Machine learning can capture these behavioral signatures from historical labeled data, retrain as scams evolve, and score new addresses in near real time.

⠀
### Dataset description and relevance

The core training data is the **“Labeled transactions-based dataset on the Ethereum network”** from an academic benchmark repository. The dataset comes from a publicly released benchmark of real Ethereum transactions, organized at the transaction level with roughly 70k labeled examples.

Dataset repository: https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network

Each row contains the core transactional fields—hash, nonce, addresses, value, gas attributes, input data, timestamps, block metadata, and scam‑related fields such as from_scam, to_scam, from_category, and to_category.

⠀
From this transaction-level data, I construct an **address-level feature table** that summarizes behavior of each wallet over time (counts, amounts, timing, gas usage, etc.). The target label **Scam** is 1 if an address ever appears as a labeled scam in the source data.

To test whether models trained on this benchmark transfer to real-world signals, I will also build a **secondary evaluation dataset** by:
* Pulling **known scam addresses** from a public regulator (e.g., State of California’s published list of fraudulent crypto wallets).
* Fetching their on-chain transactions and mixing them with background traffic.
* Scoring these addresses with the trained model as an **external hold-out test set**.

⠀
This two‑dataset design allows me to evaluate models under controlled conditions while also testing whether the learned patterns transfer to real regulatory data.

⠀
### Success criteria

**Technical success (model-level):**
Technical success is measured primarily by Average Precision (AP), which summarizes performance across all classification thresholds and is well‑suited for heavily imbalanced fraud problems. Secondary measures such as precision, recall, F1, ROC‑AUC, and confusion‑matrix behavior help validate robustness and guide operational threshold selection.

⠀
**Business success (stakeholder-level):**

In practice, a wallet provider succeeds when the model achieves high recall on scam addresses at a tolerable false‑positive rate, flags a meaningful share of scam‑related transaction volume before it is sent, reduces investigator workload through better prioritization, and enables an internally maintained, continuously updated scam list.

⠀
⸻

### 2. Problem-Solving Process

### Data acquisition and understanding
* **Acquisition**
  * Download benchmark **labeled Ethereum transaction dataset** from its public GitHub / paper repository.
  * Fetch reference scam addresses from the California DFPI Crypto Scam Tracker: https://dfpi.ca.gov/consumers/crypto/crypto-scam-tracker/
  * Build a **second, real-world evaluation dataset** by:
    * Pulling scam addresses from California’s public scam-wallet list.
    * Fetching on-chain transactions for those addresses plus background traffic.
* **Initial data quality checks & EDA**

⠀(Much of this is already in the notebook.)
	* Inspect schema, types, and missingness.
	* Run **timestamp normalization and sanity checks** (this already caught a major format leak between scam and non-scam records, which I’ll explicitly avoid in the final pipeline).
	* Explore:
		* Transaction timestamp distribution and coverage.
		* Inter-transaction gap distributions.
		* Gas, gas-price, and value distributions (log histograms, hexbin).
		* Address-level activity (Zipf plots, degree CDFs).
		* Scam vs non-scam differences in value, gas, and volume.
* **Preliminary visualization strategy**
  * Time-series plots (daily volume).
  * Histograms and CDFs for gaps, value, gas, and degrees.
  * Comparison plots for **scam vs non-scam** behavior.

⠀
### Data preparation and feature engineering

Data cleaning and feature engineering are already prototyped in the notebook; the production version will:
1. **Robust timestamp parsing**
   * Normalize multiple timestamp string formats to UTC.
   * Convert to a numeric **seconds-since-dataset-start** field.
   * Explicitly verify that **timestamp formats themselves do not encode the label** (i.e., avoid the earlier bug where scam rows used a unique format).
2. **Numeric cleaning (zero-loss)**
   * Coerce numeric string columns (value, gas, gas_price, etc.) to numeric, fill invalids with 0, retain all rows.
3. **Aggregate to address-level feature table**

⠀For each address (sender or receiver), compute:
	* **Degree / connectivity**
		* in_degree, out_degree, all_degree
		* unique in_degree, unique out_degree
	* **Amount behavior**
		* Incoming: mean, sum, max, min.
		* Outgoing: mean, sum, max, min.
	* **Temporal behavior**
		* Avg time incoming, Avg time outgoing
		* Active Duration, Total transaction time
		* Mean / Max / Min time interval between transactions
		* Burstiness = max_gap / median_gap
		* Tx count, Activity Density (tx per unit active time)
		* Incoming count, Outgoing count, In/Out Ratio
		* Hour mean and Hour entropy (time of day and spread)
		* Last seen and Recency
	* **Gas behavior**
		* Avg gas price, Avg gas limit
	* **Graph structure**
		* Undirected graph clustering coefficient (NetworkX).
4. **Label construction**
   * Define **Scam = 1** if an address ever appears as a scam in any of:
     * from_scam, to_scam
     * from_category, to_category containing scam-words (scam, fraud, phish, etc.).
   * All other addresses: Scam = 0.
5. **Export**
   * Save engineered feature table as CSV/Parquet for modeling.
6. **scikit-learn pipeline**

⠀Conceptually:
	* **Preprocessing:**
		* Train/val/test split with **stratification** (≈70/15/15).
		* **StandardScaler** for models that need scaling (logistic regression, MLP).
	* **Modeling:**
		* Fit multiple algorithms from the same feature table.
	* **Evaluation:**
		* Compute metrics and plots via shared utilities (ROC, PR, threshold sweeps, calibration, confusion matrix).

⠀In the final project I can wrap this into a Pipeline / function set so the same steps are reusable when scoring the external California-based dataset.

### Modeling strategy

**Algorithms to evaluate (≥3):**

From the notebook, I already have:
* **Logistic Regression** (with class_weight="balanced").
* **Random Forest**.
* **ExtraTrees**.
* **XGBoost**.
* **MLPClassifier** (shallow neural net).

Additionally, as a stretch experiment, I will explore the SGAN-based semi-supervised approach described in “ATD-SGAN: Address Transaction Distribution–Semi-Supervised Generative Adversarial Network for Fraud Detection in Ethereum” (IEEE, 2023). This method augments the dataset by generating synthetic minority-class samples aligned with the statistical structure of known fraudulent addresses. If time allows, I will reproduce a simplified version of this approach and document its impact on model calibration and precision–recall behavior.

⠀
I will treat:
* Logistic Regression as a **simple linear baseline**.
* RandomForest and ExtraTrees as **non-linear tree ensembles**.
* **XGBoost** as the main model, since it already shows the best precision–recall performance.
* MLP as exploratory / likely to be dropped if it doesn’t outperform the trees.

⠀
**Cross-validation strategy**
* Use a **train/val/test split** (≈70/15/15) with stratification.
* For hyperparameter tuning:
  * Use **RandomizedSearchCV with 3-fold CV**, scoring on **Average Precision** to directly optimize ranking of scams.

⠀
**Hyperparameter tuning**
* Stage 1: broad RandomizedSearch on XGBoost, RF, and ExtraTrees to identify promising regions.
* Stage 2: **narrowed RandomizedSearch around the best XGBoost baseline**, focusing on:
  * max_depth, learning_rate, n_estimators
  * subsample, colsample_bytree
  * gamma, reg_alpha, reg_lambda, min_child_weight
* Use scale_pos_weight in XGBoost and class_weight="balanced" in tree / linear models to address class imbalance.

⠀
**Evaluation metrics**
* Primary: **Average Precision (AP)** on validation and test splits.
* Secondary:
  * Precision, recall, F1 at a fixed threshold (starting at 0.5, later potentially optimized).
  * ROC-AUC, confusion matrix, calibration curves.
* For the **California external dataset**, report:
  * AP, precision, recall at an operational threshold chosen to keep false-positive rate manageable.

⠀
### Results interpretation and communication
* Use **feature importance and SHAP**:
  Feature‑importance and SHAP analyses highlight patterns such as unusual time‑of‑day behavior, characteristic incoming transaction structures, dense bursts of activity, gas‑usage anomalies, and recency effects that distinguish scam addresses from normal accounts.
* Translate to business terms:
  * “Scam addresses tend to transact in short, concentrated bursts at unusual hours, with characteristic incoming deposits and gas patterns.”
  * Provide examples of **top-ranked scam-like addresses** and how a wallet provider could:
    * Show a **warning banner** before sending.
    * Route them to **manual review**.
* Visualization for stakeholders:
  * PR curves and ROC curves for baseline vs tuned model.
  * Bar plots of top SHAP features.
  * Simple confusion-matrix-style breakdown: “Of N known scams, the model flagged K” at a specific threshold.
  * If time allows, a **mock UI screenshot** showing how a model score could appear in a wallet product.

⠀
### Conceptual framework (PUML flowchart)


![Flowchart](./flowchart.png)



### 3. Timeline and Scope (This Week: Mon–Fri)

Assuming one project week left, here’s a concrete plan mapped to the rubric bullets.

### Monday – Dataset Finalization & Problem Formulation
* Lock **business problem statement** and stakeholders (wallet providers + users).
* Finalize **project repo structure** and documentation skeleton.
* Finish **clean EDA notebook** with corrected timestamp handling.
* Start building **California scam-based dataset**:
  * Scrape / download scam wallets from the CA site.
  * Outline script to fetch their Ethereum transactions.

⠀
### Tuesday – EDA & Data Preprocessing
* Complete **EDA** on both:
  * Benchmark dataset (already mostly done; just clean up and re-run after timestamp fix).
  * Early slice of CA-based data if available.
* Finalize **data cleaning + feature-engineering code** into a reusable module / notebook:
  * Robust timestamp parsing.
  * Numeric coercion.
  * Address-level feature table build.
* Generate and save **address_features.csv/parquet** as the canonical modeling dataset.

⠀
### Wednesday – Model Development
* Implement **baseline models** (LogReg, RF, ExtraTrees, XGBoost, MLP) on the engineered features.
* Run **train/val/test split + scaling** as in the current notebook.
* Capture comparison table across models (AP, precision, recall, F1).
* Decide which models move forward (likely **XGBoost + one tree ensemble as backup**).

⠀
### Thursday – Model Evaluation & Refinement
* Run **narrow RandomizedSearchCV** on XGBoost (and optionally RF/ET) using AP as the scoring metric.
* Evaluate tuned models on **validation + test** sets.
* Generate **evaluation plots** for the final chosen model:
  * ROC, PR, threshold curves, confusion matrix, calibration.
* Run **SHAP analysis** on the tuned XGBoost and export:
  * Beeswarm and bar plots.
  * Ranked feature-importance table.
* If time allows, explore the **semi-supervised SGAN / ATD-SGAN** approach as an experimental extension:
  * Prototype small experiment where synthetic samples are added.
  * Document results qualitatively, even if not fully production-ready.

⠀
### Friday – Documentation, Reporting, Final Review
* **Documentation & reporting**
  * Clean notebooks (EDA, feature engineering, modeling).
  * Write the **technical report** and **executive-level summary** (focused on business problem, method, and key findings).
  * Add diagrams (flowchart, plots, SHAP charts) to the report / slides.
* **External evaluation**
  * Run tuned model on **California scam-wallet dataset** (if fully ready) and summarize performance.
* **Final review & submission**
  * QA: rerun notebooks from top to bottom.
  * Record project video / presentation.
  * Package repo, report, and slides for submission.

⠀
### Anticipated challenges / learning needs
* **Label leakage & data quality**

⠀Ensuring features like timestamp formats don’t accidentally encode the label (as discovered in the initial bug) and that any pre-processing is strictly label-safe.
* **Class imbalance & threshold choice**

⠀Tuning for realistic business thresholds where false positives are tolerable but not overwhelming.
* **Domain shift: synthetic → real**

⠀Evaluating how well patterns learned from the benchmark dataset generalize to the California scam-wallet dataset; may require:
	* Additional feature engineering, or
	* Re-weighting / fine-tuning on a small subset of real data.
* **Advanced models (SGAN / ATD-SGAN)**

⠀Implementing semi-supervised GAN approaches may require extra reading and careful experimentation; I’ll treat this as a **stretch goal**, not core scope, and document any partial progress.


![Banner](https://github.com/LittleHouse75/flatiron-resources/raw/main/NevitsBanner.png)
# Machine Learning Modeling and Pipeline: Synthetic Data for Modeling Fraud in Ethereum Transactions

## 1. Business Problem Scenario

### Business problem

Fraud on Ethereum isn’t just some abstract issue buried in a block explorer. When people lose money, they lose confidence, and that erodes the whole point of having an open financial network. Wallet providers, exchanges, and analytics platforms all try to keep their own internal “bad address lists,” but these lists are private, uncoordinated, and often based on different internal rules. On top of that, scams evolve constantly, and simple rule‑based filters can only chase yesterday’s tricks.

For a wallet provider, this creates a very real set of problems. Users may unknowingly send money to addresses that are already known to be malicious elsewhere. Providers carry reputational and compliance risk if they keep facilitating transfers to wallets tied to fraud. And fraud‑operations teams end up spending time manually triaging suspicious activity instead of doing deeper investigative work.

### Business goal

The goal of this project is straightforward: use transaction behavior to predict which Ethereum addresses are likely associated with scams before a user sends funds to them. A wallet provider could warn the user, escalate the case to an internal review queue, or fold the result into an evolving in‑house scam list. The key idea is to create an adaptive signal that doesn’t require sharing proprietary labels across companies.

### Why machine learning?

Ethereum generates an enormous stream of dense, time‑dependent, somewhat messy data. Human reviewers can only look at small slices of it, and scam patterns shift faster than rule systems can adapt. Some scams operate in timing bursts, others rely on specific gas‑price tricks, and many use recognizable fans of inbound transactions. Machine learning is simply better suited to spotting these irregular patterns at scale. A model can learn the behavioral fingerprints of scam addresses, retrain as new scam types appear, and score new addresses in real time.

### Dataset description and relevance

The main dataset comes from a publicly released benchmark of Ethereum transactions. It has roughly 70,000 labeled rows, each representing a single blockchain transaction. Every row includes standard Ethereum fields—things like hash, nonce, sender and receiver, value, gas usage, calldata, timestamps, block metadata—as well as the fields marking whether the sender or receiver is tied to a known scam (from_scam, to_scam) and whatever scam category is available (“phishing” and “scamming”).

From these transactions, I build an address‑level feature table. This aggregates a wallet’s behavior over time: how often it transacts, how much value it tends to move, what time of day it sends or receives funds, how long it stays active, how bursty its habits are, and so on. An address is labeled **Scam = 1** if it ever appears in the scam fields of the source data.

To check whether a model trained on this synthetic benchmark transfers to real‑world data, I’m also building a second evaluation dataset. The State of California’s Department of Financial Protection and Innovation publishes a public list of wallets tied to crypto scams. I’ll fetch the on‑chain activity for those addresses, mix that activity with background network traffic, and run the model against it. This gives me a much more realistic performance check.

Together, these two datasets allow me to tune and test the model under clean conditions and then verify that the patterns it learns are still meaningful in the wild.

### Success criteria

On the technical side, I’m using **Average Precision (AP)** as my primary metric. It works well for heavily imbalanced problems like fraud detection because it evaluates how well the model ranks the rare positive cases across all thresholds. Secondary metrics—precision, recall, F1, ROC‑AUC—help validate that the model behaves sensibly when I choose an operational threshold.

From a business point of view, success means catching a good percentage of scam addresses without overwhelming users or internal reviewers. If the model flags high‑risk addresses before transactions are sent, captures a meaningful share of scam‑related activity, and reduces manual review overhead, it’s doing its job.

---

## 2. Problem‑Solving Process

### Data acquisition and understanding

I start with the benchmark dataset from the original research repository. To complement it, I pull published scam‑wallet addresses from California’s Crypto Scam Tracker and fetch their Ethereum transactions from a node provider. This gives me two distinct sources: synthetic benchmark data and real fraud data captured by a regulator.

For EDA, I check schema consistency, look for missing or malformed values, and verify that the timestamp formats don’t accidentally encode the scam label (something I discovered early on). I examine distributions of timestamps, value, gas usage, inter‑transaction gaps, and differences in behavior between scam and non‑scam addresses. Time‑series plots, histograms, and simple comparisons help reveal what actually separates the two groups.

### Data preparation and feature engineering

Most of the feature‑engineering work is already drafted in my notebook. The cleaned, production version will:

* Normalize timestamps into a single UTC format and convert them into numeric seconds‑since‑start.
* Coerce numeric fields like value, gas, and gas_price into proper numeric types, keeping all rows.
* Aggregate transactions into address‑level behavior summaries.

For each address, I compute connectivity (in‑degree, out‑degree), transaction amounts, timing behavior (active duration, burstiness, inter‑transaction gaps), hourly patterns, gas‑usage tendencies, and a simple graph‑structure metric using the clustering coefficient. Scam labels come directly from the provided fields and category indicators.

The final engineered dataset is saved as a clean CSV/Parquet file that feeds into the modeling stage.

### Modeling strategy

I begin with a small group of baseline models: Logistic Regression, Random Forest, ExtraTrees, XGBoost, and a lightweight MLP. These give me a sense of how both linear and non‑linear models handle the engineered features. In early experiments, XGBoost consistently produced the best precision‑recall behavior, so it’s my leading candidate.

As a stretch goal, I’m exploring the SGAN‑based approach described in *ATD‑SGAN* (IEEE 2023). The idea is to generate additional synthetic minority‑class samples in a way that reflects real scam behavior. Depending on time, I may run a simplified version of this to see whether it improves calibration or ranking performance.

For tuning, I’ll run RandomizedSearchCV with AP as the scoring metric. First, I’ll run a broad sweep over multiple models. Then, I’ll narrow in on the most promising XGBoost hyperparameters, especially depth, learning rate, subsampling, and the regularization parameters. Class imbalance is handled through `scale_pos_weight` or model‑specific class weighting.

Evaluation includes AP, precision, recall, F1, ROC‑AUC, threshold curves, and calibration behavior. I’ll also test the final tuned model on the California external dataset to see how well it generalizes to real fraud.

### Results interpretation and communication

SHAP and feature importance plots help explain which behaviors push an address toward being classified as a scam. Some patterns already stand out: odd time‑of‑day activity, bursts of incoming transactions, unusual gas usage, and a distinctive recency pattern compared with normal accounts.

For a non‑technical stakeholder, the takeaway is simpler: certain wallets behave in ways that consistently match the patterns seen in known scams. The model highlights those patterns so the product or fraud team can warn users or escalate reviews. PR curves, ROC curves, and a few illustrative examples make the findings easy to communicate.

I may also include a mock UI screenshot showing how a “High‑Risk Address” warning could appear in a wallet application.

---

## 3. Timeline and Scope (Mon–Fri)

### Monday — Finalizing Problem and Data
I’ll lock down the business framing, finish the clean EDA notebook (especially the fixed timestamp handling), and start building the California‑based dataset by scraping addresses and preparing the transaction‑fetching script.

### Tuesday — EDA and Preprocessing
I’ll complete EDA for both datasets and finalize the feature‑engineering module. By the end of the day, I expect to export a clean, address‑level dataset ready for modeling.

### Wednesday — Baseline Models
I’ll train the baseline models, compare their performance, and select the candidates worth tuning.

### Thursday — Tuning and SHAP
I’ll run the narrowed hyperparameter search on XGBoost, evaluate the tuned results, produce the performance plots, and generate SHAP explanations. If there’s time, I’ll run a small SGAN experiment.

### Friday — Documentation and Final Checks
I’ll polish the notebooks, write the technical and executive summaries, add visualizations, run the model against the California dataset, record the presentation, and prepare everything for submission.

---

## Anticipated challenges

A few areas may need extra attention: avoiding any label leakage from timestamp formats or other quirks in the synthetic data; choosing realistic thresholds that manage the tradeoff between false alarms and missed scams; dealing with domain shift when moving from synthetic data to the California dataset; and experimenting with SGANs, which may require additional reading.

Overall, the aim is to deliver a clear, defensible approach that connects technical modeling work to a real business