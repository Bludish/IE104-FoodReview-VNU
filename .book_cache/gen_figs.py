import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Figure 325: Degenerate Loops & Mitigation
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
ax.set_facecolor('#ffffff')
fig.patch.set_facecolor('#ffffff')
ax.axis('off')

ax.text(0.5, 0.95, 'Degenerate Feedback Loops & Technical Mitigation Strategies', fontsize=14, fontweight='bold', ha='center', color='#0f172a')

# Left box: Problem
p_box = patches.FancyBboxPatch((0.05, 0.45), 0.4, 0.42, boxstyle='round,pad=0.03', ec='#b91c1c', fc='#fef2f2', lw=1.5)
ax.add_patch(p_box)
ax.text(0.25, 0.82, 'Vicious Cycle (Degenerate Loop)', fontsize=11, fontweight='bold', ha='center', color='#991b1b')
ax.text(0.25, 0.73, 'Model ranks Item A slightly higher', fontsize=9.5, ha='center', color='#1e293b')
ax.text(0.25, 0.65, 'Item A receives top screen exposure', fontsize=9.5, ha='center', color='#1e293b')
ax.text(0.25, 0.57, 'Users click Item A due to position bias', fontsize=9.5, ha='center', color='#1e293b')
ax.text(0.25, 0.49, 'Next training cycle over-indexes Item A', fontsize=9.5, ha='center', color='#991b1b', fontweight='bold')

# Right box: Solution 1 - Randomization Pool
s1_box = patches.FancyBboxPatch((0.52, 0.45), 0.43, 0.42, boxstyle='round,pad=0.03', ec='#047857', fc='#ecfdf5', lw=1.5)
ax.add_patch(s1_box)
ax.text(0.735, 0.82, 'Solution 1: Cold-Start Traffic Pool (TikTok)', fontsize=10.5, fontweight='bold', ha='center', color='#065f46')
ax.text(0.735, 0.72, 'Allocate random initial traffic (100-500 imp.)', fontsize=9, ha='center', color='#1e293b')
ax.text(0.735, 0.64, 'Measure unbiased interaction quality', fontsize=9, ha='center', color='#1e293b')
ax.text(0.735, 0.56, 'Elevate to broader pool only if metrics hold', fontsize=9, ha='center', color='#065f46', fontweight='bold')

# Bottom box: Solution 2 - Positional Decoupling
s2_box = patches.FancyBboxPatch((0.15, 0.06), 0.7, 0.32, boxstyle='round,pad=0.03', ec='#1d4ed8', fc='#eff6ff', lw=1.5)
ax.add_patch(s2_box)
ax.text(0.5, 0.31, 'Solution 2: Positional Feature Decoupling & Two-Stage Modeling', fontsize=10.5, fontweight='bold', ha='center', color='#1e40af')
ax.text(0.5, 0.22, 'Training: Pass position as feature (1st_pos=True/False) -> Model learns position weight', fontsize=9, ha='center', color='#1e293b')
ax.text(0.5, 0.15, 'Inference: Fix 1st_pos=False for all items -> Rank purely by intrinsic content relevance', fontsize=9, ha='center', color='#1e293b')
ax.text(0.5, 0.08, 'Two-Stage: P(Click|Item, Pos) = P(Examine|Pos) * P(Click|Examine, Item)', fontsize=9, ha='center', color='#1e40af', fontweight='bold')

plt.tight_layout()
plt.savefig(r'd:\Documents\IE104.R12\distilled_ml_systems_en\figures\figure_page_325_1.jpg', format='jpeg')
plt.savefig(r'd:\Documents\IE104.R12\distilled_ml_systems\figures\figure_page_325_1.jpg', format='jpeg')
plt.close()

# 2. Figure 329: Data Distribution Shifts Decomposition
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
ax.set_facecolor('#ffffff')
fig.patch.set_facecolor('#ffffff')
ax.axis('off')

ax.text(0.5, 0.95, 'Mathematical Taxonomy of Data Distribution Shifts: P(X, Y)', fontsize=14, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.5, 0.88, 'Decomposition 1: P(X, Y) = P(Y|X) * P(X)     |     Decomposition 2: P(X, Y) = P(X|Y) * P(Y)', fontsize=10, ha='center', color='#475569', fontstyle='italic')

# Box 1: Covariate Shift
b1 = patches.FancyBboxPatch((0.04, 0.18), 0.28, 0.64, boxstyle='round,pad=0.03', ec='#2563eb', fc='#f8fafc', lw=1.5)
ax.add_patch(b1)
ax.text(0.18, 0.76, 'Covariate Shift', fontsize=12, fontweight='bold', ha='center', color='#1e40af')
ax.text(0.18, 0.67, 'P(X) Changes\nP(Y|X) Invariant', fontsize=10.5, fontweight='bold', ha='center', color='#2563eb')
ax.text(0.18, 0.52, 'Input feature space shifts;\nground-truth mapping\nremains constant.', fontsize=9, ha='center', color='#334155')
ax.text(0.18, 0.37, 'Ex: Demographics change;\naudio recorded in noisy room\ninstead of quiet clinic.', fontsize=8.5, ha='center', color='#475569')
ax.text(0.18, 0.23, 'Fix: Importance Weighting\nw(x) = P_t(x) / P_s(x)', fontsize=9, fontweight='bold', ha='center', color='#1e40af')

# Box 2: Label Shift
b2 = patches.FancyBboxPatch((0.36, 0.18), 0.28, 0.64, boxstyle='round,pad=0.03', ec='#7c3aed', fc='#f8fafc', lw=1.5)
ax.add_patch(b2)
ax.text(0.50, 0.76, 'Label Shift (Prior Shift)', fontsize=12, fontweight='bold', ha='center', color='#6d28d9')
ax.text(0.50, 0.67, 'P(Y) Changes\nP(X|Y) Invariant', fontsize=10.5, fontweight='bold', ha='center', color='#7c3aed')
ax.text(0.50, 0.52, 'Label distribution changes;\nfeature distribution given\nlabel remains constant.', fontsize=9, ha='center', color='#334155')
ax.text(0.50, 0.37, 'Ex: Disease epidemic increases\ninfection rate, but symptoms\nfor infected remain identical.', fontsize=8.5, ha='center', color='#475569')
ax.text(0.50, 0.23, 'Fix: Class re-weighting,\nBBSE (Lipton et al.)', fontsize=9, fontweight='bold', ha='center', color='#6d28d9')

# Box 3: Concept Drift
b3 = patches.FancyBboxPatch((0.68, 0.18), 0.28, 0.64, boxstyle='round,pad=0.03', ec='#dc2626', fc='#f8fafc', lw=1.5)
ax.add_patch(b3)
ax.text(0.82, 0.76, 'Concept Drift', fontsize=12, fontweight='bold', ha='center', color='#b91c1c')
ax.text(0.82, 0.67, 'P(Y|X) Changes\nP(X) Invariant', fontsize=10.5, fontweight='bold', ha='center', color='#dc2626')
ax.text(0.82, 0.52, 'Same input leads to\ndifferent outputs over time\n(definition changes).', fontsize=9, ha='center', color='#334155')
ax.text(0.82, 0.37, 'Ex: SF house prices drop post-remote\nwork surge; search for \"Wuhan\"\npost-2019 query intent.', fontsize=8.5, ha='center', color='#475569')
ax.text(0.82, 0.23, 'Fix: Continual retraining,\nsliding retraining windows', fontsize=9, fontweight='bold', ha='center', color='#b91c1c')

# Bottom note
ax.text(0.5, 0.06, 'Label Schema Change: Y space changes (new classes added/split) -> Breaks final Softmax layer -> Requires full retrain', fontsize=8.5, ha='center', color='#0f172a', fontweight='bold')

plt.tight_layout()
plt.savefig(r'd:\Documents\IE104.R12\distilled_ml_systems_en\figures\figure_page_329_1.jpg', format='jpeg')
plt.savefig(r'd:\Documents\IE104.R12\distilled_ml_systems\figures\figure_page_329_1.jpg', format='jpeg')
plt.close()

# 3. Figure 332: Shift Detection Tests
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
ax.set_facecolor('#ffffff')
fig.patch.set_facecolor('#ffffff')
ax.axis('off')

ax.text(0.5, 0.95, 'Taxonomy of Statistical Tests for Production Drift Detection', fontsize=14, fontweight='bold', ha='center', color='#0f172a')

# Col 1: 1D Continuous
c1 = patches.FancyBboxPatch((0.05, 0.12), 0.27, 0.75, boxstyle='round,pad=0.03', ec='#0284c7', fc='#f0f9ff', lw=1.5)
ax.add_patch(c1)
ax.text(0.185, 0.81, '1D Continuous Data', fontsize=11.5, fontweight='bold', ha='center', color='#0369a1')
ax.text(0.185, 0.72, 'Kolmogorov-Smirnov (KS)', fontsize=10, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.185, 0.63, 'D = sup_x |F_1(x) - F_2(x)|\nNon-parametric, compares ECDFs.\nIdeal for scalar predictions P(Y_hat)', fontsize=8.5, ha='center', color='#334155')
ax.text(0.185, 0.48, 'Wasserstein Distance (EMD)', fontsize=10, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.185, 0.39, 'W_1(P, Q) = inf E[||x - y||]\nContinuous, stable, finite even with\nzero overlap (unlike KL-div).', fontsize=8.5, ha='center', color='#334155')
ax.text(0.185, 0.24, 'Population Stability Index (PSI)', fontsize=10, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.185, 0.16, 'Binned metric (Banking standard):\n<0.10: Stable | >0.25: Critical Drift', fontsize=8.5, ha='center', color='#0369a1', fontweight='bold')

# Col 2: 1D Categorical
c2 = patches.FancyBboxPatch((0.365, 0.12), 0.27, 0.75, boxstyle='round,pad=0.03', ec='#d97706', fc='#fffbeb', lw=1.5)
ax.add_patch(c2)
ax.text(0.50, 0.81, 'Categorical Data', fontsize=11.5, fontweight='bold', ha='center', color='#b45309')
ax.text(0.50, 0.72, 'Chi-Square (χ²) Goodness-of-Fit', fontsize=10, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.50, 0.63, 'χ² = sum (O_i - E_i)² / E_i\nCompares observed vs expected\nfrequencies across categories.', fontsize=8.5, ha='center', color='#334155')
ax.text(0.50, 0.48, 'Jensen-Shannon (JS) Divergence', fontsize=10, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.50, 0.39, 'Symmetric, bounded in [0, 1].\nSquare root of JSD is a true metric.', fontsize=8.5, ha='center', color='#334155')
ax.text(0.50, 0.24, 'Schema Integrity Check', fontsize=10, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.50, 0.16, 'Unseen categories detection;\nNull / Missing value rate tracking.', fontsize=8.5, ha='center', color='#b45309', fontweight='bold')

# Col 3: High-Dimensional Multivariate
c3 = patches.FancyBboxPatch((0.68, 0.12), 0.27, 0.75, boxstyle='round,pad=0.03', ec='#059669', fc='#f0fdf4', lw=1.5)
ax.add_patch(c3)
ax.text(0.815, 0.81, 'High-Dimensional (Multivariate)', fontsize=11.5, fontweight='bold', ha='center', color='#047857')
ax.text(0.815, 0.72, 'Maximum Mean Discrepancy (MMD)', fontsize=10, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.815, 0.63, 'Kernel-based distance in RKHS.\nPopular in research, compute-heavy\nin high-scale production.', fontsize=8.5, ha='center', color='#334155')
ax.text(0.815, 0.48, 'Dimensionality Reduction + 1D Tests', fontsize=10, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.815, 0.39, 'Industry Standard: Compress via PCA,\nAutoencoders, or Embeddings ->\nRun KS-test / PSI on latent space.', fontsize=8.5, ha='center', color='#047857', fontweight='bold')
ax.text(0.815, 0.24, 'Domain Classifier (Two-Sample ML)', fontsize=10, fontweight='bold', ha='center', color='#0f172a')
ax.text(0.815, 0.16, 'Train binary classifier (Train vs Serve);\nROC-AUC > 0.55 flags distinct distributions.', fontsize=8.5, ha='center', color='#334155')

plt.tight_layout()
plt.savefig(r'd:\Documents\IE104.R12\distilled_ml_systems_en\figures\figure_page_332_1.jpg', format='jpeg')
plt.savefig(r'd:\Documents\IE104.R12\distilled_ml_systems\figures\figure_page_332_1.jpg', format='jpeg')
plt.close()

# 4. Figure 335: 3-Tier Observability Architecture
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
ax.set_facecolor('#ffffff')
fig.patch.set_facecolor('#ffffff')
ax.axis('off')

ax.text(0.5, 0.95, '3-Tier Production Machine Learning Observability Framework', fontsize=14, fontweight='bold', ha='center', color='#0f172a')

# Tier 3
t3 = patches.FancyBboxPatch((0.08, 0.66), 0.84, 0.22, boxstyle='round,pad=0.03', ec='#7c3aed', fc='#faf5ff', lw=1.5)
ax.add_patch(t3)
ax.text(0.5, 0.83, 'TIER 3: Model Performance & Business KPIs (Delayed Ground Truth)', fontsize=11, fontweight='bold', ha='center', color='#6d28d9')
ax.text(0.5, 0.74, 'Delayed ML Metrics: Accuracy, F1-Score, AUC-ROC, Log-Loss, Calibration Error | Slice-based Evaluation', fontsize=8.5, ha='center', color='#1e293b')
ax.text(0.5, 0.68, 'Business Impact: Conversion Rate (CVR), Click-Through Rate (CTR), Revenue per User, Chargeback Rate', fontsize=8.5, ha='center', color='#475569')

# Tier 2
t2 = patches.FancyBboxPatch((0.08, 0.38), 0.84, 0.22, boxstyle='round,pad=0.03', ec='#0284c7', fc='#f0f9ff', lw=1.5)
ax.add_patch(t2)
ax.text(0.5, 0.55, 'TIER 2: Data Integrity & Prediction Drift (Real-Time Early Warning - No Labels Needed)', fontsize=11, fontweight='bold', ha='center', color='#0369a1')
ax.text(0.5, 0.46, 'Prediction Distribution P(Y_hat): Mean, Quantiles (P50, P90, P99), Softmax Entropy, PSI on Output Scores', fontsize=8.5, ha='center', color='#1e293b')
ax.text(0.5, 0.40, 'Input Feature Drift: PSI, KS-Test on raw features | Data Schema: Null rates, Unseen categories, Range breaches', fontsize=8.5, ha='center', color='#475569')

# Tier 1
t1 = patches.FancyBboxPatch((0.08, 0.10), 0.84, 0.22, boxstyle='round,pad=0.03', ec='#059669', fc='#f0fdf4', lw=1.5)
ax.add_patch(t1)
ax.text(0.5, 0.27, 'TIER 1: System & Operational Telemetry (Standard SRE / Infrastructure)', fontsize=11, fontweight='bold', ha='center', color='#047857')
ax.text(0.5, 0.18, 'Latency Metrics: P50, P95, P99 inference latency, network round-trip time | Throughput: Requests/sec (QPS)', fontsize=8.5, ha='center', color='#1e293b')
ax.text(0.5, 0.12, 'Hardware Health: GPU/CPU utilization, VRAM memory leaks, OOM crash logs, HTTP 4xx/5xx failure rates', fontsize=8.5, ha='center', color='#475569')

plt.tight_layout()
plt.savefig(r'd:\Documents\IE104.R12\distilled_ml_systems_en\figures\figure_page_335_1.jpg', format='jpeg')
plt.savefig(r'd:\Documents\IE104.R12\distilled_ml_systems\figures\figure_page_335_1.jpg', format='jpeg')
plt.close()

print('All 5 figures generated successfully!')
