import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay,
                             precision_recall_fscore_support,
                             classification_report)

os.makedirs('outputs', exist_ok=True)

# Edit to match your labels 0, 1, 2
CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2']

# ---------- shared style (readable on a projector) ----------
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
})
BLUE, ORANGE, GREY = '#4f46e5', '#f59e0b', '#9ca3af'

# ---------- load data ----------
h = json.load(open('outputs/history.json'))
p = np.load('outputs/predictions.npz')
y_true, y_pred = p['y_true'], p['y_pred']
n = len(y_true)
correct = int((y_true == y_pred).sum())
acc = correct / n
n_classes = len(CLASS_NAMES)
baseline = 1 / n_classes


def smooth(v, w=5):
    return np.convolve(v, np.ones(w) / w, mode='valid')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'outputs/{name}.png', dpi=200)
    plt.close(fig)


# ---------- 1. headline card ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axis('off')
ax.text(0.5, 0.62, f'{acc:.0%}', ha='center', va='center',
        fontsize=110, fontweight='bold', color=BLUE)
ax.text(0.5, 0.25, f'accuracy on {n} unseen examples\n'
        f'(random guessing: {baseline:.0%})',
        ha='center', va='center', fontsize=20)
save(fig, '1_headline')

# ---------- 2. accuracy curve ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(h['accuracy'], label='Train', color=BLUE, linewidth=2)
ax.plot(h['val_accuracy'], color=ORANGE, alpha=0.3)
ax.plot(range(4, 4 + len(smooth(h['val_accuracy']))), smooth(h['val_accuracy']),
        label='Validation (5-epoch avg)', color=ORANGE, linewidth=2.5)
ax.axhline(baseline, ls='--', color=GREY, label=f'Random guess ({baseline:.0%})')
ax.set(title='The model learns to classify boundaries',
       xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.05))
ax.legend(loc='center right')
save(fig, '2_accuracy_curve')

# ---------- 3. loss curve with best-epoch marker ----------
best = int(np.argmin(h['val_loss']))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(h['loss'], label='Train', color=BLUE, linewidth=2)
ax.plot(h['val_loss'], label='Validation', color=ORANGE, linewidth=2)
ax.scatter([best], [h['val_loss'][best]], color='red', zorder=5,
           label=f'Lowest val loss (epoch {best + 1})')
ax.set(title='Loss falls steadily on train and validation',
       xlabel='Epoch', ylabel='Loss')
ax.legend()
save(fig, '3_loss_curve')

# ---------- 4. confusion matrices (counts + percentages) ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, display_labels=CLASS_NAMES, cmap='Blues',
    ax=axes[0], colorbar=False)
axes[0].set_title('Counts')
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, display_labels=CLASS_NAMES, cmap='Blues',
    normalize='true', values_format='.0%', ax=axes[1], colorbar=False)
axes[1].set_title('% of each true class')
for a in axes:
    a.grid(False)
fig.suptitle(f'Test set: {correct} of {n} correct', fontsize=18, fontweight='bold')
save(fig, '4_confusion_matrix')

# ---------- 5. per-class precision / recall / F1 ----------
prec, rec, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=range(n_classes), zero_division=0)
x = np.arange(n_classes)
w = 0.26
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (vals, label, color) in enumerate(
        [(prec, 'Precision', BLUE), (rec, 'Recall', ORANGE), (f1, 'F1', GREY)]):
    bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
    ax.bar_label(bars, fmt='%.2f', fontsize=11, padding=2)
ax.set_xticks(x)
ax.set_xticklabels([f'{c}\n(n={s})' for c, s in zip(CLASS_NAMES, support)])
ax.set(title='Performance per class', ylim=(0, 1.15))
ax.legend(loc='lower right')
save(fig, '5_per_class')

# ---------- console summary ----------
print(f'Test accuracy: {acc:.1%} ({correct}/{n})')
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
print('Charts saved to outputs/')