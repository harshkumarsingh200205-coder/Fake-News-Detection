"""
Generate visualization plots for the trained model
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_model.joblib')
METRICS_PATH = os.path.join(MODELS_DIR, 'model_metrics.json')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 16, 'weight': 'bold'})
    
    plt.title('Confusion Matrix - Fake News Detection\n(TF-IDF + Logistic Regression)', 
              fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=12, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_metrics(metrics, save_path):
    """Plot and save model metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1_score', 0)
    ]
    
    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6']
    bars = axes[0].bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (90%)')
    axes[0].axhline(y=0.8, color='orange', linestyle=':', alpha=0.5, label='Good (80%)')
    
    for bar, value in zip(bars, metric_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[0].legend(loc='lower right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    cm = np.array(metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
    correct = cm[0, 0] + cm[1, 1]
    incorrect = cm[0, 1] + cm[1, 0]
    
    axes[1].pie([correct, incorrect], 
                labels=['Correct Predictions', 'Incorrect Predictions'],
                colors=['#22c55e', '#ef4444'],
                autopct='%1.1f%%',
                explode=(0.05, 0),
                shadow=True,
                startangle=90,
                textprops={'fontsize': 12})
    
    axes[1].set_title('Prediction Accuracy Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_keyword_importance(model, save_path):
    """Plot top keywords for fake and real news"""
    if not hasattr(model, 'coef_'):
        print("Model not fitted, skipping keyword plot")
        return
    
    # Get feature names from the pipeline
    feature_names = model.named_steps['tfidf'].get_feature_names_out()
    coefficients = model.named_steps['classifier'].coef_[0]
    
    # Get top keywords
    fake_indices = np.argsort(coefficients)[:15]
    real_indices = np.argsort(coefficients)[-15:][::-1]
    
    fake_keywords = [(feature_names[i], coefficients[i]) for i in fake_indices]
    real_keywords = [(feature_names[i], coefficients[i]) for i in real_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Fake news keywords
    fake_words = [k[0] for k in fake_keywords]
    fake_scores = [abs(k[1]) for k in fake_keywords]
    
    axes[0].barh(fake_words[::-1], fake_scores[::-1], color='#ef4444', edgecolor='black')
    axes[0].set_xlabel('Importance Score', fontsize=12)
    axes[0].set_title('Top Keywords for FAKE News\n(Negative coefficients)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Real news keywords
    real_words = [k[0] for k in real_keywords]
    real_scores = [k[1] for k in real_keywords]
    
    axes[1].barh(real_words[::-1], real_scores[::-1], color='#22c55e', edgecolor='black')
    axes[1].set_xlabel('Importance Score', fontsize=12)
    axes[1].set_title('Top Keywords for REAL News\n(Positive coefficients)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.suptitle('Keyword Importance Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main():
    print("\n" + "="*50)
    print("📊 GENERATING VISUALIZATION PLOTS")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("\n❌ Model not found! Run train.py first.")
        return
    
    # Check if metrics exist
    if not os.path.exists(METRICS_PATH):
        print("\n❌ Metrics not found! Run train.py first.")
        return
    
    # Load metrics
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    
    print(f"\nModel Metrics:")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
    print(f"  Precision: {metrics.get('precision', 0):.2%}")
    print(f"  Recall:    {metrics.get('recall', 0):.2%}")
    print(f"  F1 Score:  {metrics.get('f1_score', 0):.2%}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Confusion Matrix
    cm = np.array(metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
    plot_confusion_matrix(cm, os.path.join(MODELS_DIR, 'confusion_matrix.png'))
    
    # 2. Metrics Plot
    plot_metrics(metrics, os.path.join(MODELS_DIR, 'metrics_plot.png'))
    
    # 3. Keyword Importance
    model = joblib.load(MODEL_PATH)
    plot_keyword_importance(model, os.path.join(MODELS_DIR, 'keyword_importance.png'))
    
    print("\n" + "="*50)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*50)
    print(f"\nPlots saved to: {MODELS_DIR}")
    print("  - confusion_matrix.png")
    print("  - metrics_plot.png")
    print("  - keyword_importance.png")


if __name__ == "__main__":
    main()