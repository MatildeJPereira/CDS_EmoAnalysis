import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import ttest_rel


def load_data():
    csv_1 = pd.read_csv("./static/df_emotions_full_ds.csv")
    csv_2 = pd.read_csv("./static/df_emotions_full_ds-2.csv")
    csv_3 = pd.read_csv("./static/df_emotions_full_ds_3.csv")
    csv_4 = pd.read_csv("./static/df_emotions_full_ds_reduced.csv")

    emotions_scores = pd.concat([csv_1, csv_2, csv_3, csv_4])
    emotions_scores.reset_index(drop=True, inplace=True)
    emotions_scores.drop(columns="Unnamed: 0", inplace=True)
    return emotions_scores


def mean_and_std():
    emotion_comparison = pd.DataFrame({
        'Emotion': emotion_labels,
        'English Mean': data_eng.mean().values,
        'English Std': data_eng.std().values,
        'Italian Mean': data_it.mean().values,
        'Italian Std': data_it.std().values,
    })
    # print(emotion_comparison)

    # Plot Means
    (emotion_comparison[['Emotion', 'English Mean', 'Italian Mean']]
     .set_index('Emotion')
     .plot(kind='bar', figsize=(10, 6), colormap='coolwarm'))
    plt.title("Average Emotion Scores by Language")
    plt.ylabel("Mean Emotion Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/img/mean_emotion_scores.png")

    # Plot Stds
    (emotion_comparison[['Emotion', 'English Std', 'Italian Std']]
     .set_index('Emotion')
     .plot(kind='bar', figsize=(10, 6), colormap='coolwarm'))
    plt.title("Standard Deviation of Emotion Scores by Language")
    plt.ylabel("Standard Deviation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/img/std_emotion_scores_std.png")


def emotion_shifts():
    # Emotion shifts: IT - ENG
    emotion_diffs = pd.DataFrame()
    for emotion in emotion_labels:
        emotion_diffs[f'diff_{emotion}'] = data[f'it_{emotion}'] - data[f'eng_{emotion}']

    diff_cols = [col for col in emotion_diffs.columns if col.startswith('diff_')]
    emotion_diffs[diff_cols].plot(kind='box', figsize=(10, 6), color='darkred')
    plt.title("Distribution of Emotion Shifts (Italian - English)")
    plt.xticks(rotation=45)
    plt.axhline(0, linestyle='--', color='gray')
    plt.tight_layout()
    plt.savefig("static/img/emotion_shifts.png")

    avg_diffs = emotion_diffs[diff_cols].mean().rename(lambda x: x.replace('diff_', ''))
    plt.figure(figsize=(8, 5))
    sns.heatmap(avg_diffs.to_frame().T, annot=True, cmap='RdBu_r', center=0)
    plt.title("Average Emotion Shift (IT - ENG)")
    plt.yticks([])
    plt.savefig("static/img/avg_emotion_shifts.png")


def pca_analysis():
    # Standerdize
    X_eng_scaled = StandardScaler().fit_transform(data_eng)
    X_it_scaled = StandardScaler().fit_transform(data_it)

    # Apply PCA
    pca = PCA(n_components=2)
    X_eng_pca = pca.fit_transform(X_eng_scaled)
    X_it_pca = pca.transform(X_it_scaled)

    pca_df = pd.DataFrame(np.vstack([X_eng_pca, X_it_pca]), columns=['PC1', 'PC2'])
    pca_df['Language'] = ['English'] * len(data) + ['Italian'] * len(data)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Language', alpha=0.7,
                    palette={'English': 'blue', 'Italian': 'red'})
    plt.title("PCA of Emotion Profiles by Language")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/img/pca_analysis.png")

def paired_ttest():
    results = []
    for emotion in emotion_labels:
        eng = data[f'eng_{emotion}']
        it = data[f'it_{emotion}']
        t_stat, p_val = ttest_rel(eng, it)
        mean_diff = (it-eng).mean()
        results.append({
            'Emotion': emotion,
            'Mean_EN': eng.mean(),
            'Mean_IT': it.mean(),
            'Mean_diff_IT-EN': mean_diff,
            't_stat': t_stat,
            'p_value': p_val
        })
    ttest_results = pd.DataFrame(results)
    ttest_results['Significant'] = ttest_results['p_value'] < 0.05

    plt.figure(figsize=(10, 6))
    sns.barplot(data=ttest_results, x='Emotion', y='Mean_diff_IT-EN', palette='coolwarm', hue='Significant', dodge=False)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title("Mean Emotion Difference (IT - EN) with Significance")
    plt.ylabel("Mean Difference")
    plt.xlabel("Emotion")
    plt.xticks(rotation=45)
    plt.legend(title='Significant', loc='upper right')
    plt.tight_layout()
    plt.savefig("static/img/ttest_results.png")

def radar_plot():
    # Compute means
    eng_means = [data[f'eng_{emotion}'].mean() for emotion in emotion_labels]
    it_means = [data[f'it_{emotion}'].mean() for emotion in emotion_labels]

    # Radar chart setup
    labels = emotion_labels
    num_vars = len(labels)

    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Prepare data
    eng_means += eng_means[:1]
    it_means += it_means[:1]

    # Plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, eng_means, color='blue', linewidth=2, label='English')
    ax.fill(angles, eng_means, color='blue', alpha=0.25)

    ax.plot(angles, it_means, color='red', linewidth=2, label='Italian')
    ax.fill(angles, it_means, color='red', alpha=0.25)

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    # Set y-labels (optional: turn off to declutter)
    ax.set_yticklabels([])
    ax.set_title("Average Emotion Profiles by Language", size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.tight_layout()
    plt.savefig("static/img/radar_plot.png")


if __name__ == '__main__':
    data = load_data()

    english_columns = [col for col in data if col.startswith("eng_")]
    italian_columns = [col for col in data if col.startswith("it_")]
    emotion_labels = [col.replace("eng_", "") for col in english_columns]

    data_eng = data[english_columns]
    data_it = data[italian_columns]

    mean_and_std()
    emotion_shifts()
    pca_analysis() # TODO Verificar: Se calhar não faz muito sentido
    paired_ttest()
    radar_plot()

    # There is a relevant difference between all emotions besides anger and fear.
    # Trust has the highest difference, english considerably conveys more trust than italian

    # Interestingly, sadness, disgust and surprise are stronger in the italian responses, and especially interesting
    # is the two relevant differences in "negative" emotions, sadness and disgust, this may be due to differences in
    # cultural norms of expression, language structure or training data composition.”




