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
    print("\nData Analysis")
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
    plt.show()

    # Plot Stds
    (emotion_comparison[['Emotion', 'English Std', 'Italian Std']]
     .set_index('Emotion')
     .plot(kind='bar', figsize=(10, 6), colormap='coolwarm'))
    plt.title("Average Emotion Scores by Language")
    plt.ylabel("Mean Emotion Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


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
    plt.show()

    avg_diffs = emotion_diffs[diff_cols].mean().rename(lambda x: x.replace('diff_', ''))
    plt.figure(figsize=(8, 5))
    sns.heatmap(avg_diffs.to_frame().T, annot=True, cmap='RdBu_r', center=0)
    plt.title("Average Emotion Shift (IT - ENG)")
    plt.yticks([])
    plt.show()


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
    plt.show()


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

    # Hypothesis: Trust score is higher in English than Italian
    # Verify that the visual difference is relevant
    t_stat, p_val = ttest_rel(data['eng_trust'], data['it_trust'])
    print(f"T-test result: t = {t_stat:.3f}, p = {p_val:.4f}")

    # Data Analysis
    # T-test result: t = 8.965, p = 0.0000
    # That is VERY relevant

    # TODO T-test para todas as emoções ?


