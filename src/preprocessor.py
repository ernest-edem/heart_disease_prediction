import pandas as pd
from pathlib import Path


#Dataset Info
def dataset_info(df):
    print(f"\nDataset Information")
    print("-" * 20)
    print(df.info())

    print(f"\nDataset Shape: {df.shape}")

    # Missing and Duplicate Values
    print(f"\n{df.isnull().sum()}")
    print(f"\nTotal Duplicates: {df.duplicated().sum()}")
    return df


#Feature Encoding>>>>>>>

def encoding(df):

    df_new = df.copy()
    # binary_encoding
    df_new["Sex"] = df_new["Sex"].map({
        "M": 1,
        "F": 0
    })

    df_new["ExerciseAngina"] = df_new["ExerciseAngina"].map({
        "Y": 1,
        "N": 0
    })
    return df_new

# one_hot_encoding
def hot_encoding(df_new):

    df_new = pd.get_dummies(df_new, columns=["RestingECG"], drop_first=True)

    df_new = pd.get_dummies(df_new, columns=["ChestPainType"], drop_first=True)

    df_new= pd.get_dummies(df_new, columns=["ST_Slope"], drop_first=True)
    return df_new

# Save Encoded Dataset
def save_encoded_dataset(df_new):
    BASE_DIR = Path(__file__).resolve().parent.parent

    dataset_path = BASE_DIR / "dataset" / "encoded_dataset.csv"

    df_new.to_csv(dataset_path)
    return

# Dataset Statistics
def dataset_statistics(df_new):
    dataset_stats = pd.DataFrame({
        "Metrics": [
            "Total Missing Values",
            "Total Duplicates",
            "Shape",
            "Highest Age",
            "Lowest Age",
            "Average Age",
            "Number of Males",
            "Number of Females"
        ],
        "Values": [
            f"{df_new.isnull().sum().sum()}",
            f"{df_new.duplicated().sum()}",
            f"{df_new.shape}",
            f"{df_new['Age'].max()}",
            f"{df_new['Age'].min()}",
            f"{df_new['Age'].mean():.2f}",
            f"{(df_new['Sex'] == 1).sum()}",
            f"{(df_new['Sex'] == 0).sum()}"
        ]
    })
    return dataset_stats



