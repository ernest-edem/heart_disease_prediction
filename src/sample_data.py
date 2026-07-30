"""
Sample data module.
"""

import pandas as pd


def create_patient(
    age: int,
    sex: int,
    resting_bp: int,
    cholesterol: int,
    fasting_bs: int,
    max_hr: int,
    exercise_angina: int,
    oldpeak: float,
    resting_ecg_normal: int,
    resting_ecg_st: int,
    chest_pain_ata: int,
    chest_pain_nap: int,
    chest_pain_ta: int,
    st_slope_flat: int,
    st_slope_up: int,
) -> pd.DataFrame:
    """
    Create a patient record.

    Returns:
        DataFrame containing one patient.
    """

    return pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "RestingECG_Normal": resting_ecg_normal,
        "RestingECG_ST": resting_ecg_st,
        "ChestPainType_ATA": chest_pain_ata,
        "ChestPainType_NAP": chest_pain_nap,
        "ChestPainType_TA": chest_pain_ta,
        "ST_Slope_Flat": st_slope_flat,
        "ST_Slope_Up": st_slope_up,
    }])