"""Feature engineering pipeline applied identically to train and test."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import config


# Module-level label encoder so it can be reused across train/test
_label_encoder = None


def engineer_features(df, fit_encoder=False):
    """
    Apply all feature engineering transformations.
    
    Args:
        df: Raw dataframe with original features.
        fit_encoder: If True, fit the label encoder (use for training).
                     If False, use the already-fitted encoder (use for test).
    
    Returns:
        Tuple of (engineered dataframe, list of feature column names).
    """
    global _label_encoder
    d = df.copy()

    # ── Binary threshold flags ──────────────────────────────────────────
    d['is_night'] = d['transaction_hour'].isin(config.NIGHT_HOURS).astype(int)
    d['is_night_broad'] = d['transaction_hour'].isin(config.NIGHT_HOURS_BROAD).astype(int)
    d['low_device_trust'] = (d['device_trust_score'] < config.DEVICE_TRUST_THRESHOLD).astype(int)
    d['low_device_trust_v2'] = (d['device_trust_score'] < config.DEVICE_TRUST_THRESHOLD_V2).astype(int)
    d['high_velocity'] = (d['velocity_last_24h'] >= config.VELOCITY_THRESHOLD).astype(int)
    d['high_amount'] = (d['amount'] > config.HIGH_AMOUNT_THRESHOLD).astype(int)
    d['very_high_amount'] = (d['amount'] > config.VERY_HIGH_AMOUNT_THRESHOLD).astype(int)

    # ── Composite risk scores ───────────────────────────────────────────
    d['risk_flags'] = (d['foreign_transaction'] + d['location_mismatch'] +
                       d['is_night'] + d['low_device_trust'] + d['high_velocity'])
    d['risk_flags_v2'] = (d['foreign_transaction'] + d['location_mismatch'] +
                          d['is_night_broad'] + d['low_device_trust_v2'] + d['high_velocity'])

    # ── Interaction features (multiplicative) ───────────────────────────
    d['foreign_loc_mismatch'] = d['foreign_transaction'] * d['location_mismatch']
    d['foreign_high_vel'] = d['foreign_transaction'] * d['high_velocity']
    d['night_foreign'] = d['is_night'] * d['foreign_transaction']
    d['night_low_trust'] = d['is_night'] * d['low_device_trust']
    d['night_loc_mismatch'] = d['is_night'] * d['location_mismatch']
    d['foreign_low_trust'] = d['foreign_transaction'] * d['low_device_trust_v2']
    d['night_high_amount'] = d['is_night'] * d['high_amount']
    d['night_low_trust_v2'] = d['is_night'] * d['low_device_trust_v2']

    # ── Transformations ─────────────────────────────────────────────────
    d['amount_log'] = np.log1p(d['amount'])
    d['amount_squared'] = d['amount'] ** 2
    d['trust_velocity_ratio'] = d['device_trust_score'] / (d['velocity_last_24h'] + 1)
    d['amount_per_velocity'] = d['amount'] / (d['velocity_last_24h'] + 1)

    # ── Categorical encoding ────────────────────────────────────────────
    if fit_encoder:
        _label_encoder = LabelEncoder()
        d['merchant_cat_encoded'] = _label_encoder.fit_transform(d['merchant_category'])
    else:
        if _label_encoder is None:
            raise RuntimeError("Label encoder not fitted. Call with fit_encoder=True first.")
        d['merchant_cat_encoded'] = _label_encoder.transform(d['merchant_category'])

    return d


def get_feature_columns():
    """Return the list of feature columns to use for modeling."""
    return [
        # Original numeric features
        'amount', 'transaction_hour', 'foreign_transaction',
        'location_mismatch', 'device_trust_score', 'velocity_last_24h',
        'cardholder_age',
        # Encoded categorical
        'merchant_cat_encoded',
        # Binary flags
        'is_night', 'is_night_broad', 'low_device_trust', 'low_device_trust_v2',
        'high_velocity', 'high_amount', 'very_high_amount',
        # Risk scores
        'risk_flags', 'risk_flags_v2',
        # Interactions
        'foreign_loc_mismatch', 'foreign_high_vel', 'night_foreign',
        'night_low_trust', 'night_loc_mismatch', 'foreign_low_trust',
        'night_high_amount', 'night_low_trust_v2',
        # Transformations
        'amount_log', 'amount_squared', 'trust_velocity_ratio',
        'amount_per_velocity',
    ]
