import numpy as np
import lightgbm as lgb


def get_lgbm(cfg: dict, scale_pos_weight: float = 1.0):
    rs = cfg["training"]["random_state"]
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=rs,
        n_jobs=cfg["training"]["n_jobs"],
        verbose=-1,
    )


def train_lgbm(model, X_train, y_train, X_val, y_val, cfg: dict):
    early = cfg["training"]["early_stopping_rounds"]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(period=-1)],
    )
    return model
