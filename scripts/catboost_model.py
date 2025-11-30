from catboost import CatBoostClassifier
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import os 


train_path = "aluminum_coldRoll_train.csv"
test_path = "aluminum_coldRoll_testNoY.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


y = train["y_passXtremeDurability"]
X = train.drop(columns=["y_passXtremeDurability", "ID"])

test_id = test["ID"]
X_test = test.drop(columns=["ID"])


cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns detected:", cat_features)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
val_pool   = Pool(data=X_val,   label=y_val,   cat_features=cat_features)
test_pool  = Pool(data=X_test,  cat_features=cat_features)


model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Logloss",
    iterations=3000,         
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100,
    od_type="Iter",
    od_wait=150              
)


print("\nTraining CatBoost model...")
model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True
)

print("\nBest iteration:", model.get_best_iteration())
print("Best validation Logloss:", model.get_best_score()["validation"]["Logloss"])


preds = model.predict_proba(test_pool)[:, 1]


preds = preds.clip(1e-6, 1 - 1e-6)


submission = pd.DataFrame({
    "ID": test_id,
    "y_passXtremeDurability": preds
})

os.makedirs("submissions", exist_ok=True)
submission_path = "submissions/submission_catboost.csv"
submission.to_csv(submission_path, index=False)

print(f"\nSubmission saved → {submission_path}")



