import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from imblearn.over_sampling import SMOTE
import joblib

output_folder = "C:/Users/felip/Documents/DSMM_Application_Design_for_Big_Data_BDM_1034/dataset us accidents/"
os.makedirs(output_folder, exist_ok=True)

df1 = pd.read_csv("C:/Users/felip/Documents/DSMM_Application_Design_for_Big_Data_BDM_1034/dataset us accidents/US_Accidents_March23 - Copy.csv")

columns_to_drop = ['City', 'State', 'ID', 'Source', 'County', 'Start_Time', 'End_Time', 'Start_Lat', 
                   'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)', 'Description', 'Street', 
                   'Zipcode', 'Country', 'Timezone', 'Airport_Code', 'Amenity', 'Bump', 'Crossing', 
                   'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
                   'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
df_remo = df1.drop(columns=columns_to_drop, axis=1)

df_remo = df_remo.dropna()

categorical_cols = ['Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
numerical_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

df = df_remo

mayoritaria = df[df['Severity'] == 2]
resto = df[df['Severity'] != 2]
mayoritaria_muestreada = mayoritaria.sample(frac=0.15, random_state=42)
df_muestreado = pd.concat([mayoritaria_muestreada, resto])
df_muestreado = df_muestreado.sample(frac=1, random_state=42).reset_index(drop=True)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_muestreado[col] = le.fit_transform(df_muestreado[col])
    label_encoders[col] = le

X = df_muestreado[categorical_cols + numerical_cols]
y = df_muestreado['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=15)
model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

try:
    joblib.dump(model, os.path.join(output_folder, "random_forest_model.pkl"), compress=3)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")

try:
    joblib.dump(label_encoders, os.path.join(output_folder, "label_encoders.pkl"))
    print("Encoders saved successfully.")
except Exception as e:
    print(f"Error saving the encoders: {e}")

config = {
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols
}

try:
    joblib.dump(config, os.path.join(output_folder, "model_config.pkl"))
    print("Configuration saved successfully.")
except Exception as e:
    print(f"Error saving the configuration: {e}")

importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title("Importance of Features")
plt.xlabel("Importance")
plt.ylabel("Characteristics")
plt.show()

y_test_binarized = label_binarize(y_test, classes=[1, 2, 3, 4])
y_pred_proba = model.predict_proba(X_test)

plt.figure(figsize=(10, 6))
for i in range(y_test_binarized.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f"Clase {i + 1} (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves by Class")
plt.legend(loc="best")
plt.show()