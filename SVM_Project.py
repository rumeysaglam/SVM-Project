import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

np.random.seed(42)

n_samples = 200

yearsofExperience = np.random.randint(0, 10, n_samples)
technicalScore = np.random.randint(0, 100, n_samples)

labels = []

for experience, score in zip(yearsofExperience, technicalScore):
    if experience < 2 or score < 60:
        labels.append(0) #İşe alınmadı
    else:
        labels.append(1) #İşe alındı

df = pd.DataFrame({
    'experience_years': yearsofExperience,
    'technical_score': technicalScore,
    'label': labels
})

print(df)

X = df[["experience_years", "technical_score"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #Eğitimi temel alarak öğrenir ve dönüştürür
#Eğitim verisi üzerinde önce fit (ortalama ve std sapma hesaplar), sonra transform (dönüştürür).
X_test_scaled = scaler.transform(X_test) #Aynı kuralla test verisini dönüştürür
#Burada sadece transform var. Çünkü test verisini eğitim setinin ortalama ve std sapmasına göre ölçekliyoruz.
# Bu doğru olan yöntemdir. Model eğitim verisinin dağılımına göre çalışmalı.

model = SVC(kernel="linear")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Model ve scaler'ı kaydet
joblib.dump(model, r"C:/TurkcellGYK_YZ/Machine Learning/Supervised Learning/5. SVM/SVM_Project/model.joblib")
joblib.dump(scaler, r"C:/TurkcellGYK_YZ/Machine Learning/Supervised Learning/5. SVM/SVM_Project/scaler.joblib")

# Test et
test_cases = [
    [1, 50],  # Az tecrübe, düşük skor → İşe alınmamalı (0)
    [5, 85],  # İyi tecrübe, yüksek skor → İşe alınmalı (1)
    [0, 30],  # Hiç tecrübe, çok düşük skor → İşe alınmamalı (0)
    [8, 95]   # Çok tecrübe, çok yüksek skor → İşe alınmalı (1)
]

print("\nTest Tahminleri:")
for case in test_cases:
    scaled_input = scaler.transform([case])
    pred = model.predict(scaled_input)
    result = "✅ Hired" if pred[0] == 1 else "❌ Not Hired"
    print(f"Tecrübe: {case[0]}, Skor: {case[1]} → {result}")

# Eğitim verisinden x ve y eksenlerini ayır
X_vis = X_train_scaled
y_vis = y_train

# Meshgrid (karar sınırı için zemin oluştur)
h = 0.02
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Modelin karar fonksiyonu
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Çizim
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel("Tecrübe (scaled)")
plt.ylabel("Teknik Puan (scaled)")
plt.title("SVM Karar Sınırı (Linear Kernel)")
plt.show() 