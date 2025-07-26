import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Membaca data dari file Excel
df = pd.read_excel('D:/SKRIPSI/dataset/confusion_matrix.xlsx')

# Ambil kolom True_Label dan Predicted_Label
y_true = df['true_label']  # Sesuaikan dengan nama kolom yang sesuai di file Excel Anda
y_pred = df['predicted_label']  # Sesuaikan dengan nama kolom yang sesuai di file Excel Anda

# Hitung Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix sebagai Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['cari kode', 'jelaskan kode', 'lainnya'], yticklabels=['cari kode', 'jelaskan kode', 'lainnya'], cbar=True)

# Menambahkan label dan judul
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Menampilkan plot Confusion Matrix
plt.show()

# Hitung dan Tampilkan Metrik Evaluasi
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Metrik evaluasi
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]

# Membuat figure untuk metrik
fig, ax = plt.subplots(figsize=(10, 6))

# Menambahkan kotak dengan nilai-nilai di dalamnya
for i, metric in enumerate(metrics):
    # Menambahkan kotak untuk setiap metrik
    ax.text(0.2 + i*0.25, 0.5, f'{metric}: {values[i]:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold', color='black', bbox=dict(facecolor='#FFD700', alpha=0.8, boxstyle='round,pad=1'))

# Menghapus sumbu x dan y pada plot metrik
ax.axis('off')

# Menampilkan plot metrik
plt.title('Model Evaluation Metrics', fontsize=16, fontweight='bold')
plt.show()
