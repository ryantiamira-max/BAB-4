
# Example 4.23 - Sentiment Analysis

from transformers import pipeline

# Membuat model untuk analisis sentimen
classifier = pipeline("sentiment-analysis")

# Menganalisis sentimen kalimat
result = classifier("This is a good movie.")

# Menampilkan hasil
print(result)

