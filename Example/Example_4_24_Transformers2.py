
# Example 4.24 - Question Answering

from transformers import pipeline

# Membuat sistem Question Answering
question_answerer = pipeline("question-answering")

# Menentukan pertanyaan dan teks sumber
result = question_answerer({
    "question": "What is the name of the company?",
    "context": "We created Biox Systems Ltd company back in the year of 2000."
})

# Menampilkan jawaban
print(result)

