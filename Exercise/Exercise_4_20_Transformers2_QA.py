# Exercise 4.20 - Example 4.24 dengan teks dan pertanyaan berbeda
from transformers import pipeline
question_answerer = pipeline('question-answering')
context = 'Perry Xiao wrote a book about artificial intelligence programming with Python. The book was published in 2022 by Wiley.'
questions = [
    'Who wrote the book?',
    'What is the book about?',
    'When was the book published?',
    'Who published the book?',
]
for q in questions:
    print(q, '->', question_answerer({'question': q, 'context': context}))
