
from transformers import pipeline

question_answerer = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)

context = (
    "Perry Xiao wrote a book about artificial intelligence "
    "programming with Python. The book was published in 2022 by Wiley."
)

questions = [
    "Who wrote the book?",
    "What is the book about?",
    "When was the book published?",
    "Who published the book?",
]

for q in questions:
    result = question_answerer(question=q, context=context)
    print(q, "->", result["answer"])
