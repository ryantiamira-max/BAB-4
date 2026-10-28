# Exercise 4.19 - Example 4.23 dengan beberapa kalimat positif dan negatif
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
sentences = [
    'This is a good movie.',
    'I love this phone, the battery lasts all day.',
    'The service was excellent and the staff were friendly.',
    'This is a terrible movie.',
    'I hate waiting in long queues.',
    'The food was cold and the waiter was rude.',
]
for s in sentences:
    print(s, '->', classifier(s))
