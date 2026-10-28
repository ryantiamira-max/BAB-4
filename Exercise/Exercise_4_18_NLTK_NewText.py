# Exercise 4.18 - Example 4.22 dengan teks berbeda
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = """
Barack Obama was born in Hawaii. He worked for Google in California
and later visited the United Nations headquarters in New York.
"""
tokens = nltk.word_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)
