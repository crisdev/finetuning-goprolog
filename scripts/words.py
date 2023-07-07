"""
This script generates a bar chart of the top 20 most common words in a dataset.

You may change the filepath variable if you want to use a different dataset.

To run it, type in the terminal: python words.py
"""


import jsonlines
from collections import Counter
import re
import matplotlib.pyplot as plt


# Define the path to the json-lines file
filepath = '../datasets/prolog/test.json'

# Create an empty list to store all the "nl" values
nls = []

# Read the json-lines file and append all the "nl" values to the list
with jsonlines.open(filepath) as reader:
    for obj in reader:
        nls.append(obj['nl'])

# Join all the "nl" values into a single string
all_text = " ".join(nls)

# Remove all special characters from the text
all_text = re.sub(r'[^\w\s]', '', all_text)

# Convert to lowercase
all_text = all_text.lower()

# Split the string into words and count the frequency of each word
word_counts = Counter(all_text.split())

# Get the top 20 most common words
top_20_words = dict(word_counts.most_common(20))

# Print the top 20 most common words
print("20 palabras más usadas: ")
for word, count in top_20_words.items():
    print(f"{word}: {count}")

# Plot a bar chart of the top 20 most common words
plt.bar(top_20_words.keys(), top_20_words.values())
plt.xticks(rotation=75)
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.title("20 palabras más usadas")
plt.show()
