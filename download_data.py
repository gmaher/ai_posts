from sklearn.datasets import fetch_20newsgroups

# Download the full 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', download_if_missing=True)

# newsgroups is a Bunch object, which is similar to a dictionary.
# Access the data and target like this:
documents = newsgroups.data  # List of document texts
labels = newsgroups.target   # List of category labels for each document

# Optionally, print some information about the dataset
print(f"Number of documents: {len(documents)}")
print(f"Number of categories: {len(newsgroups.target_names)}")
print(f"Categories: {newsgroups.target_names}")
print("Sample document:")
print(documents[0])