from sklearn.datasets import fetch_20newsgroups

train = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes"),
    download_if_missing=True
)

test = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes"),
    download_if_missing=True
)

print("Dataset downloaded successfully!")