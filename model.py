from typing import List, Tuple, Set
import logging

import faiss
import numpy as np
from numpy.typing import NDArray
import polars as pl
from rich.logging import RichHandler
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")


def to_transaction(date, account, description, amount):
    return f"Account: {account}\nDescription: {description}\nAmount: {amount}"
    # return description


class TransactionCategorizer:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    K = 10  # number of neighbors for k-NN

    def __init__(
        self,
        label_space: Set[str],
        transactions: NDArray[np.str_] = np.array([]),
        labels: NDArray[np.str_] = np.array([]),
    ):
        self.label_space = label_space
        self.transactions = transactions
        self.labels = labels

        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)

        self.index = faiss.IndexFlatIP(
            self.embedding_model.get_sentence_embedding_dimension())
        if len(self.transactions) > 0:
            self.embeddings = self.embedding_model.encode(
                self.transactions)
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
        else:
            self.embeddings = np.array([])

    def add(
        self,
        transaction: str,
        label: str,
    ) -> None:
        new_embedding = self.embedding_model.encode(
            [transaction]).reshape(1, -1)
        faiss.normalize_L2(new_embedding)
        self.index.add(new_embedding)

        self.embeddings = np.append(self.embeddings, new_embedding, axis=0)
        self.transactions = np.append(
            self.transactions, transaction)
        self.labels = np.append(self.labels, label)

    def predict(self, transaction: str) -> List[Tuple[str, float]]:
        """
        Predict the category of a transaction description.

        Args:
            transaction (str): The transaction string to categorize.

        Returns:
            A list of tuples containing predicted categories and their confidence
            scores. The highest match is guaranteed to be the first element, and the
            rest are sorted in descending order of confidence.
        """
        if len(self.embeddings) == 0:
            return [(label, 1 / len(self.label_space)) for label in self.label_space]

        emb = self.embedding_model.encode(
            [transaction]).reshape(1, -1)
        faiss.normalize_L2(emb)

        distances, indices = self.index.search(
            emb, min(self.K, len(self.embeddings)))

        nearest_idxs = indices[0]

        nearest_labels = [self.labels[i] for i in nearest_idxs]

        # Use softmax over similarities to create weights
        sims = distances[0]
        weights = np.exp(sims) / np.sum(np.exp(sims))

        # Aggregate weights per label
        label_scores = {}
        for label, weight in zip(nearest_labels, weights):
            label_scores[label] = label_scores.get(label, 0.0) + weight

        # Sort labels we saw by descending score
        ranked_seen = sorted(label_scores.items(),
                             key=lambda x: x[1], reverse=True)

        # Add missing labels with confidence 0.0
        missing = [(label, 0.0)
                   for label in self.label_space if label not in label_scores]

        return ranked_seen + missing


DATA_FILEPATH = "finance.xlsx"

INPUT_LABELS_SHEET = "categories"
INPUT_LABELS_CATEGORY_COLUMN = "Categories"

INPUT_DATA_SHEET = "all"
INPUT_DATA_DATE_COLUMN = "Posted Date"
INPUT_DATA_ACCOUNT_COLUMN = "Account"
INPUT_DATA_DESCRIPTION_COLUMN = "Description"
INPUT_DATA_AMOUNT_COLUMN = "Amount"
INPUT_DATA_CATEGORY_COLUMN = "Category"


def load_data():
    """Returns a tuple of (df, categories). all_df is a Dataframe with transaction and category columsns,
    where transaction is the string representation, using `to_transaction` of the transaction. Categories
    is a set of all possible categories."""
    log.info("Loading data...")
    categories_df = pl.read_excel(
        DATA_FILEPATH, sheet_name=INPUT_LABELS_SHEET).select([
            INPUT_LABELS_CATEGORY_COLUMN
        ]).drop_nulls()
    all_df = pl.read_excel(
        DATA_FILEPATH, sheet_name=INPUT_DATA_SHEET).select([
            INPUT_DATA_DATE_COLUMN,
            INPUT_DATA_ACCOUNT_COLUMN,
            INPUT_DATA_DESCRIPTION_COLUMN,
            INPUT_DATA_AMOUNT_COLUMN,
            INPUT_DATA_CATEGORY_COLUMN,
        ]).drop_nulls()

    dates = all_df[INPUT_DATA_DATE_COLUMN].to_numpy()
    accounts = all_df[INPUT_DATA_ACCOUNT_COLUMN].to_numpy()
    descriptions = all_df[INPUT_DATA_DESCRIPTION_COLUMN].to_numpy()
    amounts = all_df[INPUT_DATA_AMOUNT_COLUMN].to_numpy()

    to_transaction_vec = np.vectorize(to_transaction)
    transactions = to_transaction_vec(dates, accounts, descriptions, amounts)

    df = pl.DataFrame({
        "transaction": transactions,
        "category": all_df[INPUT_DATA_CATEGORY_COLUMN]
    })
    categories = set(
        categories_df[INPUT_LABELS_CATEGORY_COLUMN].unique())

    return df, categories


def load_model() -> TransactionCategorizer:
    df, categories = load_data()

    log.info("Loading model...")
    return TransactionCategorizer(
        categories,
        transactions=df["transaction"].to_numpy(),
        labels=df["category"].to_numpy()
    )


if __name__ == "__main__":
    df, categories = load_data()

    df = df.sample(fraction=1, shuffle=True)
    training_size = int(0.9 * len(df))
    training_df, testing_df = df.head(
        training_size), df.head(-training_size)

    training_transactions = training_df["transaction"].to_numpy()
    training_labels = training_df["category"].to_numpy()
    test_transactions = testing_df["transaction"].to_numpy()
    test_labels = testing_df["category"].to_numpy()

    categorizer = TransactionCategorizer(
        categories,
        transactions=training_transactions,
        labels=training_labels
    )

    correct = 0
    total = len(testing_df)

    for transaction, true_label in zip(test_transactions, test_labels):
        predictions = categorizer.predict(transaction)
        predicted_label = predictions[0][0]

        if predicted_label == true_label:
            correct += 1
        else:
            print(
                f"{transaction}, {true_label}, {list(predictions)}")

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
