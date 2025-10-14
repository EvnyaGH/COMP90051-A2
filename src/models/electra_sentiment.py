#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELECTRA Fine-tuning Implementation for Sentiment Classification

This module implements ELECTRA fine-tuning for sentiment classification that works
with our cross-validation framework.

Features:
- Uses pre-trained ELECTRA model
- Fine-tunes on sentiment classification task
- Compatible with our CV framework
- Handles text tokenization and preprocessing
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from typing import Dict, Any, List, Optional
import random


class ElectraSentimentDataset(Dataset):
    """Dataset class for ELECTRA sentiment classification."""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text samples
            labels: List of binary labels (0 or 1)
            tokenizer: ELECTRA tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class ElectraSentiment:
    """
    ELECTRA-based sentiment classifier.

    This class provides a consistent interface for our cross-validation framework
    while using ELECTRA for fine-tuning.
    """

    def __init__(self, **params):
        """
        Initialize ELECTRA sentiment classifier.

        Args:
            **params: Hyperparameters including:
                     - learning_rate: Learning rate for fine-tuning
                     - batch_size: Batch size for training
                     - max_length: Maximum sequence length
                     - epochs: Number of training epochs
        """
        self.params = params
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set default parameters
        self.learning_rate = params.get("learning_rate", 2e-5)
        self.batch_size = params.get("batch_size", 16)
        self.max_length = params.get("max_length", 256)
        self.epochs = params.get("epochs", 3)

    def fit(self, texts, labels):
        """
        Fine-tune ELECTRA model on sentiment classification.

        Args:
            texts: List of text samples
            labels: List of binary labels (0 or 1)
        """
        # Convert to lists if needed
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        # Load pre-trained model and tokenizer
        model_name = "google/electra-small-discriminator"
        self.tokenizer = ElectraTokenizerFast.from_pretrained(model_name)
        self.model = ElectraForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

        # Move model to device
        self.model.to(self.device)

        # Create dataset
        dataset = ElectraSentimentDataset(
            texts, labels, self.tokenizer, self.max_length
        )

        # Create data loader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Set up training
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def predict(self, texts):
        """
        Make predictions on new text data.

        Args:
            texts: List of text samples

        Returns:
            predictions: Binary predictions (0 or 1)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to list if needed
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for text in texts:
                # Tokenize text
                encoding = self.tokenizer(
                    str(text),
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Move to device
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                # Get prediction
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get predicted class
                pred = torch.argmax(outputs.logits, dim=-1).item()
                predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, texts):
        """
        Get prediction probabilities.

        Args:
            texts: List of text samples

        Returns:
            probabilities: Prediction probabilities for each class
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to list if needed
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for text in texts:
                # Tokenize text
                encoding = self.tokenizer(
                    str(text),
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Move to device
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                # Get prediction
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get probabilities
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                probabilities.append(probs[0])

        return np.array(probabilities)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        self.params.update(params)
        # Update instance variables
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.batch_size = params.get("batch_size", self.batch_size)
        self.max_length = params.get("max_length", self.max_length)
        self.epochs = params.get("epochs", self.epochs)
        return self


def create_electra_factory(hyperparams: Dict[str, Any]):
    """
    Create a factory function for ELECTRA that works with our CV framework.

    Args:
        hyperparams: Dictionary of hyperparameters to test

    Returns:
        factory_function: Function that creates ElectraSentiment instances
    """

    def factory(params):
        return ElectraSentiment(**params)

    return factory


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    print("Testing ELECTRA Sentiment Classification...")

    # Sample data
    texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "I hate this film. It's terrible and boring.",
        "Amazing performance by all the actors. Highly recommended!",
        "The plot was confusing and the acting was poor.",
        "One of the best movies I've ever seen. Perfect!",
        "Waste of time. Don't watch this movie.",
        "Great cinematography and excellent storytelling.",
        "Boring and predictable. Not worth watching.",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]

    # Test different hyperparameters
    params_list = [
        {"learning_rate": 1e-5, "batch_size": 8, "max_length": 128, "epochs": 2},
        {"learning_rate": 2e-5, "batch_size": 16, "max_length": 256, "epochs": 2},
        {"learning_rate": 5e-5, "batch_size": 32, "max_length": 512, "epochs": 2},
    ]

    for i, params in enumerate(params_list):
        print(f"\nTesting params {i+1}: {params}")

        try:
            # Create and train model
            model = ElectraSentiment(**params)
            model.fit(texts, labels)

            # Test predictions
            test_texts = [
                "This is an amazing movie!",
                "I don't like this film at all.",
                "Outstanding performance by the cast.",
            ]

            predictions = model.predict(test_texts)
            probabilities = model.predict_proba(test_texts)

            print(f"Test texts: {test_texts}")
            print(f"Predictions: {predictions}")
            print(f"Probabilities shape: {probabilities.shape}")

        except Exception as e:
            print(f"Error with params {params}: {e}")

    print("\nELECTRA implementation completed!")
