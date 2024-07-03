# Dutch or Italian

This application determines if a given input of words are Dutch or Italian.
The input is read from .txt files where each line contains 10, 20, or 50 words from Dutch or Italian.

# Running application

## Training

The application will train two different models using the `examples_10000_10DEC.txt` file.

Usage: `python train`

After training each model (decision tree and decision stumps), the user will be prompted to save the model if desired.

## Predicting

The application will use a trained, saved model to predict the language of the input file.
Models are saved to .txt files during training, and recreated by reading them in.
The files (models) to be used must be named 'best_model_tree.txt', 'best_model_stumps.txt', or 'best_model_overall.txt'.
Pretrained models are provided.

Usage: `python predict {model} {input file}`

- `model`: 'tree', 'stumps', or 'best'
  - These are the models that were saved after training.
- `input file`: text file containing words.
  - The provided file 'test_examples_10DEC.txt' contains word lists not present in the training data.