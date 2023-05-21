#CS6910 Assignment 3: Transliteration Task
This assignment focuses on the transliteration task, specifically translating from English to various Indic languages. The implementation provided here is for translating from English to Tamil.

Files
The code development was carried out using Google Colab notebooks. The training was performed using Python files, which are available on GitHub. The following files are included:

cs6910_Assignment3_PartA_V5.ipynb: Implementation of Part A (Vanilla RNN)
cs6910_Assignment3_PartB_V2.ipynb: Implementation of Part B (Attention mechanism)
.py files: Used for training on GPUs and running WandB sweeps, containing the respective implementations

Part A: Neural Transliteration Task using Vanilla RNN
In this part of the project, a Vanilla RNN has been implemented. The program consists of the following sections:

File download: Codes for downloading the required files.
Data processing: Extraction, tokenization, and padding of the sequences to make them of equal length for the dataloaders to work on.
Encoder-Decoder architecture: Implementation of the encoder, decoder, and Seq2Seq model to feed the inputs and receive the output.
Evaluation: The correct_sequences_count function is used to calculate the number of correctly predicted sequences in a batch, and the Accuracy function predicts the accuracy during a specific epoch.
Training loop: The hyperparameters are passed to the instantiated models, and the model is trained for a specific number of epochs.

Part B: Transfer Learning using a Pre-Trained Model
In this part of the project, transliteration is performed using sequence models with attention mechanism. The program follows a similar structure to Part A:

File download: Codes for downloading the required files.
Data processing: Extraction, tokenization, and padding of the sequences to make them of equal length for the dataloaders.
Encoder-Decoder architecture with attention: Implementation of the encoder, decoder, and Seq2Seq model with attention mechanism.
Evaluation: The correct_sequences_count function calculates the number of correctly predicted sequences in a batch, and the Accuracy function predicts the accuracy during a specific epoch.
Training loop: The hyperparameters are passed to the instantiated models, and the model is trained for a specific number of epochs.
Evaluation Metrics
The performance of the models was evaluated using word-level accuracy as the metric. The accuracy was calculated as the ratio of correctly predicted Tamil sequences to the total number of sequences.

Conclusion
In this assignment, we explored the transliteration task and implemented both vanilla RNN and transfer learning approaches for translating from Tamil to English. The models were trained and evaluated based on word-level accuracy. The provided code and notebooks can be used to reproduce the experiments and further explore the task.
