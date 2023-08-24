CS6910 Assignment 3: Transliteration Task

This assignment focuses on the transliteration task, specifically translating from English to various Indic languages. The implementation provided here is for translating from English to Tamil.

Code development was carried out using Google Colab notebooks. The training was performed using Python files, which are available in the repository. The following files are included:

cs6910_Assignment3_PartA_V5.ipynb: Implementation of Part A (Vanilla RNN)

cs6910_Assignment3_PartB_V2.ipynb: Implementation of Part B (Attention mechanism)

Note:
.py files were used for training on GPUs and running WandB sweeps

_**Part A: Neural Machine Transliteration Task using Vanilla RNN**_

In this part of the project, a Vanilla RNN has been implemented. The program consists of the following sections:


File download: Codes for downloading the required files.

Data pre-processing: Extraction, tokenization, and padding of the sequences to make them of equal length for the dataloaders to work on.

Encoder-Decoder architecture: Implementation of the encoder, decoder, and Seq2Seq model to feed the inputs and receive the output.

Evaluation: The correct_sequences_count function is used to calculate the number of correctly predicted sequences in a batch, and the Accuracy function predicts the accuracy during a specific epoch.

Training loop: The hyperparameters are passed to the instantiated models, and the model is trained for a specific number of epochs.

_**Part B: Neural Machine Transliteration Task with Attention Mechanism**_

In this part of the project, transliteration is performed using sequence models with attention mechanism. The program follows a similar structure to Part A:

File download: Codes for downloading the required files.

Data pre-processing: Extraction, tokenization, and padding of the sequences to make them of equal length for the dataloaders.

Encoder-Decoder architecture with attention: Implementation of the encoder, decoder, and Seq2Seq model with attention mechanism.

Evaluation: The correct_sequences_count function calculates the number of correctly predicted sequences in a batch, and the Accuracy function predicts the accuracy during a specific epoch.

Training loop: The hyperparameters are passed to the instantiated models, and the model is trained for a specific number of epochs.


_**Evaluation Metrics:**_

The performance of the models was evaluated using word-level accuracy as the metric. Word-level accuracy was calculated as the ratio of correctly predicted Tamil sequences to total sequences.


_**Conclusion:**_

In this assignment, we explored the neural machine transliteration task and implemented both sequence-to-sequence models and sequence-to-sequence models with an attention mechanism for translating from English to Tamil language. The models were trained and evaluated based on word-level accuracy. The provided code and notebooks can be used to reproduce the experiments and further explore the task.
