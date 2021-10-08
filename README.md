# Learning methods for speech-based TBI detection with limited data

This repository contains codes used to conduct experiments in Learning from Limited Data for Speech-basedTraumatic Brain Injury (TBI) Detection (ICMLA 2021).

We utilized external datasets to aid the learning of TBI detection task (Coelho's corpus). The external dataset are GoogleAudio set, Librispeech corpus and WOZ-DAIC dataset. We also consider two audio-based DNN models, which are SincNet and WAV2VEC. The three learning methods are experimented as follows:

## Transfer learning
The parameters in each backbone network were pre-trained on the source task and used as the initial weights of the TBI assessment model. Only parameters in the GRU were optimized during the domain adaptation step. After the training loss stopped decreasing after 10 epochs, a scheduled learning rate that was increased from $1\times 10^{-5}$ to $1\times 10^{-3}$ in the first ten epochs was initially used for fine-tuning. Thereafter, a decay of $1\times 10^{-6}$ was utilized until convergence.

## Multi-task learning
The first 3 layers of Wav2Vec, and first 6 layers of SincNet were considered shared layers between TBI detection and other source tasks, which were jointly trained. For subsequent task-specific layers, FC layers were applied with activation functions specific to the source task as follows. GA has two FC layers with softmax activation function. As in the original paper describing the SincNet model, Libri has two FCs with a softmax activation function. Lastly, the WOZ dataset is connected to the same classifier with the Sigmoid activation function as in the TBI classifier. The gradient update in each task was balanced using GradNorm, which normalizes the gradients and weights the loss in each training batch.

## Meta-learning: We adopted the MAML++ implementation, which improved on the training stability of the original MAML. Similar to MTL, all combinations of the source datasets were used to train MAML and the target task was used in the meta-testing step. In meta-testing, only training data of TBI speech were used to optimize the model's parameters.

## Dependency
* Librosa=0.7.2
* pytorch=1.3.1
* scikit-learn
