# Extending Neural Network Models with Adaptive Activation Functions

Implementation of the experiment as published in the paper "Extending Neural
Network Models with Adaptive Activation Functions" by Serhii Kostiuk.

## Goals of the experiment

The experiment:

- demonstrates the method of extending pre-trained artificial neural network
  models with adaptive activation functions on the KerasNet [^1] example;
- demonstrates the activation function fine-tuning as a method to improve the
  performance and compensate for the loss caused by imperfect replacements;
- evaluates the performance of the base network, the patched network, the
  fine-tuned network, and the derived network trained from scratch;
- demonstrates the effectiveness of activation function fine-tuning when all
  other elements of the model are fixed (frozen);
- evaluates performance of the KerasNet variants with different activation
  functions (adaptive and non-adaptive) trained in different regimes;
- evaluates approximation errors in connection to the floating point accuracy;
- demonstrates the correctness of the method by using different seed values.

## Description of the experiment

The experiment consists of the following steps:

1. Train the base KerasNet network on the CIFAR-10 [^2] dataset for 100 epochs
   using the standard training procedure and RMSprop. 4 variants of the network:
   are trained: with ReLU [^3], SiLU [^4], Tanh and Sigmoid [^5] activation
   functions. Save the pre-trained network. Evaluate performance of the base
   pre-trained network on the test set of CIFAR-10.
2. Load the base pre-trained network and replace all activation functions with
   the corresponding adaptive alternatives (ReLU, SiLU -> AHAF [^6]; Sigmoid,
   Tanh -> F-Neuron Activation [^7]). Evaluate performance of the base derived
   network on the test set of CIFAR-10.
3. Fine-tune the adaptive activation functions on the CIFAR-10 dataset.
   Evaluate the network performance after the activation function fine-tuning.
4. Train the reference networks with adaptive activations for 100 epochs.
   Evaluate the performance of such networks trained from scratch.
5. Repeat the experiments with different seed values (affects the starting
   synaptic weights in the CNN and FFN layers).
6. Compare the evaluation results collected on steps 1-5.

## Running experiments

1. NVIDIA GPU recommended with at least 2 GiB of VRAM.
2. Install the requirements from `requirements.txt`.
3. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment variables.
4. Use the root of this repository as the current directory.
5. Add the current directory to `PYTHONPATH` so it can find the modules

This repository contains a wrapper script that sets all the required
environment variables: [run_experiment.sh](./run_experiment.sh). Use the bash shell to
execute the experiment using the wrapper script:

Example:

```shell
user@host:~/repo_path$ ./run_experiment.sh experiments/train_new_base.py
```

## Reproducing the results from the paper

1. Training the base KerasNet networks:

   ```shell
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             base --net KerasNet --ds CIFAR-10 --acts all \
             --opt rmsprop --seed 42 --bs 64 --dev gpu \
             --start_ep 0 --end_ep 100
   ```
   
2. Patching the KerasNet networks with new adaptive activation functions:

   ```shell
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             ahaf --net KerasNet --ds CIFAR-10 --acts all_lus \
             --opt rmsprop --seed 42 --bs 64 --dev gpu \
             --start_ep 100 --end_ep 100 --patch_base
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             fuzzy_ffn --net KerasNet --ds CIFAR-10 --acts all_bfs \
             --opt rmsprop --seed 42 --bs 64 --dev gpu \
             --start_ep 100 --end_ep 100 --patch_base
   ```

3. Fine-tuning the patched KerasNet networks with adaptive AFs:

   ```shell
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             ahaf --net KerasNet --ds CIFAR-10 --acts all_lus \
             --opt rmsprop --seed 42 --bs 64 --dev gpu \
             --start_ep 100 --end_ep 150 --patched --tune_aaf
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             fuzzy_ffn --net KerasNet --ds CIFAR-10 --acts all_bfs \
             --opt rmsprop --seed 42 --bs 64 --dev gpu \
             --start_ep 100 --end_ep 150 --patched --tune_aaf
   ```

4. Training the reference KerasNet networks with adaptive AFs from scratch:

   ```shell
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             ahaf --net KerasNet --ds CIFAR-10 --acts all_lus \
             --opt rmsprop --seed 42 --bs 64 --dev gpu \
             --start_ep 0 --end_ep 100
   user@host:~/repo_path$ ./run_experiment.sh experiments/train_individual.py \
             fuzzy_ffn --net KerasNet --ds CIFAR-10 --acts all_bfs \
             --opt rmsprop --seed 42 --bs 64 --dev gpu \
             --start_ep 0 --end_ep 100
   ```

5. Evaluating approximation errors:

   ```shell
   user@host:~/repo_path$ ./run_experiment.sh \
             post_experiment/show_af_diff.py
   ```

6. On the effect of synaptic weights initialization. Repeat the experiments
   1-4 once per each of the seed values:

   ```shell
   user@host:~/repo_path$ ./run_experiment_multiseed.sh \
             experiments/train_individual.py \
             # ... all arguments without the seed values
   ```

   Seed values to evaluate: 100, 128, 1999, 7823, 42.

## Visualization of experiment results

Use tools from the [post_experiment](./post_experiment) directory to visualize
training process, create the training result summary tables and visualize the
activation function form for AHAF compared to the corresponding base
activations.

## References

[^1]: Chollet, F., et al. (2015) Train a simple deep CNN on the CIFAR10 small
      images dataset. https://github.com/keras-team/keras/blob/1.2.2/examples/cifar10_cnn.py

[^2]: Krizhevsky, A. (2009) Learning Multiple Layers of Features from Tiny
      Images. Technical Report TR-2009, University of Toronto, Toronto.

[^3]: Agarap, A. F. (2018). Deep Learning using Rectified Linear Units (ReLU).
      https://doi.org/10.48550/ARXIV.1803.08375

[^4]: Elfwing, S., Uchibe, E., & Doya, K. (2017). Sigmoid-Weighted Linear Units
      for Neural Network Function Approximation in Reinforcement Learning.
      CoRR, abs/1702.03118. Retrieved from http://arxiv.org/abs/1702.03118

[^5]: Cybenko, G. Approximation by superpositions of a sigmoidal function. Math.
      Control Signal Systems 2, 303–314 (1989). https://doi.org/10.1007/BF02551274

[^6]: Bodyanskiy, Y., & Kostiuk, S. (2022). Adaptive hybrid activation function
      for deep neural networks. In System research and information technologies
      (Issue 1, pp. 87–96). Kyiv Politechnic Institute.
      https://doi.org/10.20535/srit.2308-8893.2022.1.07 

[^7]: Bodyanskiy, Y., & Kostiuk, S. (2022). Deep neural network based on
      F-neurons and its learning. Research Square Platform LLC.
      https://doi.org/10.21203/rs.3.rs-2032768/v1 
