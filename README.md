# Mixture of Experts from scratch

This code implements a **Mixture of Experts (MoE)** model in PyTorch, which is a neural network architecture that uses multiple specialised "expert" networks and a routing mechanism to decide which experts should process each input.

## To Do

- Weight calculations for inference v full model

- Introduce capacity limits per expert

- Add noise (similar to epsilon-greedy in RL, encourages exploration)

- GPT with MoE instead of the normal FFN

## Components

**Expert**: A simple feedforward network that transforms input through a hidden layer with GELU activation.

**Router**: The key component that decides which experts to use for each input.

**MoE**: The main model that combines routing and experts, plus an output layer for classification.

## Router Explanation

The Router implements a **top-k gating mechanism**:

1. **Gating**: `gate` is a linear layer that produces logits for each expert
2. **Scoring**: Softmax converts logits to probabilities (scores) - these represent how much each expert should contribute
3. **Selection**: `topk` selects the k highest-scoring experts for each input
4. **Dispatch mask**: A binary mask indicating which experts are active for each input

The key insight is that instead of using all experts for every input, we only use the top-k most relevant experts, making the model more efficient and specialised.

## Auxiliary Loss

The auxiliary loss is designed to encourage **load balancing** among experts. Here's the mathematical formulation:

$$
\mathcal{L}_{aux} = \frac{E}{B^2} \sum_{i=1}^{E} f_i \cdot P_i
$$


Where:
- $E$ = number of experts
- $B$ = batch size  
- $f_i$ = fraction of tokens routed to expert $i$ (load)
- $P_i$ = sum of routing probabilities for expert $i$ (importance)


## Why This Loss Matters

Without load balancing, the model might:
- Overuse some experts while ignoring others
- Create training instability
- Reduce the benefits of having multiple experts

The auxiliary loss penalises scenarios where:
1. An expert gets high routing probabilities (high importance) AND
2. Gets assigned many tokens (high load)

This encourages the router to distribute work more evenly across experts while still allowing specialisation. The loss is typically added to the main task loss with a small coefficient (e.g., I've done 0.01 by default) during training.

The $\frac{E}{B^2}$ normalisation ensures the loss scale is reasonable regardless of batch size or number of experts.