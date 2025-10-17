    It's present in https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
    
    Loss is computed per sample. scale each sample's loss by weight (larger weight for rare classes)

    In our case, each BXT is a sample (per token)
    Each token predicts a next token. Let's assume the grouth truth is yb,t. CE is -log of the probability your token gave for this grouth truth 
    each of those is multiplied by ce_weight which is a V vector 
    Total is divided by 

# Cross-Entropy for LLMs, Class Weights, and Reductions 

## Setup & Notation

* Logits: `z` shape `(B, T, V)`
* Targets (class indices): `y` shape `(B, T)`
* Valid-token mask from `ignore_index`:
  `m[b,t] = 1 if y[b,t] != ignore_index else 0`
* Softmax probs:
  `p[b,t,c] = exp(z[b,t,c]) / sum_j exp(z[b,t,j])`

Per-token cross-entropy for a valid token `(b,t)` with true class `y[b,t]`:

```
CE[b,t] = -log( p[b,t, y[b,t]] )
```

---

## Default Reduction (token average over B × T*)

PyTorch `CrossEntropyLoss(reduction="mean")` averages over **all valid tokens**:

```
L = ( sum_{b,t} m[b,t] * ( -log p[b,t, y[b,t]] ) )
    / ( sum_{b,t} m[b,t] )
```

The m[b,t] is just to make sure we are only caring about tokens that are not padding. 


This is a mean across **tokens** (over both batch B and time T*), not just across sequences.

---

## Class Weights (`weight`)

Let `w[c] >= 0` be the class weight for class `c`. PyTorch multiplies each valid token’s loss by `w[y[b,t]]` **and** uses a weighted denominator:

```
L = ( sum_{b,t} m[b,t] * w[y[b,t]] * ( -log p[b,t, y[b,t]] ) )
    / ( sum_{b,t} m[b,t] * w[y[b,t]] )
```

Log for each sample is multiplied by how rare the class is. Then the sum is divided by sum of weights.

Notes:

* Setting `w[c] = 0` effectively ignores class `c`.
* Useful for class/token imbalance or emphasizing certain tokens.


