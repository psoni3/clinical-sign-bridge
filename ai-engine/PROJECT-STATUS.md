## Classification Baseline V1

**Dataset:** INCLUDE-50  
**Training samples:** 662  
**Validation samples:** 77  
**Classes:** 50  
**Input:** 21 pooled statistical features extracted from Holistic landmarks  
**Model:** MLP (64 → 32 → 50)  
**Optimizer:** Adam, learning rate 0.001  
**Best epoch:** 86

### Validation metrics

| Metric | Result |
|---|---:|
| Macro F1 | 0.7040 |
| Weighted F1 | 0.6762 |
| Top-1 accuracy | 70.13% |
| Top-5 accuracy | 97.40% |
| Macro recall | 74.00% |

### Limitations

- Temporal order and movement information discarded.
- Individual landmark relationships discarded through pooling.
- Validation set contains only 77 samples.
- Signer-independent generalization has not been established.
- Test set remains untouched.

### Decision

Freeze Baseline V1. Do not tune further.

Next experiment: BiLSTM using the original temporal landmark sequences `(T, 75, 3)` and detection masks.
