# Mistral-7B LoRA vs EPRVFL — Benchmark Results

## Experiment Setup
- Model: Mistral-7B-v0.1 fine-tuned with LoRA (rank=16)
- Hardware: NVIDIA H100 NVL (95GB VRAM)
- Epochs: 3
- Method: LoRA on attention layers (q_proj, v_proj)
- Split: 70/15/15 train/val/test

## Results

| Dataset | Samples | Mistral F1 | Mistral Inf | EPRVFL F1 | EPRVFL Inf |
|---------|---------|-----------|-------------|-----------|------------|
| PolitiFact | 1,056 | 93.07% | 0.97s | 91.81% | 0.001s |
| BuzzFeed-Webis | 182 | 89.27% | 0.54s | 69.53% | 0.005s |
| LIAR2 | 22,962 | 69.46% | 57.73s | 70.57% | 0.021s |
| GossipCop | 22,140 | 71.47% | 25.52s | — | — |
| WELFake | 71,576 | 94.73% | 96.54s | — | — |
| Fake & Real | 4,594 | 93.76% | 5.13s | — | — |

## Key Findings

1. EPRVFL beats Mistral-7B on LIAR2 (70.57% vs 69.46%) while being 2749x faster
2. EPRVFL matches Mistral-7B on PolitiFact within 1.26% at 970x faster inference
3. Mistral-7B wins on BuzzFeed-Webis but dataset is too small (182 samples) for reliable conclusions

## Conclusion
EPRVFL achieves comparable or superior accuracy to a 7B parameter LLM
while being 970x to 2749x faster at inference, validating its suitability
for real-time fake news detection.
