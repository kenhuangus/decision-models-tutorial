# Decision Models in Production: Architecture, Empirical Use Cases, System Integration & Fine-Tuning

> **Instructor:** **Ken Huang, CISSP**  
> *Adjunct Professor, University of San Francisco (USF)*  
> *CEO, DistributedApps.ai · Project Lead, OWASP Agentic AI Vulnerability Scoring System (AIVSS)*  
> *Fellow & Co-Chair, Cloud Security Alliance (CSA) AI Safety Working Groups*

---

## 1. Overview & Architectural Principles

Modern AI Agent systems frequently struggle when relying exclusively on generative autoregressive Large Language Models (LLMs) for deterministic routing, tool selection, security gates, and policy enforcement. Generative models:
- Suffer from non-zero hallucination rates and output format parsing drift (1.5%–5.0% JSON syntax failures).
- Incur high latency overhead (1,200ms–3,500ms) by sequentially decoding 40–120 tokens step-by-step.
- Produce uncalibrated token logits where the softmax probability of a single token does not represent true statistical calibration.

**Decision Models** (such as Perplexity's `pplx-decider-v1.1-27b` and `pplx-decider-flash-v1.2`) solve this fundamental architectural challenge by:
1. **Lifting the causal attention mask:** Allowing full bi-directional cross-attention across the entire input state without token-position decay.
2. **Direct readout projection:** Projecting the final hidden state through a specialized readout matrix ($W_{\text{readout}} \in \mathbb{R}^{255 \times 5120}$) onto exact choice tokens or booleans (`noul`).
3. **Consuming strictly 0 output tokens:** Eliminating autoregressive decoding loops entirely and executing in a single forward pass.
4. **Providing mathematically calibrated probabilities:** Guaranteeing that probability scores map directly to empirical expectation values.

---

## 2. Interactive Presentation Deck

The tutorial slide deck is built directly on the **MAESTRO** slide presentation framework:
- **Presentation:** [index.html](index.html)
- **Features:** Keyboard navigation (`ArrowRight`, `ArrowLeft`, `PageUp`, `PageDown`, `Home`, `End`), full-screen toggle (`F`), direct slide deep linking (`#1` to `#26`), dynamic progress bar, and touch swipe gestures.

### Slide Structure (26 Slides):
1. **Title Slide:** Decision Models: The Deterministic Engine for AI Agents
2. **About the Instructor:** Ken Huang, USF, OWASP, CSA Credentials
3. **The Core Problem:** Why Generative LLMs Fail at Decision-Making
4. **Architectural Taxonomy:** Generative vs. Decision Architectures
5. **Deep Dive:** Anatomy of the `pplx-decider` Protocol & Readout Head
6. **Empirical Hardware Baseline:** Apple Silicon MPS Deployment on Mac Mini
7. **Use Case 1:** AI Agent Tool Selection & Execution Routing (Live Apple Silicon MPS Result)
8. **Use Case 2:** Customer Support Escalation Gate (Live Apple Silicon MPS Result)
9. **Use Case 3:** RAG Retrieval Chunk Re-ranking & Filtering (Live Apple Silicon MPS Result)
10. **Use Case 4:** Hallucination Verification & Factuality NLI (Live Apple Silicon MPS Result)
11. **Use Case 5:** Prompt Injection & Adversarial Attack Gate (Live Apple Silicon MPS Result)
12. **Use Case 6:** E-Commerce Intent & Conversion Readiness (Live Apple Silicon MPS Result)
13. **Use Case 7:** Automated Code Review & Deployment Risk Gate (Live Apple Silicon MPS Result)
14. **Use Case 8:** Multi-Class Content Moderation & Compliance (Live Apple Silicon MPS Result)
15. **Use Case 9:** Agentic Planning & Tree-of-Thoughts Branch Selection (Live Apple Silicon MPS Result)
16. **Use Case 10:** Sensitive Data & Secret Leakage Prevention (Live Apple Silicon MPS Result)
17. **System Topology:** The Sandwich Guardrail Architecture Pattern
18. **Multi-Agent Orchestration:** LangGraph & CrewAI Conditional Edge Router Code
19. **Quantitative Performance:** Benchmark Comparison (Decision vs Generative LLM)
20. **Fine-Tuning Architecture:** Parameter Isolation & Readout Adaptation Strategy
21. **Dataset Engineering:** Supervised State-Question-Target Schema & Compound Brier Loss
22. **Pipeline Code:** End-to-End PyTorch Fine-Tuning Script (`finetune_decider.py`)
23. **Empirical Mac Mini Run:** Training Trajectory & Validation Probability Shifts
24. **Diagnostic Engineering Log:** 4 Production Failure Modes Discovered & Resolved
25. **Course Synthesis:** The 4 Golden Rules of Decision Models
26. **Connect & Academic Resources:** References, Course Links, and Professional Contact

---

## 3. Empirical Test Results from Apple Silicon Metal Node

All 10 production use cases were deployed and verified live on an Apple Silicon node (ARM64, Darwin 27.0.0, 64 GB Unified RAM) running `perplexity-ai/pplx-decider-flash-v1.2` over Metal Performance Shaders (`mps`). Output tokens consumed across all 10 use cases was strictly **0 tokens**.

| ID | Use Case Pattern | Schema | Decision Output | Empirical Probability | Latency (Metal MPS) |
|---|---|---|---|---|---|
| **01** | Tool Selection & Routing | Choice | `chart_generator` | **75.65%** | 1,279.8 ms |
| **02** | Support Escalation Gate | Noul (Boolean) | Escalation Required | **97.82%** | 912.0 ms |
| **03** | RAG Chunk Re-ranking | Choice | `highly_relevant` | **93.35%** | 950.8 ms |
| **04** | Hallucination NLI Gate | Choice | `contradicted` | **95.28%** | 618.5 ms |
| **05** | Prompt Injection Gate | Noul (Boolean) | Adversarial Attack | **99.94%** | 473.6 ms |
| **06** | E-Commerce Intent | Choice | `ready_to_buy` | **99.83%** | 938.4 ms |
| **07** | Code Review Risk Gate | Choice | `critical_risk` | **98.87%** | 862.9 ms |
| **08** | Content Moderation | Choice | `ip_violation` | **97.67%** | 635.8 ms |
| **09** | Planning Branch Selection | Choice | `branch_b` | **99.72%** | 1,100.3 ms |
| **10** | Secret Leakage Prevention | Noul (Boolean) | Secret Leaked | **99.53%** | 571.8 ms |

*Full machine-readable test records:* [decision_test_results.json](decision_test_results.json)

---

## 4. System Topology: The Sandwich Guardrail Architecture

In enterprise AI Agent applications, Decision Models and Generative LLMs operate in complementary, decoupled tiers:

```
[ Raw User Request ]
         │
         ▼
┌──────────────────────────────────────────────┐
│  1. INGRESS GUARDRAIL (Decision Model)       │  <-- 0 Tokens, 470ms
│  - Prompt Injection & Jailbreak Gate         │
│  - Policy Compliance & Toxicity Filter       │
└──────────────────────┬───────────────────────┘
                       │ Validated Clean State
                       ▼
┌──────────────────────────────────────────────┐
│  2. REASONING & ROUTING (Decision Model)     │  <-- 0 Tokens, 850ms
│  - Tool Selection (Strict Enum Validity)     │
│  - Multi-Agent Orchestration Routing         │
└──────────────────────┬───────────────────────┘
                       │ Targeted Subagent Context
                       ▼
┌──────────────────────────────────────────────┐
│  3. GENERATIVE CORE (Claude / GPT / Qwen)    │  <-- Full Context Generation
│  - Creative Synthesis & Code Implementation  │
│  - Technical Documentation & Dialogue        │
└──────────────────────┬───────────────────────┘
                       │ Raw Generated Output
                       ▼
┌──────────────────────────────────────────────┐
│  4. EGRESS VERIFIER (Decision Model)         │  <-- 0 Tokens, 570ms
│  - Hallucination / Entailment NLI Gate       │
│  - Credential & Secret Leakage Inspection    │
└──────────────────────┬───────────────────────┘
                       │ Certified Secure Output
                       ▼
           [ Safe Response to Client ]
```

---

## 5. Decision Model Fine-Tuning Pipeline

### Architectural Formulation: Parameter Isolation
Unlike generative language models where full fine-tuning or LoRA adapts attention projections to alter token transition probabilities, decision model adaptation isolates the **Readout Head**:
- **Backbone (Frozen):** The 4B-parameter bi-directional linear attention network is wrapped with `torch.no_grad()`. This eliminates catastrophic forgetting and retains language representation priors.
- **Readout Head (Tuned):** $W_{\text{readout}} \in \mathbb{R}^{255 \times 5120}$ (2,611,280 bytes BF16/FP32).

### The Compound Decision Loss Function
Minimizing Cross-Entropy alone produces extreme overconfidence on small domain datasets. We employ a compound loss incorporating a quadratic Brier score penalty:

$$\mathcal{L}_{\text{total}} = \underbrace{-\sum_{i=1}^{K} y_i \log p_i}_{\mathcal{L}_{\text{CE}}} + \lambda_{\text{Brier}} \underbrace{\sum_{i=1}^{K} (p_i - y_i)^2}_{\mathcal{L}_{\text{Brier}}}$$

Where $\lambda_{\text{Brier}} = 0.15$ guarantees that predicted probabilities maintain calibrated confidence spreads.

### Empirical Training Progression on Mac Mini
Executed via [finetune_decider.py](finetune_decider.py) on Apple Silicon Metal Node:
- **Initial Loss (Epoch 1):** `1.9283` (CE: 1.7610, Brier: 1.1153)
- **Epoch 3:** `0.1664` (CE: 0.1527, Brier: 0.0909)
- **Epoch 6:** `0.0040` (CE: 0.0040, Brier: 0.0003)
- **Epoch 9:** `0.0013` (CE: 0.0013, Brier: 0.0000)
- **Final Loss (Epoch 12):** `0.0010` (CE: 0.0010, Brier: 0.0000)

### Empirical Validation Shifts
- **Ambiguous PowerShell Download Alert:** Baseline `investigation` probability shifted from **75.77%** to **99.98%** (+24.21% gain).
- **Authorized Red-Team Phishing Exercise:** Baseline `false_positive` shifted from **99.65%** to **100.00%** (zero false alarm).
- **Lateral Movement Host Isolation:** `containment` was smoothly calibrated to **75.03%** under the Brier score constraint.
- **Saved Model Checkpoint:** `finetuned_readout.safetensors` (2,611,280 bytes).

---

## 6. Diagnostic Engineering Log: Real-World Issues & Resolutions

During live remote deployment, protocol implementation, and fine-tuning on Apple Silicon, four distinct runtime failure modes were identified and systematically resolved:

1. **`Encoder.encode` Signature Mismatch:**
   - *Problem:* Calling `decider.encoder.encode([req], max_length=...)` fails with `TypeError: unexpected keyword argument 'max_length'`.
   - *Cause:* The `pplx_decider` protocol encoder operates synchronously on single `Request` dictionary objects.
   - *Resolution:* Encode requests individually: `chunk = decider.encoder.encode(req); chunks = [chunk]`, then collate using `pplx_decider.protocol.left_pad(chunks, pad_id)`.

2. **Absence of CUDA/Triton Kernels on macOS Metal (`mps`):**
   - *Problem:* PyTorch issues warnings that `causal_conv1d_fn` and `chunk_gated_delta_rule` fall back to reference PyTorch implementations.
   - *Cause:* Optimized CUDA/Triton C++ kernels for linear attention are uncompiled for Darwin ARM64 MPS.
   - *Resolution:* Verified that PyTorch reference implementations run with 100% numerical parity and zero divergence on MPS, delivering ~850ms per forward pass without stability degradation.

3. **Probability Saturation & Calibration Drift:**
   - *Problem:* Standard Cross-Entropy drove output probabilities to >0.999 after only 2 epochs on small incident triage datasets, destroying uncertainty estimation on ambiguous edge cases.
   - *Resolution:* Formulated compound loss with quadratic Brier penalty ($\lambda = 0.15$), enforcing smooth probability curvature.

4. **VRAM Exhaustion During Recurrent Backprop:**
   - *Problem:* Unfreezing full 4B parameters requires holding backward attention caches across recurrent delta layers, threatening unified memory exhaustion.
   - *Resolution:* Froze the entire backbone with `torch.no_grad()`. Backpropagated exclusively into the 2.6 MB readout head, maintaining peak RSS below 5.48 GB.

---

## 7. Artifacts Index

- Presentation Slides: [index.html](index.html)
- Style Sheet: [assets/style.css](assets/style.css)
- Controller Script: [assets/deck.js](assets/deck.js)
- 10 Empirical Test Cases Data: [decision_test_results.json](decision_test_results.json)
- Fine-Tuning Pipeline Script: [finetune_decider.py](finetune_decider.py)
- Fine-Tuning Metrics & Diagnostic Data: [finetune_results.json](finetune_results.json)
