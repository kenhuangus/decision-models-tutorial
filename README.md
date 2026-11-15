# Decision Models in Production: Architecture, Empirical Use Cases, System Integration & Fine-Tuning

🌐 **Live Interactive Tutorial Site (GitHub Pages):** [https://kenhuangus.github.io/decision-models-tutorial/](https://kenhuangus.github.io/decision-models-tutorial/)

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

**Decision Models** (such as Perplexity's `pplx-decider-v1.3`, `pplx-decider-flash-v1.2`, and `pplx-decider-v1.1-27b`) solve this fundamental architectural challenge by:
1. **Lifting the causal attention mask:** Allowing full bi-directional cross-attention across the entire input state without token-position decay.
2. **Direct readout projection:** Projecting the final hidden state through a specialized readout matrix onto exact choice tokens, booleans (`noul`), or text spans (`locate`).
3. **Consuming strictly 0 generated text tokens:** Eliminating autoregressive decoding loops entirely and executing in a single forward pass while emitting discrete **decision tokens**.
4. **Providing mathematically calibrated probabilities:** Guaranteeing that probability scores map directly to empirical expectation values through compound Brier loss calibration.

---

## 2. Interactive Presentation Deck (47 Slides · Pure Light Theme)

The interactive tutorial slide deck is deployed on GitHub Pages:
- **Live Slide Deck URL:** [https://kenhuangus.github.io/decision-models-tutorial/](https://kenhuangus.github.io/decision-models-tutorial/)
- **Local File Entrypoint:** [index.html](index.html)
- **Features:** Strict pure light theme (zero dark backgrounds), keyboard navigation (`ArrowRight`, `ArrowLeft`, `PageUp`, `PageDown`, `Home`, `End`), full-screen toggle (`F`), direct slide deep linking (`#1` to `#47`), dynamic progress bar, and touch swipe gestures.

### Slide Structure Overview (47 Slides):
- **Section A · Foundations & Architecture (Slides 01–06):** Masterclass Title, Ken Huang USF Leadership, Why Generative LLMs Fail at Decisions, Lifting the Causal Mask, Anatomy of Perplexity Decision Models, and Apple Silicon Metal MPS Deployment.
- **Section B · Acquisition, Weights & Runtime Setup (Slides 07–10):** Student Environment Setup, `huggingface-cli` & `hf_transfer` Download Deep-Dive, Git-LFS & Programmatic Python Snapshot Downloads, and Memory Allocation Tiers (BF16 vs. INT4).
- **Section C · CLI Execution & Prompt Schemas (Slides 11–14):** CLI for Running the Model Locally (Python, vLLM, SGLang, llama-server), PyTorch Execution Engine, Prompt Framing for Choice and Locate Heads, and `run_usecase.py` CLI Tool Reference.
- **Section D · API Endpoints, Schemas & Client Integration (Slides 15–18):** Where is the API Endpoint? (Localhost, LAN, Tailscale, Cloud), REST Specification for `POST /v1/decide` and `POST /v1/locate`, Testing Endpoints with cURL/HTTPie, and Asynchronous Client SDKs (Python `httpx` & TypeScript).
- **Section E · Serving Daemons & Batch Processing (Slides 19–21):** Complete Production FastAPI Microservice (`server.py`), macOS Persistent Daemon (`launchd`), and High-Throughput Batch Decision Processing (1,400+ decisions/sec).
- **Section F · 10 Production Use Cases with Real Data & CLI (Slides 22–31):**
  - **Use Case 01:** AI Agent Tool Selection & Routing (`python run_usecase.py --case 1 --device mps`)
  - **Use Case 02:** Customer Support Escalation Gate (`python run_usecase.py --case 2 --device mps`)
  - **Use Case 03:** RAG Retrieval Chunk Re-ranking (`python run_usecase.py --case 3 --format table`)
  - **Use Case 04:** Hallucination & Factuality Verification (`python run_usecase.py --case 4 --device mps`)
  - **Use Case 05:** Prompt Injection & Adversarial Attack Gate (`python run_usecase.py --case 5 --benchmark 10`)
  - **Use Case 06:** E-Commerce Intent & Conversion Readiness (`python run_usecase.py --case 6 --device mps`)
  - **Use Case 07:** Automated Code Review & Security Gate (`python run_usecase.py --case 7 --format table`)
  - **Use Case 08:** Multi-Class Content Moderation (`python run_usecase.py --case 8 --device mps`)
  - **Use Case 09:** Agentic Planning Branch Selection (`python run_usecase.py --case 9 --device mps`)
  - **Use Case 10:** Sensitive Data Leakage Prevention with Locate (`python run_usecase.py --case 10 --locate`)
- **Section G · Hardware Benchmarks & Agent Frameworks (Slides 32–36):** Mac Mini M4 Empirical Benchmarks, Ingress/Branching/Egress Topology, LangGraph Decision Routing Node, LlamaIndex Chunk Pruner Node, and CrewAI/AutoGen Tool Authorization Guardrail.
- **Section H · End-to-End Fine-Tuning Curriculum (Slides 37–42):** Parameter Isolation Architecture (Freezing 27B Backbone, Tuning 2.6 MB Head), Compound Brier Loss Formulation, PyTorch MPS Training Script (`finetune_decider.py`), Live Mac Mini Training Logs, Diagnostic Engineering Log, and Calibration Curves (ECE).
- **Section I · Production Operations, Labs & Synthesis (Slides 43–47):** Prometheus Telemetry & Brier Score Drift Alerts, Zero-Trust Reverse Proxy Hardening, 3 Hands-On Student Lab Exercises, Architectural Decision Matrix, and Course Summary (The 4 Golden Rules).

---

## 3. Empirical Test Results from Apple Silicon Metal Node

All 10 production use cases were deployed and verified live on an Apple Silicon node (ARM64, Darwin 27.0.0, 64 GB Unified RAM) running `perplexity-ai/pplx-decider-flash-v1.2` over Metal Performance Shaders (`mps`). Output tokens generated across all 10 use cases was strictly **0 generated text tokens** (1 emitted decision token):

| ID | Use Case Pattern | Schema | Decision Output | Empirical Probability | Latency (Metal MPS) | CLI Execution Command |
|---|---|---|---|---|---|---|
| **01** | Tool Selection & Routing | Choice | `chart_generator` | **75.65%** | 1,279.8 ms | `python run_usecase.py --case 1 --device mps` |
| **02** | Support Escalation Gate | Noul (Boolean) | Escalation Required | **97.82%** | 912.0 ms | `python run_usecase.py --case 2 --device mps` |
| **03** | RAG Chunk Re-ranking | Choice | `highly_relevant` | **93.35%** | 950.8 ms | `python run_usecase.py --case 3 --format table` |
| **04** | Hallucination NLI Gate | Choice | `contradicted` | **95.28%** | 618.5 ms | `python run_usecase.py --case 4 --device mps` |
| **05** | Prompt Injection Gate | Noul (Boolean) | Adversarial Attack | **99.94%** | 473.6 ms | `python run_usecase.py --case 5 --benchmark 10` |
| **06** | E-Commerce Intent | Choice | `ready_to_buy` | **99.83%** | 938.4 ms | `python run_usecase.py --case 6 --device mps` |
| **07** | Code Review Risk Gate | Choice | `critical_risk` | **98.87%** | 862.9 ms | `python run_usecase.py --case 7 --format table` |
| **08** | Content Moderation | Choice | `ip_violation` | **97.67%** | 635.8 ms | `python run_usecase.py --case 8 --device mps` |
| **09** | Planning Branch Selection | Choice | `branch_b` | **99.72%** | 1,100.3 ms | `python run_usecase.py --case 9 --device mps` |
| **10** | Secret Leakage Prevention | Locate Span | Secret Leaked | **99.53%** | 571.8 ms | `python run_usecase.py --case 10 --locate` |

*Full machine-readable test records:* [decision_test_results.json](decision_test_results.json)

---

## 4. System Topology: The Sandwich Guardrail Architecture

In enterprise AI Agent applications, Decision Models and Generative LLMs operate in complementary, decoupled tiers:

```
[ Raw User Request ]
         │
         ▼
┌──────────────────────────────────────────────┐
│  1. INGRESS GUARDRAIL (Decision Model)       │  <-- 0 Text Tokens, 470ms
│  - Prompt Injection & Jailbreak Gate         │
│  - Policy Compliance & Toxicity Filter       │
└──────────────────────┬───────────────────────┘
                       │ Validated Clean State
                       ▼
┌──────────────────────────────────────────────┐
│  2. REASONING & ROUTING (Decision Model)     │  <-- 0 Text Tokens, 850ms
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
│  4. EGRESS VERIFIER (Decision Model)         │  <-- 0 Text Tokens, 570ms
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

- Live GitHub Pages Website: [https://kenhuangus.github.io/decision-models-tutorial/](https://kenhuangus.github.io/decision-models-tutorial/)
- Presentation Slides Source: [index.html](index.html)
- Pure Light Style Sheet: [assets/style.css](assets/style.css)
- Controller Script: [assets/deck.js](assets/deck.js)
- Runnable Student CLI Test Harness: [run_usecase.py](run_usecase.py)
- 10 Empirical Test Cases Data: [decision_test_results.json](decision_test_results.json)
- Fine-Tuning Pipeline Script: [finetune_decider.py](finetune_decider.py)
- Fine-Tuning Metrics & Diagnostic Data: [finetune_results.json](finetune_results.json)
