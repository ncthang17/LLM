# LLM Report Summary
## 📌 GPT-4: Technical Summary
### 📖 Overview
GPT-4 is a large multimodal model developed by OpenAI that can process both text and image inputs and generate text outputs. It demonstrates human-level performance on a variety of professional and academic tasks, significantly surpassing its predecessor GPT-3.5 in accuracy, reasoning, and language understanding. The architecture is based on the Transformer model and trained using next-token prediction, followed by Reinforcement Learning from Human Feedback (RLHF).

### 💡 Capabilities
- Multimodal Input: Accepts both text and images.

- Exam Simulation: Scored in the top 10% on the Uniform Bar Exam, compared to GPT-3.5's bottom 10%.

- Academic Benchmarking: Achieved state-of-the-art results on:

  - MMLU (86.4%)
  - HumanEval (67% pass rate)
  - ARC, HellaSwag, GSM-8K, etc.

- Multilingual Proficiency: Outperformed GPT-3.5 in 24 out of 26 languages on MMLU.

- Few-shot and Zero-shot learning: Excels at reasoning and instruction-following even with minimal examples.

- Visual Reasoning: Demonstrates strong understanding of images with text (e.g., humor detection in comic-style images).

### ✅ Strengths
- High factuality: 19% improvement in internal factuality evals over GPT-3.5.

- Safety improvements: Stronger refusals for harmful prompts (e.g., bomb-making, drug synthesis).

- Language generalization: Handles low-resource languages (e.g., Swahili, Welsh).

- Predictable scaling: Performance and loss metrics can be accurately extrapolated from smaller models.

- Code generation: Significant improvement in coding tasks (e.g., HumanEval pass rate of 67%).

### ⚠️ Weaknesses
- Hallucination: Still occasionally generates false or misleading information.

- Reasoning errors: May fail in simple or adversarial reasoning tasks.

- Lack of real-time learning: Cannot learn from post-training experience or user feedback dynamically.

- Limited context window: Constrained by the amount of text it can process at once.

- Calibration loss: After RLHF, becomes less confident even when correct (lower calibration).

- Susceptibility to jailbreaks: Can be tricked into generating unsafe content under adversarial prompts.

### 💼 Use Cases
- Education: Tutoring, exam prep, summarization, and knowledge explanation.

- Programming: Code generation, debugging, and explanation (especially Python).

- Business: Market analysis, report generation, customer service chatbots.

- Creative writing: Story generation, scriptwriting, editing assistance.

- Translation and multilingual support: Handles diverse languages effectively.

- Vision + Text: Document QA, meme explanation, form understanding (e.g., receipts or tables).

### 📜 Licensing
- GPT-4 is a proprietary model developed by OpenAI, licensed under commercial terms.

- Public usage is available via:

  - ChatGPT (Plus) – includes GPT-4 access.
  - OpenAI API – metered access for developers.

- The exact architecture, dataset composition, and model size are not disclosed due to safety and competitive reasons.

### 🧠 Notable Innovations
- Predictable Scaling Laws: Infrastructure enables loss and capability forecasting using small-scale model runs.

- System Cards: Transparent documentation of model behavior, risks, and safety.

- Rule-Based Reward Models (RBRMs): Used to enforce appropriate refusals or non-refusals in the RLHF stage.

- Adversarial Red Teaming: Involved 50+ domain experts testing edge-case risks (e.g., cybersecurity, biohazards).

### 📉 Safety and Risk Mitigation
- RLHF alignment: Tuned to follow user intent more reliably.

- Reduced refusal rates: Smarter distinction between harmful and benign prompts.

- Toxicity control: Generates toxic content only 0.73% of the time (vs. GPT-3.5's 6.48%).

- Truthfulness: Outperforms GPT-3.5 and Anthropic models on TruthfulQA.

- Monitoring pipeline: Continuous evaluation and model updates to reduce misuse.

### 📚 Citation
OpenAI (2023). GPT-4 Technical Report. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)

## 📌 EXAONE 4.0: Technical Summary
### 📖 Overview
EXAONE 4.0 is a next-generation unified large language model developed by LG AI Research that integrates both REASONING and NON-REASONING modes. It aims to deliver high performance in both general usability and deep reasoning tasks like mathematics and coding. It supports multilingual use (Korean, English, Spanish) and is designed for the agentic AI era through enhanced tool use, long-context understanding (up to 128K tokens), and modular training.

### 💡 Capabilities
- Dual-mode operation: Seamlessly switches between:
  - 🧠 Reasoning Mode (deep, accurate thinking)
  - ⚡ Non-reasoning Mode (fast, general usability)

- Multilingual: English, Korean, Spanish.

- Agentic tool use: Can simulate external API/tool calling—essential for future AI agents.

- Long-context support: Handles up to 128K tokens (32B model), 64K for 1.2B model.

- Hybrid attention mechanism: Combines global and local attention for efficient long-context processing.

- Modular training:
    - Supervised fine-tuning
    - Reasoning-specific reinforcement learning
    - Preference learning (conciseness, consistency)

✅ Strengths
- World Knowledge: Outperforms or matches frontier models in MMLU benchmarks.

- Math & Coding: Excels in Olympiad and coding tasks:
  - AIME 2025: 85.3%
  - HMMT FEB 2025: 72.9%
  - LiveCodeBench V5: 72.6%

- Instruction Following: IFEVAL: 83.7%, MULTI-IF (EN): 73.5%

- Multilingual Mastery: Top-tier Korean/Spanish performance (e.g., KSM (KO): 87.6%)

- Tool Use: Competitive with much larger models in TAU-BENCH and BFCL-V3.

- Reasoning Budget Efficiency: Maintains strong performance even when token usage is reduced (32K vs. 64K).

### ⚠️ Weaknesses / Limitations
- Bias & Safety: Still vulnerable to generating biased or harmful outputs.

- Factual inaccuracies: Especially under reasoning constraints or long sequences.

- Training contamination risk: Risks of overlap in training/evaluation corpora noted (e.g., KMMLU).

- Non-commercial license: Limits deployment in commercial applications (see license below).

- Compute: Long context and reasoning ability require significant compute.

### 💼 Use Cases
- Education: Math tutoring, scientific reasoning, language learning.

- Enterprise: Document Q&A, legal/technical content summarization.

- Research: Tool-use simulation, long-context benchmarking, multilingual NLP.

- Code generation: Full-stack development and debugging.

- Agent simulation: Task-planning, tool integration, API orchestration.

### 📝 Model Variants
|Version	|Parameters	|Context Length       	|Primary Use|
|:----------:|:----------:|:-------------------:|:----------:|
|EXAONE 4.0 32B	|32B	|128K |High-performance agent, server-side deployment
|EXAONE 4.0 1.2B	|1.2B	|64K |On-device and lightweight applications

### 📜 License
- Type: EXAONE AI Model License Agreement 1.2 – NC (Non-Commercial)

- Allowed:
  - Research & education
  - Model modification & redistribution (with terms)
  - Publishing results

-Prohibited:
  - Commercial use (apps, services, monetization)
  - Model competition without explicit permission
  - Reverse engineering or competing model development

- License text: [EXAONE License PDF](https://huggingface.co/LGAI-EXAONE)

### 📊 Benchmarks Summary
|Benchmark|	EXAONE 4.0 (32B)|	Compared Models|
|:-:|:-:|:-:|
|MMLU-REDUX |92.3	|GPT-4.1: 93.4|
|GPQA-DIAMOND (expert)	|75.4	|Qwen-235B: 71.1|
|AIME 2025 (math)	|85.3	|DeepSeek-R1: 87.5|
|LiveCodeBench V6 (code)	|66.7	|Qwen-235B: 70.3|
|IFEVAL (instr. following)	|83.7	|Phi 4: 84.9|
|KMMLU-REDUX (Korean)	|72.7	|Llama 4: 77.0|
|MMMLU (Spanish)	|85.6	|GPT-4.1: 88.2|
|Tool use: BFCL-V3	|63.9	|DeepSeek-R1: 64.7|

*EXAONE performs similarly to or better than some frontier models, despite smaller size (32B vs. 235B/400B+)*

### 🧠 Innovations
- AGAPO: A new RL algorithm combining asymmetric sampling and group/global reward calculation for enhanced reasoning.

- Unified Mode Training: Simultaneous training of REASONING and NON-REASONING with balanced datasets.

- Hybrid attention: Efficient long-context support with a 3:1 local-to-global ratio.

- Post-training preference optimization: Conciseness, correctness, language consistency.

### 📚 Citation
LG AI Research (2025). EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes. [arXiv:2507.11407v1](https://arxiv.org/abs/2507.11407)
