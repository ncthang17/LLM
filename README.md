# LLM Report Summary
## 🤖 LLM Comparison Table

| Feature | GPT-4 | EXAONE 4.0 | Meta LLaMA (3.1) | HyperCLOVA | HyperCLOVA X THINK |
|:--------:|:--------:|:--------------:|:--------------------:|:--------------:|:--------------------:|
| **Developer** | OpenAI | LG AI Research | Meta AI | NAVER CLOVA | NAVER AI Lab |
| **Release Year** | 2023 | 2025 | 2023–2024 | 2021–2023 | 2025 |
| **Model Sizes** | Undisclosed | 1.2B, 32B | 8B, 70B, 405B | 13B–204B | Not disclosed |
| **Multimodal** | ✅ (text + image) | ✅ (Vision Model) | ❌ (text-only) | ❌ | ✅ (Vision support) |
| **Multilingual** | ✅ (24+ langs) | ✅ (KO, EN, ES) | ✅ (8 langs) | ✅ (KO, EN, JA) | ✅ (KO, EN) |
| **Context Length** | 8K–128K | Up to 128K | 128K | Not stated | 128K |
| **Reasoning Strength** | High (MMLU: 86.4) | High (MMLU-Redux: 92.3) | Very High (MMLU: 87.3) | Moderate | Very High (KCSAT +9.4% vs GPT-4.1) |
| **Tool Use / API** | ✅ | ✅ | ✅ | Limited | ✅ |
| **Code Support** | ✅ (67%) | ✅ | ✅ (89%) | Some | ✅ |
| **Vision Support** | ✅ | ✅ | ❌ | ❌ | ✅ |
| **License** | Proprietary | Non-commercial | Community License | Commercial API | Planned open (pruned) |
| **Use Cases** | Education, coding, business writing, vision tasks | STEM tutoring, coding, agent use | Multilingual agents, research, assistants | Korean chatbots, search + generation | Korean CSAT agents, visual STEM QA |
| **Strengths** | Multimodal, strong alignment, few-shot | Bilingual, reasoning, tool use | Open, multilingual, efficient | Korean-native, RAG via Cue | Reasoning, vision-text, length control |
| **Limitations** | Closed, hallucination risk | Non-commercial, training contamination risk | Text-only, 8-languages, license limits | Language-limited, closed | Korean focus, smaller ecosystem |

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

- Prohibited:
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
## 📌 Meta LLaMA
### 📖 Overview
LLaMA (Large Language Model Meta AI) is a family of open-weight large language models developed by Meta AI. LLaMA is designed to democratize access to powerful foundation models, with a strong emphasis on transparency, efficiency, multilingual support, and open research. Meta has released three major iterations so far: LLaMA 1 (2023), LLaMA 2 (July 2023), and LLaMA 3 / 3.1 (April & July 2024).

### 🧠 Model Evolution
|Version	|Release Date	|Sizes (B)	|Key Features|
|:-:|:-:|:-:|:-:|
|LLaMA 1	|Feb 2023	|7B, 13B, 33B, 65B	|Academic release, strong performance at small scale|
|LLaMA 2	|July 2023	|7B, 13B, 70B	|Open commercial use, improved alignment|
|LLaMA 3	|April 2024	|8B, 70B	|Instruction-tuned, long context, better CoT|
|LLaMA 3.1	|July 2024	|8B, 70B, 405B	|Multilingual support, tool use, 128K context|

### 🔧 Architecture & Design
- Type: Auto-regressive, decoder-only transformer
- Attention Mechanism: Switched from standard attention to Grouped Query Attention (GQA) for faster inference in LLaMA 3+
- Context Length:
  - LLaMA 1 & 2: ~4K to 32K tokens
  - LLaMA 3.x: 128K tokens

- Tokenizer: Custom sentencepiece-based tokenizer (token count varies slightly across versions)

- Training Data: Publicly available and licensed data (no Meta user data)
  - LLaMA 3/3.1 trained on over 15 trillion tokens
  - Fine-tuning with 25M+ synthetic instructions in 3.1

### 💡 Capabilities Across the LLaMA Family
- 🧾 Text Generation: High-quality coherent text in multiple languages.

- 🧠 Instruction Following: Strong alignment via RLHF (LLaMA 2+) and preference optimization.

- 💻 Code Generation: Effective in Python and multiple languages, improved across releases.

- 📚 Multilingual Tasks: LLaMA 3.1 supports 8 major languages, with potential for others.

- 🧰 Tool Use & Agents: Built-in support for function calling and structured tool APIs (3.1).

- 📏 Long-context understanding: From 32K in LLaMA 2 to 128K in 3.1, great for document QA and agents.

### ✅ Strengths
- 📖 Open-weight: Available for research and commercial use (under license).

- 🧪 Efficient: Smaller models (8B, 13B) outperform much larger models (e.g., GPT-3) on several tasks.

- 🌍 Multilingual: LLaMA 3.1 supports English, Spanish, French, German, Italian, Portuguese, Hindi, and Thai.

- 🛠️ Developer-friendly: Works with Hugging Face, llama.cpp, and other open frameworks.

- 🧠 Reasoning: Supports Chain-of-Thought prompting and high performance in benchmarks like MMLU and GSM-8K.

- 🔧 Tool compatibility: Supports tool use pipelines, prompt templates, and inference acceleration.

### ❌ Weaknesses
- 🔒 Not permissively licensed: Community License limits certain types of use (e.g., foundation for competing models).

- 🌐 Language limitations: Only 8 languages officially supported in LLaMA 3.1.

- 📸 No vision input: Unlike GPT-4 or EXAONE, LLaMA is text-only.

- 🧯 Still hallucinates: Prone to factual inaccuracies in absence of retrieval-based grounding.

### 📊 Benchmarks (LLaMA 2 & 3.1)
**MMLU (5-shot accuracy)**
|Model	|Size	|Score (%)|
|:-:|:-:|:-:|
|LLaMA 2	|70B	|~76|
|LLaMA 3	|70B	|~83|
|LLaMA 3.1	|405B	|87.3|

**HumanEval (0-shot code pass@1)**
|Model	|Size	|Score (%)|
|:-:|:-:|:-:|
|LLaMA 2	|13B	|~33|
|LLaMA 3	|70B	|~82|
|LLaMA 3.1	|405B	|89.0|

### 🔐 License
- LLaMA 1: Research use only

- LLaMA 2 & 3: Meta LLaMA Community License

  - ✅ Commercial use allowed with conditions
  - ❌ Not allowed for creating competing foundation models
  - ❌ Redistribution only with license compliance

- [License Link (GitHub)](https://github.com/meta-llama/llama-models/blob/main/LICENSE)

### 🧰 Use Cases
|Area	|Applications|
|:-:|:-:|
|Research	|NLP modeling, transfer learning, prompt engineering|
|Assistants	|Customer support bots, tutors, productivity apps|
|Coding	|Pair programming, debugging, documentation generation|
|Enterprise NLP	|Document classification, summarization, RAG systems|
|Education	|Language learning, STEM explanation, test prep|

### 🌱 Efficiency & Emissions
- Meta used H100 GPUs for LLaMA 3.1 training with 39.3M GPU hours.

- Estimated location-based GHG emissions: 11,390 tons CO₂eq.

- But market-based GHG emissions: 0 tons CO₂eq, due to 100% renewable energy.

### 🧠 Community Ecosystem
- LLaMA models power many open projects: OpenChat, Nous Hermes, Zephyr, etc.

- Fully integrated with:

  - 🧪 Hugging Face Transformers
  - ⚡ llama.cpp (quantized inference)
  - 🧱 LangChain, OpenDevin, AutoGen, and other LLM agent frameworks

### 📚 Citation
Meta AI (2023–2024). Meta LLaMA Model Family Reports.
Access: https://github.com/meta-llama

## 🧠 HyperCLOVA X THINK (NAVER AI)
### 📖 Overview
HyperCLOVA X THINK is a reasoning-specialized large language model developed by NAVER AI Lab, as part of the HyperCLOVA X family. It focuses on high-performance reasoning, particularly in Korean and English, and is capable of length-controllable, verifiable, and step-by-step thinking. THINK supports both text-only and multimodal (vision-language) versions and is designed for education, science, and multilingual applications.

### 💡 Capabilities
- Bilingual Reasoning: Fluent in Korean and English; excels in Korean STEM and CSAT benchmarks.

- Length Control: Users can choose concise or detailed (step-by-step) responses via simple prompts.

- Long-context understanding: Supports context lengths up to 128K tokens.

- Multimodal vision-text model: THINK with Vision handles image-text reasoning tasks.

- Tool use support: THINK supports tool calling for agentic tasks (similar to APIs).

- RLVR (Reinforcement Learning with Verifiable Rewards): New alignment method using deterministic reward signals for truthfulness, conciseness, and reasoning quality.

### ✅ Strengths
- 🇰🇷 Sovereign AI for Korea: SOTA performance on Korean-specific benchmarks (e.g., KCSAT STEM).
- 🧠 Reasoning performance: Outperforms GPT-4.1 on Korean CSAT STEM vision-language benchmark.
- 🧾 Concise vs. step-by-step outputs: Controlled via prompt; good for exams or summaries.
- 📊 Efficiency: Outperforms larger models while using fewer training GPU hours.
- 👁️ Vision-language integration: Includes a powerful visual reasoning version (THINK with Vision).

### ⚠️ Weaknesses & Limitations
- 🌐 Multilingual limitations: Strong in Korean/English, but not yet broadly multilingual like GPT-4 or LLaMA 3.1.

- 💬 Model availability: Full weights are not publicly released yet; only a pruned open version is planned.

- ⚙️ Ecosystem adoption: Smaller global developer ecosystem compared to Meta or OpenAI.

- 🧪 Benchmarks localized: Some evaluations are unique to Korean academic and reasoning tasks.

### 💼 Use Cases
- STEM education & tutoring (especially for Korean students)

- AI exam assistants for Korean CSAT or multilingual tasks

- Vision-language reasoning (e.g., interpreting graphs, exam figures)

- Public sector AI: Language-sovereign applications in Korea

- Agentic assistants: Use in tools, automation, and long-term task planning

### 📊 Performance Highlights
|Task / Benchmark	|THINK Model Size	|Performance Note|
|:-:|:-:|:-:|
|KCSAT STEM Vision QA	|Not specified	|Outperformed GPT-4.1 by +9.4% (Vision tasks)|
|GSM8K (Math Reasoning)	|128K tokens	|Competitive with global SOTA|
|Commonsense Reasoning (EN)	|128K tokens	|High accuracy in multiturn logical questions|
|Concise vs. Step-by-step	|Prompt control	|User-controllable response depth|
|RLVR Alignment	|All sizes	|High stability and factuality improvements|

### 🏗️ Architecture & Design
- Transformer-based architecture (Auto-regressive decoder)

- Tokenizer: Korean-optimized with multilingual support

- Training:

  - Pretraining on public Korean + English corpora
  - Alignment via RLVR, enhancing factuality and verifiability
  - Visual training with stage-wise curriculum for vision-language models

### 🔒 License & Availability
- Status: Full weights not released yet

- Planned: A pruned & distilled open model under a business-friendly license (similar to LLaMA community license)

- Access:
  - THINK API via CLOVA Studio (NAVER)
  - Open model release expected soon on Hugging Face or GitHub

### 🧠 Innovations & Highlights
- RLVR (Reinforcement Learning with Verifiable Rewards):
  - Deterministic alignment reward function for:
    - Truthfulness
    - Conciseness
    - Reasoning integrity

- Vision-Language Stage Curriculum:
  - Pretraining → reasoning alignment → multimodal alignment

- Length-control prompting:
  - Example: Add "Concise:" or "Step-by-step:" at the beginning of the prompt

### 🌱 Efficiency
  - Comparable performance to GPT-4.1 with fewer parameters and less training compute.
  - Optimized for Korean infrastructure with green AI design focus.

### 📚 Citation
NAVER AI Lab (2025). HyperCLOVA X THINK Technical Report. [arXiv:2506.22403](https://arxiv.org/abs/2506.22403)
