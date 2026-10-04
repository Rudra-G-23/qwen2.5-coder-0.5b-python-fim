Here are the direct paper links for your reading list, in the same order:

## 1. FIM (Fill-in-the-Middle) training
- **Efficient Training of Language Models to Fill in the Middle** (Bavarian et al., 2022)  
  https://arxiv.org/abs/2207.14255 [arxiv](https://arxiv.org/abs/2207.14255)

## 2. LoRA
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)  
  https://arxiv.org/abs/2106.09685 [arxiv](https://arxiv.org/abs/2106.09685)

## 3. QLoRA
- **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)  
  https://arxiv.org/abs/2305.14314 [arxiv](https://arxiv.org/abs/2305.14314)

## 4. SAFIM (syntax-aware FIM for code)
- **Evaluation of LLMs on Syntax-Aware Code Fill-in-the-Middle Tasks** (Gong et al., 2024)  
  https://arxiv.org/abs/2403.04814 [arxiv](https://arxiv.org/abs/2403.04814)

## 5. Sequence-level knowledge distillation for LLMs
The foundational “practical” sequence-level KD paper commonly used for LLM distillation is:

- **Sequence-Level Knowledge Distillation** (Kim & Rush, 2016)  
  https://arxiv.org/abs/1606.07947 [arxiv](https://arxiv.org/abs/1606.07947)

For a more LLM-focused, practical overview that explicitly treats this as the standard SeqKD/SFT approach, see:

- **A Survey on Knowledge Distillation of Large Language Models** (2024) – discusses SeqKD as the basic, effective method:  
  https://arxiv.org/abs/2402.13116 [arxiv](https://arxiv.org/html/2402.13116v1)

If you want a specific modern LLM distillation architecture paper (e.g., MiniLLM), I can add that too.

## 6. PEFT / Transformers / Unsloth docs
- **Hugging Face PEFT docs**: https://huggingface.co/docs/peft [huggingface](https://huggingface.co/docs/peft/index)
- **PEFT + Transformers integration guide**: https://huggingface.co/docs/transformers/v4.56.2/en/peft [huggingface](https://huggingface.co/docs/transformers/v4.56.2/en/peft)
- **Transformers PEFT & adapter integration**: https://deepwiki.com/huggingface/transformers/6.3-peft-and-adapter-integration [deepwiki](https://deepwiki.com/huggingface/transformers/6.3-peft-and-adapter-integration)

For Unsloth’s own documentation and examples, you’ll want their GitHub and docs site (e.g., https://github.com/unslothai/unsloth and https://docs.unsloth.ai/); if you tell me your exact use-case (model size, GPU, 4-bit vs 16-bit), I can point you to the most relevant Unsloth pages and example notebooks.
