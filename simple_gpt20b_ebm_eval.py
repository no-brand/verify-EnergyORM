import argparse
import json
import math
import random
import re
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ebm_model import TransEBM


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_numeric_answer(text: str) -> Optional[str]:
    if not text:
        return None
    if "####" in text:
        candidate = text.split("####")[-1].strip()
        if candidate:
            return candidate.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if matches:
        return matches[-1]
    return None


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return math.isclose(float(pred), float(gold), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return pred.strip() == gold.strip()


def build_prompt(question: str) -> str:
    return (
        "Solve the following GSM8K math problem step by step. "
        "Show your reasoning and place the numeric answer after '####'.\n\n"
        f"Question: {question}\nReasoning:\n"
    )


def generate_thoughts(generator, tokenizer, question: str, args) -> List[str]:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(generator.device)
    outputs = generator.generate(
        **inputs,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_paths,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def setup_ebm_tokenizer(name: str, max_length: int):
    tok = AutoTokenizer.from_pretrained(name, use_fast=False)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    if tok.bos_token_id is not None:
        cls_id = tok.bos_token_id
    elif tok.cls_token_id is not None:
        cls_id = tok.cls_token_id
    elif tok.eos_token_id is not None:
        cls_id = tok.eos_token_id
    else:
        raise ValueError("Tokenizer must have BOS/CLS/EOS token.")
    tok.model_max_length = max_length
    return tok, tok.pad_token_id, cls_id


def encode_pair(tok, cls_id, question: str, thought: str, max_length: int):
    sep = tok.eos_token or "\n"
    combined = f"{question}{sep}{thought}"
    ids = tok.encode(
        combined,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 1,
    )
    return torch.tensor([cls_id] + ids, dtype=torch.long)


def pad_batch(seqs: List[torch.Tensor], pad_id: int):
    max_len = max(len(seq) for seq in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = seq
        mask[i, : len(seq)] = 1
    return ids, mask


def score_with_ebm(model, tok, cls_id, pad_id, question, thoughts, max_length, device):
    seqs = [encode_pair(tok, cls_id, question, t, max_length) for t in thoughts]
    ids, mask = pad_batch(seqs, pad_id)
    ids = ids.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        energies = model(ids, mask)
    return energies.cpu()


def main(args):
    set_seed(args.seed)

    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.num_questions:
        dataset = dataset.select(range(min(args.num_questions, len(dataset))))

    gen_tokenizer = AutoTokenizer.from_pretrained(
        args.generator_model,
        use_fast=False,
        padding_side="left",
        trust_remote_code=True,
    )
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    generator = AutoModelForCausalLM.from_pretrained(
        args.generator_model,
        torch_dtype=getattr(torch, args.generator_dtype),
        device_map="auto",
        trust_remote_code=True,
    )

    ebm_tok, pad_id, cls_id = setup_ebm_tokenizer(args.ebm_tokenizer_path, args.ebm_max_length)
    ebm = TransEBM(
        vocab_size=len(ebm_tok),
        d_model=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(args.ebm_device)
    ebm.load_state_dict(torch.load(args.ebm_model_path, map_location=args.ebm_device))
    ebm.eval()

    naive_correct = 0
    ebm_correct = 0
    total = 0

    detailed = []

    for sample in dataset:
        question = sample["question"]
        gold = extract_numeric_answer(sample["answer"])
        thoughts = generate_thoughts(generator, gen_tokenizer, question, args)

        labels = []
        for thought in thoughts:
            pred = extract_numeric_answer(thought)
            labels.append(1 if answers_match(pred, gold) else 0)

        if not thoughts:
            continue

        total += 1
        if labels[0] == 1:
            naive_correct += 1

        energies = score_with_ebm(
            ebm,
            ebm_tok,
            cls_id,
            pad_id,
            question,
            thoughts,
            args.ebm_max_length,
            torch.device(args.ebm_device),
        )
        best_idx = int(torch.argmin(energies).item())
        if labels[best_idx] == 1:
            ebm_correct += 1

        detailed.append({
            "question": question,
            "gold_answer": gold,
            "thoughts": thoughts,
            "labels": labels,
            "energies": energies.tolist(),
            "chosen_by_ebm": best_idx,
            "chosen_by_naive": 0,
        })

    summary = {
        "total_questions": total,
        "ebm_accuracy": 100.0 * ebm_correct / total if total else 0.0,
        "naive_accuracy": 100.0 * naive_correct / total if total else 0.0,
        "accuracy_delta": (100.0 * (ebm_correct - naive_correct) / total) if total else 0.0,
        "generator_model": args.generator_model,
        "num_paths": args.num_paths,
        "split": args.split,
    }

    print(json.dumps(summary, indent=2))

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": detailed}, f, ensure_ascii=False, indent=2)
        print(f"Saved detailed log to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simple GPT-OSS 20B vs TransEBM evaluation on GSM8K")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--num_paths", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--generator_model", default="open-gpt-20b")
    parser.add_argument("--generator_dtype", default="bfloat16")

    parser.add_argument("--ebm_model_path", default="ebm_llama_gsm_model.pt")
    parser.add_argument("--ebm_tokenizer_path", default="gpt2")
    parser.add_argument("--ebm_max_length", type=int, default=2048)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ebm_device", default="cuda")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path")
    args = parser.parse_args()
    main(args)
