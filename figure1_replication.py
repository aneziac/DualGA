import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm

from models import SRLConfig, DualGAConfig, ExperimentResult
from watermarking import generate_srl, generate_dualga, compute_z_score


def load_prompts(n_prompts: int, min_prompt_tokens: int, tokenizer: PreTrainedTokenizer) -> list[list[int]]:
    """Load prompts from C4 realnewslike dataset."""
    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    prompts = []
    for example in dataset:
        text = example["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= min_prompt_tokens * 2:
            # Use first half as prompt
            prompt_tokens = tokens[:min_prompt_tokens]
            prompts.append(prompt_tokens)
            if len(prompts) >= n_prompts:
                break
    return prompts


def run_experiment(model_name: str, n_prompts: int, max_new_tokens: int, min_prompt_tokens: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    prompts = load_prompts(n_prompts, min_prompt_tokens, tokenizer)

    srl_configs = [
        SRLConfig(gamma=0.25, delta=1.0,  label="SRL γ=0.25 δ=1"),
        SRLConfig(gamma=0.25, delta=2.0,  label="SRL γ=0.25 δ=2"),
        SRLConfig(gamma=0.25, delta=5.0,  label="SRL γ=0.25 δ=5"),
        SRLConfig(gamma=0.25, delta=10.0, label="SRL γ=0.25 δ=10"),
        SRLConfig(gamma=0.5,  delta=1.0,  label="SRL γ=0.5  δ=1"),
        SRLConfig(gamma=0.5,  delta=2.0,  label="SRL γ=0.5  δ=2"),
        SRLConfig(gamma=0.5,  delta=5.0,  label="SRL γ=0.5  δ=5"),
        SRLConfig(gamma=0.5,  delta=10.0, label="SRL γ=0.5  δ=10"),
    ]

    dualga_configs = [
        DualGAConfig(gamma=0.325, D=0.2, init_lambda=0.8, autoeta=10, label="DualGA Δ=0.2"),
        DualGAConfig(gamma=0.325, D=0.3, init_lambda=1.5, autoeta=10, label="DualGA Δ=0.3"),
        DualGAConfig(gamma=0.325, D=0.4, init_lambda=2.2, autoeta=10, label="DualGA Δ=0.4"),
        DualGAConfig(gamma=0.325, D=0.5, init_lambda=4.0, autoeta=10, label="DualGA Δ=0.5"),
    ]

    results: list[ExperimentResult] = []

    # Run SRL
    for cfg in srl_configs:
        print(f"\nRunning {cfg.label}...")
        z_scores = []
        mean_dgs = []
        mean_kls = []
        for prompt_tokens in tqdm(prompts):
            input_ids = torch.tensor([prompt_tokens], device=device)
            _, green_hits, dg_vals, kl_vals = generate_srl(
                model, tokenizer, input_ids, max_new_tokens,
                cfg.gamma, cfg.delta, device
            )
            z_scores.append(compute_z_score(green_hits, cfg.gamma))
            mean_dgs.append(np.mean(dg_vals))
            mean_kls.append(np.mean(kl_vals))

        results.append(ExperimentResult(
            label=cfg.label,
            type="SRL",
            z_scores=z_scores,
            mean_dgs=mean_dgs,
            mean_kls=mean_kls,
        ))

    # Run DualGA
    for cfg in dualga_configs:
        print(f"\nRunning {cfg.label}...")
        z_scores = []
        mean_dgs = []
        mean_kls = []
        for prompt_tokens in tqdm(prompts):
            input_ids = torch.tensor([prompt_tokens], device=device)
            _, green_hits, dg_vals, kl_vals = generate_dualga(
                model, tokenizer, input_ids, max_new_tokens,
                cfg.gamma, cfg.D, cfg.init_lambda, cfg.autoeta, device
            )
            z = compute_z_score(green_hits, cfg.gamma)
            z_scores.append(z)
            mean_dgs.append(np.mean(dg_vals))
            mean_kls.append(np.mean(kl_vals))
        results.append(ExperimentResult(
            label=cfg.label,
            type="DualGA",
            z_scores=z_scores,
            mean_dgs=mean_dgs,
            mean_kls=mean_kls,
        ))

    # Plot Figure 1: z-score vs realized DG
    _, ax = plt.subplots(1, 1, figsize=(10, 7))

    srl_colors = np.concatenate([
      plt.cm.Blues(np.linspace(0.4, 0.8, 4)),  # type: ignore
      plt.cm.Greens(np.linspace(0.4, 0.8, 4))  # type: ignore
    ])

    dualga_colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(dualga_configs)))  # type: ignore

    for i, r in enumerate([res for res in results if res.type == "SRL"]):
        ax.scatter(r.mean_dgs, r.z_scores, label=r.label,
                   marker='o', color=srl_colors[i], alpha=0.6, s=20)

    for i, r in enumerate([res for res in results if res.type == "DualGA"]):
        ax.scatter(r.mean_dgs, r.z_scores, label=r.label,
                   marker='o', color=dualga_colors[i], alpha=0.6, s=20)

    ax.set_xlabel("Realized DG", fontsize=13)
    ax.set_ylabel("Z-score", fontsize=13)
    ax.legend(fontsize=13, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure1_replication.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure1_replication.png")


if __name__ == "__main__":
    run_experiment(
        model_name="facebook/opt-350m",
        n_prompts=20,
        max_new_tokens=200,
        min_prompt_tokens=50
    )
