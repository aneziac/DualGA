import torch
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm

from models import SRLConfig, DualGAConfig, ExperimentResult


def get_green_list(token: int, vocab_size: int, gamma: float, salt_key: int = 35317, seed: int = 0) -> set[int]:
    """Compute green list via pseudorandom partition seeded by previous token."""
    rng = torch.Generator()
    rng.manual_seed((seed * salt_key + token) % (2**63))  # torch generator needs non-negative int64
    perm = torch.randperm(vocab_size, generator=rng)
    green_size = int(gamma * vocab_size)
    return set(perm[:green_size].tolist())

def generate_srl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: float,
    delta: float,
    device: torch.device
) -> tuple[torch.Tensor, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate watermarked text using SRL (Kirchenbauer et al.)"""
    model.eval()
    generated = input_ids.clone()
    vocab_size = model.config.vocab_size

    kl_values = []
    dg_values = []
    green_hits = []

    with torch.no_grad():
        past_key_values = None
        next_input = generated

        for _ in range(max_new_tokens):
            outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # 3. Get green list based on previous token
            prev_token = int(generated[0, -1:].item())
            green_set = get_green_list(prev_token, vocab_size, gamma)
            green_mask = torch.zeros(vocab_size, device=device)
            green_indices = torch.tensor(list(green_set), device=device, dtype=torch.long)
            green_mask[green_indices] = 1.0

            # p_t: original probabilities
            original_probs = torch.softmax(logits, dim=-1).squeeze(0)

            # Add delta to green list logits
            watermarked_logits = logits.clone()
            watermarked_logits[0, green_indices] += delta

            # 4. Watermarked probabilities
            wm_probs = torch.softmax(watermarked_logits, dim=-1).squeeze(0)

            # Compute DG_t = sum_{v in G} q_{t,v} - sum_{v in G} p_{t,v}
            green_prob_original = (original_probs * green_mask).sum().item()
            green_prob_wm = (wm_probs * green_mask).sum().item()
            dg_t = green_prob_wm - green_prob_original
            dg_values.append(dg_t)

            # Compute KL(q_t || p_t)
            kl_t = (wm_probs * (torch.log(wm_probs + 1e-30) - torch.log(original_probs + 1e-30))).sum().item()
            kl_values.append(kl_t)

            # 5. Sample from watermarked distribution
            next_token = torch.multinomial(wm_probs.unsqueeze(0), num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            next_input = next_token

            # Track if sampled token is green
            is_green = next_token.item() in green_set
            green_hits.append(1.0 if is_green else 0.0)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated, np.array(green_hits), np.array(dg_values), np.array(kl_values)

def generate_dualga(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: float,
    D_target: float,
    init_lambda: float,
    autoeta: float,
    device: torch.device
) -> tuple[torch.Tensor, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate watermarked text using DualGA."""
    model.eval()
    generated = input_ids.clone()
    vocab_size = model.config.vocab_size

    kl_values = []
    dg_values = []
    green_hits = []
    lambda_t = init_lambda

    with torch.no_grad():
        past_key_values = None
        next_input = generated

        for t in range(max_new_tokens):
            outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # 3. Get green list based on previous token
            prev_token = int(generated[0, -1:].item())
            green_set = get_green_list(prev_token, vocab_size, gamma)
            green_mask = torch.zeros(vocab_size, device=device)
            green_indices = torch.tensor(list(green_set), device=device, dtype=torch.long)
            green_mask[green_indices] = 1.0

            # p_t: original probabilities
            original_probs = torch.softmax(logits, dim=-1).squeeze(0)

            # g = sum of original probs on green list
            g = (original_probs * green_mask).sum()

            # 4. delta = lambda
            delta_t = max(0.0, min(lambda_t, 15.0))

            # Add delta to green list logits
            watermarked_logits = logits.clone()
            watermarked_logits[0, green_indices] += delta_t

            # 5. Watermarked probabilities
            wm_probs = torch.softmax(watermarked_logits, dim=-1).squeeze(0)

            # Compute DG_t
            green_prob_original = (original_probs * green_mask).sum().item()
            green_prob_wm = (wm_probs * green_mask).sum().item()
            dg_t_val = green_prob_wm - green_prob_original
            dg_values.append(dg_t_val)

            # Compute KL(q_t || p_t)
            kl_t = (wm_probs * (torch.log(wm_probs + 1e-30) - torch.log(original_probs + 1e-30))).sum().item()
            kl_values.append(kl_t)

            # 6. Sample from watermarked distribution
            next_token = torch.multinomial(wm_probs.unsqueeze(0), num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            next_input = next_token

            # Track if sampled token is green
            is_green = next_token.item() in green_set
            green_hits.append(1.0 if is_green else 0.0)

            # DG_t as a function of delta and g.
            # Not sure if this is explicitly given in the paper but it's in their code
            g_val = g.item()
            delta_tensor = torch.tensor(delta_t)
            g_tensor = torch.tensor(g_val)
            denom = g_tensor * torch.exp(delta_tensor) - g_tensor + 1
            dg_closed = (1 - g_tensor) * (1 - 1 / denom)

            # Gradient: d g_t / d lambda = Delta - DG_t(lambda)
            gradient = D_target - dg_closed.item()

            # Step size: eta = autoeta / sqrt(t+1)
            eta = autoeta / np.sqrt(t + 1)

            # Update lambda (projected gradient ascent)
            lambda_t = max(0.0, min(lambda_t + eta * gradient, 15.0))

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated, np.array(green_hits), np.array(dg_values), np.array(kl_values)

def compute_z_score(green_hits: npt.NDArray[np.float64], gamma: float) -> float:
    """
    Compute z-score for watermark detection.
    z = (|s|_G - gamma * T) / sqrt(T * gamma * (1 - gamma))
    where |s|_G is the number of green tokens.
    """
    T = len(green_hits)
    if T == 0:
        return 0.0
    n_green = np.sum(green_hits)
    expected = gamma * T
    std = np.sqrt(T * gamma * (1 - gamma))
    if std < 1e-10:
        return 0.0
    return (n_green - expected) / std


# ============================================================
# Main experiment
# ============================================================

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
