import torch
import numpy as np
import numpy.typing as npt
from transformers import PreTrainedModel, PreTrainedTokenizer

from models import WatermarkResult


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
) -> WatermarkResult:
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
            next_token = torch.multinomial(wm_probs.unsqueeze(0).cpu(), num_samples=1).to(device)
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
) -> WatermarkResult:
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
            next_token = torch.multinomial(wm_probs.unsqueeze(0).cpu(), num_samples=1).to(device)
            generated = torch.cat([generated, next_token], dim=-1)
            next_input = next_token

            # Track if sampled token is green
            is_green = next_token.item() in green_set
            green_hits.append(1.0 if is_green else 0.0)

            # DG_t as a function of delta and g.
            # Not sure if this is explicitly given in the paper but it's in their code
            g_val = g.item()
            delta_tensor = torch.tensor(delta_t, device=device)
            g_tensor = torch.tensor(g_val, device=device)
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