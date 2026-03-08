import torch
from rich.console import Console
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermarking import get_green_list, generate_srl


def generate_normal(model, tokenizer, input_ids, max_new_tokens):
    """Generate normal text without watermarking."""
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated


def highlight_tokens(tokenizer, tokens, prompt_length, gamma, vocab_size):
    """Create rich Text object with green tokens highlighted."""
    text = Text()

    for i in range(prompt_length, len(tokens)):
        if i == prompt_length:
            continue  # Skip first generated token (no previous context)

        prev_token = tokens[i - 1]
        current_token = tokens[i]
        token_text = tokenizer.decode([current_token])

        green_set = get_green_list(prev_token, vocab_size, gamma)
        is_green = current_token in green_set

        if is_green:
            text.append(token_text, style="bold green")
        else:
            text.append(token_text, style="dim")

    return text


def main():
    console = Console()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/opt-350m"

    console.print(f"Loading {model_name}...", style="blue")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]
    max_new_tokens = 100
    gamma = 0.25
    vocab_size = model.config.vocab_size

    console.print(f"\nPrompt: {prompt}", style="bold")
    console.print("=" * 50)

    # Normal generation
    console.print("\nNormal generation:", style="yellow")
    normal_tokens = generate_normal(model, tokenizer, input_ids, max_new_tokens)
    normal_text = highlight_tokens(tokenizer, normal_tokens[0].tolist(), prompt_length, gamma, vocab_size)
    console.print(normal_text)

    # Count green tokens for normal
    normal_token_list = normal_tokens[0].tolist()
    normal_green_count = 0
    normal_total = 0

    for i in range(prompt_length + 1, len(normal_token_list)):
        prev_token = normal_token_list[i - 1]
        current_token = normal_token_list[i]
        green_set = get_green_list(prev_token, vocab_size, gamma)
        if current_token in green_set:
            normal_green_count += 1
        normal_total += 1

    normal_ratio = normal_green_count / normal_total if normal_total > 0 else 0
    console.print(f"Green: {normal_green_count}/{normal_total} ({normal_ratio:.1%})", style="dim")

    # Watermarked generation
    console.print("\nWatermarked generation:", style="yellow")
    wm_tokens, _, _, _ = generate_srl(model, tokenizer, input_ids, max_new_tokens, gamma, delta=2.0, device=device)
    wm_text = highlight_tokens(tokenizer, wm_tokens[0].tolist(), prompt_length, gamma, vocab_size)
    console.print(wm_text)

    # Count green tokens for watermarked
    wm_token_list = wm_tokens[0].tolist()
    wm_green_count = 0
    wm_total = 0

    for i in range(prompt_length + 1, len(wm_token_list)):
        prev_token = wm_token_list[i - 1]
        current_token = wm_token_list[i]
        green_set = get_green_list(prev_token, vocab_size, gamma)
        if current_token in green_set:
            wm_green_count += 1
        wm_total += 1

    wm_ratio = wm_green_count / wm_total if wm_total > 0 else 0
    console.print(f"Green: {wm_green_count}/{wm_total} ({wm_ratio:.1%})", style="dim")

    console.print(f"\nExpected green ratio: {gamma:.1%}", style="blue")


if __name__ == "__main__":
    main()
