import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """
    Load the Qwen2.5-7B-Instruct model and tokenizer from Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder="./offload",
        offload_state_dict=True,
    )
    return tokenizer, model


def pad_right_align(input_ids: torch.Tensor, pad_id: int, max_length: int) -> torch.Tensor:
    """
    Right-align a sequence of input_ids by padding with pad_id on the left.
    input_ids: (seq_len,)
    Returns: (max_length,)
    """
    seq_len = input_ids.size(0)
    if seq_len >= max_length:
        return input_ids[-max_length:]
    pad_len = max_length - seq_len
    pad = torch.full((pad_len,), pad_id, dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([pad, input_ids], dim=0)


def interpolate_prompts(
    tokenizer,
    model,
    prompt_a: str,
    prompt_b: str,
    alpha: float = 0.5,
    max_new_tokens: int = 100,
    temperature: float = 1,
):
    """
    Interpolate between two prompts by mixing their embeddings with parameter alpha.
    Right-align prompts using bos_token_id to pad on the left.
    """
    # Tokenize without special tokens
    ids_a = tokenizer(prompt_a, add_special_tokens=False)["input_ids"]
    ids_b = tokenizer(prompt_b, add_special_tokens=False)["input_ids"]

    tensor_a = torch.tensor(ids_a, device=model.device)
    tensor_b = torch.tensor(ids_b, device=model.device)

    # Determine max length for alignment
    max_len = max(tensor_a.size(0), tensor_b.size(0))
    pad_id = tokenizer.bos_token_id or tokenizer.pad_token_id

    # Right-align
    aligned_a = pad_right_align(tensor_a, pad_id, max_len)
    aligned_b = pad_right_align(tensor_b, pad_id, max_len)

    # Get embeddings
    emb_a = model.model.embed_tokens(aligned_a.unsqueeze(0))  # (1, max_len, d)
    emb_b = model.model.embed_tokens(aligned_b.unsqueeze(0))  # (1, max_len, d)

    # Interpolate
    emb_mix = (alpha) * emb_a + (1 - alpha) * emb_b

    # Prepare generation config
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
    )

    # Generate using inputs_embeds
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=emb_mix,
            generation_config=gen_config,
            attention_mask=(emb_mix.abs().sum(-1) != 0).long(),
            return_dict_in_generate=True,
            output_attentions=True,
        )
        attentions = outputs['attentions'] # attn map for each layer
    # Decode and strip any leading special tokens
    text = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
    return text, attentions, max_len


def main():
    import csv
    import os
    from datetime import datetime
    models = [
        "Qwen/Qwen3-1.7B", 
        "Qwen/Qwen3-4B", 
        "Qwen/Qwen3-8B", 
        "Qwen/Qwen3-14B",
    ]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    prompt_pairs = {
        "language": ("Explain transformer attention in english", "Explain transformer attention in french"), # Language
        "object": ("Explain why the sky is blue", "Explain why the water is blue"), # Object
        "factual": ("Explain why the elephant is pink", "Explain why the elephant is gray"), # factual  
        "math": ("What is 2 + 2", "What is 10 / 2"), # Math

    }
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/generations", exist_ok=True)
    os.makedirs("outputs/attention", exist_ok=True)

    attention_results = {}
    metadata = []

    for model_name in models:
        print(f"Processing model: {model_name}")
        model_short_name = model_name.split('/')[-1]
        tokenizer, model = load_model(model_name)
        
        for prompt_type, (prompt_a, prompt_b) in prompt_pairs.items():
            print(f"Processing prompt pair: {prompt_type}")
            
            # Create CSV file for this model and prompt pair
            csv_file = f"outputs/generations/{model_short_name}_{prompt_type}.csv"
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['prompt_a', 'prompt_b', 'alpha', 'response'])
                
                for alpha in alphas:
                    response, last_layer_attention, num_input_tokens = interpolate_prompts(
                        tokenizer, model, prompt_a, prompt_b, alpha
                    )
                    
                    # Save main results to CSV
                    writer.writerow([prompt_a, prompt_b, f"{alpha:.2f}", response])
                    
                    # Save attention for this specific combination
                    attention_file = f"outputs/attention/{model_short_name}_{prompt_type}_alpha{alpha:.2f}.pt"
                    torch.save({
                        'attention': last_layer_attention,
                        'num_input_tokens': num_input_tokens,
                        'model': model_name,
                        'prompt_type': prompt_type,
                        'alpha': alpha,
                        'prompt_a': prompt_a,
                        'prompt_b': prompt_b
                    }, attention_file)
                    
                    print(f"Alpha={alpha:.2f}\n{response}\n{'-'*40}")
            
            # Free up memory after each prompt pair
            torch.cuda.empty_cache()
        
        # Free up memory after each model
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":

    main()
