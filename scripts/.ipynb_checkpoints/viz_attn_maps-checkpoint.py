import torch 
# Save using sns heatmap 
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_attn_map(path: str) -> torch.Tensor:
    state_dict = torch.load(path, map_location="cpu") 
    attn_map = state_dict['attention']
    return attn_map 


prompt_keys = ["language", "object", "factual", "math"]
alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
models = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
state_dict_file_format = "outputs/attention/{model_name}_{prompt_key}_alpha{alpha:.2f}.pth"


def generate_one_attn_heatmap(attn_map: torch.Tensor) -> torch.Tensor:
    num_tokens_generated = len(attn_map)
    total_tokens = attn_map[-1][-1].shape[-1]
    total_heads = attn_map[-1][-1].shape[1]
    num_layers = len(attn_map[0])
    midpoint_layer_idx = num_layers // 2 
    start_heatmap = torch.zeros((total_heads, total_tokens, total_tokens))
    end_heatmap = torch.zeros((total_heads, total_tokens, total_tokens))
    midpoint_heatmap = torch.zeros((total_heads, total_tokens, total_tokens))
    # in attn map the indexing is [tokens, layer, heads, query, key] 
    for h in range(total_heads):
        # Fill the initial attention scores 
        start_layer_attn_scores = attn_map[0][0][0][h].squeeze(0)
        end_layer_attn_scores = attn_map[0][-1][0][h].squeeze(0)
        midpoint_layer_attn_scores = attn_map[0][midpoint_layer_idx][0][h].squeeze(0)
        start_heatmap[h, :len(start_layer_attn_scores), :len(start_layer_attn_scores)] = start_layer_attn_scores
        end_heatmap[h, :len(end_layer_attn_scores), :len(end_layer_attn_scores)] = end_layer_attn_scores
        midpoint_heatmap[h, :len(midpoint_layer_attn_scores), :len(midpoint_layer_attn_scores)] = midpoint_layer_attn_scores


    for i in range(1, num_tokens_generated):
        for h in range(total_heads):
            # start layer heat map 
            start_layer_attn_scores = attn_map[i][0][0][h].squeeze(0)
            # end layer heat map 
            end_layer_attn_scores = attn_map[i][-1][0][h].squeeze(0)
            # midpoint layer heat map 
            midpoint_layer_attn_scores = attn_map[i][midpoint_layer_idx][0][h].squeeze(0)
            start_heatmap[h, i, :len(start_layer_attn_scores)] = start_layer_attn_scores
            end_heatmap[h, i, :len(end_layer_attn_scores)] = end_layer_attn_scores
            midpoint_heatmap[h, i, :len(midpoint_layer_attn_scores)] = midpoint_layer_attn_scores
    
    # save the heatmaps 
    os.makedirs("outputs/heatmaps-materialized", exist_ok=True)
    torch.save({
        "start_heatmap": start_heatmap,
        "end_heatmap": end_heatmap,
        "midpoint_heatmap": midpoint_heatmap
    }, f"outputs/heatmaps-materialized/{model_name}_{prompt_key}_alpha{alpha:.2f}.pt")
    return start_heatmap, end_heatmap, midpoint_heatmap


if __name__ == "__main__":
    os.makedirs("outputs/heatmaps", exist_ok=True)
    
    for model_name in models:
        for prompt_key in prompt_keys:
            for alpha in alpha_values:
                path = f"outputs/attention/{model_name}_{prompt_key}_alpha{alpha:.2f}.pt"
                attn_map = load_attn_map(path)
                start_heatmap, end_heatmap, midpoint_heatmap = generate_one_attn_heatmap(attn_map)
                
                # Get number of attention heads
                num_heads = start_heatmap.shape[0]
                
                for h in range(num_heads):
                    print(f"Generating for model: {model_name}, prompt: {prompt_key}, alpha: {alpha:.2f}, head: {h}")
                    # Check if the heatmap file already exists
                    if os.path.exists(f"outputs/heatmaps/{model_name}_{prompt_key}_alpha{alpha:.2f}_head{h}_start_layer.png"):
                        print(f"Heatmap for model: {model_name}, prompt: {prompt_key}, alpha: {alpha:.2f}, head: {h} already exists")
                    else:
                        # Plot start layer heatmap
                        plt.figure(figsize=(10,8))
                        sns.heatmap(start_heatmap[h], cmap="YlGnBu")
                        plt.title(f"Start Layer - Model: {model_name}, Prompt: {prompt_key}, Alpha: {alpha:.2f}, Head: {h}")
                        plt.xlabel("Keys")
                        plt.ylabel("Query")
                        plt.tight_layout()
                        plt.savefig(f"outputs/heatmaps/{model_name}_{prompt_key}_alpha{alpha:.2f}_head{h}_start_layer.png")
                        plt.close()
                    
                    if os.path.exists(f"outputs/heatmaps/{model_name}_{prompt_key}_alpha{alpha:.2f}_head{h}_end_layer.png"):
                        print(f"Heatmap for model: {model_name}, prompt: {prompt_key}, alpha: {alpha:.2f}, head: {h} already exists")
                    else:
                        # Plot end layer heatmap
                        plt.figure(figsize=(10,8))
                        sns.heatmap(end_heatmap[h], cmap="YlGnBu")
                        plt.title(f"End Layer - Model: {model_name}, Prompt: {prompt_key}, Alpha: {alpha:.2f}, Head: {h}")
                        plt.xlabel("Keys")
                        plt.ylabel("Query")
                        plt.tight_layout()
                        plt.savefig(f"outputs/heatmaps/{model_name}_{prompt_key}_alpha{alpha:.2f}_head{h}_end_layer.png")
                        plt.close()
                    

                    if os.path.exists(f"outputs/heatmaps/{model_name}_{prompt_key}_alpha{alpha:.2f}_head{h}_midpoint_layer.png"):
                        print(f"Heatmap for model: {model_name}, prompt: {prompt_key}, alpha: {alpha:.2f}, head: {h} already exists")
                    else:
                        # Plot middle layer heatmap
                        plt.figure(figsize=(10,8))
                        sns.heatmap(midpoint_heatmap[h], cmap="YlGnBu")
                        plt.title(f"Middle Layer - Model: {model_name}, Prompt: {prompt_key}, Alpha: {alpha:.2f}, Head: {h}")
                        plt.xlabel("Keys")
                        plt.ylabel("Query")
                        plt.tight_layout()
                        plt.savefig(f"outputs/heatmaps/{model_name}_{prompt_key}_alpha{alpha:.2f}_head{h}_midpoint_layer.png")
                        plt.close()
                    
                # Free up memory
                torch.cuda.empty_cache()












