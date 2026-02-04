"""
MAS-Orchestra Harmony Models — HuggingFace Space
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "Harmony GRPO-7B (step 180)": "rychin/harmony-grpo-7b-global-step-180",
    "Harmony Medium GRPO-7B — BrowseComp+ (step 140)": "rychin/harmony-medium-grpo-7b-browse-comp-plus-global-step-140",
    "Harmony Medium GRPO-7B — HotpotQA (step 250)": "rychin/harmony-medium-grpo-7b-hotpot-global-step-250",
}

# Load default model at startup
DEFAULT_REPO = "rychin/harmony-grpo-7b-global-step-180"
print(f"Loading tokenizer from {DEFAULT_REPO}...")
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_REPO, trust_remote_code=True)

print(f"Loading model from {DEFAULT_REPO}...")
model = AutoModelForCausalLM.from_pretrained(
    DEFAULT_REPO,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,  # Stream to GPU without staging in RAM
)
print("Model loaded and ready!")

# Cache for other models if user switches
loaded_models = {DEFAULT_REPO: model}


def get_model(repo: str):
    """Get model from cache or load it."""
    if repo not in loaded_models:
        print(f"Loading model from {repo}...")
        loaded_models[repo] = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    return loaded_models[repo]


def generate(prompt, model_name, temperature):
    repo = MODELS[model_name]
    current_model = get_model(repo)

    # Check if prompt is already formatted (API mode)
    if "<|im_start|>" in prompt:
        text = prompt
    else:
        # UI mode: apply chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = tokenizer(text, return_tensors="pt").to(current_model.device)

    with torch.no_grad():
        outputs = current_model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=float(temperature) if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(result, skip_special_tokens=True)


# UI
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(theme=theme, title="MAS-Orchestra Harmony") as demo:

    gr.Markdown("# MAS-Orchestra · Harmony Models\nMulti-agent reasoning checkpoints trained with GRPO")

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(list(MODELS.keys()), value=list(MODELS.keys())[0], label="Model")
            prompt_box = gr.Textbox(lines=6, placeholder="Enter your question...", label="Prompt")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_box = gr.Textbox(lines=18, label="Response", show_copy_button=True)

    btn.click(generate, [prompt_box, model_dropdown, temperature], output_box)
    prompt_box.submit(generate, [prompt_box, model_dropdown, temperature], output_box)

demo.queue()
demo.launch(show_error=True)
