"""
MAS-Orchestra Harmony Models — HuggingFace Space
"""

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "Harmony GRPO-7B (step 180)": "alvinming/harmony-grpo-7b-global-step-180",
    "Harmony Medium GRPO-7B — BrowseComp+ (step 140)": "alvinming/harmony-medium-grpo-7b-browse-comp-plus-global-step-140",
    "Harmony Medium GRPO-7B — HotpotQA (step 250)": "alvinming/harmony-medium-grpo-7b-hotpot-global-step-250",
}

# pre-load tokenizers (CPU, cheap)
tokenizers = {}
for name, repo_id in MODELS.items():
    tokenizers[repo_id] = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)


@spaces.GPU
def generate(prompt: str, model_name: str, max_tokens: int, temperature: float):
    repo_id = MODELS[model_name]
    tokenizer = tokenizers[repo_id]

    # load model fresh onto GPU each call (ZeroGPU releases between calls)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature) if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── light theme ──────────────────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

css = """
#header { text-align: center; margin-bottom: 0.5rem; }
#header h1 { font-size: 1.6rem; font-weight: 700; color: #1e293b; }
#header p  { color: #64748b; font-size: 0.95rem; }
"""

# ── layout ───────────────────────────────────────────────────────────
with gr.Blocks(theme=theme, css=css, title="MAS-Orchestra Harmony") as demo:

    gr.HTML(
        '<div id="header">'
        "<h1>MAS-Orchestra &middot; Harmony Models</h1>"
        "<p>Multi-agent reasoning checkpoints trained with GRPO</p>"
        "</div>"
    )

    with gr.Row(equal_height=True):

        # left panel — inputs
        with gr.Column(scale=1):
            model_name = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="Model",
            )
            prompt = gr.Textbox(
                lines=6,
                placeholder="Enter your question or reasoning problem...",
                label="Prompt",
            )
            with gr.Row():
                max_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            run_btn = gr.Button("Generate", variant="primary")

        # right panel — output
        with gr.Column(scale=1):
            output = gr.Textbox(label="Response", lines=18, interactive=False, show_copy_button=True)

    run_btn.click(generate, inputs=[prompt, model_name, max_tokens, temperature], outputs=output)
    prompt.submit(generate, inputs=[prompt, model_name, max_tokens, temperature], outputs=output)

    gr.Markdown(
        "<center style='color:#94a3b8; font-size:0.8rem; margin-top:1rem;'>"
        "Built with MAS-Orchestra &middot; Salesforce Research"
        "</center>"
    )

demo.queue()
demo.launch()
