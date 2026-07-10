import gradio as gr
from src.tokenizer import SanskritBPETokenizer
import os
import random

# Initialize tokenizer
tokenizer = SanskritBPETokenizer(
    merges_path='data/vocab',
    token_path='data/vocab'
)

def generate_color(token_id: int) -> str:
    """Generate a consistent color for a token ID"""
    random.seed(token_id)  # Make color consistent for same token
    hue = random.randint(0, 360)
    return f"hsl({hue}, 80%, 80%)"

def colorize_tokens(text: str) -> str:
    """Convert text to HTML with colored token spans"""
    if not text.strip():
        return ""
        
    tokens = tokenizer.encode(text)
    decoded_pieces = []
    
    for i, token_id in enumerate(tokens):
        decoded_text = tokenizer.decode([token_id])
        color = generate_color(token_id)
        span = f'<span style="background-color: {color}; color: black; padding: 0 2px; border-radius: 3px; margin: 0 1px;" title="Token {token_id}">{decoded_text}</span>'
        decoded_pieces.append(span)
    
    return "".join(decoded_pieces)

def count_tokens(text: str, show_tokens: bool = False) -> tuple:
    """Count tokens and return token visualization"""
    if not text.strip():
        return "0 tokens", ""
        
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    
    if show_tokens:
        decoded = tokenizer.decode(tokens)
        token_info = f"{token_count} tokens\nTokens: {tokens}\nDecoded: {decoded}"
    else:
        token_info = f"{token_count} tokens"
        
    colored_text = colorize_tokens(text)
    return token_info, colored_text

# Custom CSS for better visualization
custom_css = """
footer {visibility: hidden}
.token-text {
    font-family: monospace;
    line-height: 1.8;
    padding: 10px;
    border-radius: 5px;
    background: white;
    margin: 10px 0;
    color: black;
}
.gradio-container {
    max-width: 1000px !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(
        """
        # Sanskrit BPE Tokenizer
        
        Test how the Sanskrit BPE tokenizer processes text. Enter Sanskrit text below to see how many tokens it uses.
        Each colored span represents one token.
        """
    )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Content",
                placeholder="Enter Sanskrit text here...",
                lines=5
            )
            show_tokens = gr.Checkbox(
                label="Show token IDs and decoded text",
                value=False
            )
        
        with gr.Column():
            token_count = gr.Textbox(
                label="Token count",
                lines=2,
                interactive=False
            )
            token_viz = gr.HTML(
                label="Token visualization",
                elem_classes=["token-text"]
            )
    
    # Update token count and visualization when text changes or checkbox is toggled
    text_input.change(
        fn=count_tokens,
        inputs=[text_input, show_tokens],
        outputs=[token_count, token_viz]
    )
    show_tokens.change(
        fn=count_tokens,
        inputs=[text_input, show_tokens],
        outputs=[token_count, token_viz]
    )

    gr.Markdown(
        """
        ### Examples
        Try these Sanskrit text samples:
        """
    )
    
    gr.Examples(
        examples=[
            ["विश्वामित्रवचः श्रुत्वा राघवः सहलक्ष्मणः।"],
            ["धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।"],
            ["यदा यदा हि धर्मस्य ग्लानिर्भवति भारत।"],
        ],
        inputs=text_input
    )

    gr.Markdown(
        """
        ---
        Built with [Gradio](https://gradio.app) | [GitHub Repository](https://github.com/PRIYE/SanskritBPETokenizer)
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch() 