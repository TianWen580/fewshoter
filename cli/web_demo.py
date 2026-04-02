#!/usr/bin/env python3
"""
Gradio Web Demo for Fewshoter - Few-Shot Image Classification

A clean, minimal interface following Anthropic design principles:
- Clear hierarchy and visual flow
- Warm, trustworthy color palette
- Minimal decoration, maximum utility
- Generous whitespace

Usage:
    fewshoter-web-demo --port 7860
"""

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from ..core.config import Config
from ..features.feature_extractor import create_feature_extractor
from ..data.support_manager import SupportSetManager
from ..engine.classifier import create_classifier
from ..core.utils import setup_logging


# Anthropic-inspired warm color palette (module-level for reuse)
ANTHROPIC_THEME = gr.themes.Soft(
    primary_hue="orange",  # Warm coral-like tone
    secondary_hue="neutral",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    # Warm, trustworthy colors
    body_background_fill="*neutral_50",
    background_fill_primary="*neutral_100",
    background_fill_secondary="*neutral_200",
    border_color_primary="*neutral_300",
    # Button styling
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    button_secondary_background_fill="*neutral_200",
    button_secondary_background_fill_hover="*neutral_300",
)


# Preset configurations for common use cases
PRESETS = {
    "General Purpose": {
        "model": "ViT-B/32",
        "device": "auto",
        "description": "Fast and efficient for most tasks",
    },
    "High Accuracy": {
        "model": "ViT-L/14",
        "device": "auto",
        "description": "Larger model for better accuracy",
    },
    "Biology/Species (BioCLIP)": {
        "model": "BioCLIP",
        "device": "auto",
        "description": "Optimized for biodiversity and species classification",
    },
    "Biology/Species (BioCLIP-2)": {
        "model": "BioCLIP-2",
        "device": "auto",
        "description": "Best for biology - trained on 214M images, 952K taxa",
    },
    "Biology/Species (BioCLIP-2.5)": {
        "model": "BioCLIP-2.5",
        "device": "auto",
        "description": "Latest biology model with 18% improvement",
    },
    "Camera Trap Species": {
        "model": "ViT-H-14-quickgelu",
        "device": "auto",
        "description": "Optimized for wildlife camera trap images",
    },
    "Bird Classification": {
        "model": "ViT-L-14-quickgelu",
        "device": "auto",
        "description": "Good for fine-grained bird species recognition",
    },
    "Insect Classification": {
        "model": "ViT-L-14-quickgelu",
        "device": "auto",
        "description": "Optimized for insect and arthropod classification",
    },
}


# Global state
class AppState:
    """Simple state container for the demo"""

    config: Optional[Config] = None
    feature_extractor: Optional[object] = None
    support_manager: Optional[SupportSetManager] = None
    classifier: Optional[object] = None
    prototypes_path: Optional[str] = None
    support_images: Dict[str, List[str]] = {}


state = AppState()


def initialize_system(model_name: str = "ViT-B/32", device: str = "auto") -> str:
    """Initialize the CLIP model and feature extractor"""
    try:
        state.config = Config()
        state.config.model.clip_model_name = model_name
        state.config.model.device = device

        state.feature_extractor = create_feature_extractor(state.config)

        return f"Model loaded: {model_name} on {state.config.model.device}"
    except Exception as e:
        logging.error(f"Failed to initialize: {e}")
        return f"Error: {str(e)}"


def add_support_images(class_name: str, images: List[str]) -> Tuple[str, gr.Gallery]:
    """Add support images for a class"""
    if not images:
        return "No images uploaded", None

    if state.feature_extractor is None:
        return "Please initialize model first", None

    class_name = class_name.strip() or "unknown"

    if class_name not in state.support_images:
        state.support_images[class_name] = []

    temp_dir = Path(tempfile.gettempdir()) / "fewshoter_support"
    temp_dir.mkdir(exist_ok=True)
    class_dir = temp_dir / class_name
    class_dir.mkdir(exist_ok=True)

    for img_path in images:
        img = Image.open(img_path)
        save_path = class_dir / Path(img_path).name
        img.save(save_path)
        state.support_images[class_name].append(str(save_path))

    gallery_data = []
    for cls, paths in state.support_images.items():
        for p in paths:
            gallery_data.append((p, cls))

    return f"Added {len(images)} images to '{class_name}' (total: {len(state.support_images[class_name])})", gallery_data


def build_prototypes() -> str:
    """Build prototypes from support images"""
    if not state.support_images:
        return "No support images added"

    if state.feature_extractor is None:
        return "Please initialize model first"

    try:
        temp_dir = Path(tempfile.gettempdir()) / "fewshoter_support"

        state.support_manager = SupportSetManager(
            support_dir=str(temp_dir),
            feature_extractor=state.feature_extractor,
            config=state.config,
        )
        # Load and process support set to build prototypes
        state.support_manager.load_support_set()
        state.support_manager.process_support_set()

        output_dir = Path(tempfile.gettempdir()) / "fewshoter_outputs"
        output_dir.mkdir(exist_ok=True)
        state.prototypes_path = str(output_dir / "prototypes.json")
        state.support_manager.save_prototypes(state.prototypes_path)

        state.classifier = create_classifier(state.support_manager, state.config)

        class_list = list(state.support_images.keys())
        return f"Prototypes built for {len(class_list)} classes: {', '.join(class_list)}"
    except Exception as e:
        logging.error(f"Failed to build prototypes: {e}")
        return f"Error: {str(e)}"


def classify_image(image: np.ndarray, top_k: int = 5) -> Tuple[gr.Label, str, str]:
    """Classify an image and return results"""
    if state.classifier is None:
        return {}, "Please build prototypes first", ""

    if image is None:
        return {}, "No image provided", ""

    try:
        temp_img = Path(tempfile.gettempdir()) / "fewshoter_query.jpg"
        Image.fromarray(image).save(temp_img)

        predicted_class, details = state.classifier.classify(
            str(temp_img), return_scores=True
        )

        top_preds = state.classifier.get_top_k_predictions(details["all_scores"], k=top_k)

        label_output = {cls: float(score) for cls, score in top_preds}

        confidence_analysis = state.classifier.evaluate_confidence(details["all_scores"])

        confidence_val = details["confidence"]
        confidence_desc = "High" if confidence_val > 0.8 else "Medium" if confidence_val > 0.5 else "Low"

        details_text = f"""
**Prediction:** {predicted_class}

**Confidence:** {confidence_val:.2f} ({confidence_desc})

**Status:** {confidence_analysis.get('status', 'N/A')}

---

**Top {top_k} matches:**
"""
        for cls, score in top_preds:
            details_text += f"\n{cls}: {score:.2%}"

        temp_img.unlink()

        return label_output, details_text, f"{predicted_class}"

    except Exception as e:
        logging.error(f"Classification failed: {e}")
        return {}, f"Error: {str(e)}", ""


def clear_support_set() -> Tuple[str, gr.Gallery]:
    """Clear all support images"""
    state.support_images = {}
    state.support_manager = None
    state.classifier = None

    temp_dir = Path(tempfile.gettempdir()) / "fewshoter_support"
    if temp_dir.exists():
        for item in temp_dir.iterdir():
            if item.is_dir():
                for f in item.iterdir():
                    f.unlink()
                item.rmdir()

    return "Support set cleared", None


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface with Anthropic-inspired design"""

    with gr.Blocks(
        title="Fewshoter",
    ) as demo:
        # Clean header
        gr.Markdown(
            """
            # Fewshoter

            Few-shot image classification. Upload 2-5 examples per class, then classify new images.
            """
        )

        # Step 1: Model Selection
        gr.Markdown("""### Step 1: Choose Model""")

        # Preset selector
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value=None,
                label="Quick Presets",
                info="Select a preset for common use cases",
                scale=2,
            )
            preset_info = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                placeholder="Select a preset to see description...",
                scale=2,
            )

        with gr.Row(equal_height=True):
            model_dropdown = gr.Dropdown(
                choices=[
                    "ViT-B/32",
                    "ViT-B/16",
                    "ViT-L/14",
                    "ViT-L/14@336px",
                    "ViT-H-14-quickgelu",
                    "ViT-L-14-quickgelu",
                    "BioCLIP",
                    "BioCLIP-2",
                    "BioCLIP-2.5",
                ],
                value="ViT-B/32",
                label="CLIP Model",
                info="BioCLIP models are optimized for biology/biodiversity",
                scale=2,
            )
            device_dropdown = gr.Dropdown(
                choices=["auto", "cuda", "mps", "cpu"],
                value="auto",
                label="Device",
                info="Auto selects best available (cuda > mps > cpu)",
                scale=1,
            )
            init_btn = gr.Button(
                "Initialize",
                variant="primary",
                scale=1,
            )

        init_status = gr.Textbox(
            label="",
            show_label=False,
            interactive=False,
            placeholder="Click Initialize to load model...",
        )

        # Preset handler
        def apply_preset(preset_name):
            if preset_name and preset_name in PRESETS:
                preset = PRESETS[preset_name]
                return preset["model"], preset["device"], preset["description"]
            return gr.update(), gr.update(), ""

        preset_dropdown.change(
            apply_preset,
            inputs=[preset_dropdown],
            outputs=[model_dropdown, device_dropdown, preset_info],
        )

        init_btn.click(
            initialize_system,
            inputs=[model_dropdown, device_dropdown],
            outputs=init_status,
        )

        gr.Markdown("")  # Spacing

        # Step 2: Support Set
        gr.Markdown("""### Step 2: Add Examples""")

        with gr.Row():
            with gr.Column(scale=3):
                class_input = gr.Textbox(
                    label="Class name",
                    placeholder="e.g., robin, sparrow, eagle",
                    info="Name for the images you're uploading",
                )

                support_upload = gr.File(
                    label="Upload example images",
                    file_count="multiple",
                    file_types=["image"],
                )

                with gr.Row():
                    add_btn = gr.Button("Add examples", variant="secondary")
                    build_btn = gr.Button("Build prototypes", variant="primary")
                    clear_btn = gr.Button("Clear all", variant="stop")

                support_status = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    placeholder="Add 2-5 images per class...",
                )

            with gr.Column(scale=2):
                support_gallery = gr.Gallery(
                    label="Your examples",
                    columns=3,
                    rows=2,
                    height="auto",
                    show_label=True,
                )

        add_btn.click(
            add_support_images,
            inputs=[class_input, support_upload],
            outputs=[support_status, support_gallery],
        )
        build_btn.click(build_prototypes, outputs=support_status)
        clear_btn.click(clear_support_set, outputs=[support_status, support_gallery])

        gr.Markdown("")  # Spacing

        # Step 3: Classification
        gr.Markdown("""### Step 3: Classify""")

        with gr.Row():
            with gr.Column(scale=2):
                query_image = gr.Image(
                    label="Image to classify",
                    type="numpy",
                    height=350,
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Show top predictions",
                    info="Number of predictions to display",
                )

                classify_btn = gr.Button(
                    "Classify image",
                    variant="primary",
                )

            with gr.Column(scale=1):
                prediction_label = gr.Label(
                    label="Predictions",
                    num_top_classes=5,
                )

                prediction_summary = gr.Textbox(
                    label="Top prediction",
                    show_label=True,
                    interactive=False,
                )

                details_text = gr.Markdown(
                    label="Details",
                )

        classify_btn.click(
            classify_image,
            inputs=[query_image, top_k_slider],
            outputs=[prediction_label, details_text, prediction_summary],
        )

        gr.Markdown("")  # Spacing

        # Minimal footer
        gr.Markdown(
            """
            ---

            Built with [Fewshoter](https://github.com/TianWen580/fewshoter) | CLIP-based few-shot classification

            *No training required. Works with 2-5 examples per class.*
            """
        )

    return demo


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fewshoter Web Demo"
    )

    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address"
    )

    parser.add_argument("--port", type=int, default=7860, help="Port")

    parser.add_argument(
        "--share", action="store_true", help="Create public share link"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main function to run the web demo"""
    args = parse_arguments()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Creating Fewshoter Web Demo...")

    demo = create_demo()

    logger.info(f"Launching on {args.host}:{args.port}")
    if args.share:
        logger.info("Public share link enabled")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=ANTHROPIC_THEME,
        css="""
        .contain { max-width: 1200px !important; margin: auto !important; }
        .block { padding: 24px !important; }
        """,
    )

    return 0


if __name__ == "__main__":
    exit(main())