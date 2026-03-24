"""
shared_utils.py
Shared helpers for assistant-axis projection scripts.
"""

import json
import os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

from assistant_axis import load_axis, get_config
from assistant_axis.internals import ProbingModel, ConversationEncoder, SpanMapper


def init_env():
    """Load .env and login to HuggingFace."""
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in .env — get one at https://huggingface.co/settings/tokens")
    # huggingface_hub respects HF_TOKEN env var automatically
    os.environ["HF_TOKEN"] = token

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B")
    model_short = os.environ.get("MODEL_SHORT", "qwen-3-32b")
    repo_id = os.environ.get("REPO_ID", "lu-christina/assistant-axis-vectors")
    return model_name, model_short, repo_id


def load_model_and_axis(model_name, model_short, repo_id):
    """Load the ProbingModel, config, and normalized axis vector."""
    config = get_config(model_name)
    target_layer = config["target_layer"]
    print(f"Model: {model_name}  |  Target layer: {target_layer}")

    print("Loading model...")
    pm = ProbingModel(model_name)
    print("Model loaded!")

    axis_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{model_short}/assistant_axis.pt",
        repo_type="dataset",
    )
    axis = load_axis(axis_path)
    axis_vec = F.normalize(axis[target_layer].float(), dim=0)
    print(f"Axis shape: {axis.shape}  |  Axis vector shape: {axis_vec.shape}")

    return pm, axis_vec, target_layer


def compute_trajectory(pm, conversation, axis_vec, layer):
    """
    Compute projection for each assistant turn in a conversation.
    Returns a numpy array of projection values.
    """
    encoder = ConversationEncoder(pm.tokenizer)
    span_mapper = SpanMapper(pm.tokenizer)

    mean_acts = span_mapper.mean_all_turn_activations(pm, encoder, conversation, layer=layer)
    assistant_acts = mean_acts[1::2].float()
    assistant_acts_normalized = F.normalize(assistant_acts, dim=-1)
    projections = (assistant_acts_normalized @ axis_vec).numpy()
    return projections


def compute_projection_pair(pm, entry, axis_vec, target_layer):
    """Compute aligned and misaligned projections for a single paired entry."""
    aligned_proj = compute_trajectory(pm, entry["aligned"]["conversation"], axis_vec, target_layer)
    misaligned_proj = compute_trajectory(pm, entry["misaligned"]["conversation"], axis_vec, target_layer)
    return aligned_proj, misaligned_proj