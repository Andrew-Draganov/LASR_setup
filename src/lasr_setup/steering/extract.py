"""Core logic for extracting steering vectors from model activations."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset(path: str) -> list[dict]:
    """Load a JSONL dataset with text and label fields."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def get_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer: int,
) -> torch.Tensor:
    """Extract activations at a specific layer for a list of texts.

    Returns the mean activation across tokens for each text.
    """
    activations = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get activations at specified layer, mean over sequence length
        hidden = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
        mean_activation = hidden.mean(dim=1).squeeze(0)  # (hidden_dim,)
        activations.append(mean_activation)

    return torch.stack(activations)  # (num_samples, hidden_dim)


def project_to_orthogonal_subspace(
    activations: torch.Tensor,
    existing_vectors: list[torch.Tensor],
) -> torch.Tensor:
    """Project activations to the subspace orthogonal to existing steering vectors."""
    # Use fp32 for numerical stability
    original_dtype = activations.dtype
    activations = activations.float()

    for v in existing_vectors:
        v = v.to(activations.device).float()
        v = v / v.norm()
        # a_proj = a - (a · v) * v
        dots = activations @ v  # (num_samples,)
        activations = activations - dots.unsqueeze(1) * v.unsqueeze(0)

    return activations.to(original_dtype)


def extract_steering_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: list[dict],
    layer: int,
    existing_vectors: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Extract a steering vector for a binary classification task.

    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: List of dicts with 'text' and 'label' (0 or 1) fields
        layer: Which layer to extract activations from
        existing_vectors: Previously found steering vectors to be orthogonal to

    Returns:
        A steering vector (the direction separating label=0 from label=1)
    """
    if existing_vectors is None:
        existing_vectors = []

    # Split by label
    texts_0 = [s["text"] for s in dataset if s["label"] == 0]
    texts_1 = [s["text"] for s in dataset if s["label"] == 1]

    print(f"  Processing {len(texts_0)} negative samples...")
    activations_0 = get_activations(model, tokenizer, texts_0, layer)

    print(f"  Processing {len(texts_1)} positive samples...")
    activations_1 = get_activations(model, tokenizer, texts_1, layer)

    # Project to orthogonal subspace if we have existing vectors
    if existing_vectors:
        print(f"  Projecting to subspace orthogonal to {len(existing_vectors)} existing vectors...")
        activations_0 = project_to_orthogonal_subspace(activations_0, existing_vectors)
        activations_1 = project_to_orthogonal_subspace(activations_1, existing_vectors)

    # Steering vector = mean(positive) - mean(negative)
    mean_0 = activations_0.mean(dim=0)
    mean_1 = activations_1.mean(dim=0)
    steering_vector = mean_1 - mean_0

    # Normalize
    steering_vector = steering_vector / steering_vector.norm()

    return steering_vector


def extract_multiple_steering_vectors(
    model_name: str,
    dataset_path: str,
    layer: int,
    num_vectors: int,
    output_dir: str,
) -> list[torch.Tensor]:
    """Extract multiple orthogonal steering vectors.

    Args:
        model_name: HuggingFace model name (e.g., 'Qwen/Qwen3-0.6B')
        dataset_path: Path to JSONL dataset
        layer: Which layer to extract from
        num_vectors: How many steering vectors to find
        output_dir: Where to save the vectors

    Returns:
        List of steering vectors
    """
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model = model.cuda()
    model.eval()

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    steering_vectors = []

    for i in range(num_vectors):
        print(f"\nExtracting steering vector {i + 1}/{num_vectors}...")

        vector = extract_steering_vector(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layer=layer,
            existing_vectors=steering_vectors,
        )

        steering_vectors.append(vector)

        # Save this vector
        save_path = output_path / f"steering_vector_{i}_layer{layer}.pt"
        torch.save(vector.cpu(), save_path)
        print(f"  Saved to {save_path}")

    return steering_vectors
