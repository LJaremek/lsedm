from dataclasses import dataclass

from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer
import torch


@dataclass
class ClipConfig:
    model: str
    processor: str
    tokenizer: str
    device: str = "cpu"


def get_clip(model: str, processor: str, tokenizer: str, device: str) -> tuple:
    model = AutoModel.from_pretrained(model, trust_remote_code=True).to(device)
    processor = AutoFeatureExtractor.from_pretrained(processor)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    return model, processor, tokenizer


def get_clip_from_clip_config(config: ClipConfig) -> tuple:
    model = AutoModel.from_pretrained(
        config.model, trust_remote_code=True
        ).to(config.device)
    processor = AutoFeatureExtractor.from_pretrained(config.processor)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

    return model, processor, tokenizer


def query_clip(
        image: torch.Tensor,
        texts: list[str],
        model,
        processor,
        tokenizer
        ) -> list[list[float]]:

    device = model.device
    image_input = processor(image, return_tensors="pt").to(device)
    text_input = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**image_input, **text_input)
        text_probs = outputs.logits_per_image.softmax(dim=-1)

        return text_probs


def get_top_n_indices(numbers: list[float], n: int) -> list[int]:
    if n <= 0:
        return []

    return sorted(
        range(len(numbers)),
        key=lambda i: numbers[i],
        reverse=True
    )[:n]


class CLIPS:
    laion5b_roberta = ClipConfig(
        "calpt/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "xlm-roberta-base"
    )
