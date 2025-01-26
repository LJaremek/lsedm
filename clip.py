from transformers import CLIPProcessor, CLIPModel
import torch


def get_clip_with_processor(
        name: str,
        implementation: str,
        device: str
        ) -> tuple[CLIPModel, CLIPProcessor]:

    if device == "cuda":
        torch_dtype = torch.float16

        model = CLIPModel.from_pretrained(
            name,
            attn_implementation=implementation,
            device_map=device,
            torch_dtype=torch_dtype
        )
    else:
        torch_dtype = torch.float32

        model = CLIPModel.from_pretrained(
            name,
            device_map={"": device},
            torch_dtype=torch_dtype
        )

    processor = CLIPProcessor.from_pretrained(name)

    return model, processor


def query_clip(
        image: torch.Tensor,
        text: list[str],
        model: CLIPModel,
        processor: CLIPProcessor,
        device: str
        ) -> list[list[float]]:

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    return logits_per_image.softmax(dim=-1).tolist()


def get_top_n_indices(numbers: list[float], n: int) -> list[int]:
    if n <= 0:
        return []

    return sorted(
        range(len(numbers)),
        key=lambda i: numbers[i],
        reverse=True
    )[:n]
