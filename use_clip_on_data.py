from pathlib import Path
import os

from torch.utils.data import Dataset
from dotenv import load_dotenv
from tqdm import tqdm

from clip import get_clip_with_processor, query_clip
from diffae.dataset import CelebADataset
from most_common_words import get_mscoco


def _create_file_if_not_exists(
        file_path: str,
        columns: list[str],
        delimiter: str = ","
        ) -> None:

    path = Path(file_path)
    if not path.exists():
        with open(file_path, "w", -1, "utf-8") as file:
            print(delimiter.join(columns), file=file)


def _add_record_to_csv(file_path: str, record: str) -> None:
    with open(file_path, "a", -1, "utf-8") as file:
        print(record, file=file)


def use_clip_on_dataset(
        dataset: Dataset,
        words: list[str],
        device: str
        ) -> list[float]:

    clip, processor = get_clip_with_processor(
        "openai/clip-vit-base-patch32",
        "flash_attention_2",
        device
    )

    for image, _, image_id in dataset:
        probs = query_clip(image, words, clip, processor, device)
        yield image_id, probs[0]


if __name__ == "__main__":
    load_dotenv()
    os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE")
    device = "cuda"

    dataset = CelebADataset(
        os.getenv("CELEBA_DATA_PATH"),
        os.getenv("CELEBA_LABELS_PATH")
    )
    results_path = "results.csv"
    words = get_mscoco()[:50]
    _create_file_if_not_exists(results_path, ["image_id"] + words)

    tqdm_process = tqdm(
        use_clip_on_dataset(dataset, words, device),
        desc="Processing images",
        unit="image",
        total=len(dataset)
    )

    records_to_add = []
    for image_id, probs in tqdm_process:
        probs = [image_id.split(".")[0]] + [str(float(prob)) for prob in probs]
        records_to_add.append(",".join(probs))
        if len(records_to_add) == 10_000:
            new_records = "\n".join(records_to_add)
            records_to_add = []
            _add_record_to_csv(results_path, new_records)

    if len(records_to_add) > 0:
        new_records = "\n".join(records_to_add)
        records_to_add = []
        _add_record_to_csv(results_path, new_records)
