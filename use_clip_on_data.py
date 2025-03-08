from pathlib import Path
import os

from torch.utils.data import Dataset
from dotenv import load_dotenv
from tqdm import tqdm
import torch

from clip import CLIPS, get_clip_from_clip_config, query_clip
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
        clip_model,
        processor,
        tokenizer,
        template: str = "{X}",
        batch_size: int = 256
        ) -> list[float]:

    words = [
        template.replace("{X}", word)
        for word in words
    ]

    batch_images, batch_ids = [], []
    for image, _, image_id in dataset:
        batch_images.append(image)
        batch_ids.append(image_id)

        if len(batch_images) == batch_size:
            probs = query_clip(
                batch_images, words,
                clip_model, processor, tokenizer
                )
            for i, image_id in enumerate(batch_ids):
                yield image_id, probs[i]

            batch_images, batch_ids = [], []

    if batch_images:
        probs = query_clip(
            torch.stack(batch_images), words,
            clip_model, processor, tokenizer
            )
        for i, image_id in enumerate(batch_ids):
            yield image_id, probs[i]


if __name__ == "__main__":
    load_dotenv()
    device = "cuda"
    images_limit = 29_000
    labels_limit = 1000
    how_often_save = 1_000

    dataset = CelebADataset(
        os.getenv("CELEBA_DATA_PATH"),
        os.getenv("CELEBA_LABELS_PATH"),
        image_size=256,
        images_limit=images_limit
    )
    print("LEN:", len(dataset))
    results_path = "celebahq256_clip_laion5b_mscoco.csv"
    words = get_mscoco()[:labels_limit]
    # categories = [  # better
    #     "man", "woman",
    #     "bald person", "person with long hair", "person with short hair",
    #     "person wiht blond hair", "person wiht black hair",
    #     "person with red hair", "person with brown hair",
    #     "person with glasses", "person without glasses",
    #     "person with hat", "person without hat",
    #     "cat", "dog", "horse", "frog", "ant", "snail", "monkey"
    # ]
    # categories = [
    #     "Man", "Woman",
    #     "Bald", "Curly hair", "Straight hair", "Wavy hair",
    #     "Black hair", "Blonde hair", "Brown hair", "Red hair", "Gray hair",
    #     "Beard", "Mustache", "Sideburns",
    #     "Glasses", "Sunglasses",
    #     "Earrings",
    #     "Necklace", "Scarf",
    #     "Hat", "Cap", "Hood", "Headphones",
    #     "Tie", "Bow tie", "Collar"
    #     ]

    _create_file_if_not_exists(results_path, ["image_id"] + words)

    model_config = CLIPS.laion5b_roberta
    model_config.device = "cuda"
    clip_model, processor, tokenizer = get_clip_from_clip_config(model_config)

    tqdm_process = tqdm(
        use_clip_on_dataset(
            dataset, words,
            clip_model, processor, tokenizer,
            "photo of a {X}",
            1
            ),
        desc="Processing images",
        unit="image",
        total=len(dataset)
    )

    records_to_add = []
    for image_id, probs in tqdm_process:
        probs = [image_id.split(".")[0]] + [str(float(prob)) for prob in probs]
        records_to_add.append(",".join(probs))
        if len(records_to_add) == how_often_save:
            new_records = "\n".join(records_to_add)
            records_to_add = []
            _add_record_to_csv(results_path, new_records)

    if len(records_to_add) > 0:
        new_records = "\n".join(records_to_add)
        records_to_add = []
        _add_record_to_csv(results_path, new_records)
