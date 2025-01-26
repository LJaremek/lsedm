def _get_file_records(file_name: str) -> list[str]:
    with open(file_name, "r", -1, "utf-8") as file:
        return [
            line.strip()
            for line in file.readlines()
            if line.strip() != ""
            ][::-1]


def get_laion_bigrams() -> list[str]:
    return _get_file_records(
        "most_common_words/prepared_data/laion_bigrams.txt"
        )


def get_laion() -> list[str]:
    return _get_file_records(
        "most_common_words/prepared_data/laion_bigrams.txt"
        )


def get_mscoco() -> list[str]:
    return _get_file_records(
        "most_common_words/prepared_data/mscoco.txt"
        )
