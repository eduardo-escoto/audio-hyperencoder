from .audio import AudioDataset


class PDMXDataset(AudioDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        return super().__getitem__(idx)

    def __get_metadata__(self, idx: int):
        # TODO: Implement metadata retrieval from PDMX -- need to figure out how to do this with miditoolkit