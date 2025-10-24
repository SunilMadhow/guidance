from typing import Sequence, List

class TextImageScorer:
    name: str
    version: str
    def score_pairs(self, image_paths: Sequence[str], prompts: Sequence[str], batch_size: int = 32) -> List[float]:
        """Return raw scores, same length as inputs."""
        raise NotImplementedError
