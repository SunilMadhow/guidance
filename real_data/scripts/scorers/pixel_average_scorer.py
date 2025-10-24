from base import TextImageScorer
from typing import Sequence, List
import tqdm
from PIL import Image
from PIL import ImageStat


class PixelAvgMockScorer(TextImageScorer):
    """
    Dependency-free mock scorer:
    - score = normalized mean pixel intensity * small prompt-length factor
    Use ONLY for plumbing tests. Not meaningful for training.
    """
    name = "pixel-avg-mock"
    version = "v1"

    def __init__(self):
        pass

    def score_pairs(self, image_paths: Sequence[str], prompts: Sequence[str], batch_size: int = 64) -> List[float]:
        out: List[float] = []
        for pth, prmpt in tqdm.tqdm(zip(image_paths, prompts), total=len(image_paths), desc="Scoring (pixel-avg)", leave=False):
            im = Image.open(pth).convert("RGB")
            stat = ImageStat.Stat(im)
            # Mean over channels, normalize to ~[0,1]
            mean = sum(stat.mean) / (3.0 * 255.0)
            # Tiny prompt influence to ensure prompt-dependence in tests
            factor = 1.0 + min(len(prmpt), 300) / 3000.0
            out.append(float(mean * factor * 100.0))  # scale to a "score-like" range
        return out
