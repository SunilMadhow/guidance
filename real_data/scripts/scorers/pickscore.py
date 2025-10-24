import torch
from PIL import Image
from typing import Sequence, List
from transformers import AutoProcessor, AutoModel

from base import TextImageScorer

class PickScoreScorer(TextImageScorer):
    name = "PickScore"
    version = "YOUR_VERSION_OR_HASH"
    def __init__(self, device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        model_kwargs = {}
        if self.device.startswith("cuda") and dtype != torch.float32:
            model_kwargs["torch_dtype"] = dtype
        self.proc = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1", **model_kwargs).to(self.device)
        self.model.eval()
        self._text_cache = {}

    def _encode_text(self, prompts: Sequence[str]):
        missing = [p for p in prompts if p not in self._text_cache]
        if missing:
            # Deduplicate while preserving first occurrence order
            seen = set()
            unique_missing = [p for p in missing if not (p in seen or seen.add(p))]
            if unique_missing:
                inputs = self.proc(text=unique_missing, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeds = self.model.get_text_features(**inputs)
                embeds = self._normalize(embeds.to(self.dtype))
                for prompt, emb in zip(unique_missing, embeds):
                    self._text_cache[prompt] = emb.detach()
        ordered = [self._text_cache[prompt] for prompt in prompts]
        return torch.stack(ordered, dim=0).to(self.device, dtype=self.dtype)

    def _encode_images(self, batch_imgs_tensor):
        inputs = self.proc(images=batch_imgs_tensor, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        embeds = self.model.get_image_features(pixel_values=pixel_values)
        return self._normalize(embeds.to(self.dtype))

    def _score_embeds(self, img_emb, txt_emb):
        scale = self.model.logit_scale.exp().to(img_emb.dtype)
        return (img_emb * txt_emb).sum(dim=-1) * scale

    @staticmethod
    def _normalize(embeds: torch.Tensor) -> torch.Tensor:
        return embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @torch.no_grad()
    def score_pairs(self, image_paths: Sequence[str], prompts: Sequence[str], batch_size: int = 32) -> List[float]:
        assert len(image_paths) == len(prompts)
        text_embeds = self._encode_text(prompts)
        scores: List[float] = []
        total = len(image_paths)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_paths = image_paths[start:end]
            images: List[Image.Image] = []
            for path in batch_paths:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            img_embeds = self._encode_images(images)
            txt_embeds = text_embeds[start:end]
            batch_scores = self._score_embeds(img_embeds, txt_embeds)
            scores.extend(batch_scores.detach().cpu().float().tolist())
        return scores
