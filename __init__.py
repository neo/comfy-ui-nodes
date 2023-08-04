import torch

class EmptyLatentImageSDXL:
    # https://platform.stability.ai/docs/features/api-parameters#about-dimensions
    SUPPORTED_DIMENSIONS = {
        "12:5 (1536 x 640)": (1536, 640), # 2.4
        "7:4 (1344 x 768)": (1344, 768), # 1.75
        "19:13 (1216 x 832)": (1216, 832), # 1.4615
        "9:7 (1152 x 896)": (1152, 896), # 1.2857
        "1:1 (1024 x 1024)": (1024, 1024), # 1
        "7:9 (896 x 1152)": (896, 1152),
        "13:19 (832 x 1216)": (832, 1216),
        "4:7 (768 x 1344)": (768, 1344),
        "5:12 (640 x 1536)": (640, 1536),
    }

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        dimensions = list(s.SUPPORTED_DIMENSIONS)
        return {"required": { "aspect_ratio": (dimensions, {"default": dimensions[len(dimensions) // 2]}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}) }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, aspect_ratio, batch_size=1):
        width, height = self.SUPPORTED_DIMENSIONS[aspect_ratio]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples":latent}, )

NODE_CLASS_MAPPINGS = {
  'Empty Latent Image (SDXL)': EmptyLatentImageSDXL,
}
