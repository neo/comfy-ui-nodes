import torch

import comfy.model_management


class EmptyLatentImageQwen:
    # https://huggingface.co/Qwen/Qwen-Image#quick-start
    SUPPORTED_DIMENSIONS = {
        "16:9 (1664 x 928)": (1664, 928),
        "3:2 (1584 x 1056)": (1584, 1056),
        "4:3 (1472 x 1140)": (1472, 1140),
        "1:1 (1328 x 1328)": (1328, 1328),
        "3:4 (1140 x 1472)": (1140, 1472),
        "2:3 (1056 x 1584)": (1056, 1584),
        "9:16 (928 x 1664)": (928, 1664),
    }

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        dimensions = list(s.SUPPORTED_DIMENSIONS)
        return {"required": { "aspect_ratio": (dimensions, {"default": dimensions[len(dimensions) // 2]}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}) }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, aspect_ratio, batch_size=1):
        width, height = self.SUPPORTED_DIMENSIONS[aspect_ratio]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )
        

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

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        dimensions = list(s.SUPPORTED_DIMENSIONS)
        return {"required": { "aspect_ratio": (dimensions, {"default": dimensions[len(dimensions) // 2]}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}) }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, aspect_ratio, batch_size=1):
        width, height = self.SUPPORTED_DIMENSIONS[aspect_ratio]
        latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )


class ToggleDifferentText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "forceInput": True}),
                             "use_different_text": ("BOOLEAN", {"default": True}),
                             "different_text": ("STRING", {"multiline": True}),}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "choose"

    CATEGORY = "utils"

    def choose(self, text, use_different_text, different_text):
        return text if not use_different_text else different_text

NODE_CLASS_MAPPINGS = {
  'Empty Latent Image (Qwen)': EmptyLatentImageQwen,
  'Empty Latent Image (SDXL)': EmptyLatentImageSDXL,
  'Toggle Different Text': ToggleDifferentText,
}
