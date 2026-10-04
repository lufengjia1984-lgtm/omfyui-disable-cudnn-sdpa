import torch


def _disable_cudnn_sdpa():
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
            print("[disable-cudnn-sdpa] cuDNN SDPA backend disabled.")
        except Exception as e:
            print("[disable-cudnn-sdpa] failed:", e)


_disable_cudnn_sdpa()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
