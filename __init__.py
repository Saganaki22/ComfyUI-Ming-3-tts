from .nodes import MingExtension


async def comfy_entrypoint():
    return MingExtension()
