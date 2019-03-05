from model import meta_arch


def build_model(cfg, mode):
    model_factory = getattr(meta_arch, cfg.MODEL.META_ARCHITECTURE)
    model = model_factory(cfg, mode)
    return model
