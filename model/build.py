from model import meta_arch


def build_model(cfg):
    model_factory = getattr(meta_arch, cfg.MODEL.META_ARCHITECTURE)
    model = model_factory(cfg)
    return model
