from .PSVMANet import build_PSVMANet

_GZSL_META_ARCHITECTURES = {
    "Model": build_PSVMANet,
}

def build_gzsl_pipeline(cfg):
    meta_arch = _GZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)