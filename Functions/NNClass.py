from tensorflow.keras import Sequential

class PersonalizedSequential(Sequential):
    def __init__(self, layers = None, config: dict = {}, **kwargs):
        super().__init__(layers=layers, **kwargs) 
        self.config = config          # <- custom attribute

    # ------- 1.  put custom stuff in the config  -----------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"config": self.config})
        return cfg
        
    # ------- 2.  reconstruct from config -------------------------
    @classmethod
    def from_config(cls, cfg):
        # pull the custom field out BEFORE recreating the Sequential:
        config = cfg.pop("config")
        # let the built-in Sequential re-build the layer stack:
        obj = super(PersonalizedSequential, cls).from_config(cfg)
        # super() returns a plain Sequential; cast it to our subclass
        obj.__class__ = cls
        obj.config = config
        return obj