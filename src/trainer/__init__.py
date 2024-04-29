

def get_trainer(model_name, h_params):
    if model_name == "lightgbm":
        from .lightgbm import LightGBMTrainer
        return LightGBMTrainer(h_params)
    elif model_name == "linearregression":
        from .linearreg import LinearRegTrainer
        return LinearRegTrainer(h_params)
    elif model_name == "elasticnet":
        from .elasticnet import ElasticNetTrainer
        return ElasticNetTrainer(h_params)
    elif "former" in model_name:
        from .transformer import TransformerTrainer
        from src.models.transformer.args import args
        args.model = model_name
        return TransformerTrainer(args)
    return None

