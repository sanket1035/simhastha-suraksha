"""Model factory shared by training, evaluation and export."""


def build_model(name, activation="relu", dropout=0.0, input_size=128, pretrained=True, freeze_frontend=False):
    if name == "mlp":
        from .mlp import CountMLP
        return CountMLP(input_size=input_size, activation=activation, dropout=dropout)
    if name == "cnn":
        from .cnn import CountCNN
        return CountCNN(activation=activation, dropout=dropout)
    if name == "csrnet":
        from .csrnet import CSRNet
        return CSRNet(pretrained=pretrained, freeze_frontend=freeze_frontend)
    raise ValueError("Unknown model %r (use mlp, cnn or csrnet)" % name)
