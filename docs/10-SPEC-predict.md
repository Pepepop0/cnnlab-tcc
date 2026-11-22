# Spec — core/predict.py

## Responsabilidade

Carregar uma rede salva e classificar imagens.

## Interface

```python
@dataclass
class Prediction:
    predicted: str
    confidence: float
    top5: list[tuple[str, float]]


def load_model(run_id: str, device: torch.device) -> tuple[nn.Module, list[str]]: ...
def predict_tensor(model: nn.Module, x: Tensor, class_names: list[str]) -> Prediction: ...
def predict_file(model: nn.Module, path: Path, class_names: list[str], device: torch.device) -> Prediction: ...
def sample_from_dataset(data: DataBundle, index: int | None = None) -> tuple[Tensor, str]: ...
```

## Regras

1. `load_model` le `best.pt`, reconstroi via `build_cnn(ModelConfig(**payload["model_config"]))`
   e carrega o `state_dict`. Nunca usa pickle de objeto.
2. `predict_file`: abrir com PIL, `.convert("RGB")`, redimensionar para 32x32 com
   `Image.BILINEAR`, para tensor, normalizar com as **mesmas** medias e desvios
   do treino. Divergencia aqui e a causa numero um de predicao ruim em imagem
   propria — deixe explicito no codigo com um comentario.
3. `model.eval()` e `torch.no_grad()` sempre.
4. Softmax sobre os logits para obter confianca.
5. Registrar cada predicao em `predictions.jsonl`.
6. Avisar na interface quando a run foi treinada com `class_subset`: o modelo so
   conhece N classes, e uma foto arbitraria sera forcada numa delas.

## Aceite

- [ ] Carregar a melhor rede de uma run concluida e classificar uma amostra do dataset
- [ ] Classificar um PNG arbitrario sem erro
- [ ] `top5` soma proximo de 1.0 quando `num_classes >= 5`
- [ ] Predicoes aparecem no historico depois de reabrir o app
