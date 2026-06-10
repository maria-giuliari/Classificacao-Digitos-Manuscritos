# Classificador de Dígitos Manuscritos — MLP + MNIST

Atividade da disciplina de **Introdução a Redes Neurais Artificiais**.  
Implementação de um Perceptron Multicamadas (MLP) para reconhecimento de dígitos manuscritos (0–9), com interface interativa para desenhar e classificar dígitos em tempo real.

---

## Sobre o Projeto

O modelo é treinado no dataset **MNIST**, que contém 70.000 imagens em escala de cinza (28×28 pixels) de dígitos escritos à mão:
- **60.000** imagens para treino
- **10.000** imagens para teste

Após o treinamento, uma interface web é aberta no navegador onde o usuário pode **desenhar um dígito com o mouse** e a rede neural classifica em tempo real, exibindo a probabilidade para cada classe.

---

## Arquitetura da Rede

```
Entrada: 784 neurônios (28×28 pixels achatados)
    ↓  Dense(512) + ReLU
    ↓  Dropout(0.3)
    ↓  Dense(256) + ReLU
    ↓  Dropout(0.3)
    ↓  Dense(128) + ReLU
    ↓  Dropout(0.2)
Saída: 10 neurônios + Softmax (dígitos 0–9)
```

**Total de parâmetros:** 567.434

---

## Resultados

| Métrica | Valor |
|---|---|
| Acurácia no teste | **98.30%** |
| Loss no teste | 0.0676 |
| Épocas treinadas | 16 (EarlyStopping) |

---

## Estrutura do Projeto

```
📁 redes_neurais/
├── mnist_mlp.py       # Script principal (treinamento + servidor web)
├── interface.html     # Interface interativa de desenho
└── README.md
```

---

## Como Executar

**1. Instalar dependências:**
```bash
pip install tensorflow matplotlib pillow scikit-learn numpy
```

**2. Rodar o script:**
```bash
python mnist_mlp.py
```

O script vai automaticamente:
1. Baixar o dataset MNIST (~11 MB)
2. Treinar o modelo (~2–5 minutos)
3. Gerar gráficos de desempenho
4. Abrir o navegador com a interface de desenho

---

## Interface Interativa

Após o treinamento, o navegador abre com um canvas onde você pode:
- Desenhar um dígito com o mouse (ou toque no celular)
- Clicar em **Classificar** (ou pressionar `Enter`)
- Ver a probabilidade para cada dígito (0–9)
- Limpar o canvas (ou pressionar `Esc`)

---

## Referências

- [MNIST Classification using Multilayer Perceptron — Kaggle](https://www.kaggle.com/code/jonathankristanto/mnist-classification-using-multilayer-perceptron)
- [Digit Recognizer — Kaggle Competition](https://www.kaggle.com/c/digit-recognizer/data)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/api_docs)
