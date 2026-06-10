"""
=============================================================
  CLASSIFICAÇÃO DE DÍGITOS MANUSCRITOS COM REDE MLP
  Dataset: MNIST  |  Framework: TensorFlow / Keras
=============================================================

Este script implementa um Perceptron Multicamadas (MLP) para
reconhecer dígitos de 0 a 9 escritos à mão.

Fluxo completo:
  1. Carregamento e exploração do dataset MNIST
  2. Pré-processamento dos dados
  3. Construção da arquitetura MLP
  4. Treinamento e avaliação do modelo
  5. Visualizações de desempenho
  6. Servidor web com canvas interativo para desenhar dígitos

Como executar:
  pip install tensorflow matplotlib pillow scikit-learn numpy
  python mnist_mlp.py
"""

# ─── Importações ────────────────────────────────────────────────
import os
import sys
import json
import base64
import threading
import webbrowser
import http.server

import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sem janela gráfica (salva em arquivo)
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silencia logs verbose do TF
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report


# ╔═══════════════════════════════════════════════════════════╗
# ║  1. CARREGAMENTO DO DATASET MNIST                         ║
# ╚═══════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("  CLASSIFICAÇÃO DE DÍGITOS MANUSCRITOS — MLP + MNIST")
print("="*60)
print("\n[1/5] Carregando o dataset MNIST...")


def carregar_mnist():
    """
    Tenta carregar o MNIST de múltiplas fontes.
    Prioridade:
      1. Keras nativo (mais rápido se já em cache local)
      2. sklearn fetch_openml (espelho alternativo)
    Retorna:
      (X_treino, y_treino), (X_teste, y_teste)
      com imagens shape (N, 28, 28) e valores [0, 255].
    """

    # ── Tentativa 1: Keras ─────────────────────────────────────
    try:
        (Xt, yt), (Xte, yte) = keras.datasets.mnist.load_data()
        print("      ✓ Carregado via Keras (TensorFlow)")
        return (Xt, yt), (Xte, yte)
    except Exception as e:
        print(f"      ✗ Keras falhou ({e}). Tentando sklearn...")

    # ── Tentativa 2: sklearn / OpenML ──────────────────────────
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype("uint8").reshape(-1, 28, 28)
        y = mnist.target.astype("int")
        # divisão padrão: 60k treino / 10k teste
        return (X[:60000], y[:60000]), (X[60000:], y[60000:])
    except Exception as e:
        print(f"      ✗ sklearn falhou ({e}).")
        print()
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║  AÇÃO NECESSÁRIA                                     ║")
        print("  ║  Baixe o arquivo 'mnist.npz' manualmente em:        ║")
        print("  ║  https://storage.googleapis.com/tensorflow/         ║")
        print("  ║    tf-keras-datasets/mnist.npz                      ║")
        print("  ║  Coloque no mesmo diretório deste script e           ║")
        print("  ║  execute novamente.                                  ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        sys.exit(1)


(X_treino, y_treino), (X_teste, y_teste) = carregar_mnist()

print(f"      Treino : {X_treino.shape[0]:>6} imagens — shape: {X_treino.shape}")
print(f"      Teste  : {X_teste.shape[0]:>6} imagens — shape: {X_teste.shape}")
print(f"      Classes: {sorted(np.unique(y_treino).tolist())}")

# ── Visualização de amostras do dataset ─────────────────────────
fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
fig.suptitle("Amostras do Dataset MNIST (uma por dígito, treino/teste)",
             fontsize=12, fontweight="bold")

for digito in range(10):
    for linha, conj, rots in [(0, X_treino, y_treino), (1, X_teste, y_teste)]:
        idx = np.where(rots == digito)[0][0]
        ax  = axes[linha, digito]
        ax.imshow(conj[idx], cmap="gray")
        ax.set_title(str(digito), fontsize=10)
        ax.axis("off")

axes[0, 0].set_ylabel("Treino", fontsize=9)
axes[1, 0].set_ylabel("Teste",  fontsize=9)
plt.tight_layout()
plt.savefig("amostras_mnist.png", dpi=120, bbox_inches="tight")
plt.close()
print("      ✓ Figura salva: amostras_mnist.png")


# ╔═══════════════════════════════════════════════════════════╗
# ║  2. PRÉ-PROCESSAMENTO                                     ║
# ╚═══════════════════════════════════════════════════════════╝

print("\n[2/5] Pré-processando os dados...")

# ── 2a. Normalização: [0, 255] → [0.0, 1.0] ───────────────────
#   Valores pequenos evitam gradientes muito grandes durante
#   a retropropagação, acelerando a convergência do treinamento.
X_treino = X_treino.astype("float32") / 255.0
X_teste  = X_teste.astype("float32")  / 255.0

# ── 2b. Achatamento: (28, 28) → (784,) ────────────────────────
#   A MLP "fully-connected" opera sobre vetores 1-D.
#   Cada pixel vira uma feature de entrada.
X_treino_flat = X_treino.reshape(-1, 784)
X_teste_flat  = X_teste.reshape(-1, 784)

# ── 2c. One-hot encoding dos rótulos ──────────────────────────
#   Exemplo: 7 → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#   Necessário para a loss categorical_crossentropy.
y_treino_oh = keras.utils.to_categorical(y_treino, 10)
y_teste_oh  = keras.utils.to_categorical(y_teste,  10)

print(f"      Shape de entrada (treino) : {X_treino_flat.shape}")
print(f"      Shape de rótulos (treino) : {y_treino_oh.shape}  (one-hot, 10 classes)")


# ╔═══════════════════════════════════════════════════════════╗
# ║  3. ARQUITETURA MLP                                       ║
# ╠═══════════════════════════════════════════════════════════╣
# ║                                                           ║
# ║   [entrada: 784 pixels]                                   ║
# ║       ↓  Dense(512) + ReLU                               ║
# ║       ↓  Dropout(0.3)  ← desativa 30% dos neurônios      ║
# ║       ↓  Dense(256) + ReLU                               ║
# ║       ↓  Dropout(0.3)                                    ║
# ║       ↓  Dense(128) + ReLU                               ║
# ║       ↓  Dropout(0.2)  ← desativa 20% dos neurônios      ║
# ║   [saída: 10 neurônios] + Softmax                         ║
# ║                                                           ║
# ║  ReLU    → f(x) = max(0, x)                              ║
# ║            Evita gradiente desvanecente; treina rápido.   ║
# ║  Dropout → Regularização: reduz overfitting               ║
# ║  Softmax → Converte logits em probabilidades (soma = 1)   ║
# ╚═══════════════════════════════════════════════════════════╝

print("\n[3/5] Construindo o modelo MLP...")

modelo = keras.Sequential([
    # Camada de entrada — define a dimensão (784 pixels achatados)
    keras.Input(shape=(784,), name="entrada"),

    # ── Bloco oculto 1 ──────────────────────────────────────
    layers.Dense(512, activation="relu",  name="oculta_1"),
    layers.Dropout(0.3,                   name="dropout_1"),

    # ── Bloco oculto 2 ──────────────────────────────────────
    layers.Dense(256, activation="relu",  name="oculta_2"),
    layers.Dropout(0.3,                   name="dropout_2"),

    # ── Bloco oculto 3 ──────────────────────────────────────
    layers.Dense(128, activation="relu",  name="oculta_3"),
    layers.Dropout(0.2,                   name="dropout_3"),

    # ── Camada de saída — 10 classes (dígitos 0–9) ──────────
    layers.Dense(10,  activation="softmax", name="saida"),

], name="MLP_MNIST")

modelo.summary()   # imprime o resumo da arquitetura no terminal

# ── Compilação ──────────────────────────────────────────────────
# • Adam: otimizador adaptativo, ajusta lr individualmente
# • categorical_crossentropy: loss padrão para multi-classe
# • accuracy: porcentagem de acertos — fácil de interpretar
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ╔═══════════════════════════════════════════════════════════╗
# ║  4. TREINAMENTO                                           ║
# ╚═══════════════════════════════════════════════════════════╝

print("\n[4/5] Treinando o modelo...")

# ── Callbacks ───────────────────────────────────────────────────
# EarlyStopping: para o treino se val_loss não melhorar em
#   5 épocas seguidas, e restaura os melhores pesos encontrados.
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau: reduz a learning rate pela metade quando
#   val_loss ficar estagnada por 3 épocas consecutivas.
reduz_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

historico = modelo.fit(
    X_treino_flat, y_treino_oh,
    epochs=30,
    batch_size=256,         # amostras por passo de gradiente
    validation_split=0.1,   # 10% do treino vira validação interna
    callbacks=[early_stop, reduz_lr],
    verbose=1
)

# ── Avaliação no conjunto de teste ──────────────────────────────
loss_teste, acc_teste = modelo.evaluate(X_teste_flat, y_teste_oh, verbose=0)
print(f"\n      ✓ Acurácia no Teste : {acc_teste * 100:.2f}%")
print(f"      ✓ Loss no Teste     : {loss_teste:.4f}")

# ── Salva o modelo treinado ──────────────────────────────────────
modelo.save("modelo_mnist.keras")
print("      ✓ Modelo salvo: modelo_mnist.keras")


# ╔═══════════════════════════════════════════════════════════╗
# ║  5. VISUALIZAÇÕES DE DESEMPENHO                           ║
# ╚═══════════════════════════════════════════════════════════╝

print("\n[5/5] Gerando visualizações...")

# ── 5a. Curvas de treino / validação ────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Histórico de Treinamento", fontsize=13, fontweight="bold")
epocas = range(1, len(historico.history["loss"]) + 1)

ax1.plot(epocas, historico.history["loss"],
         label="Treino",    color="#2563eb", lw=2)
ax1.plot(epocas, historico.history["val_loss"],
         label="Validação", color="#dc2626", lw=2, linestyle="--")
ax1.set_title("Loss (Cross-Entropy Categórica)")
ax1.set_xlabel("Época"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epocas, historico.history["accuracy"],
         label="Treino",    color="#16a34a", lw=2)
ax2.plot(epocas, historico.history["val_accuracy"],
         label="Validação", color="#ea580c", lw=2, linestyle="--")
ax2.set_title("Acurácia")
ax2.set_xlabel("Época"); ax2.set_ylabel("Acurácia")
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("historico_treino.png", dpi=120, bbox_inches="tight")
plt.close()
print("      ✓ Figura salva: historico_treino.png")

# ── 5b. Matriz de Confusão ──────────────────────────────────────
#   Linhas = classe real, colunas = classe predita.
#   Valores altos na diagonal principal = bom desempenho.
y_pred = np.argmax(modelo.predict(X_teste_flat, verbose=0), axis=1)
cm = confusion_matrix(y_teste, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
fig.colorbar(im, ax=ax)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(range(10)); ax.set_yticklabels(range(10))
ax.set_xlabel("Dígito Predito", fontsize=11)
ax.set_ylabel("Dígito Real",    fontsize=11)
ax.set_title("Matriz de Confusão — Conjunto de Teste",
             fontsize=12, fontweight="bold")
for i in range(10):
    for j in range(10):
        cor = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color=cor, fontsize=9)
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=120, bbox_inches="tight")
plt.close()
print("      ✓ Figura salva: matriz_confusao.png")

# ── 5c. Relatório por classe ─────────────────────────────────────
print("\n── Relatório de Classificação por Dígito ──")
print(classification_report(y_teste, y_pred,
                             target_names=[str(i) for i in range(10)]))


# ╔═══════════════════════════════════════════════════════════╗
# ║  6. INTERFACE INTERATIVA — SERVIDOR WEB COM CANVAS        ║
# ╠═══════════════════════════════════════════════════════════╣
# ║  O usuário abre o navegador, desenha um dígito com o      ║
# ║  mouse (ou toque), e o servidor Python recebe a imagem    ║
# ║  via POST HTTP, pré-processa para 28×28 pixels e          ║
# ║  retorna a previsão + probabilidades para as 10 classes.  ║
# ╚═══════════════════════════════════════════════════════════╝

# ──────────────────────────────────────────────────────────────────
# Chamando a interface web
# ──────────────────────────────────────────────────────────────────

with open("interface.html", "r", encoding="utf-8") as f:
    HTML_PAGINA = f.read()


# ──────────────────────────────────────────────────────────────────
# Pré-processamento da imagem vinda do canvas
# ──────────────────────────────────────────────────────────────────
def preprocessar_imagem_canvas(dados_base64: str) -> np.ndarray:
    """
    Converte a imagem do canvas HTML em um vetor pronto para inferência.

    Etapas:
      1. Decodifica base64 → bytes PNG
      2. Abre com Pillow em escala de cinza (canal L)
      3. Redimensiona para 28×28 pixels (padrão MNIST)
      4. Normaliza pixels para [0.0, 1.0]
      5. Achata para shape (1, 784)

    Nota: o canvas usa fundo preto + traço branco,
    idêntico ao MNIST — não é necessário inverter cores.
    """
    from PIL import Image
    import io

    # Remove o cabeçalho "data:image/png;base64,"
    _, dados_puros = dados_base64.split(",", 1)
    imagem_bytes   = base64.b64decode(dados_puros)

    img = Image.open(io.BytesIO(imagem_bytes)).convert("L")  # escala de cinza
    img = img.resize((28, 28), Image.LANCZOS)                # 28×28 pixels

    arr = np.array(img, dtype="float32") / 255.0             # normalização
    return arr.reshape(1, 784)                               # vetor (1, 784)


# ──────────────────────────────────────────────────────────────────
# Handler HTTP (rotas GET e POST)
# ──────────────────────────────────────────────────────────────────
class ManipuladorHTTP(http.server.BaseHTTPRequestHandler):
    """Gerencia as requisições HTTP do servidor local."""

    def log_message(self, fmt, *args):
        """Suprime os logs padrão — deixa o terminal mais limpo."""
        pass

    def do_GET(self):
        """Serve a interface HTML para qualquer rota GET."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGINA.encode("utf-8"))

    def do_POST(self):
        """
        Rota POST /classificar:
          recebe JSON { imagem: "<base64>" }
          retorna JSON { digito, confianca, probabilidades }
        """
        if self.path != "/classificar":
            self.send_response(404)
            self.end_headers()
            return

        tamanho = int(self.headers.get("Content-Length", 0))
        corpo   = json.loads(self.rfile.read(tamanho))

        try:
            entrada  = preprocessar_imagem_canvas(corpo["imagem"])
            probs    = modelo.predict(entrada, verbose=0)[0].tolist()
            digito   = int(np.argmax(probs))
            resposta = {
                "digito"        : digito,
                "confianca"     : probs[digito],
                "probabilidades": probs
            }
        except Exception as exc:
            resposta = {"erro": str(exc)}

        corpo_resp = json.dumps(resposta).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(corpo_resp))
        self.end_headers()
        self.wfile.write(corpo_resp)


# ── Inicia servidor em thread separada ────────────────────────────
PORTA = 8765

def iniciar_servidor():
    servidor = http.server.HTTPServer(("", PORTA), ManipuladorHTTP)
    servidor.serve_forever()

thread_servidor = threading.Thread(target=iniciar_servidor, daemon=True)
thread_servidor.start()

url = f"http://localhost:{PORTA}"
print("\n" + "="*60)
print("    Servidor iniciado com sucesso!")
print(f"     Acesse: {url}")
print("     O navegador abrirá automaticamente.")
print("     Pressione Ctrl+C para encerrar.")
print("="*60 + "\n")

webbrowser.open(url)

# Mantém o script rodando enquanto o servidor atende requisições
try:
    thread_servidor.join()
except KeyboardInterrupt:
    print("\n  Servidor encerrado. Até mais!")
