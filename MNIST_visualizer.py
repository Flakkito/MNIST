"""
MNIST Neural Network Real-Time Visualizer
==========================================
Muestra las activaciones de cada capa mientras la red predice digitos.

Controles:
  SPACE / → : siguiente imagen
  ←         : imagen anterior
  R         : imagen aleatoria
  Q / ESC   : salir

Requisitos: pip install pygame tensorflow tensorflow-datasets
"""
import sys
import numpy as np
import pygame

# ── Dimensiones de la ventana ─────────────────────────────────────────────────
W, H       = 1400, 760
LEFT_W     = 290    # panel izquierdo: imagen del digito
MID_W      = 820    # panel central:  red neuronal
RIGHT_W    = 290    # panel derecho:  barras de prediccion

# ── Paleta de colores ─────────────────────────────────────────────────────────
BG         = ( 10,  12,  22)
PANEL      = ( 18,  20,  36)
WHITE      = (225, 230, 255)
GRAY       = ( 85,  90, 115)
DIM        = ( 35,  40,  58)
GREEN      = ( 50, 210,  90)
RED_COL    = (215,  65,  55)
BLUE_COL   = ( 55, 120, 210)
DIVIDER    = ( 40,  45,  65)

# ── Red neuronal - parametros visuales ───────────────────────────────────────
N_SHOW   = 22       # neuronas visibles por capa oculta
N_OUT    = 10
R_HID    = 9        # radio neuronas ocultas
R_OUT    = 13       # radio neuronas de salida
V_SPACE  = 28       # espaciado vertical entre neuronas ocultas
V_SPACE_OUT = 42    # espaciado vertical neuronas de salida
AUTO_MS  = 2600     # ms entre avances automaticos


# ── Utilidades de color ───────────────────────────────────────────────────────
def neuron_color(t: float):
    t = max(0.0, min(1.0, t))
    return (int(10 + 235*t), int(25 + 205*t), int(95 + 160*t))


def conn_brightness(t: float) -> int:
    return int(8 + 60 * max(0.0, min(1.0, t)))


# ── Carga del modelo y datos ──────────────────────────────────────────────────
def load_everything():
    import tensorflow as tf

    print("Cargando modelo mnist_model.keras ...")
    model = tf.keras.models.load_model("mnist_model.keras")

    # Forzar la construccion del grafo con un input dummy
    model(np.zeros((1, 28, 28, 1), dtype="float32"))

    act_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in model.layers],
    )

    print("Cargando datos MNIST test ...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalizar y agregar dimension de canal: (N, 28, 28, 1)
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., np.newaxis]
    return act_model, x_test, y_test


def get_activations(act_model, image_np):
    """Devuelve lista de arrays 1-D con activaciones de cada capa."""
    out = act_model.predict(image_np[np.newaxis, ...], verbose=0)
    return [o.flatten() for o in out]


# ── Panel izquierdo: imagen del digito ───────────────────────────────────────
def draw_digit_panel(surf, image_np, true_lbl, font_b, font_s):
    p = pygame.Surface((LEFT_W, H))
    p.fill(PANEL)

    t = font_b.render("Entrada", True, WHITE)
    p.blit(t, (LEFT_W // 2 - t.get_width() // 2, 18))

    img = image_np.squeeze()
    ps  = 8
    ix  = LEFT_W // 2 - 28 * ps // 2
    iy  = 65
    for r in range(28):
        for c in range(28):
            v = int(img[r, c] * 255)
            pygame.draw.rect(p, (v, v, v), (ix + c*ps, iy + r*ps, ps, ps))
    pygame.draw.rect(p, GRAY, (ix - 2, iy - 2, 28*ps + 4, 28*ps + 4), 2)

    lbl = font_b.render(f"Digito real: {true_lbl}", True, WHITE)
    p.blit(lbl, (LEFT_W // 2 - lbl.get_width() // 2, iy + 28*ps + 18))

    for i, hint in enumerate(["SPACE / flechas: siguiente",
                               "R: aleatorio   Q/ESC: salir"]):
        ht = font_s.render(hint, True, GRAY)
        p.blit(ht, (LEFT_W // 2 - ht.get_width() // 2, H - 52 + i * 22))

    surf.blit(p, (0, 0))


# ── Panel central: red neuronal ───────────────────────────────────────────────
def layer_positions(n, x, y_center, spacing):
    total = (n - 1) * spacing
    y0    = y_center - total // 2
    return [(x, y0 + i * spacing) for i in range(n)]


def draw_network_panel(surf, acts, font_s, ox):
    p = pygame.Surface((MID_W, H))
    p.fill(PANEL)

    title = font_s.render("Activaciones de la Red Neuronal en tiempo real", True, WHITE)
    p.blit(title, (MID_W // 2 - title.get_width() // 2, 14))

    flat_a = acts[0]           # Flatten  (784,)
    h1_a   = acts[1]           # Dense 1  (200,) ReLU
    h2_a   = acts[2]           # Dense 2  (200,) ReLU
    h3_a   = acts[3]           # Dense 3  (200,) Sigmoid
    out_a  = acts[4]           # Dense 4  (10,)  Softmax

    inp_idx = np.linspace(0, 783, N_SHOW, dtype=int)
    h1_idx  = np.argsort(h1_a)[::-1][:N_SHOW]
    h2_idx  = np.argsort(h2_a)[::-1][:N_SHOW]
    h3_idx  = np.argsort(h3_a)[::-1][:N_SHOW]

    layers = [
        ("Entrada",         inp_idx,        flat_a[inp_idx],    N_SHOW,  R_HID,  V_SPACE),
        ("Oculta 1 (ReLU)", h1_idx,         h1_a[h1_idx],       N_SHOW,  R_HID,  V_SPACE),
        ("Oculta 2 (ReLU)", h2_idx,         h2_a[h2_idx],       N_SHOW,  R_HID,  V_SPACE),
        ("Oculta 3 (Sig.)", h3_idx,         h3_a[h3_idx],       N_SHOW,  R_HID,  V_SPACE),
        ("Salida (Softmax)",np.arange(N_OUT),out_a,             N_OUT,   R_OUT,  V_SPACE_OUT),
    ]

    xs       = [70, 235, 400, 565, 720]
    y_center = H // 2
    all_pos  = [layer_positions(l[3], xs[li], y_center, l[5])
                for li, l in enumerate(layers)]

    # Conexiones
    for li in range(len(layers) - 1):
        a_from = layers[li][2]
        for fi, (x1, y1) in enumerate(all_pos[li]):
            b = conn_brightness(float(a_from[fi]))
            col = (b, b, b + 18)
            for (x2, y2) in all_pos[li + 1]:
                pygame.draw.line(p, col, (x1, y1), (x2, y2), 1)

    # Neuronas
    for li, (name, idx, acts_l, n, r, _) in enumerate(layers):
        for ni, (x, y) in enumerate(all_pos[li]):
            t     = float(np.clip(acts_l[ni], 0, 1))
            fill  = neuron_color(t)
            edge  = tuple(min(255, c + 35) for c in fill)
            pygame.draw.circle(p, fill, (x, y), r)
            pygame.draw.circle(p, edge, (x, y), r, 2)
            if li == 4:                              # etiqueta del digito
                d = font_s.render(str(ni), True, WHITE)
                p.blit(d, (x + r + 4, y - d.get_height() // 2))

        # Nombre de la capa
        nt = font_s.render(name, True, GRAY)
        p.blit(nt, (xs[li] - nt.get_width() // 2, 40))

    surf.blit(p, (ox, 0))


# ── Panel derecho: prediccion ─────────────────────────────────────────────────
def draw_prediction_panel(surf, out_a, predicted, true_lbl, font_b, font_s, ox):
    p = pygame.Surface((RIGHT_W, H))
    p.fill(PANEL)

    t = font_b.render("Prediccion", True, WHITE)
    p.blit(t, (RIGHT_W // 2 - t.get_width() // 2, 18))

    bar_max_w = RIGHT_W - 80
    bar_h     = 38
    y_start   = 60
    y_gap     = 52

    for i, prob in enumerate(out_a):
        y = y_start + i * y_gap
        if   i == true_lbl:  col = GREEN
        elif i == predicted: col = RED_COL
        else:                col = BLUE_COL

        pygame.draw.rect(p, DIM, (58, y, bar_max_w, bar_h), border_radius=5)
        bw = int(bar_max_w * float(prob))
        if bw > 0:
            pygame.draw.rect(p, col, (58, y, bw, bar_h), border_radius=5)

        d = font_b.render(str(i), True, WHITE)
        p.blit(d, (28 - d.get_width() // 2, y + bar_h // 2 - d.get_height() // 2))

        pct = font_s.render(f"{prob*100:.1f}%", True, WHITE)
        p.blit(pct, (62 + bar_max_w - pct.get_width() - 4,
                     y + bar_h // 2 - pct.get_height() // 2))

    # Leyenda
    yl = H - 105
    for col, lbl in [(GREEN, "Correcto"), (RED_COL, "Predicho (X)"), (BLUE_COL, "Otros")]:
        pygame.draw.rect(p, col, (14, yl, 18, 15), border_radius=3)
        lt = font_s.render(lbl, True, GRAY)
        p.blit(lt, (38, yl))
        yl += 24

    # Veredicto final
    ok        = predicted == true_lbl
    status    = "CORRECTO" if ok else "INCORRECTO"
    v_col     = GREEN if ok else RED_COL
    vt        = font_b.render(f"Predijo: {predicted}  [{status}]", True, v_col)
    p.blit(vt, (RIGHT_W // 2 - vt.get_width() // 2, H - 34))

    surf.blit(p, (ox, 0))


# ── Separadores ───────────────────────────────────────────────────────────────
def draw_dividers(surf):
    pygame.draw.line(surf, DIVIDER, (LEFT_W, 0), (LEFT_W, H), 1)
    pygame.draw.line(surf, DIVIDER, (LEFT_W + MID_W, 0), (LEFT_W + MID_W, H), 1)


# ── Bucle principal ───────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MNIST - Visualizador de Red Neuronal en Tiempo Real")
    clock = pygame.time.Clock()

    font_b = pygame.font.SysFont("segoeui", 19, bold=True)
    font_s = pygame.font.SysFont("segoeui", 15)

    act_model, images, labels = load_everything()
    rng       = np.random.default_rng(42)
    idx       = 0
    last_time = pygame.time.get_ticks()

    while True:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                    idx = (idx + 1) % len(images); last_time = now
                if event.key == pygame.K_LEFT:
                    idx = (idx - 1) % len(images); last_time = now
                if event.key == pygame.K_r:
                    idx = int(rng.integers(0, len(images))); last_time = now

        if now - last_time > AUTO_MS:
            idx = (idx + 1) % len(images)
            last_time = now

        image_np  = images[idx]
        true_lbl  = int(labels[idx])
        acts      = get_activations(act_model, image_np)
        out_a     = acts[4]
        predicted = int(np.argmax(out_a))

        screen.fill(BG)
        draw_digit_panel(screen, image_np, true_lbl, font_b, font_s)
        draw_network_panel(screen, acts, font_s, LEFT_W)
        draw_prediction_panel(screen, out_a, predicted, true_lbl,
                               font_b, font_s, LEFT_W + MID_W)
        draw_dividers(screen)

        info = font_s.render(
            f"Imagen #{idx} / {len(images)}    FPS: {clock.get_fps():.0f}", True, GRAY)
        screen.blit(info, (5, H - 20))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
