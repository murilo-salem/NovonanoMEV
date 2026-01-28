import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose import models
from skimage.measure import regionprops_table, find_contours, regionprops
from skimage.color import label2rgb
from scipy.ndimage import binary_fill_holes
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import torch
import warnings

# ==============================================================================
# 🚑 MONKEY PATCH: CORREÇÃO CRÍTICA DE COMPATIBILIDADE
# ==============================================================================
import streamlit.elements.image as st_image
from streamlit.elements.lib.image_utils import image_to_url as real_image_to_url
import inspect

def patched_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="auto", image_id=None, allow_emoji=False):
    sig = inspect.signature(real_image_to_url)
    params = sig.parameters.keys()
    
    kwargs = {
        'image': image,
        'clamp': clamp,
        'channels': channels,
        'output_format': output_format,
    }
    
    if 'image_id' in params:
        kwargs['image_id'] = image_id
    if 'allow_emoji' in params:
        kwargs['allow_emoji'] = allow_emoji
    
    if 'layout_config' in params:
        try:
            from streamlit.elements.lib.image_utils import LayoutConfig
            kwargs['layout_config'] = LayoutConfig(width=width) if width else LayoutConfig()
        except ImportError:
            class SimpleLayoutConfig:
                def __init__(self, width=None):
                    self.width = width
            kwargs['layout_config'] = SimpleLayoutConfig(width=width)
    
    return real_image_to_url(**kwargs)

st_image.image_to_url = patched_image_to_url
# ==============================================================================

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Nióbio Lab", page_icon="🔬", layout="wide")

# --- ESTADO (SESSION STATE) ---
if 'zoom_factor' not in st.session_state:
    st.session_state.zoom_factor = 0.0
if 'final_masks' not in st.session_state:
    st.session_state.final_masks = None
if 'original_raw' not in st.session_state:
    st.session_state.original_raw = None
if 'processed_view' not in st.session_state:
    st.session_state.processed_view = None
if 'editor_mode' not in st.session_state:
    st.session_state.editor_mode = "delete"  # "delete", "paint" ou "new_particle"

st.title("🔬 Análise de Nióbio: Ajuste Fino & Editor Visual")
st.markdown("---")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Pré-Processamento (Tempo Real)")
    
    clahe_clip = st.slider("CLAHE Clip (Contraste Local)", 0.0, 10.0, 1.7, 0.1)
    gamma_val = st.slider("Gamma (Brilho/Contraste)", 0.1, 3.0, 0.5, 0.1)
    blur_k = st.slider("Suavização (Blur)", 0, 10, 0, 1, help="0 = Desligado.")
    
    st.markdown("---")
    st.subheader("🖼️ Fundo Falso (Zoom Out)")
    c1, c2 = st.columns(2)
    if c1.button("➖ Menor"):
        st.session_state.zoom_factor = max(0.0, st.session_state.zoom_factor - 0.1)
    if c2.button("➕ Maior"):
        st.session_state.zoom_factor = min(2.0, st.session_state.zoom_factor + 0.1)
    st.caption(f"Borda: {int(st.session_state.zoom_factor * 100)}%")
    
    st.markdown("---")
    st.subheader("Parâmetros do Modelo")
    microns_per_pixel = st.number_input("µm / Pixel", value=0.05, format="%.4f")
    flow_threshold = st.number_input("Flow Thresh", value=1.0)
    cellprob_threshold = st.number_input("Cellprob Thresh", value=-50.0)
    min_size_px = st.number_input("Tam. Mínimo (px)", value=10)
    
    st.markdown("---")
    run_btn = st.button("🚀 RODAR CELLPOSE", type="primary")
    
    if st.button("🗑️ Resetar Tudo"):
        st.session_state.final_masks = None
        st.session_state.original_raw = None
        st.session_state.editor_mode = "delete"
        st.rerun()

# --- FUNÇÕES ---
@st.cache_resource
def load_model():
    use_gpu = torch.cuda.is_available()
    return models.CellposeModel(gpu=use_gpu, model_type="cyto3"), "GPU" if use_gpu else "CPU"

def to_8bit_display(img):
    """Converte qualquer imagem para 8-bit seguro para exibição."""
    if img is None:
        return None
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

def apply_preprocessing(img_gray, clip, gamma, blur):
    """Aplica filtros na imagem bruta em tempo real."""
    img_8bit = to_8bit_display(img_gray)
    
    # CLAHE
    if clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        img_eq = clahe.apply(img_8bit)
    else:
        img_eq = img_8bit.copy()
    
    # Normaliza Float (0-1)
    img_norm = img_eq.astype(np.float32) / 255.0
    
    # Inversão automática
    if np.mean(img_norm) > 0.5:
        img_norm = 1 - img_norm
    
    # Gamma
    img_norm = np.clip(np.power(img_norm, gamma) * 1.15, 0, 1)
    
    # Blur
    if blur > 0:
        k = blur | 1
        img_norm = cv2.GaussianBlur(img_norm, (k, k), 0)
    
    return img_norm

def apply_fake_background(img_norm, factor):
    if factor <= 0.01:
        return img_norm, 0, 0
    h, w = img_norm.shape[:2]
    pad_h, pad_w = int(h * factor), int(w * factor)
    bg_color = 29.0 / 255.0
    img_padded = cv2.copyMakeBorder(img_norm, pad_h, pad_h, pad_w, pad_w,
                                    cv2.BORDER_CONSTANT, value=bg_color)
    return img_padded, pad_h, pad_w

def create_detailed_overlay(masks, img_norm, df_filtered):
    """Cria overlay detalhado com contornos coloridos, eixos e centróides."""
    overlay_rgb = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
        (255, 100, 255), (100, 255, 255), (255, 200, 100), (200, 100, 255),
    ]
    
    for idx, (_, region) in enumerate(df_filtered.iterrows()):
        label = int(region['label'])
        mask_r = masks == label
        color = colors[idx % len(colors)]
        
        # Desenhar contorno
        contours = find_contours(mask_r.astype(float), 0.5)
        for c in contours:
            if len(c) > 2:
                c = np.flip(c, axis=1).astype(np.int32)
                cv2.polylines(overlay_rgb, [c], True, color, 2)
                cv2.polylines(overlay_rgb, [c], True, (255, 255, 255), 1)
        
        # Desenhar eixo maior
        y0, x0 = region['centroid-0'], region['centroid-1']
        angle = region['orientation']
        length = region['major_axis_length'] / 2
        
        dx = np.cos(angle) * length
        dy = -np.sin(angle) * length
        
        p1 = (int(x0 - dx), int(y0 - dy))
        p2 = (int(x0 + dx), int(y0 + dy))
        
        cv2.line(overlay_rgb, p1, p2, (255, 255, 255), 1)
        cv2.line(overlay_rgb, p1, p2, color, 1)
        
        # Desenhar centróide
        cv2.circle(overlay_rgb, (int(x0), int(y0)), 4, (255, 255, 255), -1)
        cv2.circle(overlay_rgb, (int(x0), int(y0)), 2, color, -1)
    
    return overlay_rgb

def create_dashboard(df_filtered, img_shape, microns_per_pixel):
    """Cria dashboard com 6 gráficos estatísticos."""
    if len(df_filtered) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Nenhuma partícula detectada',
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig
    
    # Limpar dados inválidos (NaN, inf)
    df_clean = df_filtered.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=['area_um2', 'circularity', 'aspect_ratio', 'diameter_um'])
    
    if len(df_clean) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Dados inválidos após limpeza',
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Histograma de tamanhos
    ax1 = fig.add_subplot(gs[0, 0])
    n_bins = min(30, len(df_clean))
    ax1.hist(df_clean['area_um2'], bins=n_bins,
            edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(df_clean['area_um2'].mean(), color='red', linestyle='--',
               label=f'Média: {df_clean["area_um2"].mean():.2f} µm²')
    ax1.axvline(df_clean['area_um2'].median(), color='green', linestyle=':',
               label=f'Mediana: {df_clean["area_um2"].median():.2f} µm²')
    ax1.set_xlabel('Área (µm²)', fontsize=10)
    ax1.set_ylabel('Frequência', fontsize=10)
    ax1.set_title('DISTRIBUIÇÃO DE TAMANHOS', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Circularidade vs Área
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(df_clean['area_um2'], df_clean['circularity'],
                         c=df_clean['aspect_ratio'], cmap='viridis',
                         alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Área (µm²)', fontsize=10)
    ax2.set_ylabel('Circularidade', fontsize=10)
    ax2.set_title('CIRCULARIDADE vs ÁREA', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Razão de Aspecto', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    
    # 3. Forma vs Tamanho
    ax3 = fig.add_subplot(gs[0, 2])
    scatter2 = ax3.scatter(df_clean['diameter_um'], df_clean['aspect_ratio'],
                          c=df_clean['circularity'], cmap='plasma',
                          alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Diâmetro (µm)', fontsize=10)
    ax3.set_ylabel('Razão de Aspecto', fontsize=10)
    ax3.set_title('FORMA vs TAMANHO', fontsize=11, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax3)
    cbar2.set_label('Circularidade', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # 4. Gráfico de Pizza
    ax4 = fig.add_subplot(gs[1, 0])
    max_area = df_clean['area_um2'].max()
    if max_area < 1:
        bins = [0, 0.1, 0.2, 0.5, 1.0]
        labels = ['<0.1', '0.1-0.2', '0.2-0.5', '0.5-1.0']
    elif max_area < 5:
        bins = [0, 0.5, 1.0, 2.0, 5.0]
        labels = ['<0.5', '0.5-1.0', '1.0-2.0', '2.0-5.0']
    else:
        bins = [0, 1.0, 2.0, 5.0, 10.0, 20.0]
        labels = ['<1.0', '1.0-2.0', '2.0-5.0', '5.0-10.0', '>10.0']
    
    try:
        df_clean['size_bin'] = pd.cut(df_clean['area_um2'], bins=bins, labels=labels)
        size_dist = df_clean['size_bin'].value_counts().sort_index()
        
        # Remover categorias vazias
        size_dist = size_dist[size_dist > 0]
        
        if len(size_dist) > 0:
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(size_dist)))
            wedges, texts, autotexts = ax4.pie(size_dist.values, labels=size_dist.index,
                                              autopct='%1.1f%%', colors=colors_pie,
                                              startangle=90, counterclock=False)
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
        else:
            ax4.text(0.5, 0.5, 'Sem dados', ha='center', va='center', fontsize=12)
    except Exception as e:
        ax4.text(0.5, 0.5, f'Erro no gráfico de pizza', ha='center', va='center', fontsize=10)
    
    ax4.set_title('DISTRIBUIÇÃO POR TAMANHO', fontsize=11, fontweight='bold')
    
    # 5. Boxplot de métricas
    ax5 = fig.add_subplot(gs[1, 1])
    metrics_to_plot = ['area_um2', 'circularity', 'aspect_ratio']
    data_to_plot = [df_clean[metric] for metric in metrics_to_plot]
    labels_box = ['Área (µm²)', 'Circularidade', 'Razão Aspecto']
    
    bp = ax5.boxplot(data_to_plot, labels=labels_box, patch_artist=True)
    
    colors_box = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    
    ax5.set_title('DISTRIBUIÇÃO DAS MÉTRICAS', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylabel('Valor', fontsize=10)
    ax5.tick_params(axis='x', labelsize=8)
    
    for i, metric in enumerate(metrics_to_plot):
        mean_val = df_clean[metric].mean()
        ax5.text(i+1, mean_val, f'{mean_val:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 6. Estatísticas textuais
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
ESTATÍSTICAS GERAIS

Total de partículas: {len(df_clean)}

Área média: {df_clean['area_um2'].mean():.3f} ± {df_clean['area_um2'].std():.3f} µm²

Área mediana: {df_clean['area_um2'].median():.3f} µm²

Diâmetro médio: {df_clean['diameter_um'].mean():.3f} ± {df_clean['diameter_um'].std():.3f} µm

Circularidade média: {df_clean['circularity'].mean():.3f} ± {df_clean['circularity'].std():.3f}

Razão de aspecto média: {df_clean['aspect_ratio'].mean():.3f} ± {df_clean['aspect_ratio'].std():.3f}
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('ANÁLISE COMPLETA DE PARTÍCULAS DE NIÓBIO',
                fontsize=13, fontweight='bold', y=0.98)
    
    return fig

# --- LÓGICA PRINCIPAL ---
uploaded_file = st.file_uploader("Arraste sua imagem aqui...", type=["tif", "png", "jpg"])

if uploaded_file:
    # Carregar RAW apenas uma vez
    if st.session_state.original_raw is None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if img_bgr.ndim == 3:
            st.session_state.original_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.shape[2] == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
        else:
            st.session_state.original_raw = img_bgr
    
    # Aplicar Filtros
    img_processed = apply_preprocessing(st.session_state.original_raw, clahe_clip, gamma_val, blur_k)
    st.session_state.processed_view = img_processed
    
    # Aplicar Zoom Out
    img_input, ph, pw = apply_fake_background(img_processed, st.session_state.zoom_factor)
    
    # Mostrar Preview
    with st.expander("👀 Visualizar Entrada do Modelo (Ajuste os sliders ao lado)", expanded=True):
        c1, c2 = st.columns(2)
        c1.image(to_8bit_display(st.session_state.original_raw),
                caption="Original RAW", use_container_width=True, clamp=True)
        c2.image(img_input, caption="Pré-processada + Fundo Falso",
                use_container_width=True, clamp=True)
    
    # --- EXECUÇÃO ---
    if run_btn:
        with st.spinner("Analisando..."):
            model, device_name = load_model()
            masks_pred, _, _ = model.eval(
                img_input,
                diameter=None,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size_px,
                augment=True
            )
            
            if ph > 0:
                h, w = masks_pred.shape
                masks_final = masks_pred[ph:h-ph, pw:w-pw]
            else:
                masks_final = masks_pred
            
            st.session_state.final_masks = masks_final
            st.rerun()
    
    # --- EDITOR VISUAL ---
    if st.session_state.final_masks is not None:
        st.markdown("### 🖌️ Editor Visual de Partículas")
        
        # Seletor de Modo - 3 Botões
        col_mode1, col_mode2, col_mode3 = st.columns(3)
        with col_mode1:
            if st.button("🗑️ Exclusão", type="primary" if st.session_state.editor_mode == "delete" else "secondary", use_container_width=True):
                st.session_state.editor_mode = "delete"
                st.rerun()
        with col_mode2:
            if st.button("🎨 Pincel (Expandir/Fundir)", type="primary" if st.session_state.editor_mode == "paint" else "secondary", use_container_width=True):
                st.session_state.editor_mode = "paint"
                st.rerun()
        with col_mode3:
            if st.button("✨ Nova Partícula", type="primary" if st.session_state.editor_mode == "new_particle" else "secondary", use_container_width=True):
                st.session_state.editor_mode = "new_particle"
                st.rerun()
        
        # Preparar overlay base
        overlay = label2rgb(st.session_state.final_masks,
                           image=st.session_state.processed_view,
                           bg_label=0, alpha=0.3)
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_uint8)
        
        canvas_width = 700
        scale_factor = st.session_state.processed_view.shape[1] / canvas_width if st.session_state.processed_view.shape[1] > 0 else 1
        
        # ==================== MODO EXCLUSÃO ====================
        if st.session_state.editor_mode == "delete":
            st.info("🗑️ **Modo Exclusão:** Pinte sobre partículas que deseja remover ou reduzir.")
            
            # Controle de tamanho do pincel
            stroke_width_delete = st.slider("🖌️ Tamanho do Pincel", 1, 50, 15, key="brush_size_delete")
            
            canvas_delete = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=stroke_width_delete,
                stroke_color="#ff0000",
                background_image=overlay_pil,
                update_streamlit=True,
                height=int(st.session_state.processed_view.shape[0] / scale_factor),
                width=canvas_width,
                drawing_mode="freedraw",
                key=f"canvas_delete_{stroke_width_delete}",
            )
            
            if st.button("✂️ Excluir Áreas Pintadas"):
                if canvas_delete.image_data is not None:
                    # Extrair canal alpha (onde foi pintado)
                    painted_alpha = canvas_delete.image_data[:, :, 3]
                    painted_mask = (painted_alpha > 0).astype(np.uint8)
                    
                    # Verificar se há pintura
                    if np.sum(painted_mask) == 0:
                        st.warning("⚠️ Nenhuma área foi pintada.")
                    else:
                        # Redimensionar para o tamanho original
                        h_orig, w_orig = st.session_state.final_masks.shape
                        painted_resized = cv2.resize(painted_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                        painted_coords = painted_resized > 0
                        
                        # Identificar quais labels foram interceptados
                        intercepted_labels = np.unique(st.session_state.final_masks[painted_coords])
                        intercepted_labels = intercepted_labels[intercepted_labels > 0]  # Remove background (0)
                        
                        if len(intercepted_labels) == 0:
                            st.warning("⚠️ Nenhuma partícula foi interceptada pela pintura.")
                        else:
                            # REMOVER as áreas pintadas (zerar pixels)
                            st.session_state.final_masks[painted_coords] = 0
                            
                            # Verificar se alguma partícula foi completamente removida
                            remaining_labels = np.unique(st.session_state.final_masks)
                            removed_completely = [lbl for lbl in intercepted_labels if lbl not in remaining_labels]
                            
                            if len(removed_completely) > 0:
                                st.success(f"🗑️ {len(removed_completely)} partícula(s) removida(s) completamente!")
                            
                            if len(intercepted_labels) > len(removed_completely):
                                st.info(f"✂️ {len(intercepted_labels) - len(removed_completely)} partícula(s) reduzida(s) parcialmente!")
                            
                            st.rerun()
                else:
                    st.warning("⚠️ Nenhum dado de pintura disponível.")
        
        # ==================== MODO PINCEL ====================
        elif st.session_state.editor_mode == "paint":
            st.info("🎨 **Modo Pincel:** Pinte para expandir partículas, fundir várias ou criar novas.")
            
            # Controle de tamanho do pincel
            stroke_width = st.slider("🖌️ Tamanho do Pincel", 1, 50, 15, key="brush_size")
            
            canvas_paint = st_canvas(
                fill_color="rgba(0, 255, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color="#00ff00",
                background_image=overlay_pil,
                update_streamlit=True,
                height=int(st.session_state.processed_view.shape[0] / scale_factor),
                width=canvas_width,
                drawing_mode="freedraw",
                key=f"canvas_paint_{stroke_width}",
            )
            
            if st.button("✅ Confirmar Pintura"):
                if canvas_paint.image_data is not None:
                    # Extrair canal alpha (onde foi pintado)
                    painted_alpha = canvas_paint.image_data[:, :, 3]
                    painted_mask = (painted_alpha > 0).astype(np.uint8)
                    
                    # Verificar se há pintura
                    if np.sum(painted_mask) == 0:
                        st.warning("⚠️ Nenhuma área foi pintada.")
                    else:
                        # Redimensionar para o tamanho original
                        h_orig, w_orig = st.session_state.final_masks.shape
                        painted_resized = cv2.resize(painted_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                        painted_coords = painted_resized > 0
                        
                        # Identificar quais labels existentes foram interceptados
                        intercepted_labels = np.unique(st.session_state.final_masks[painted_coords])
                        intercepted_labels = intercepted_labels[intercepted_labels > 0]  # Remove background (0)
                        
                        # LÓGICA DE EXPANSÃO/FUSÃO/CRIAÇÃO
                        if len(intercepted_labels) == 0:
                            # CRIAR NOVA PARTÍCULA
                            new_id = st.session_state.final_masks.max() + 1
                            st.session_state.final_masks[painted_coords] = new_id
                            st.success(f"✨ Nova partícula criada com ID {new_id}!")
                        
                        elif len(intercepted_labels) == 1:
                            # EXPANDIR PARTÍCULA EXISTENTE
                            target_id = intercepted_labels[0]
                            st.session_state.final_masks[painted_coords] = target_id
                            st.success(f"📈 Partícula {target_id} expandida!")
                        
                        else:
                            # FUSÃO DE MÚLTIPLAS PARTÍCULAS
                            target_id = intercepted_labels[0]
                            for old_id in intercepted_labels[1:]:
                                st.session_state.final_masks[st.session_state.final_masks == old_id] = target_id
                            st.session_state.final_masks[painted_coords] = target_id
                            st.success(f"🔗 {len(intercepted_labels)} partículas fundidas em ID {target_id}!")
                        
                        st.rerun()
                else:
                    st.warning("⚠️ Nenhum dado de pintura disponível.")
        
        # ==================== MODO NOVA PARTÍCULA ====================
        elif st.session_state.editor_mode == "new_particle":
            st.info("✨ **Modo Nova Partícula:** Pinte para criar partículas completamente novas (não expande nem funde).")
            
            # Controle de tamanho do pincel
            stroke_width_new = st.slider("🖌️ Tamanho do Pincel", 1, 50, 15, key="brush_size_new")
            
            canvas_new = st_canvas(
                fill_color="rgba(255, 215, 0, 0.3)",  # Dourado/Amarelo
                stroke_width=stroke_width_new,
                stroke_color="#FFD700",
                background_image=overlay_pil,
                update_streamlit=True,
                height=int(st.session_state.processed_view.shape[0] / scale_factor),
                width=canvas_width,
                drawing_mode="freedraw",
                key=f"canvas_new_{stroke_width_new}",
            )
            
            if st.button("✅ Criar Nova Partícula"):
                if canvas_new.image_data is not None:
                    # Extrair canal alpha (onde foi pintado)
                    painted_alpha = canvas_new.image_data[:, :, 3]
                    painted_mask = (painted_alpha > 0).astype(np.uint8)
                    
                    # Verificar se há pintura
                    if np.sum(painted_mask) == 0:
                        st.warning("⚠️ Nenhuma área foi pintada.")
                    else:
                        # Redimensionar para o tamanho original
                        h_orig, w_orig = st.session_state.final_masks.shape
                        painted_resized = cv2.resize(painted_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                        painted_binary = painted_resized > 0
                        
                        # Verificar se área mínima foi pintada
                        if np.sum(painted_binary) < 5:
                            st.warning("⚠️ Área pintada muito pequena. Tente pintar uma área maior.")
                        else:
                            # CRIAR PARTÍCULAS ÚNICAS PARA CADA REGIÃO DESCONECTADA
                            # Usar connected components para separar regiões desconectadas
                            from scipy import ndimage
                            
                            # Identificar componentes conectados na área pintada
                            labeled_painted, num_components = ndimage.label(painted_binary)
                            
                            if num_components == 0:
                                st.warning("⚠️ Nenhuma região válida foi identificada.")
                            else:
                                # Criar ID único para cada componente
                                base_id = st.session_state.final_masks.max() + 1
                                created_count = 0
                                total_area = 0
                                
                                for component_id in range(1, num_components + 1):
                                    # Máscara para este componente específico
                                    component_mask = (labeled_painted == component_id)
                                    
                                    # Verificar tamanho mínimo do componente
                                    component_area = np.sum(component_mask)
                                    if component_area >= 5:
                                        # Atribuir novo ID único e sequencial
                                        new_id = base_id + created_count
                                        st.session_state.final_masks[component_mask] = new_id
                                        created_count += 1
                                        total_area += component_area
                                
                                if created_count == 0:
                                    st.warning("⚠️ Todas as regiões pintadas eram muito pequenas (<5 pixels).")
                                else:
                                    # Calcular área total
                                    area_um2 = total_area * (microns_per_pixel ** 2)
                                    
                                    if created_count == 1:
                                        st.success(f"✨ Nova partícula criada com ID {base_id}!")
                                        st.info(f"📊 Área: {area_um2:.3f} µm² ({total_area} pixels)")
                                    else:
                                        st.success(f"✨ {created_count} novas partículas criadas (IDs {base_id} a {base_id + created_count - 1})!")
                                        st.info(f"📊 Área total: {area_um2:.3f} µm² ({total_area} pixels)")
                                    
                                    st.rerun()
                else:
                    st.warning("⚠️ Nenhum dado de pintura disponível.")
        
        # --- RELATÓRIO COMPLETO ---
        st.markdown("---")
        st.subheader("📊 Relatório Final com Visualizações Avançadas")
        
        # Calcular propriedades
        props = regionprops_table(
            st.session_state.final_masks,
            properties=['label', 'area', 'perimeter', 'major_axis_length',
                       'minor_axis_length', 'equivalent_diameter_area', 'eccentricity',
                       'centroid', 'orientation']
        )
        df = pd.DataFrame(props)
        
        if len(df) > 0:
            # Calcular métricas
            df['area_um2'] = df['area'] * (microns_per_pixel ** 2)
            df['diameter_um'] = df['equivalent_diameter_area'] * microns_per_pixel
            
            # Proteger contra divisão por zero no cálculo de circularidade
            df['circularity'] = np.where(
                df['perimeter'] > 0,
                (4 * np.pi * df['area']) / (df['perimeter'] ** 2),
                0
            )
            
            # Proteger contra divisão por zero no cálculo de aspect_ratio
            df['aspect_ratio'] = np.where(
                df['minor_axis_length'] > 0,
                df['major_axis_length'] / df['minor_axis_length'],
                0
            )
            
            # Remover partículas com perímetro zero ou métricas inválidas
            df = df[df['perimeter'] > 0]
            df = df[df['minor_axis_length'] > 0]
            
            # Filtros de área na sidebar
            min_a = st.sidebar.number_input("Min Área (µm²)", 0.0, 100.0, 0.005, 0.001)
            max_a = st.sidebar.number_input("Max Área (µm²)", 0.0, 10000.0, 500.0, 10.0)
            
            df_filtered = df[(df['area_um2'] >= min_a) & (df['area_um2'] <= max_a)]
            
            # OVERLAY DETALHADO
            st.markdown("#### 🎨 Visualização Detalhada com Contornos")
            with st.spinner("Gerando overlay detalhado..."):
                overlay_detailed = create_detailed_overlay(
                    st.session_state.final_masks,
                    st.session_state.processed_view,
                    df_filtered
                )
            st.image(overlay_detailed, caption=f"Overlay Detalhado - {len(df_filtered)} partículas",
                    use_container_width=True, clamp=True)
            
            # Botão de download para overlay
            col_download_overlay1, col_download_overlay2 = st.columns(2)
            with col_download_overlay1:
                overlay_bytes = cv2.imencode('.png', cv2.cvtColor(overlay_detailed, cv2.COLOR_RGB2BGR))[1].tobytes()
                st.download_button("💾 Baixar Overlay (PNG 300 DPI)",
                                 overlay_bytes,
                                 "overlay_detalhado_300dpi.png",
                                 "image/png",
                                 use_container_width=True)
            with col_download_overlay2:
                # Salvar overlay em alta resolução
                import io
                from PIL import Image
                
                overlay_pil_hq = Image.fromarray(overlay_detailed)
                buf_overlay = io.BytesIO()
                overlay_pil_hq.save(buf_overlay, format='PNG', dpi=(300, 300))
                buf_overlay.seek(0)
                
                st.download_button("💾 Baixar Overlay (PNG 300 DPI - Alta Qualidade)",
                                 buf_overlay.getvalue(),
                                 "overlay_detalhado_300dpi_hq.png",
                                 "image/png",
                                 use_container_width=True)
            
            # DASHBOARD ESTATÍSTICO
            st.markdown("#### 📈 Dashboard Estatístico Completo")
            with st.spinner("Gerando gráficos estatísticos..."):
                fig_dashboard = create_dashboard(df_filtered, st.session_state.processed_view.shape, microns_per_pixel)
            st.pyplot(fig_dashboard)
            
            # Botões de download para dashboard
            col_download_dash1, col_download_dash2, col_download_dash3 = st.columns(3)
            with col_download_dash1:
                # Salvar dashboard em PNG 300 DPI
                buf_png = io.BytesIO()
                fig_dashboard.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
                buf_png.seek(0)
                st.download_button("💾 Dashboard PNG (300 DPI)",
                                 buf_png.getvalue(),
                                 "dashboard_300dpi.png",
                                 "image/png",
                                 use_container_width=True)
            
            with col_download_dash2:
                # Salvar dashboard em PDF (vetorial)
                buf_pdf = io.BytesIO()
                fig_dashboard.savefig(buf_pdf, format='pdf', dpi=300, bbox_inches='tight')
                buf_pdf.seek(0)
                st.download_button("📄 Dashboard PDF (Vetorial)",
                                 buf_pdf.getvalue(),
                                 "dashboard_vetorial.pdf",
                                 "application/pdf",
                                 use_container_width=True)
            
            with col_download_dash3:
                # Salvar dashboard em SVG (vetorial escalável)
                buf_svg = io.BytesIO()
                fig_dashboard.savefig(buf_svg, format='svg', bbox_inches='tight')
                buf_svg.seek(0)
                st.download_button("🖼️ Dashboard SVG (Vetorial)",
                                 buf_svg.getvalue(),
                                 "dashboard_vetorial.svg",
                                 "image/svg+xml",
                                 use_container_width=True)
            
            plt.close(fig_dashboard)  # Liberar memória
            
            # TABELA E DOWNLOADS
            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric("Total Partículas", len(df_filtered))
                st.dataframe(df_filtered[['label', 'area_um2', 'diameter_um', 'circularity']], height=300)
                
                # Downloads
                st.download_button("💾 Baixar CSV Completo",
                                 df_filtered.to_csv(index=False).encode('utf-8'),
                                 "dados_niobio.csv", "text/csv",
                                 use_container_width=True)
            
            with c2:
                # Estatísticas resumidas
                st.markdown("**📊 Estatísticas Resumidas:**")
                stats_cols = st.columns(3)
                stats_cols[0].metric("Área Média", f"{df_filtered['area_um2'].mean():.3f} µm²")
                stats_cols[1].metric("Diâmetro Médio", f"{df_filtered['diameter_um'].mean():.3f} µm")
                stats_cols[2].metric("Circularidade", f"{df_filtered['circularity'].mean():.3f}")
                
                # Histograma rápido
                fig_quick, ax = plt.subplots(figsize=(8, 3))
                ax.hist(df_filtered['area_um2'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax.set_xlabel('Área (µm²)')
                ax.set_ylabel('Frequência')
                ax.set_title('Distribuição Rápida de Tamanhos')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_quick)
                plt.close(fig_quick)