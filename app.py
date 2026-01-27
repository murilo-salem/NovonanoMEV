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
    """
    Cria overlay detalhado com contornos coloridos, eixos e centróides.
    Baseado no código de referência (linhas 235-275).
    """
    # Criar imagem RGB base
    overlay_rgb = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Paleta de cores otimizada
    colors = [
        (255, 100, 100),  # Vermelho claro
        (100, 255, 100),  # Verde claro
        (100, 100, 255),  # Azul claro
        (255, 255, 100),  # Amarelo
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Ciano
        (255, 200, 100),  # Laranja
        (200, 100, 255),  # Roxo
    ]
    
    for idx, (_, region) in enumerate(df_filtered.iterrows()):
        label = int(region['label'])
        mask_r = masks == label
        
        # Escolher cor (cíclica)
        color = colors[idx % len(colors)]
        
        # Desenhar contorno
        contours = find_contours(mask_r.astype(float), 0.5)
        for c in contours:
            if len(c) > 2:
                c = np.flip(c, axis=1).astype(np.int32)
                # Contorno externo mais espesso
                cv2.polylines(overlay_rgb, [c], True, color, 2)
                # Contorno interno mais fino para definição
                cv2.polylines(overlay_rgb, [c], True, (255, 255, 255), 1)
        
        # Desenhar eixo maior (orientação)
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
    """
    Cria dashboard com 5 gráficos estatísticos.
    Baseado no código de referência (linhas 280-380).
    """
    if len(df_filtered) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Nenhuma partícula detectada', 
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Histograma de tamanhos
    ax1 = fig.add_subplot(gs[0, 0])
    n_bins = min(30, len(df_filtered))
    ax1.hist(df_filtered['area_um2'], bins=n_bins, 
            edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(df_filtered['area_um2'].mean(), color='red', linestyle='--', 
               label=f'Média: {df_filtered["area_um2"].mean():.2f} µm²')
    ax1.axvline(df_filtered['area_um2'].median(), color='green', linestyle=':', 
               label=f'Mediana: {df_filtered["area_um2"].median():.2f} µm²')
    ax1.set_xlabel('Área (µm²)', fontsize=10)
    ax1.set_ylabel('Frequência', fontsize=10)
    ax1.set_title('DISTRIBUIÇÃO DE TAMANHOS', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Circularidade vs Área
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(df_filtered['area_um2'], df_filtered['circularity'], 
                         c=df_filtered['aspect_ratio'], cmap='viridis', 
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
    scatter2 = ax3.scatter(df_filtered['diameter_um'], df_filtered['aspect_ratio'], 
                          c=df_filtered['circularity'], cmap='plasma', 
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
    max_area = df_filtered['area_um2'].max()
    if max_area < 1:
        bins = [0, 0.1, 0.2, 0.5, 1.0]
        labels = ['<0.1', '0.1-0.2', '0.2-0.5', '0.5-1.0']
    elif max_area < 5:
        bins = [0, 0.5, 1.0, 2.0, 5.0]
        labels = ['<0.5', '0.5-1.0', '1.0-2.0', '2.0-5.0']
    else:
        bins = [0, 1.0, 2.0, 5.0, 10.0, 20.0]
        labels = ['<1.0', '1.0-2.0', '2.0-5.0', '5.0-10.0', '>10.0']
    
    df_filtered['size_bin'] = pd.cut(df_filtered['area_um2'], bins=bins, labels=labels)
    size_dist = df_filtered['size_bin'].value_counts().sort_index()
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(size_dist)))
    wedges, texts, autotexts = ax4.pie(size_dist.values, labels=size_dist.index, 
                                      autopct='%1.1f%%', colors=colors_pie,
                                      startangle=90, counterclock=False)
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    ax4.set_title('DISTRIBUIÇÃO POR TAMANHO', fontsize=11, fontweight='bold')
    
    # 5. Boxplot de métricas
    ax5 = fig.add_subplot(gs[1, 1])
    metrics_to_plot = ['area_um2', 'circularity', 'aspect_ratio']
    data_to_plot = [df_filtered[metric] for metric in metrics_to_plot]
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
        mean_val = df_filtered[metric].mean()
        ax5.text(i+1, mean_val, f'{mean_val:.2f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 6. Estatísticas textuais
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
ESTATÍSTICAS GERAIS

Total de partículas: {len(df_filtered)}

Área média: {df_filtered['area_um2'].mean():.3f} ± {df_filtered['area_um2'].std():.3f} µm²

Área mediana: {df_filtered['area_um2'].median():.3f} µm²

Diâmetro médio: {df_filtered['diameter_um'].mean():.3f} ± {df_filtered['diameter_um'].std():.3f} µm

Circularidade média: {df_filtered['circularity'].mean():.3f} ± {df_filtered['circularity'].std():.3f}

Razão de aspecto média: {df_filtered['aspect_ratio'].mean():.3f} ± {df_filtered['aspect_ratio'].std():.3f}
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
        st.markdown("### 🖌️ Editor Visual")
        st.info("Desenhe **retângulos** sobre as partículas que deseja remover e clique no botão abaixo.")
        
        # Overlay simples para o canvas
        overlay = label2rgb(st.session_state.final_masks, 
                           image=st.session_state.processed_view,
                           bg_label=0, alpha=0.3)
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_uint8)
        
        canvas_width = 700
        scale_factor = st.session_state.processed_view.shape[1] / canvas_width if st.session_state.processed_view.shape[1] > 0 else 1
        
        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=overlay_pil,
            update_streamlit=True,
            height=int(st.session_state.processed_view.shape[0] / scale_factor),
            width=canvas_width,
            drawing_mode="rect",
            key="canvas",
        )
        
        if st.button("✂️ Excluir Áreas Selecionadas"):
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                exclusion_mask = np.zeros(st.session_state.final_masks.shape, dtype=bool)
                objects = canvas_result.json_data["objects"]
                
                for obj in objects:
                    left = int(obj["left"] * scale_factor)
                    top = int(obj["top"] * scale_factor)
                    width = int(obj["width"] * scale_factor)
                    height = int(obj["height"] * scale_factor)
                    
                    r1 = max(0, top)
                    r2 = min(exclusion_mask.shape[0], top + height)
                    c1 = max(0, left)
                    c2 = min(exclusion_mask.shape[1], left + width)
                    
                    exclusion_mask[r1:r2, c1:c2] = True
                
                props = regionprops(st.session_state.final_masks)
                ids_to_remove = []
                
                for prop in props:
                    y, x = prop.centroid
                    if exclusion_mask[int(y), int(x)]:
                        ids_to_remove.append(prop.label)
                
                if ids_to_remove:
                    mask_data = st.session_state.final_masks
                    for uid in ids_to_remove:
                        mask_data[mask_data == uid] = 0
                    st.session_state.final_masks = mask_data
                    st.success(f"Removidos {len(ids_to_remove)} objetos!")
                    st.rerun()
                else:
                    st.warning("Nenhum centróide encontrado na seleção.")
            else:
                st.warning("Desenhe um retângulo primeiro.")
        
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
            df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-8)
            df['aspect_ratio'] = df['major_axis_length'] / (df['minor_axis_length'] + 1e-8)
            
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
            
            # DASHBOARD ESTATÍSTICO
            st.markdown("#### 📈 Dashboard Estatístico Completo")
            with st.spinner("Gerando gráficos estatísticos..."):
                fig_dashboard = create_dashboard(df_filtered, st.session_state.processed_view.shape, microns_per_pixel)
            st.pyplot(fig_dashboard)
            
            # TABELA E DOWNLOADS
            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric("Total Partículas", len(df_filtered))
                st.dataframe(df_filtered[['label', 'area_um2', 'diameter_um', 'circularity']], height=300)
                
                # Downloads
                st.download_button("💾 Baixar CSV Completo", 
                                 df_filtered.to_csv(index=False).encode('utf-8'),
                                 "dados_niobio.csv", "text/csv")
                
                # Salvar overlay
                overlay_bytes = cv2.imencode('.png', cv2.cvtColor(overlay_detailed, cv2.COLOR_RGB2BGR))[1].tobytes()
                st.download_button("🖼️ Baixar Overlay PNG",
                                 overlay_bytes,
                                 "overlay_detalhado.png",
                                 "image/png")
            
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