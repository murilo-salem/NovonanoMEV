
# 🔬 Nióbio Lab — Análise Avançada de Partículas com IA

O **Nióbio Lab** é uma aplicação interativa construída com **Streamlit** para análise avançada de imagens microscópicas de partículas de nióbio.  
Ela combina **pré-processamento em tempo real**, **segmentação automática com Cellpose**, **edição visual manual**, e **relatórios estatísticos completos**, tudo em uma única interface web.

Projeto pensado para uso científico, pesquisa de materiais e análise exploratória de microestruturas.

---

## ✨ Principais Funcionalidades

### 🧪 Pré-processamento em tempo real
- Ajuste dinâmico de:
  - **CLAHE** (contraste local)
  - **Gamma** (brilho/contraste)
  - **Blur gaussiano**
- Inversão automática de contraste para imagens claras/escuras
- Visualização imediata do impacto dos parâmetros

### 🔍 Segmentação automática (Cellpose)
- Modelo **Cellpose cyto3**
- Execução automática em **GPU** (se disponível) ou **CPU**
- Parâmetros ajustáveis:
  - `flow_threshold`
  - `cellprob_threshold`
  - tamanho mínimo do objeto
  - escala física (µm/pixel)

### 🔎 Fundo falso (Zoom Out)
- Simula um *zoom out* adicionando bordas artificiais
- Melhora a segmentação de partículas próximas às bordas
- Controle fino via interface

### 🖌️ Editor visual interativo
- Canvas para **remoção manual de partículas**
- Exclusão baseada no centróide do objeto
- Ideal para corrigir falsos positivos

### 🎨 Visualização detalhada
- Overlays RGB com:
  - Contornos coloridos
  - Eixo maior (orientação)
  - Centróide
- Paleta de cores cíclica para facilitar inspeção visual

### 📊 Dashboard estatístico completo
Inclui:
- Histograma de áreas
- Circularidade vs área
- Forma vs tamanho
- Distribuição por classes (pizza)
- Boxplot de métricas
- Estatísticas textuais consolidadas

### 💾 Exportação de resultados
- Download de:
  - **CSV** com todas as métricas
  - **PNG** do overlay detalhado
- Tabela interativa no próprio app

---

## 📐 Métricas Calculadas

Para cada partícula segmentada:

- Área (µm²)
- Diâmetro equivalente (µm)
- Circularidade
- Razão de aspecto
- Eixos maior e menor
- Orientação
- Centróide

---

## 🧰 Tecnologias Utilizadas

- **Python 3.9+**
- **Streamlit**
- **OpenCV**
- **NumPy / Pandas**
- **Matplotlib**
- **Cellpose**
- **scikit-image**
- **SciPy**
- **PyTorch**
- **streamlit-drawable-canvas**

---

## 🚀 Como Executar

### 1️⃣ Criar ambiente (opcional, mas recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 2️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 3️⃣ Rodar a aplicação
```bash
streamlit run app.py
```

---

## 🖼️ Formatos de Imagem Suportados
- `.tif`
- `.png`
- `.jpg`

---

## ⚠️ Observações Importantes

- O projeto inclui um **monkey patch** para compatibilidade com versões recentes do Streamlit (renderização de imagens).
- Para melhor desempenho, recomenda-se:
  - GPU com CUDA
  - Imagens em escala de cinza de boa qualidade
- Projetado para análise exploratória e científica — **não é uma ferramenta médica**.

---

## 📌 Casos de Uso

- Pesquisa em ciência dos materiais
- Análise morfológica de partículas
- Estudos estatísticos de microestruturas
- Inspeção visual assistida por IA

---

## 🧠 Autor

Desenvolvido para o laboratório Novonano - UFPel.
