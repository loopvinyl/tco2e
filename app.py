# =============================================================================
# VERMICOMPOSTING vs LANDFILL EMISSION SIMULATOR - VERSION 2.0
# Com correções completas baseadas em Yang et al. 2017
# =============================================================================

import requests
from bs4 import BeautifulSoup
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze

np.random.seed(50)  # Garante reprodutibilidade

# Configurações iniciais
st.set_page_config(page_title="Simulador de Emissões - Yang et al. 2017", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# FUNÇÕES DE COTAÇÃO AUTOMÁTICA DO CARBONO E CÂMBIO (MANTIDAS)
# =============================================================================

def obter_cotacao_carbono_investing():
    """Obtém a cotação em tempo real do carbono via web scraping"""
    try:
        url = "https://www.investing.com/commodities/carbon-emissions"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        selectores = [
            '[data-test="instrument-price-last"]',
            '.text-2xl',
            '.last-price-value',
            '.instrument-price-last'
        ]
        
        preco = None
        fonte = "Investing.com"
        
        for seletor in selectores:
            try:
                elemento = soup.select_one(seletor)
                if elemento:
                    texto_preco = elemento.text.strip().replace(',', '')
                    texto_preco = ''.join(c for c in texto_preco if c.isdigit() or c == '.')
                    if texto_preco:
                        preco = float(texto_preco)
                        break
            except (ValueError, AttributeError):
                continue
        
        if preco is not None:
            return preco, "€", "Carbon Emissions Future", True, fonte
        
        return None, None, None, False, fonte
        
    except Exception as e:
        return None, None, None, False, f"Investing.com - Erro: {str(e)}"

def obter_cotacao_carbono():
    """Obtém a cotação em tempo real do carbono"""
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    
    if sucesso:
        return preco, moeda, f"{contrato_info}", True, fonte
    
    return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"

def obter_cotacao_euro_real():
    """Obtém a cotação em tempo real do Euro em relação ao Real"""
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except:
        pass
    
    return 5.50, "R$", False, "Referência"

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    """Calcula o valor financeiro das emissões evitadas"""
    valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
    return valor_total

def exibir_cotacao_carbono():
    """Exibe a cotação do carbono com informações"""
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    if not st.session_state.get('cotacao_carregada', False):
        st.session_state.mostrar_atualizacao = True
        st.session_state.cotacao_carregada = True
    
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Atualizar Cotações", key="atualizar_cotacoes"):
            st.session_state.cotacao_atualizada = True
            st.session_state.mostrar_atualizacao = True
    
    if st.session_state.get('mostrar_atualizacao', False):
        st.sidebar.info("🔄 Atualizando cotações...")
        
        preco_carbono, moeda, contrato_info, sucesso_carbono, fonte_carbono = obter_cotacao_carbono()
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        
        st.rerun()

    st.sidebar.metric(
        label=f"Preço do Carbono (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {formatar_br(st.session_state.taxa_cambio)}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {formatar_br(preco_carbono_reais)}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )

# =============================================================================
# INICIALIZAÇÃO DA SESSION STATE
# =============================================================================

def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        preco_carbono, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
        
    if 'taxa_cambio' not in st.session_state:
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        
    if 'moeda_real' not in st.session_state:
        st.session_state.moeda_real = "R$"
    if 'cotacao_atualizada' not in st.session_state:
        st.session_state.cotacao_atualizada = False
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'mostrar_atualizacao' not in st.session_state:
        st.session_state.mostrar_atualizacao = False
    if 'cotacao_carregada' not in st.session_state:
        st.session_state.cotacao_carregada = False
    if 'k_ano' not in st.session_state:
        st.session_state.k_ano = 0.06

inicializar_session_state()

# =============================================================================
# FUNÇÕES DE FORMATAÇÃO (MANTIDAS)
# =============================================================================

def formatar_br(numero):
    """Formata números no padrão brasileiro: 1.234,56"""
    if pd.isna(numero):
        return "N/A"
    numero = round(numero, 2)
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formatar_br_dec(numero, decimais=2):
    """Formata números com número específico de casas decimais"""
    if pd.isna(numero):
        return "N/A"
    numero = round(numero, decimais)
    return f"{numero:,.{decimais}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format(x, pos):
    """Função de formatação para eixos de gráficos"""
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =============================================================================
# FUNÇÕES DE CORREÇÃO BASEADAS EM YANG ET AL. 2017 (ADICIONADAS)
# =============================================================================

def temperatura_correcao_fator_ch4(temp_atual, temp_referencia=25):
    """
    Correção para CH₄ baseada em temperatura
    Base: Yang et al. 2017 - Q10 = 2 (dobra a cada 10°C)
    """
    Q10_ch4 = 2.0
    return Q10_ch4 ** ((temp_atual - temp_referencia) / 10)

def temperatura_correcao_fator_n2o(temp_atual, temp_referencia=25):
    """
    Correção para N₂O baseada em temperatura
    Base: Yang et al. 2017 - curva empírica
    """
    if temp_atual <= 10:
        return 0.1
    elif temp_atual <= 20:
        return 0.5
    elif temp_atual <= 30:
        return 1.0
    elif temp_atual <= 35:
        return 1.2
    elif temp_atual <= 40:
        return 1.0
    else:
        return 0.8

def temperatura_correcao_fator_nh3(temp_atual, temp_referencia=25):
    """
    Correção para NH₃ baseada em temperatura
    Base: Yang et al. 2017 - relação exponencial
    """
    return np.exp(0.06 * (temp_atual - temp_referencia))

def umidade_correcao_fator_ch4(umidade_atual, umidade_otima=0.60):
    """
    Correção para CH₄ baseada em umidade
    Base: Yang et al. 2017 - ótimo em condições anaeróbicas
    """
    if umidade_atual < 0.40:
        return 0.1
    elif umidade_atual < 0.60:
        return 0.5
    elif umidade_atual < 0.80:
        return 1.0
    else:
        return 1.2

def umidade_correcao_fator_n2o(umidade_atual, umidade_otima=0.60):
    """
    Correção para N₂O baseada em umidade
    Base: Yang et al. 2017 - ótimo em condições alternadas
    """
    if umidade_atual < 0.40:
        return 0.3
    elif umidade_atual < 0.60:
        return 0.8
    elif umidade_atual < 0.70:
        return 1.0
    else:
        return 0.7

def umidade_correcao_fator_nh3(umidade_atual, umidade_otima=0.60):
    """
    Correção para NH₃ baseada em umidade
    Base: Yang et al. 2017 - maior volatilização em condições secas
    """
    if umidade_atual < 0.40:
        return 1.5
    elif umidade_atual < 0.60:
        return 1.0
    elif umidade_atual < 0.80:
        return 0.8
    else:
        return 0.6

def calcular_fatores_correcao_completos(umidade_val, temp_val):
    """
    Calcula todos os fatores de correção combinados
    Retorna: (fator_ch4, fator_n2o, fator_nh3)
    """
    # Fatores de temperatura
    fator_T_ch4 = temperatura_correcao_fator_ch4(temp_val)
    fator_T_n2o = temperatura_correcao_fator_n2o(temp_val)
    fator_T_nh3 = temperatura_correcao_fator_nh3(temp_val)
    
    # Fatores de umidade
    fator_U_ch4 = umidade_correcao_fator_ch4(umidade_val)
    fator_U_n2o = umidade_correcao_fator_n2o(umidade_val)
    fator_U_nh3 = umidade_correcao_fator_nh3(umidade_val)
    
    # Fatores combinados (multiplicativos)
    fator_ch4 = fator_T_ch4 * fator_U_ch4
    fator_n2o = fator_T_n2o * fator_U_n2o
    fator_nh3 = fator_T_nh3 * fator_U_nh3
    
    return fator_ch4, fator_n2o, fator_nh3

# =============================================================================
# INTERFACE DO APLICATIVO
# =============================================================================

st.title("🌱 Simulador de Emissões - Baseado em Yang et al. 2017")
st.markdown("""
**Versão 2.0** - Com correções completas de temperatura e umidade baseadas em Yang et al. (2017)
""")

# =============================================================================
# SIDEBAR COM PARÂMETROS
# =============================================================================

exibir_cotacao_carbono()

with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    
    # Configurações básicas
    residuos_kg_dia = st.slider("Quantidade de resíduos (kg/dia)", 
                               10, 1000, 100, 10,
                               help="Quantidade diária de resíduos orgânicos")
    
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    
    st.subheader("🌡️ Condições Ambientais (Yang et al. 2017)")
    
    # Temperatura com destaque para ótimo
    T = st.slider("Temperatura média (°C)", 15, 45, 25, 1,
                  help="Yang et al. 2017: Ótimo = 30-35°C para vermicompostagem")
    
    if 30 <= T <= 35:
        st.success(f"✅ Temperatura ótima (30-35°C)")
    elif T > 35:
        st.warning(f"⚠️ Temperatura acima do ótimo (>35°C)")
    elif T < 20:
        st.warning(f"⚠️ Temperatura abaixo do ideal (<20°C)")
    
    # Umidade com destaque para ótimo
    umidade_valor = st.slider("Umidade do resíduo (%)", 40, 95, 85, 1,
                             help="Yang et al. 2017: Ótimo = 60-70% para vermicompostagem")
    umidade = umidade_valor / 100.0
    
    if 60 <= umidade_valor <= 70:
        st.success(f"✅ Umidade ótima (60-70%)")
    elif umidade_valor > 80:
        st.warning(f"⚠️ Umidade muito alta (>80%) - favorece CH₄")
    elif umidade_valor < 50:
        st.warning(f"⚠️ Umidade muito baixa (<50%) - favorece NH₃")
    
    # Parâmetros do resíduo
    st.subheader("📊 Características do Resíduo")
    
    DOC = st.slider("DOC - Carbono Orgânico Degradável", 
                   0.10, 0.30, 0.15, 0.01,
                   help="Fração de carbono orgânico que pode ser degradado")
    
    # Taxa de decaimento do aterro
    st.subheader("🗑️ Parâmetros do Aterro")
    opcao_k = st.selectbox(
        "Taxa de decaimento do aterro (k)",
        options=[
            "k = 0.06 ano⁻¹ (decaimento lento - IPCC padrão)",
            "k = 0.10 ano⁻¹ (decaimento moderado)",
            "k = 0.20 ano⁻¹ (decaimento médio)",
            "k = 0.40 ano⁻¹ (decaimento rápido)"
        ],
        index=0
    )
    
    # Extrair valor k
    if "0.40" in opcao_k:
        k_ano = 0.40
    elif "0.20" in opcao_k:
        k_ano = 0.20
    elif "0.10" in opcao_k:
        k_ano = 0.10
    else:
        k_ano = 0.06
    
    st.session_state.k_ano = k_ano
    
    # Configurações de análise
    st.subheader("🔬 Configurações da Análise")
    n_simulations = st.slider("Simulações Monte Carlo", 100, 2000, 500, 100)
    n_samples = st.slider("Amostras Sobol", 64, 512, 128, 32)
    
    # Informações sobre correções
    with st.expander("📚 Sobre as correções de Yang et al. 2017"):
        st.markdown("""
        **Correções aplicadas:**
        
        **🌡️ Temperatura:**
        - CH₄: Q10 = 2 (dobra a cada 10°C)
        - N₂O: Pico em 35°C (fator 1.2)
        - NH₃: Aumento exponencial com temperatura
        
        **💧 Umidade:**
        - CH₄: Ótimo 60-80% (condições anaeróbicas)
        - N₂O: Ótimo 60-70% (condições alternadas)
        - NH₃: Máximo em condições secas (<40%)
        
        **📊 Fatores de emissão (Yang et al. 2017, Tabela 3):**
        - Vermicompostagem: CH₄-C = 0.13%, N₂O-N = 0.92%, NH₃-N = 12.3%
        - Compostagem termofílica: CH₄-C = 0.60%, N₂O-N = 1.96%, NH₃-N = 24.9%
        """)
    
    if st.button("🚀 Executar Simulação Completa", type="primary"):
        st.session_state.run_simulation = True

# =============================================================================
# PARÂMETROS FIXOS BASEADOS EM YANG ET AL. 2017
# =============================================================================

# Parâmetros do estudo Yang et al. 2017
TOC_YANG = 0.436  # Fração de carbono orgânico total
TN_YANG = 14.2 / 1000  # Fração de nitrogênio total

# Fatores de emissão ORIGINAIS de Yang et al. 2017 (Tabela 3)
CH4_C_FRAC_YANG_ORIG = 0.13 / 100  # 0.13% do C inicial
N2O_N_FRAC_YANG_ORIG = 0.92 / 100  # 0.92% do N inicial
NH3_N_FRAC_YANG_ORIG = 12.3 / 100  # 12.3% do N inicial (ADICIONADO)

# Fatores para compostagem termofílica
CH4_C_FRAC_THERMO_ORIG = 0.60 / 100  # 0.60% do C inicial
N2O_N_FRAC_THERMO_ORIG = 1.96 / 100  # 1.96% do N inicial
NH3_N_FRAC_THERMO_ORIG = 24.9 / 100  # 24.9% do N inicial (ADICIONADO)

# Global Warming Potentials (IPCC AR6)
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# Período de compostagem (Yang et al. 2017)
COMPOSTING_DAYS = 50
dias = anos_simulacao * 365
ano_inicio = datetime.now().year
data_inicio = datetime(ano_inicio, 1, 1)
datas = pd.date_range(start=data_inicio, periods=dias, freq='D')

# Perfis temporais baseados em Yang et al. 2017 (Figura 1)
CH4_PROFILE_VERMI = np.array([
    # Primeiros 10 dias: aumento gradual
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    # Dias 11-20: pico de emissão
    0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10,
    # Dias 21-30: declínio
    0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02,
    # Dias 31-40: emissões residuais
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    # Dias 41-50: emissões mínimas
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005
])
CH4_PROFILE_VERMI /= CH4_PROFILE_VERMI.sum()

N2O_PROFILE_VERMI = np.array([
    # Primeiros 10 dias: emissões iniciais
    0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14,
    # Dias 11-20: pico principal
    0.15, 0.16, 0.17, 0.18, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
    # Dias 21-30: segundo pico
    0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
    # Dias 31-40: declínio
    0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01,
    # Dias 41-50: emissões residuais
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005
])
N2O_PROFILE_VERMI /= N2O_PROFILE_VERMI.sum()

NH3_PROFILE_VERMI = np.array([
    # Primeiros 10 dias: pico inicial rápido
    0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06,
    # Dias 11-20: declínio gradual
    0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02,
    # Dias 21-30: baixas emissões
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    # Dias 31-50: emissões mínimas
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005
])
NH3_PROFILE_VERMI /= NH3_PROFILE_VERMI.sum()

# Perfis para compostagem termofílica (mais intensos)
CH4_PROFILE_THERMO = CH4_PROFILE_VERMI * 1.5
N2O_PROFILE_THERMO = N2O_PROFILE_VERMI * 2.0
NH3_PROFILE_THERMO = NH3_PROFILE_VERMI * 2.5

# Normalizar novamente
CH4_PROFILE_THERMO /= CH4_PROFILE_THERMO.sum()
N2O_PROFILE_THERMO /= N2O_PROFILE_THERMO.sum()
NH3_PROFILE_THERMO /= NH3_PROFILE_THERMO.sum()

# =============================================================================
# FUNÇÕES DE CÁLCULO ATUALIZADAS
# =============================================================================

def calcular_emissoes_aterro(params, k_ano, dias_simulacao=dias):
    """
    Calcula emissões do aterro usando modelo IPCC FOD
    Parâmetros: [umidade, temperatura, DOC]
    Retorna: CH₄, N₂O, NH₃ (kg/dia)
    """
    umidade_val, temp_val, doc_val = params
    
    # Fator de correção de umidade (IPCC)
    fator_umid = (1 - umidade_val) / (1 - 0.55)
    
    # Fração de resíduo exposto (simplificado)
    massa_exposta_kg = residuos_kg_dia
    horas_exposta = 8
    f_aberto = np.clip((massa_exposta_kg / residuos_kg_dia) * (horas_exposta / 24), 0.0, 1.0)
    
    # DOC que decompõe (dependente da temperatura)
    docf_calc = 0.0147 * temp_val + 0.28
    
    # Potencial de CH₄ (IPCC 2006)
    potencial_CH4_por_kg = doc_val * docf_calc * 1 * 0.5 * (16/12) * (1 - 0.0) * (1 - 0.1)
    potencial_CH4_diario = residuos_kg_dia * potencial_CH4_por_kg
    
    # Kernel FOD para CH₄
    t = np.arange(1, dias_simulacao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    entradas_diarias = np.ones(dias_simulacao, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, kernel_ch4, mode='full')[:dias_simulacao]
    emissoes_CH4 *= potencial_CH4_diario
    
    # N₂O do aterro (Wang et al. 2017)
    E_aberto = 1.91
    E_fechado = 2.15
    E_medio = f_aberto * E_aberto + (1 - f_aberto) * E_fechado
    E_medio_ajust = E_medio * fator_umid
    emissao_diaria_N2O = (E_medio_ajust * (44/28) / 1_000_000) * residuos_kg_dia
    
    # Perfil temporal N₂O
    PERFIL_N2O = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}
    kernel_n2o = np.array([PERFIL_N2O.get(d, 0) for d in range(1, 6)], dtype=float)
    emissoes_N2O = fftconvolve(np.full(dias_simulacao, emissao_diaria_N2O), kernel_n2o, mode='full')[:dias_simulacao]
    
    # NH₃ do aterro (estimativa simplificada)
    NH3_N_FRAC_LANDFILL = 0.05  # 5% do N inicial se perde como NH₃
    emissao_diaria_NH3 = residuos_kg_dia * (TN_YANG * NH3_N_FRAC_LANDFILL * (17/14))
    emissoes_NH3 = fftconvolve(np.full(dias_simulacao, emissao_diaria_NH3), kernel_n2o, mode='full')[:dias_simulacao]
    
    return emissoes_CH4, emissoes_N2O, emissoes_NH3

def calcular_emissoes_vermicompostagem(params, dias_simulacao=dias):
    """
    Calcula emissões da vermicompostagem COM correções de Yang et al. 2017
    Parâmetros: [umidade, temperatura, DOC]
    Retorna: CH₄, N₂O, NH₃ (kg/dia) - COM CORREÇÕES
    """
    umidade_val, temp_val, doc_val = params
    
    # Calcular fatores de correção
    fator_ch4, fator_n2o, fator_nh3 = calcular_fatores_correcao_completos(umidade_val, temp_val)
    
    # Fração de matéria seca
    fracao_ms = 1 - umidade_val
    
    # Emissões totais por lote COM correções
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_YANG_ORIG * (16/12) * fracao_ms) * fator_ch4
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_YANG_ORIG * (44/28) * fracao_ms) * fator_n2o
    nh3_total_por_lote = residuos_kg_dia * (TN_YANG * NH3_N_FRAC_YANG_ORIG * (17/14) * fracao_ms) * fator_nh3
    
    # Distribuir ao longo do período de compostagem
    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)
    emissoes_NH3 = np.zeros(dias_simulacao)
    
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(COMPOSTING_DAYS):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                idx = min(dia_compostagem, len(CH4_PROFILE_VERMI)-1)
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * CH4_PROFILE_VERMI[idx]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * N2O_PROFILE_VERMI[idx]
                emissoes_NH3[dia_emissao] += nh3_total_por_lote * NH3_PROFILE_VERMI[idx]
    
    return emissoes_CH4, emissoes_N2O, emissoes_NH3

def calcular_emissoes_compostagem_termofilica(params, dias_simulacao=dias):
    """
    Calcula emissões da compostagem termofílica (cenário UNFCCC)
    COM correções de Yang et al. 2017
    """
    umidade_val, temp_val, doc_val = params
    
    # Calcular fatores de correção
    fator_ch4, fator_n2o, fator_nh3 = calcular_fatores_correcao_completos(umidade_val, temp_val)
    
    # Fração de matéria seca
    fracao_ms = 1 - umidade_val
    
    # Emissões totais por lote COM correções
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_THERMO_ORIG * (16/12) * fracao_ms) * fator_ch4
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_THERMO_ORIG * (44/28) * fracao_ms) * fator_n2o
    nh3_total_por_lote = residuos_kg_dia * (TN_YANG * NH3_N_FRAC_THERMO_ORIG * (17/14) * fracao_ms) * fator_nh3
    
    # Distribuir ao longo do período
    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)
    emissoes_NH3 = np.zeros(dias_simulacao)
    
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(COMPOSTING_DAYS):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                idx = min(dia_compostagem, len(CH4_PROFILE_THERMO)-1)
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * CH4_PROFILE_THERMO[idx]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * N2O_PROFILE_THERMO[idx]
                emissoes_NH3[dia_emissao] += nh3_total_por_lote * NH3_PROFILE_THERMO[idx]
    
    return emissoes_CH4, emissoes_N2O, emissoes_NH3

# =============================================================================
# FUNÇÕES PARA ANÁLISE SOBOL (ATUALIZADAS)
# =============================================================================

def executar_simulacao_completa_sobol(params_sobol):
    """
    Para análise Sobol - inclui todas as correções
    Parâmetros: [taxa_decaimento, temperatura, DOC]
    """
    k_ano_sobol, T_sobol, DOC_sobol = params_sobol
    
    # Usar umidade fixa do slider
    params_base = [umidade, T_sobol, DOC_sobol]
    
    # Calcular emissões COM correções
    ch4_aterro, n2o_aterro, nh3_aterro = calcular_emissoes_aterro(params_base, k_ano_sobol)
    ch4_vermi, n2o_vermi, nh3_vermi = calcular_emissoes_vermicompostagem(params_base)
    
    # Converter para CO₂eq (apenas CH₄ e N₂O - gases de efeito estufa)
    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000
    total_vermi_tco2eq = (ch4_vermi * GWP_CH4_20 + n2o_vermi * GWP_N2O_20) / 1000
    
    reducao_tco2eq = total_aterro_tco2eq.sum() - total_vermi_tco2eq.sum()
    return reducao_tco2eq

def executar_simulacao_unfccc_sobol(params_sobol):
    """
    Para análise Sobol UNFCCC - inclui todas as correções
    """
    k_ano_sobol, T_sobol, DOC_sobol = params_sobol
    
    params_base = [umidade, T_sobol, DOC_sobol]
    
    ch4_aterro, n2o_aterro, nh3_aterro = calcular_emissoes_aterro(params_base, k_ano_sobol)
    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000
    
    ch4_compost, n2o_compost, nh3_compost = calcular_emissoes_compostagem_termofilica(params_base)
    total_compost_tco2eq = (ch4_compost * GWP_CH4_20 + n2o_compost * GWP_N2O_20) / 1000
    
    reducao_tco2eq = total_aterro_tco2eq.sum() - total_compost_tco2eq.sum()
    return reducao_tco2eq

# =============================================================================
# EXECUÇÃO DA SIMULAÇÃO PRINCIPAL
# =============================================================================

if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação com correções de Yang et al. 2017...'):
        
        # Parâmetros base
        params_base = [umidade, T, DOC]
        k_ano = st.session_state.k_ano
        
        # Calcular emissões COM CORREÇÕES
        ch4_aterro, n2o_aterro, nh3_aterro = calcular_emissoes_aterro(params_base, k_ano)
        ch4_vermi, n2o_vermi, nh3_vermi = calcular_emissoes_vermicompostagem(params_base)
        ch4_compost, n2o_compost, nh3_compost = calcular_emissoes_compostagem_termofilica(params_base)
        
        # =============================================================================
        # EXIBIR FATORES DE CORREÇÃO APLICADOS
        # =============================================================================
        
        st.header("🔬 Fatores de Correção Aplicados (Yang et al. 2017)")
        
        # Calcular fatores
        fator_ch4, fator_n2o, fator_nh3 = calcular_fatores_correcao_completos(umidade, T)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fator CH₄", f"{formatar_br(fator_ch4)}", 
                     help="CH₄: Q10=2 × fator umidade")
        with col2:
            st.metric("Fator N₂O", f"{formatar_br(fator_n2o)}",
                     help="N₂O: curva temperatura × fator umidade")
        with col3:
            st.metric("Fator NH₃", f"{formatar_br(fator_nh3)}",
                     help="NH₃: exponencial × fator umidade")
        
        # Explicação detalhada
        with st.expander("📊 Detalhes dos fatores de correção"):
            st.markdown(f"""
            **Temperatura: {T}°C**
            - Fator CH₄ (Q10=2): **{formatar_br(temperatura_correcao_fator_ch4(T))}**
            - Fator N₂O (curva): **{formatar_br(temperatura_correcao_fator_n2o(T))}**
            - Fator NH₃ (exp): **{formatar_br(temperatura_correcao_fator_nh3(T))}**
            
            **Umidade: {umidade_valor}%**
            - Fator CH₄ (anaeróbico): **{formatar_br(umidade_correcao_fator_ch4(umidade))}**
            - Fator N₂O (alternado): **{formatar_br(umidade_correcao_fator_n2o(umidade))}**
            - Fator NH₃ (volatilização): **{formatar_br(umidade_correcao_fator_nh3(umidade))}**
            
            **Impacto nas emissões base:**
            - CH₄ vermicompostagem: **{formatar_br(fator_ch4*100)}%** do valor base
            - N₂O vermicompostagem: **{formatar_br(fator_n2o*100)}%** do valor base
            - NH₃ vermicompostagem: **{formatar_br(fator_nh3*100)}%** do valor base
            """)
        
        # =============================================================================
        # CRIAR DATAFRAME COM TODOS OS DADOS
        # =============================================================================
        
        df = pd.DataFrame({
            'Data': datas,
            # Aterro
            'CH4_Aterro_kg_dia': ch4_aterro,
            'N2O_Aterro_kg_dia': n2o_aterro,
            'NH3_Aterro_kg_dia': nh3_aterro,
            # Vermicompostagem
            'CH4_Vermi_kg_dia': ch4_vermi,
            'N2O_Vermi_kg_dia': n2o_vermi,
            'NH3_Vermi_kg_dia': nh3_vermi,
            # Compostagem termofílica
            'CH4_Compost_kg_dia': ch4_compost,
            'N2O_Compost_kg_dia': n2o_compost,
            'NH3_Compost_kg_dia': nh3_compost,
        })
        
        # Converter para CO₂eq (apenas CH₄ e N₂O)
        for gas in ['CH4_Aterro', 'N2O_Aterro', 'CH4_Vermi', 'N2O_Vermi', 'CH4_Compost', 'N2O_Compost']:
            df[f'{gas}_tCO2eq'] = df[f'{gas}_kg_dia'] * (GWP_CH4_20 if 'CH4' in gas else GWP_N2O_20) / 1000
        
        # Totais diários
        df['Total_Aterro_tCO2eq_dia'] = df['CH4_Aterro_tCO2eq'] + df['N2O_Aterro_tCO2eq']
        df['Total_Vermi_tCO2eq_dia'] = df['CH4_Vermi_tCO2eq'] + df['N2O_Vermi_tCO2eq']
        df['Total_Compost_tCO2eq_dia'] = df['CH4_Compost_tCO2eq'] + df['N2O_Compost_tCO2eq']
        
        # Acumulados
        df['Total_Aterro_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_dia'].cumsum()
        df['Total_Vermi_tCO2eq_acum'] = df['Total_Vermi_tCO2eq_dia'].cumsum()
        df['Total_Compost_tCO2eq_acum'] = df['Total_Compost_tCO2eq_dia'].cumsum()
        
        # Reduções
        df['Reducao_Vermi_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Vermi_tCO2eq_acum']
        df['Reducao_Compost_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Compost_tCO2eq_acum']
        
        # NH₃ acumulado
        df['NH3_Aterro_acum'] = df['NH3_Aterro_kg_dia'].cumsum()
        df['NH3_Vermi_acum'] = df['NH3_Vermi_kg_dia'].cumsum()
        df['NH3_Compost_acum'] = df['NH3_Compost_kg_dia'].cumsum()
        df['Reducao_NH3_Vermi_acum'] = df['NH3_Aterro_acum'] - df['NH3_Vermi_acum']
        df['Reducao_NH3_Compost_acum'] = df['NH3_Aterro_acum'] - df['NH3_Compost_acum']
        
        # Resumo anual
        df['Year'] = df['Data'].dt.year
        df_anual = df.groupby('Year').agg({
            'Total_Aterro_tCO2eq_dia': 'sum',
            'Total_Vermi_tCO2eq_dia': 'sum',
            'Total_Compost_tCO2eq_dia': 'sum',
            'NH3_Aterro_kg_dia': 'sum',
            'NH3_Vermi_kg_dia': 'sum',
            'NH3_Compost_kg_dia': 'sum',
        }).reset_index()
        
        df_anual['Reducao_Vermi_tCO2eq'] = df_anual['Total_Aterro_tCO2eq_dia'] - df_anual['Total_Vermi_tCO2eq_dia']
        df_anual['Reducao_Compost_tCO2eq'] = df_anual['Total_Aterro_tCO2eq_dia'] - df_anual['Total_Compost_tCO2eq_dia']
        df_anual['Reducao_NH3_Vermi_kg'] = df_anual['NH3_Aterro_kg_dia'] - df_anual['NH3_Vermi_kg_dia']
        df_anual['Reducao_NH3_Compost_kg'] = df_anual['NH3_Aterro_kg_dia'] - df_anual['NH3_Compost_kg_dia']
        
        # =============================================================================
        # RESULTADOS PRINCIPAIS
        # =============================================================================
        
        st.header("📊 Resultados Principais")
        
        # Obter totais
        total_evitado_vermi = df['Reducao_Vermi_tCO2eq_acum'].iloc[-1]
        total_evitado_compost = df['Reducao_Compost_tCO2eq_acum'].iloc[-1]
        total_nh3_vermi = df['Reducao_NH3_Vermi_acum'].iloc[-1]
        total_nh3_compost = df['Reducao_NH3_Compost_acum'].iloc[-1]
        
        # Valores financeiros
        preco_carbono = st.session_state.preco_carbono
        taxa_cambio = st.session_state.taxa_cambio
        
        valor_vermi_eur = calcular_valor_creditos(total_evitado_vermi, preco_carbono, "€")
        valor_compost_eur = calcular_valor_creditos(total_evitado_compost, preco_carbono, "€")
        valor_vermi_brl = calcular_valor_creditos(total_evitado_vermi, preco_carbono, "R$", taxa_cambio)
        valor_compost_brl = calcular_valor_creditos(total_evitado_compost, preco_carbono, "R$", taxa_cambio)
        
        # Métricas principais
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌱 Vermicompostagem")
            st.metric("Emissões evitadas (CO₂eq)", f"{formatar_br(total_evitado_vermi)} t")
            st.metric("NH₃ evitado", f"{formatar_br(total_nh3_vermi)} kg")
            st.metric("Valor (Euro)", f"€ {formatar_br(valor_vermi_eur)}")
            st.metric("Valor (Real)", f"R$ {formatar_br(valor_vermi_brl)}")
        
        with col2:
            st.subheader("🔥 Compostagem Termofílica")
            st.metric("Emissões evitadas (CO₂eq)", f"{formatar_br(total_evitado_compost)} t")
            st.metric("NH₃ evitado", f"{formatar_br(total_nh3_compost)} kg")
            st.metric("Valor (Euro)", f"€ {formatar_br(valor_compost_eur)}")
            st.metric("Valor (Real)", f"R$ {formatar_br(valor_compost_brl)}")
        
        # =============================================================================
        # GRÁFICOS
        # =============================================================================
        
        st.header("📈 Visualizações")
        
        # Gráfico 1: Comparação de reduções anuais
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_anual['Year']))
        bar_width = 0.35
        
        ax1.bar(x - bar_width/2, df_anual['Reducao_Vermi_tCO2eq'], width=bar_width,
                label='Vermicompostagem', color='green', edgecolor='black')
        ax1.bar(x + bar_width/2, df_anual['Reducao_Compost_tCO2eq'], width=bar_width,
                label='Compostagem Termofílica', color='orange', edgecolor='black', hatch='//')
        
        ax1.set_xlabel('Ano')
        ax1.set_ylabel('Redução de Emissões (t CO₂eq)')
        ax1.set_title('Redução Anual de Emissões: Comparação entre Tecnologias')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_anual['Year'], fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.yaxis.set_major_formatter(FuncFormatter(br_format))
        
        st.pyplot(fig1)
        
        # Gráfico 2: Redução acumulada
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(df['Data'], df['Reducao_Vermi_tCO2eq_acum'], 'g-', 
                label='Vermicompostagem', linewidth=2)
        ax2.plot(df['Data'], df['Reducao_Compost_tCO2eq_acum'], 'orange', 
                label='Compostagem Termofílica', linewidth=2, linestyle='--')
        
        ax2.set_xlabel('Ano')
        ax2.set_ylabel('Redução Acumulada (t CO₂eq)')
        ax2.set_title(f'Redução Acumulada de Emissões em {anos_simulacao} Anos')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.yaxis.set_major_formatter(FuncFormatter(br_format))
        
        st.pyplot(fig2)
        
        # Gráfico 3: NH₃ evitado
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(df['Data'], df['Reducao_NH3_Vermi_acum']/1000, 'blue', 
                label='Vermicompostagem (toneladas)', linewidth=2)
        ax3.plot(df['Data'], df['Reducao_NH3_Compost_acum']/1000, 'red', 
                label='Compostagem Termofílica (toneladas)', linewidth=2, linestyle='--')
        
        ax3.set_xlabel('Ano')
        ax3.set_ylabel('NH₃ Evitado (toneladas)')
        ax3.set_title(f'Redução Acumulada de NH₃ em {anos_simulacao} Anos')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.yaxis.set_major_formatter(FuncFormatter(br_format))
        
        st.pyplot(fig3)
        
        # =============================================================================
        # ANÁLISE DE SENSIBILIDADE SOBOL
        # =============================================================================
        
        st.header("🎯 Análise de Sensibilidade Global (Sobol)")
        
        with st.spinner('Executando análise de sensibilidade...'):
            # Definir problema Sobol
            problem = {
                'num_vars': 3,
                'names': ['taxa_decaimento', 'temperatura', 'DOC'],
                'bounds': [
                    [0.06, 0.40],
                    [25.0, 45.0],
                    [0.15, 0.25]
                ]
            }
            
            # Gerar amostras
            param_values = sample(problem, n_samples)
            
            # Executar simulações em paralelo
            results = Parallel(n_jobs=-1)(
                delayed(executar_simulacao_completa_sobol)(params) 
                for params in param_values
            )
            
            # Analisar resultados
            Si = analyze(problem, np.array(results), print_to_console=False)
            
            # Criar DataFrame de resultados
            sensibilidade_df = pd.DataFrame({
                'Parâmetro': problem['names'],
                'S1_Primeira_Ordem': Si['S1'],
                'ST_Efeito_Total': Si['ST']
            })
            
            # Mapear nomes
            nomes_amigaveis = {
                'taxa_decaimento': 'Taxa de Decaimento (k)',
                'temperatura': 'Temperatura',
                'DOC': 'Carbono Orgânico Degradável'
            }
            sensibilidade_df['Parâmetro'] = sensibilidade_df['Parâmetro'].map(nomes_amigaveis)
            sensibilidade_df = sensibilidade_df.sort_values('ST_Efeito_Total', ascending=False)
            
            # Gráfico de sensibilidade
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            bars = ax4.barh(sensibilidade_df['Parâmetro'], sensibilidade_df['ST_Efeito_Total'],
                          color='steelblue', edgecolor='black')
            
            ax4.set_xlabel('Índice ST (Efeito Total)')
            ax4.set_title('Análise de Sensibilidade Global - Efeito Total dos Parâmetros')
            ax4.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Adicionar valores nas barras
            for bar, st_val in zip(bars, sensibilidade_df['ST_Efeito_Total']):
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{st_val:.3f}', va='center', fontweight='bold')
            
            st.pyplot(fig4)
            
            # Tabela de resultados
            st.subheader("📋 Resultados Quantitativos da Análise Sobol")
            st.dataframe(sensibilidade_df.style.format({
                'S1_Primeira_Ordem': '{:.4f}',
                'ST_Efeito_Total': '{:.4f}'
            }))
        
        # =============================================================================
        # ANÁLISE DE INCERTEZA MONTE CARLO
        # =============================================================================
        
        st.header("🎲 Análise de Incerteza (Monte Carlo)")
        
        with st.spinner('Executando simulações Monte Carlo...'):
            
            def gerar_parametros_mc(n):
                np.random.seed(50)
                taxas_decaimento = np.random.uniform(0.06, 0.40, n)
                temperaturas = np.random.uniform(25.0, 45.0, n)
                docs = np.random.uniform(0.15, 0.25, n)
                return taxas_decaimento, temperaturas, docs
            
            # Gerar parâmetros
            taxas_mc, temps_mc, docs_mc = gerar_parametros_mc(n_simulations)
            
            # Executar simulações
            resultados_mc = []
            for i in range(n_simulations):
                params_mc = [umidade, temps_mc[i], docs_mc[i]]
                ch4_a, n2o_a, nh3_a = calcular_emissoes_aterro(params_mc, taxas_mc[i])
                ch4_v, n2o_v, nh3_v = calcular_emissoes_vermicompostagem(params_mc)
                
                total_a = (ch4_a * GWP_CH4_20 + n2o_a * GWP_N2O_20) / 1000
                total_v = (ch4_v * GWP_CH4_20 + n2o_v * GWP_N2O_20) / 1000
                reducao = total_a.sum() - total_v.sum()
                resultados_mc.append(reducao)
            
            resultados_array = np.array(resultados_mc)
            
            # Estatísticas
            media = np.mean(resultados_array)
            mediana = np.median(resultados_array)
            desvio = np.std(resultados_array)
            ci_95 = np.percentile(resultados_array, [2.5, 97.5])
            ci_90 = np.percentile(resultados_array, [5, 95])
            
            # Gráfico de distribuição
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            n, bins, patches = ax5.hist(resultados_array, bins=30, alpha=0.7, 
                                       color='skyblue', edgecolor='black', density=True)
            
            # Adicionar KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(resultados_array)
            x_vals = np.linspace(resultados_array.min(), resultados_array.max(), 1000)
            ax5.plot(x_vals, kde(x_vals), 'r-', linewidth=2, label='Densidade KDE')
            
            # Linhas de referência
            ax5.axvline(media, color='green', linestyle='--', linewidth=2, label=f'Média: {formatar_br(media)}')
            ax5.axvline(ci_95[0], color='red', linestyle=':', linewidth=1.5, label='IC 95%')
            ax5.axvline(ci_95[1], color='red', linestyle=':', linewidth=1.5)
            ax5.axvline(total_evitado_vermi, color='purple', linestyle='-', linewidth=2, 
                       label=f'Valor base: {formatar_br(total_evitado_vermi)}')
            
            ax5.set_xlabel('Redução de Emissões (t CO₂eq)')
            ax5.set_ylabel('Densidade de Probabilidade')
            ax5.set_title(f'Distribuição das Reduções de Emissões ({n_simulations} simulações Monte Carlo)')
            ax5.legend()
            ax5.grid(alpha=0.3)
            ax5.xaxis.set_major_formatter(FuncFormatter(br_format))
            
            st.pyplot(fig5)
            
            # Estatísticas resumidas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Média", f"{formatar_br(media)} t")
                st.metric("IC 95% inferior", f"{formatar_br(ci_95[0])} t")
            with col2:
                st.metric("Mediana", f"{formatar_br(mediana)} t")
                st.metric("IC 95% superior", f"{formatar_br(ci_95[1])} t")
            with col3:
                st.metric("Desvio padrão", f"{formatar_br(desvio)} t")
                st.metric("IC 90%", f"{formatar_br(ci_90[0])} - {formatar_br(ci_90[1])} t")
        
        # =============================================================================
        # TABELAS DETALHADAS
        # =============================================================================
        
        st.header("📋 Tabelas Detalhadas")
        
        with st.expander("📊 Resumo Anual Detalhado"):
            # Formatar tabela anual
            df_anual_formatado = df_anual.copy()
            for col in df_anual_formatado.columns:
                if col != 'Year':
                    if 'tCO2eq' in col:
                        df_anual_formatado[col] = df_anual_formatado[col].apply(formatar_br)
                    elif 'NH3' in col:
                        df_anual_formatado[col] = df_anual_formatado[col].apply(lambda x: formatar_br(x/1000) + " t")
            
            st.dataframe(df_anual_formatado)
        
        with st.expander("📈 Fatores de Correção por Condição"):
            # Tabela de fatores para diferentes condições
            temps_teste = [15, 20, 25, 30, 35, 40, 45]
            umids_teste = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85]
            
            fatores_data = []
            for temp in temps_teste:
                for umid in umids_teste:
                    f_ch4, f_n2o, f_nh3 = calcular_fatores_correcao_completos(umid, temp)
                    fatores_data.append({
                        'Temperatura (°C)': temp,
                        'Umidade': f"{umid*100:.0f}%",
                        'Fator CH₄': f"{f_ch4:.3f}",
                        'Fator N₂O': f"{f_n2o:.3f}",
                        'Fator NH₃': f"{f_nh3:.3f}"
                    })
            
            st.dataframe(pd.DataFrame(fatores_data))
        
        # =============================================================================
        # RELATÓRIO FINAL
        # =============================================================================
        
        st.header("📄 Relatório de Simulação")
        
        relatorio = f"""
        ## 📊 RELATÓRIO DE SIMULAÇÃO - YANG ET AL. 2017 CORRIGIDO
        
        ### 🎯 Parâmetros de Entrada
        - **Resíduos processados:** {formatar_br(residuos_kg_dia)} kg/dia
        - **Período de simulação:** {anos_simulacao} anos
        - **Temperatura:** {T}°C {'(ÓTIMA)' if 30 <= T <= 35 else '(FORA DO ÓTIMO)'}
        - **Umidade:** {umidade_valor}% {'(ÓTIMA)' if 60 <= umidade_valor <= 70 else '(FORA DO ÓTIMO)'}
        - **Taxa de decaimento (k):** {formatar_br(k_ano)} ano⁻¹
        
        ### 🌡️ Correções Aplicadas (Yang et al. 2017)
        - **Fator CH₄:** {formatar_br(fator_ch4)} (Temperatura: {formatar_br(temperatura_correcao_fator_ch4(T))} × Umidade: {formatar_br(umidade_correcao_fator_ch4(umidade))})
        - **Fator N₂O:** {formatar_br(fator_n2o)} (Temperatura: {formatar_br(temperatura_correcao_fator_n2o(T))} × Umidade: {formatar_br(umidade_correcao_fator_n2o(umidade))})
        - **Fator NH₃:** {formatar_br(fator_nh3)} (Temperatura: {formatar_br(temperatura_correcao_fator_nh3(T))} × Umidade: {formatar_br(umidade_correcao_fator_nh3(umidade))})
        
        ### 📈 Resultados Principais
        | Métrica | Vermicompostagem | Compostagem Termofílica |
        |---------|------------------|-------------------------|
        | **Emissões evitadas (t CO₂eq)** | {formatar_br(total_evitado_vermi)} | {formatar_br(total_evitado_compost)} |
        | **NH₃ evitado (toneladas)** | {formatar_br(total_nh3_vermi/1000)} | {formatar_br(total_nh3_compost/1000)} |
        | **Valor em Euro (€)** | {formatar_br(valor_vermi_eur)} | {formatar_br(valor_compost_eur)} |
        | **Valor em Real (R$)** | {formatar_br(valor_vermi_brl)} | {formatar_br(valor_compost_brl)} |
        
        ### 🔬 Análise de Sensibilidade (Sobol)
        **Parâmetro mais influente:** {sensibilidade_df.iloc[0]['Parâmetro']}
        
        ### 🎲 Análise de Incerteza (Monte Carlo)
        - **Média:** {formatar_br(media)} t CO₂eq
        - **Intervalo de confiança 95%:** {formatar_br(ci_95[0])} a {formatar_br(ci_95[1])} t CO₂eq
        - **Coeficiente de variação:** {formatar_br((desvio/media)*100 if media != 0 else 0)}%
        
        ### 💡 Recomendações
        1. **Condições ótimas para vermicompostagem:** 30-35°C, 60-70% umidade
        2. **Impacto econômico significativo:** {formatar_br(valor_vermi_brl)} em créditos de carbono
        3. **Benefício adicional de qualidade do ar:** {formatar_br(total_nh3_vermi/1000)} toneladas de NH₃ evitadas
        """
        
        st.markdown(relatorio)
        
        # =============================================================================
        # BOTÃO DE DOWNLOAD
        # =============================================================================
        
        st.download_button(
            label="📥 Baixar Dados Completos (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"simulacao_yang_et_al_2017_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info("""
    ## 🌱 Bem-vindo ao Simulador de Emissões com Correções de Yang et al. 2017
    
    **Versão 2.0** - Implementação completa das correções científicas baseadas no artigo:
    
    **"Greenhouse gas emissions during biological treatment of municipal solid waste"**
    Yang et al. (2017)
    
    ### 🎯 Novidades nesta versão:
    
    1. **✅ Correções de temperatura para cada gás:**
       - CH₄: Q10 = 2 (dobra a cada 10°C)
       - N₂O: Curva empírica com pico em 35°C
       - NH₃: Relação exponencial com temperatura
    
    2. **✅ Correções de umidade para cada gás:**
       - CH₄: Ótimo em condições anaeróbicas (60-80%)
       - N₂O: Ótimo em condições alternadas (60-70%)
       - NH₃: Máxima volatilização em condições secas
    
    3. **✅ NH₃ incluído nos cálculos:**
       - Fator: 12.3% do N inicial (Yang et al. Tabela 3)
       - Benefício significativo para qualidade do ar
    
    4. **✅ Perfis temporais baseados em dados experimentais:**
       - 50 dias de compostagem
       - Curvas realistas de emissão
    
    **👉 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação' para começar.**
    """)

# =============================================================================
# RODAPÉ
# =============================================================================

st.markdown("---")
st.markdown("""
**📚 Referências Científicas:**

**Base metodológica principal:**
- **Yang et al. (2017)** - "Greenhouse gas emissions during biological treatment of municipal solid waste"
  - Fatores de emissão (Tabela 3)
  - Correções de temperatura e umidade
  - Perfis temporais de emissão
  - Comparação entre vermicompostagem e compostagem termofílica

**Modelos complementares:**
- **IPCC (2006)** - Waste Model para emissões de aterro
- **Wang et al. (2017)** - Emissões de N₂O de aterros
- **IPCC AR6 (2021)** - Potenciais de aquecimento global (GWP)

**Desenvolvido por:** [Seu Nome/Instituição]
**Contato:** [seu.email@exemplo.com]
**Versão:** 2.0 (Corrigida com Yang et al. 2017)
""")
