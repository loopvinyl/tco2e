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
st.set_page_config(page_title="Simulador de Emissões CO₂eq", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# VARIÁVEIS GLOBAIS COM VALORES PADRÃO
# =============================================================================

# Valores padrão que serão sobrescritos pelos sliders
residuos_kg_dia = 100
massa_exposta_kg = 100
h_exposta = 8

# =============================================================================
# FUNÇÕES DE COTAÇÃO AUTOMÁTICA DO CARBONO E CÂMBIO
# =============================================================================

def obter_cotacao_carbono_investing():
    """
    Obtém a cotação em tempo real do carbono via web scraping do Investing.com
    """
    try:
        url = "https://www.investing.com/commodities/carbon-emissions"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.investing.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Várias estratégias para encontrar o preço
        selectores = [
            '[data-test="instrument-price-last"]',
            '.text-2xl',
            '.last-price-value',
            '.instrument-price-last',
            '.pid-1062510-last',
            '.float_lang_base_1',
            '.top.bold.inlineblock',
            '#last_last'
        ]
        
        preco = None
        fonte = "Investing.com"
        
        for seletor in selectores:
            try:
                elemento = soup.select_one(seletor)
                if elemento:
                    texto_preco = elemento.text.strip().replace(',', '')
                    # Remover caracteres não numéricos exceto ponto
                    texto_preco = ''.join(c for c in texto_preco if c.isdigit() or c == '.')
                    if texto_preco:
                        preco = float(texto_preco)
                        break
            except (ValueError, AttributeError):
                continue
        
        if preco is not None:
            return preco, "€", "Carbon Emissions Future", True, fonte
        
        # Tentativa alternativa: procurar por padrões numéricos no HTML
        import re
        padroes_preco = [
            r'"last":"([\d,]+)"',
            r'data-last="([\d,]+)"',
            r'last_price["\']?:\s*["\']?([\d,]+)',
            r'value["\']?:\s*["\']?([\d,]+)'
        ]
        
        html_texto = str(soup)
        for padrao in padroes_preco:
            matches = re.findall(padrao, html_texto)
            for match in matches:
                try:
                    preco_texto = match.replace(',', '')
                    preco = float(preco_texto)
                    if 50 < preco < 200:  # Faixa razoável para carbono
                        return preco, "€", "Carbon Emissions Future", True, fonte
                except ValueError:
                    continue
                    
        return None, None, None, False, fonte
        
    except Exception as e:
        return None, None, None, False, f"Investing.com - Erro: {str(e)}"

def obter_cotacao_carbono():
    """
    Obtém a cotação em tempo real do carbono - usa apenas Investing.com
    """
    # Tentar via Investing.com
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    
    if sucesso:
        return preco, moeda, f"{contrato_info}", True, fonte
    
    # Fallback para valor padrão
    return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"

def obter_cotacao_euro_real():
    """
    Obtém a cotação em tempo real do Euro em relação ao Real Brasileiro
    """
    try:
        # API do BCB
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except:
        pass
    
    try:
        # Fallback para API alternativa
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = data['rates']['BRL']
            return cotacao, "R$", True, "ExchangeRate-API"
    except:
        pass
    
    # Fallback para valor de referência
    return 5.50, "R$", False, "Referência"

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    """
    Calcula o valor financeiro das emissões evitadas baseado no preço do carbono
    """
    valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
    return valor_total

def exibir_cotacao_carbono():
    """
    Exibe a cotação do carbono com informações - ATUALIZADA AUTOMATICAMENTE
    """
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    # Atualização automática na primeira execução
    if not st.session_state.get('cotacao_carregada', False):
        st.session_state.mostrar_atualizacao = True
        st.session_state.cotacao_carregada = True
    
    # Botão para atualizar cotações
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Atualizar Cotações", key="atualizar_cotacoes"):
            st.session_state.cotacao_atualizada = True
            st.session_state.mostrar_atualizacao = True
    
    # Mostrar mensagem de atualização se necessário
    if st.session_state.get('mostrar_atualizacao', False):
        st.sidebar.info("🔄 Atualizando cotações...")
        
        # Obter cotação do carbono
        preco_carbono, moeda, contrato_info, sucesso_carbono, fonte_carbono = obter_cotacao_carbono()
        
        # Obter cotação do Euro
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        
        # Atualizar session state
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        
        # Resetar flags
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        
        st.rerun()

    # Exibe cotação atual do carbono com formatação brasileira
    st.sidebar.metric(
        label=f"Preço do Carbono (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    
    # Exibe cotação atual do Euro com formatação brasileira
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {formatar_br(st.session_state.taxa_cambio)}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    # Calcular preço do carbono em Reais com formatação brasileira
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {formatar_br(preco_carbono_reais)}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )
    
    # Informações adicionais
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**
        - **Fonte do Carbono:** {st.session_state.fonte_cotacao}
        - **Preço Atual:** {st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}/tCO₂eq
        - **Câmbio EUR/BRL:** 1 Euro = R$ {formatar_br(st.session_state.taxa_cambio)}
        - **Carbono em Reais:** R$ {formatar_br(preco_carbono_reais)}/tCO₂eq
        
        **🌍 Mercado de Referência:**
        - European Union Allowances (EUA)
        - European Emissions Trading System (EU ETS)
        - Contratos futuros de carbono
        - Preços em tempo real
        
        **🔄 Atualização:**
        - As cotações são carregadas automaticamente ao abrir o aplicativo
        - Clique em **"Atualizar Cotações"** para obter valores mais recentes
        - Em caso de falha na conexão, são utilizados valores de referência atualizados
        
        **💡 Importante:**
        - Os preços são baseados no mercado regulado da UE
        - Valores em tempo real sujeitos a variações de mercado
        - Conversão para Real utilizando câmbio comercial
        """)

# =============================================================================
# INICIALIZAÇÃO DA SESSION STATE
# =============================================================================

# Inicializar todas as variáveis de session state necessárias
def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        # Buscar cotação automaticamente na inicialização
        preco_carbono, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
        
    if 'taxa_cambio' not in st.session_state:
        # Buscar cotação do Euro automaticamente
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
    if 'run_analise_lote' not in st.session_state:
        st.session_state.run_analise_lote = False

# Chamar a inicialização
inicializar_session_state()

# =============================================================================
# FUNÇÕES ORIGINAIS DO SEU SCRIPT
# =============================================================================

# Função para formatar números no padrão brasileiro
def formatar_br(numero):
    """
    Formata números no padrão brasileiro: 1.234,56
    """
    if pd.isna(numero):
        return "N/A"
    
    # Arredonda para 2 casas decimais
    numero = round(numero, 2)
    
    # Formata como string e substitui o ponto pela vírgula
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Função para formatar com número específico de casas decimais
def formatar_br_dec(numero, decimais=2):
    """
    Formata números no padrão brasileiro com número específico de casas decimais
    """
    if pd.isna(numero):
        return "N/A"
    
    # Arredonda para o número de casas decimais especificado
    numero = round(numero, decimais)
    
    # Formata como string e substitui o ponto pela vírgula
    return f"{numero:,.{decimais}f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Função de formatação para os gráficos
def br_format(x, pos):
    """
    Função de formatação para eixos de gráficos (padrão brasileiro)
    """
    if x == 0:
        return "0"
    
    # Para valores muito pequenos, usa notação científica
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    
    # Para valores grandes, formata com separador de milhar
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    # Para valores menores, mostra duas casas decimais
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =============================================================================
# FUNÇÕES PARA ANÁLISE POR LOTE ÚNICO (100 kg) - ADICIONADAS
# =============================================================================

def calcular_potencial_metano_aterro(residuos_kg, umidade, temperatura, dias=365):
    """
    Calcula o potencial de geração de metano de um lote de resíduos no aterro
    Baseado na metodologia IPCC 2006 - CORRIGIDO: Kernel NÃO normalizado
    """
    # Parâmetros fixos (IPCC 2006)
    DOC = 0.15  # Carbono orgânico degradável (fração)
    MCF = 1.0   # Fator de correção de metano (para aterros sanitários)
    F = 0.5     # Fração de metano no biogás
    OX = 0.1    # Fator de oxidação
    Ri = 0.0    # Metano recuperado
    
    # DOCf calculado pela temperatura (DOCf = 0.0147 × T + 0.28)
    DOCf = 0.0147 * temperatura + 0.28
    
    # Cálculo do potencial de metano por kg de resíduo
    potencial_CH4_por_kg = DOC * DOCf * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    
    # Potencial total do lote
    potencial_CH4_total = residuos_kg * potencial_CH4_por_kg
    
    # CORREÇÃO: Taxa de decaimento anual
    k_ano = 0.06  # Constante de decaimento anual (6% ao ano)
    k_dia = k_ano / 365.0  # Taxa de decaimento diária
    
    # Gerar emissões ao longo do tempo
    t = np.arange(1, dias + 1, dtype=float)
    
    # CORREÇÃO: Kernel NÃO normalizado (IPCC correto)
    kernel_ch4 = np.exp(-k_dia * (t - 1)) - np.exp(-k_dia * t)
    
    # Garantir que não há valores negativos (pode ocorrer por erro numérico)
    kernel_ch4 = np.maximum(kernel_ch4, 0)
    
    # NÃO NORMALIZAR o kernel - manter a fração correta da equação diferencial
    # A soma do kernel não será 1, mas sim a fração total emitida no período
    
    # Distribuir o potencial total ao longo do tempo
    emissoes_CH4 = potencial_CH4_total * kernel_ch4
    
    # Calcular fração total emitida no período
    fracao_total_emitida = kernel_ch4.sum()
    
    return emissoes_CH4, potencial_CH4_total, DOCf, fracao_total_emitida

def calcular_emissoes_vermicompostagem_lote(residuos_kg, umidade, dias=50):
    """
    Calcula emissões de metano na vermicompostagem (Yang et al. 2017) - ANÁLISE POR LOTE
    """
    # Parâmetros fixos para vermicompostagem
    TOC = 0.436  # Fração de carbono orgânico total
    CH4_C_FRAC = 0.13 / 100  # Fração do TOC emitida como CH4-C (0.13%)
    fracao_ms = 1 - umidade  # Fração de matéria seca
    
    # Metano total por lote
    ch4_total_por_lote = residuos_kg * (TOC * CH4_C_FRAC * (16/12) * fracao_ms)
    
    # Perfil temporal baseado em Yang et al. (2017)
    perfil_ch4 = np.array([
        0.02, 0.02, 0.02, 0.03, 0.03,  # Dias 1-5
        0.04, 0.04, 0.05, 0.05, 0.06,  # Dias 6-10
        0.07, 0.08, 0.09, 0.10, 0.09,  # Dias 11-15
        0.08, 0.07, 0.06, 0.05, 0.04,  # Dias 16-20
        0.03, 0.02, 0.02, 0.01, 0.01,  # Dias 21-25
        0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 26-30
        0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 31-35
        0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 36-40
        0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 41-45
        0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
    ])
    
    # Normalizar perfil (para processos de curta duração, normalização é aceitável)
    perfil_ch4 = perfil_ch4 / perfil_ch4.sum()
    
    # Distribuir emissões
    emissoes_CH4 = ch4_total_por_lote * perfil_ch4
    
    return emissoes_CH4, ch4_total_por_lote

def calcular_emissoes_compostagem_lote(residuos_kg, umidade, dias=50):
    """
    Calcula emissões de metano na compostagem termofílica (Yang et al. 2017) - ANÁLISE POR LOTE
    """
    # Parâmetros fixos para compostagem termofílica
    TOC = 0.436  # Fração de carbono orgânico total
    CH4_C_FRAC = 0.006  # Fração do TOC emitida como CH4-C (0.6%)
    fracao_ms = 1 - umidade  # Fração de matéria seca
    
    # Metano total por lote
    ch4_total_por_lote = residuos_kg * (TOC * CH4_C_FRAC * (16/12) * fracao_ms)
    
    # Perfil temporal para compostagem termofílica
    perfil_ch4 = np.array([
        0.01, 0.02, 0.03, 0.05, 0.08,  # Dias 1-5
        0.12, 0.15, 0.18, 0.20, 0.18,  # Dias 6-10 (pico termofílico)
        0.15, 0.12, 0.10, 0.08, 0.06,  # Dias 11-15
        0.05, 0.04, 0.03, 0.02, 0.02,  # Dias 16-20
        0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 21-25
        0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
        0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
        0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
        0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
        0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
    ])
    
    # Normalizar perfil (para processos de curta duração, normalização é aceitável)
    perfil_ch4 = perfil_ch4 / perfil_ch4.sum()
    
    # Distribuir emissões
    emissoes_CH4 = ch4_total_por_lote * perfil_ch4
    
    return emissoes_CH4, ch4_total_por_lote

# =============================================================================
# PARÂMETROS FIXOS (DO CÓDIGO ORIGINAL)
# =============================================================================

T = 25  # Temperatura média (ºC)
DOC = 0.15  # Carbono orgânico degradável (fração)
DOCf_val = 0.0147 * T + 0.28
MCF = 1  # Fator de correção de metano
F = 0.5  # Fração de metano no biogás
OX = 0.1  # Fator de oxidação
Ri = 0.0  # Metano recuperado

# Constante de decaimento (fixa como no script anexo)
k_ano = 0.06  # Constante de decaimento anual

# Vermicompostagem (Yang et al. 2017) - valores fixos
TOC_YANG = 0.436  # Fração de carbono orgânico total
TN_YANG = 14.2 / 1000  # Fração de nitrogênio total
CH4_C_FRAC_YANG = 0.13 / 100  # Fração do TOC emitida como CH4-C (fixo)
N2O_N_FRAC_YANG = 0.92 / 100  # Fração do TN emitida como N2O-N (fixo)
DIAS_COMPOSTAGEM = 50  # Período total de compostagem

# Perfil temporal de emissões baseado em Yang et al. (2017)
PERFIL_CH4_VERMI = np.array([
    0.02, 0.02, 0.02, 0.03, 0.03,  # Dias 1-5
    0.04, 0.04, 0.05, 0.05, 0.06,  # Dias 6-10
    0.07, 0.08, 0.09, 0.10, 0.09,  # Dias 11-15
    0.08, 0.07, 0.06, 0.05, 0.04,  # Dias 16-20
    0.03, 0.02, 0.02, 0.01, 0.01,  # Dias 21-25
    0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 26-30
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 31-35
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 36-40
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_CH4_VERMI /= PERFIL_CH4_VERMI.sum()

PERFIL_N2O_VERMI = np.array([
    0.15, 0.10, 0.20, 0.05, 0.03,  # Dias 1-5 (pico no dia 3)
    0.03, 0.03, 0.04, 0.05, 0.06,  # Dias 6-10
    0.08, 0.09, 0.10, 0.08, 0.07,  # Dias 11-15
    0.06, 0.05, 0.04, 0.03, 0.02,  # Dias 16-20
    0.01, 0.01, 0.005, 0.005, 0.005,  # Dias 21-25
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_N2O_VERMI /= PERFIL_N2O_VERMI.sum()

# Emissões pré-descarte (Feng et al. 2020)
CH4_pre_descarte_ugC_por_kg_h_min = 0.18
CH4_pre_descarte_ugC_por_kg_h_max = 5.38
CH4_pre_descarte_ugC_por_kg_h_media = 2.78

fator_conversao_C_para_CH4 = 16/12
CH4_pre_descarte_ugCH4_por_kg_h_media = CH4_pre_descarte_ugC_por_kg_h_media * fator_conversao_C_para_CH4
CH4_pre_descarte_g_por_kg_dia = CH4_pre_descarte_ugCH4_por_kg_h_media * 24 / 1_000_000

N2O_pre_descarte_mgN_por_kg = 20.26
N2O_pre_descarte_mgN_por_kg_dia = N2O_pre_descarte_mgN_por_kg / 3
N2O_pre_descarte_g_por_kg_dia = N2O_pre_descarte_mgN_por_kg_dia * (44/28) / 1000

PERFIL_N2O_PRE_DESCARTE = {1: 0.8623, 2: 0.10, 3: 0.0377}

# GWP (IPCC AR6)
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# Período de Simulação - AGORA DENTRO DA FUNÇÃO
ano_inicio = datetime.now().year
data_inicio = datetime(ano_inicio, 1, 1)

# Perfil temporal N2O (Wang et al. 2017)
PERFIL_N2O = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}

# Valores específicos para compostagem termofílica (Yang et al. 2017) - valores fixos
CH4_C_FRAC_THERMO = 0.006  # Fixo
N2O_N_FRAC_THERMO = 0.0196  # Fixo

PERFIL_CH4_THERMO = np.array([
    0.01, 0.02, 0.03, 0.05, 0.08,  # Dias 1-5
    0.12, 0.15, 0.18, 0.20, 0.18,  # Dias 6-10 (pico termofílico)
    0.15, 0.12, 0.10, 0.08, 0.06,  # Dias 11-15
    0.05, 0.04, 0.03, 0.02, 0.02,  # Dias 16-20
    0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 21-25
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_CH4_THERMO /= PERFIL_CH4_THERMO.sum()

PERFIL_N2O_THERMO = np.array([
    0.10, 0.08, 0.15, 0.05, 0.03,  # Dias 1-5
    0.04, 0.05, 0.07, 0.10, 0.12,  # Dias 6-10
    0.15, 0.18, 0.20, 0.18, 0.15,  # Dias 11-15 (pico termofílico)
    0.12, 0.10, 0.08, 0.06, 0.05,  # Dias 16-20
    0.04, 0.03, 0.02, 0.02, 0.01,  # Dias 21-25
    0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 26-30
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 31-35
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001,   # Dias 46-50
])
PERFIL_N2O_THERMO /= PERFIL_N2O_THERMO.sum()

# =============================================================================
# FUNÇÕES DE CÁLCULO (ADAPTADAS DO SCRIPT ANEXO)
# =============================================================================

def ajustar_emissoes_pre_descarte(O2_concentracao):
    ch4_ajustado = CH4_pre_descarte_g_por_kg_dia

    if O2_concentracao == 21:
        fator_n2o = 1.0
    elif O2_concentracao == 10:
        fator_n2o = 11.11 / 20.26
    elif O2_concentracao == 1:
        fator_n2o = 7.86 / 20.26
    else:
        fator_n2o = 1.0

    n2o_ajustado = N2O_pre_descarte_g_por_kg_dia * fator_n2o
    return ch4_ajustado, n2o_ajustado

def calcular_emissoes_pre_descarte(O2_concentracao, dias_simulacao, residuos_kg_dia):
    ch4_ajustado, n2o_ajustado = ajustar_emissoes_pre_descarte(O2_concentracao)

    emissoes_CH4_pre_descarte_kg = np.full(dias_simulacao, residuos_kg_dia * ch4_ajustado / 1000)
    emissoes_N2O_pre_descarte_kg = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dias_apos_descarte, fracao in PERFIL_N2O_PRE_DESCARTE.items():
            dia_emissao = dia_entrada + dias_apos_descarte - 1
            if dia_emissao < dias_simulacao:
                emissoes_N2O_pre_descarte_kg[dia_emissao] += (
                    residuos_kg_dia * n2o_ajustado * fracao / 1000
                )

    return emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg

def calcular_emissoes_aterro(params, dias_simulacao, residuos_kg_dia, massa_exposta_kg, h_exposta):
    umidade_val, temp_val, doc_val = params

    fator_umid = (1 - umidade_val) / (1 - 0.55)
    f_aberto = np.clip((massa_exposta_kg / residuos_kg_dia) * (h_exposta / 24), 0.0, 1.0)
    docf_calc = 0.0147 * temp_val + 0.28

    potencial_CH4_por_kg = doc_val * docf_calc * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    potencial_CH4_lote_diario = residuos_kg_dia * potencial_CH4_por_kg

    t = np.arange(1, dias_simulacao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    entradas_diarias = np.ones(dias_simulacao, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, kernel_ch4, mode='full')[:dias_simulacao]
    emissoes_CH4 *= potencial_CH4_lote_diario

    E_aberto = 1.91
    E_fechado = 2.15
    E_medio = f_aberto * E_aberto + (1 - f_aberto) * E_fechado
    E_medio_ajust = E_medio * fator_umid
    emissao_diaria_N2O = (E_medio_ajust * (44/28) / 1_000_000) * residuos_kg_dia

    kernel_n2o = np.array([PERFIL_N2O.get(d, 0) for d in range(1, 6)], dtype=float)
    emissoes_N2O = fftconvolve(np.full(dias_simulacao, emissao_diaria_N2O), kernel_n2o, mode='full')[:dias_simulacao]

    O2_concentracao = 21
    emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg = calcular_emissoes_pre_descarte(O2_concentracao, dias_simulacao, residuos_kg_dia)

    total_ch4_aterro_kg = emissoes_CH4 + emissoes_CH4_pre_descarte_kg
    total_n2o_aterro_kg = emissoes_N2O + emissoes_N2O_pre_descarte_kg

    return total_ch4_aterro_kg, total_n2o_aterro_kg

def calcular_emissoes_vermi(params, dias_simulacao, residuos_kg_dia):
    umidade_val, temp_val, doc_val = params
    fracao_ms = 1 - umidade_val
    
    # Usando valores fixos para CH4_C_FRAC_YANG e N2O_N_FRAC_YANG
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_YANG * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_YANG * (44/28) * fracao_ms)

    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_VERMI)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_VERMI[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_VERMI[dia_compostagem]

    return emissoes_CH4, emissoes_N2O

def calcular_emissoes_compostagem(params, dias_simulacao, residuos_kg_dia, dias_compostagem=50):
    umidade, T, DOC = params
    fracao_ms = 1 - umidade
    
    # Usando valores fixos para CH4_C_FRAC_THERMO e N2O_N_FRAC_THERMO
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_THERMO * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_THERMO * (44/28) * fracao_ms)

    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_THERMO)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_THERMO[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_THERMO[dia_compostagem]

    return emissoes_CH4, emissoes_N2O

def executar_simulacao_completa(parametros, dias, residuos_kg_dia, massa_exposta_kg, h_exposta):
    umidade, T, DOC = parametros
    
    ch4_aterro, n2o_aterro = calcular_emissoes_aterro([umidade, T, DOC], dias, residuos_kg_dia, massa_exposta_kg, h_exposta)
    ch4_vermi, n2o_vermi = calcular_emissoes_vermi([umidade, T, DOC], dias, residuos_kg_dia)

    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000
    total_vermi_tco2eq = (ch4_vermi * GWP_CH4_20 + n2o_vermi * GWP_N2O_20) / 1000

    reducao_tco2eq = total_aterro_tco2eq.sum() - total_vermi_tco2eq.sum()
    return reducao_tco2eq

def executar_simulacao_unfccc(parametros, dias, residuos_kg_dia, massa_exposta_kg, h_exposta):
    umidade, T, DOC = parametros

    ch4_aterro, n2o_aterro = calcular_emissoes_aterro([umidade, T, DOC], dias, residuos_kg_dia, massa_exposta_kg, h_exposta)
    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000

    ch4_compost, n2o_compost = calcular_emissoes_compostagem([umidade, T, DOC], dias, residuos_kg_dia, dias_compostagem=50)
    total_compost_tco2eq = (ch4_compost * GWP_CH4_20 + n2o_compost * GWP_N2O_20) / 1000

    reducao_tco2eq = total_aterro_tco2eq.sum() - total_compost_tco2eq.sum()
    return reducao_tco2eq

# =============================================================================
# CONFIGURAÇÃO PRINCIPAL DO APLICATIVO
# =============================================================================

# Título do aplicativo
st.title("Simulador de Emissões de tCO₂eq")
st.markdown("""
Esta ferramenta projeta os Créditos de Carbono ao calcular as emissões de gases de efeito estufa para dois contextos de gestão de resíduos
""")

# Criar abas
tab1, tab2 = st.tabs(["📦 Análise por Lote Único (100 kg)", "📈 Entrada Contínua (kg/dia)"])

# =============================================================================
# ABA 1: ANÁLISE POR LOTE ÚNICO (100 kg) - NOVA SEÇÃO
# =============================================================================
with tab1:
    st.header("Análise por Lote Único de 100 kg")
    st.markdown("""
    **Análise Comparativa: Aterro vs Vermicompostagem vs Compostagem**

    Este simulador calcula o potencial de geração de metano de um lote de 100 kg de resíduos orgânicos
    em três diferentes cenários de gestão, com análise financeira baseada no mercado de carbono.
    
    **✅ CORREÇÃO APLICADA:** Kernel de decaimento NÃO normalizado para aterro (metodologia IPCC correta)
    """)
    
    # Exibir cotação do carbono
    exibir_cotacao_carbono()
    
    # Parâmetros de entrada na sidebar (apenas para aba 1)
    with st.sidebar:
        if st.session_state.get('aba_atual') != 1:
            st.session_state.aba_atual = 1
            
        st.header("⚙️ Parâmetros de Entrada - Lote Único")
        
        # Entrada principal de resíduos (fixo em 100 kg para o lote)
        st.subheader("📦 Lote de Resíduos")
        residuos_kg = st.number_input(
            "Peso do lote (kg)", 
            min_value=10, 
            max_value=1000, 
            value=100, 
            step=10,
            help="Peso do lote de resíduos orgânicos para análise",
            key="lote_residuos"
        )
        
        st.subheader("📊 Parámetros Ambientais")
        
        umidade_valor = st.slider(
            "Umidade do resíduo (%)", 
            50, 95, 85, 1,
            help="Percentual de umidade dos resíduos orgânicos",
            key="umidade_lote"
        )
        umidade = umidade_valor / 100.0
        
        temperatura = st.slider(
            "Temperatura média (°C)", 
            15, 35, 25, 1,
            help="Temperatura média ambiente (importante para cálculo do DOCf)",
            key="temp_lote"
        )
        
        st.subheader("⏰ Período de Análise")
        dias_simulacao = st.slider(
            "Dias de simulação", 
            50, 3650, 365, 50,
            help="Período total da simulação em dias (até 10 anos)",
            key="dias_lote"
        )
        
        # Adicionar aviso sobre método correto
        with st.expander("ℹ️ Informação sobre Metodologia"):
            st.info("""
            **Método Corrigido (IPCC 2006):**
            - **Aterro:** Kernel NÃO normalizado - respeita a equação diferencial do decaimento
            - **Compostagem/Vermicompostagem:** Kernel normalizado - processos curtos (<50 dias)
            
            **Para 100 kg × 365 dias:**
            - Potencial total CH₄: ~5.83 kg
            - Fração emitida em 365 dias: ~6%
            - CH₄ emitido no período: ~0.35 kg
            """)
        
        if st.button("🚀 Calcular Potencial de Metano", type="primary", key="btn_lote"):
            st.session_state.run_analise_lote = True

    # Execução da simulação para aba 1
    if st.session_state.get('run_analise_lote', False):
        with st.spinner('Calculando potencial de metano para os três cenários...'):
            
            # 1. CÁLCULO DO POTENCIAL DE METANO PARA CADA CENÁRIO
            # Aterro Sanitário (CORRIGIDO)
            emissoes_aterro, total_aterro, DOCf, fracao_emitida = calcular_potencial_metano_aterro(
                residuos_kg, umidade, temperatura, dias_simulacao
            )
            
            # Vermicompostagem (50 dias de processo)
            dias_vermi = min(50, dias_simulacao)
            emissoes_vermi_temp, total_vermi = calcular_emissoes_vermicompostagem_lote(
                residuos_kg, umidade, dias_vermi
            )
            emissoes_vermi = np.zeros(dias_simulacao)
            emissoes_vermi[:dias_vermi] = emissoes_vermi_temp
            
            # Compostagem Termofílica (50 dias de processo)
            dias_compost = min(50, dias_simulacao)
            emissoes_compost_temp, total_compost = calcular_emissoes_compostagem_lote(
                residuos_kg, umidade, dias_compost
            )
            emissoes_compost = np.zeros(dias_simulacao)
            emissoes_compost[:dias_compost] = emissoes_compost_temp
            
            # 2. CRIAR DATAFRAME COM OS RESULTADOS
            datas = pd.date_range(start=datetime.now(), periods=dias_simulacao, freq='D')
            
            df = pd.DataFrame({
                'Data': datas,
                'Aterro_CH4_kg': emissoes_aterro,
                'Vermicompostagem_CH4_kg': emissoes_vermi,
                'Compostagem_CH4_kg': emissoes_compost
            })
            
            # Calcular valores acumulados
            df['Aterro_Acumulado'] = df['Aterro_CH4_kg'].cumsum()
            df['Vermi_Acumulado'] = df['Vermicompostagem_CH4_kg'].cumsum()
            df['Compost_Acumulado'] = df['Compostagem_CH4_kg'].cumsum()
            
            # Calcular reduções (evitadas) em relação ao aterro
            df['Reducao_Vermi'] = df['Aterro_Acumulado'] - df['Vermi_Acumulado']
            df['Reducao_Compost'] = df['Aterro_Acumulado'] - df['Compost_Acumulado']
            
            # 3. EXIBIR RESULTADOS PRINCIPAIS
            st.header("📊 Resultados - Potencial de Metano por Cenário")
            
            # Informação sobre metodologia
            st.info(f"""
            **📈 Método Corrigido (Kernel NÃO normalizado):**
            - Potencial total de CH₄ no aterro: **{formatar_br(total_aterro)} kg**
            - Fração emitida em {dias_simulacao} dias: **{formatar_br(fracao_emitida*100)}%**
            - CH₄ realmente emitido no período: **{formatar_br(df['Aterro_Acumulado'].iloc[-1])} kg**
            """)
            
            # Métricas principais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Aterro Sanitário",
                    f"{formatar_br(df['Aterro_Acumulado'].iloc[-1])} kg CH₄",
                    f"Potencial: {formatar_br(total_aterro)} kg",
                    help=f"Emitido em {dias_simulacao} dias ({formatar_br(fracao_emitida*100)}% do potencial)"
                )
            
            with col2:
                reducao_vermi_kg = df['Aterro_Acumulado'].iloc[-1] - df['Vermi_Acumulado'].iloc[-1]
                reducao_vermi_perc = (1 - df['Vermi_Acumulado'].iloc[-1]/df['Aterro_Acumulado'].iloc[-1])*100 if df['Aterro_Acumulado'].iloc[-1] > 0 else 0
                st.metric(
                    "Vermicompostagem",
                    f"{formatar_br(df['Vermi_Acumulado'].iloc[-1])} kg CH₄",
                    delta=f"-{formatar_br(reducao_vermi_perc)}%",
                    delta_color="inverse",
                    help=f"Redução de {formatar_br(reducao_vermi_kg)} kg vs aterro"
                )
            
            with col3:
                reducao_compost_kg = df['Aterro_Acumulado'].iloc[-1] - df['Compost_Acumulado'].iloc[-1]
                reducao_compost_perc = (1 - df['Compost_Acumulado'].iloc[-1]/df['Aterro_Acumulado'].iloc[-1])*100 if df['Aterro_Acumulado'].iloc[-1] > 0 else 0
                st.metric(
                    "Compostagem Termofílica",
                    f"{formatar_br(df['Compost_Acumulado'].iloc[-1])} kg CH₄",
                    delta=f"-{formatar_br(reducao_compost_perc)}%",
                    delta_color="inverse",
                    help=f"Redução de {formatar_br(reducao_compost_kg)} kg vs aterro"
                )
            
            # 4. GRÁFICO: REDUÇÃO DE EMISSÕES ACUMULADA
            st.subheader("📉 Redução de Emissões Acumulada (CH₄)")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Configurar formatação
            br_formatter = FuncFormatter(br_format)
            
            # Plotar linhas de acumulado
            ax.plot(df['Data'], df['Aterro_Acumulado'], 'r-', 
                    label='Aterro Sanitário', linewidth=3, alpha=0.7)
            ax.plot(df['Data'], df['Vermi_Acumulado'], 'g-', 
                    label='Vermicompostagem', linewidth=2)
            ax.plot(df['Data'], df['Compost_Acumulado'], 'b-', 
                    label='Compostagem Termofílica', linewidth=2)
            
            # Área de redução (evitadas)
            ax.fill_between(df['Data'], df['Vermi_Acumulado'], df['Aterro_Acumulado'],
                            color='green', alpha=0.3, label='Redução Vermicompostagem')
            ax.fill_between(df['Data'], df['Compost_Acumulado'], df['Aterro_Acumulado'],
                            color='blue', alpha=0.2, label='Redução Compostagem')
            
            # Configurar gráfico
            ax.set_title(f'Acumulado de Metano em {dias_simulacao} Dias - Lote de {residuos_kg} kg (Método Corrigido)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('Metano Acumulado (kg CH₄)')
            ax.legend(title='Cenário de Gestão', loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.yaxis.set_major_formatter(br_formatter)
            
            # Rotacionar labels do eixo x
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # 5. GRÁFICO: EMISSÕES DIÁRIAS COMPARATIVAS
            st.subheader("📈 Emissões Diárias de Metano")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plotar emissões diárias (apenas primeiros 100 dias para melhor visualização)
            dias_exibir = min(100, dias_simulacao)
            
            # Criar gráfico com barras para visualizar melhor as diferenças
            x_pos = np.arange(dias_exibir)
            bar_width = 0.25
            
            # Usar barras para visualização mais clara
            ax.bar(x_pos - bar_width, df['Aterro_CH4_kg'][:dias_exibir], bar_width, 
                    label='Aterro', color='red', alpha=0.7)
            ax.bar(x_pos, df['Vermicompostagem_CH4_kg'][:dias_exibir], bar_width, 
                    label='Vermicompostagem', color='green', alpha=0.7)
            ax.bar(x_pos + bar_width, df['Compostagem_CH4_kg'][:dias_exibir], bar_width, 
                    label='Compostagem', color='blue', alpha=0.7)
            
            ax.set_xlabel('Dias')
            ax.set_ylabel('Metano (kg CH₄/dia)')
            ax.set_title(f'Emissões Diárias de Metano (Primeiros {dias_exibir} Dias) - Método Corrigido', 
                        fontsize=14, fontweight='bold')
            ax.legend(title='Cenário')
            ax.grid(True, linestyle='--', alpha=0.5, axis='y')
            ax.yaxis.set_major_formatter(br_formatter)
            
            # Ajustar ticks do eixo x
            ax.set_xticks(x_pos[::10])
            ax.set_xticklabels([f'Dia {i+1}' for i in x_pos[::10]])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # =============================================================================
            # NOVA SEÇÃO: DURAÇÃO DAS EMISSÕES - COMPARAÇÃO TEMPORAL CRÍTICA
            # =============================================================================
            
            st.header("⏰ Duração das Emissões: Diferença Crucial entre Cenários")
            
            # Criar DataFrame temporal para análise
            df_temporal = pd.DataFrame({
                'Dia': np.arange(1, dias_simulacao + 1),
                'Aterro_CH4_kg_dia': emissoes_aterro,
                'Vermicompostagem_CH4_kg_dia': emissoes_vermi,
                'Compostagem_CH4_kg_dia': emissoes_compost
            })
            
            # Calcular acumulados e percentuais
            df_temporal['Aterro_Acumulado'] = df_temporal['Aterro_CH4_kg_dia'].cumsum()
            df_temporal['Vermi_Acumulado'] = df_temporal['Vermicompostagem_CH4_kg_dia'].cumsum()
            df_temporal['Compost_Acumulado'] = df_temporal['Compostagem_CH4_kg_dia'].cumsum()
            
            # Calcular percentuais acumulados
            for cenario in ['Aterro', 'Vermi', 'Compost']:
                total = df_temporal[f'{cenario}_Acumulado'].iloc[-1]
                df_temporal[f'{cenario}_%_Acumulado'] = (df_temporal[f'{cenario}_Acumulado'] / total * 100) if total > 0 else 0
            
            # Encontrar dias para atingir certos percentuais
            resultados_temporais = []
            
            for cenario in ['Aterro', 'Vermi', 'Compost']:
                dados = {
                    'Cenário': cenario,
                    'Total_kg': df_temporal[f'{cenario}_Acumulado'].iloc[-1],
                }
                
                # Encontrar dia para 50%, 90%, 95% e 99% das emissões
                for percentual in [50, 90, 95, 99]:
                    try:
                        dia = df_temporal[df_temporal[f'{cenario}_%_Acumulado'] >= percentual]['Dia'].iloc[0]
                        dados[f'Dia_{percentual}%'] = dia
                    except:
                        dados[f'Dia_{percentual}%'] = dias_simulacao
                
                resultados_temporais.append(dados)
            
            # Criar DataFrame de resultados
            df_resultados_temp = pd.DataFrame(resultados_temporais)
            
            # Exibir métricas comparativas
            st.subheader("📊 Duração Temporal das Emissões")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                anos_aterro_95 = df_resultados_temp.loc[0, 'Dia_95%'] / 365
                st.metric(
                    "Aterro Sanitário",
                    f"{int(anos_aterro_95)} anos",
                    f"{df_resultados_temp.loc[0, 'Dia_95%']} dias para 95%",
                    help=f"Emite metano por {int(df_resultados_temp.loc[0, 'Dia_99%']/365)} anos até atingir 99%"
                )
            
            with col2:
                st.metric(
                    "Vermicompostagem",
                    "50 dias",
                    "Processo completo",
                    help="Todas as emissões ocorrem em apenas 50 dias"
                )
            
            with col3:
                st.metric(
                    "Compostagem Termofílica",
                    "50 dias",
                    "Processo completo",
                    help="Todas as emissões ocorrem em apenas 50 dias"
                )
            
            # Gráfico comparativo temporal
            st.subheader("📈 Comparação Temporal: Emissões Concentradas vs Prolongadas")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Primeiro gráfico: Emissões diárias (escala linear)
            dias_visao = min(500, dias_simulacao)  # Mostrar até 500 dias
            
            ax1.plot(df_temporal['Dia'][:dias_visao], df_temporal['Aterro_CH4_kg_dia'][:dias_visao], 
                    'r-', label='Aterro', linewidth=1.5, alpha=0.7)
            ax1.plot(df_temporal['Dia'][:dias_visao], df_temporal['Vermicompostagem_CH4_kg_dia'][:dias_visao], 
                    'g-', label='Vermicompostagem', linewidth=1.5, alpha=0.7)
            ax1.plot(df_temporal['Dia'][:dias_visao], df_temporal['Compostagem_CH4_kg_dia'][:dias_visao], 
                    'b-', label='Compostagem', linewidth=1.5, alpha=0.7)
            
            # Destacar área dos primeiros 50 dias
            ax1.axvspan(0, 50, alpha=0.1, color='green', label='Processos Biológicos (0-50 dias)')
            ax1.axvline(x=50, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
            
            ax1.set_xlabel('Dias')
            ax1.set_ylabel('Emissão Diária (kg CH₄/dia)')
            ax1.set_title('COMPARAÇÃO TEMPORAL: Aterro (Anos) vs Compostagem/Vermicompostagem (50 Dias)',
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.set_xlim([0, dias_visao])
            
            # Segundo gráfico: Percentual acumulado
            ax2.plot(df_temporal['Dia'][:dias_visao], df_temporal['Aterro_%_Acumulado'][:dias_visao], 
                    'r-', label='Aterro', linewidth=2)
            ax2.plot(df_temporal['Dia'][:dias_visao], df_temporal['Vermi_%_Acumulado'][:dias_visao], 
                    'g-', label='Vermicompostagem', linewidth=2)
            ax2.plot(df_temporal['Dia'][:dias_visao], df_temporal['Compost_%_Acumulado'][:dias_visao], 
                    'b-', label='Compostagem', linewidth=2)
            
            # Linhas de referência
            for percentual in [50, 90, 95, 99]:
                ax2.axhline(y=percentual, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
            
            # Marcar 50 dias
            ax2.axvline(x=50, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax2.text(50, 50, ' 50 dias', rotation=90, verticalalignment='center', fontsize=9)
            
            ax2.set_xlabel('Dias')
            ax2.set_ylabel('Percentual Acumulado (%)')
            ax2.set_title('Percentual de Emissões Acumulado ao Longo do Tempo',
                         fontsize=14, fontweight='bold')
            ax2.legend(loc='lower right')
            ax2.grid(True, linestyle='--', alpha=0.3)
            ax2.set_xlim([0, dias_visao])
            ax2.set_ylim([0, 100])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabela detalhada de tempos
            st.subheader("📋 Tabela Detalhada: Tempo para Emitir Percentuais das Emissões")
            
            # Criar DataFrame formatado
            df_tempos = pd.DataFrame({
                'Cenário': ['Aterro Sanitário', 'Vermicompostagem', 'Compostagem Termofílica'],
                '50% das emissões': [
                    f"{df_resultados_temp.loc[0, 'Dia_50%']} dias ({formatar_br(df_resultados_temp.loc[0, 'Dia_50%']/365)} anos)",
                    f"{df_resultados_temp.loc[1, 'Dia_50%']} dias",
                    f"{df_resultados_temp.loc[2, 'Dia_50%']} dias"
                ],
                '90% das emissões': [
                    f"{df_resultados_temp.loc[0, 'Dia_90%']} dias ({formatar_br(df_resultados_temp.loc[0, 'Dia_90%']/365)} anos)",
                    f"{df_resultados_temp.loc[1, 'Dia_90%']} dias",
                    f"{df_resultados_temp.loc[2, 'Dia_90%']} dias"
                ],
                '95% das emissões': [
                    f"{df_resultados_temp.loc[0, 'Dia_95%']} dias ({formatar_br(df_resultados_temp.loc[0, 'Dia_95%']/365)} anos)",
                    f"{df_resultados_temp.loc[1, 'Dia_95%']} dias",
                    f"{df_resultados_temp.loc[2, 'Dia_95%']} dias"
                ],
                '99% das emissões': [
                    f"{df_resultados_temp.loc[0, 'Dia_99%']} dias ({formatar_br(df_resultados_temp.loc[0, 'Dia_99%']/365)} anos)",
                    f"{df_resultados_temp.loc[1, 'Dia_99%']} dias",
                    f"{df_resultados_temp.loc[2, 'Dia_99%']} dias"
                ]
            })
            
            st.dataframe(df_tempos, use_container_width=True)
            
            # Explicação sobre a importância da diferença temporal
            with st.expander("🎯 POR QUE ESTA DIFERENÇA TEMPORAL É CRÍTICA?"):
                st.markdown(f"""
                **⚠️ DIFERENÇA FUNDAMENTAL ENTRE OS PROCESSOS:**
                
                ### 🕰️ **ATERRO SANITÁRIO:**
                - **Duração:** {int(anos_aterro_95)} **ANOS** para 95% das emissões
                - **Padrão:** Emissões prolongadas por décadas (decaimento exponencial lento)
                - **Pico:** Baixo e estendido ao longo do tempo
                - **Impacto climático:** Persistente e de longo prazo
                - **Controle:** **QUASE IMPOSSÍVEL** - emissões dispersas no tempo e espaço
                
                ### 🪱 **VERMICOMPOSTAGEM:**
                - **Duração:** **50 DIAS** para todas as emissões
                - **Padrão:** Emissões concentradas em processo controlado
                - **Pico:** Alto nas primeiras 2-3 semanas
                - **Impacto climático:** Imediato e de curta duração
                - **Controle:** **FÁCIL** - emissões em reator fechado e período definido
                
                ### 🌡️ **COMPOSTAGEM TERMOFÍLICA:**
                - **Duração:** **50 DIAS** para todas as emissões
                - **Padrão:** Emissões concentradas com pico termofílico
                - **Pico:** Muito alto na fase termofílica (dias 6-15)
                - **Impacto climático:** Imediato e de curta duração
                - **Controle:** **MUITO FÁCIL** - emissões em reator com captura possível
                
                ### 💡 **IMPLICAÇÕES PRÁTICAS IMPORTANTES:**
                
                1. **CAPTURA DE METANO:**
                   - **Aterro:** Complexa, cara e ineficiente (emissões por anos)
                   - **Compostagem:** Viável economicamente (emissões em 50 dias)
                
                2. **MONITORAMENTO:**
                   - **Aterro:** Necessário por décadas, alto custo
                   - **Compostagem:** Apenas 50 dias, baixo custo
                
                3. **CRÉDITOS DE CARBONO:**
                   - **Aterro:** Incerteza nas emissões futuras
                   - **Compostagem:** Certeza nas emissões evitadas
                
                4. **INVESTIMENTO EM TECNOLOGIA:**
                   - **Aterro:** Sistemas complexos para captura prolongada
                   - **Compostagem:** Reatores simples com captura no pico
                
                **📊 CONCLUSÃO:** A compostagem/vermicompostagem NÃO SÓ emitem MENOS metano, mas também concentram as emissões em um período MUITO CURTO (50 dias), permitindo captura e controle eficiente. O aterro, por outro lado, emite por ANOS, tornando praticamente impossível qualquer controle efetivo.
                """)
            
            # 6. CÁLCULO DE CO₂eq E VALOR FINANCEIRO
            st.header("💰 Valor Financeiro das Emissões Evitadas")
            
            # Converter metano para CO₂eq (GWP CH₄ = 27.9 para 100 anos - IPCC AR6)
            GWP_CH4 = 27.9  # kg CO₂eq per kg CH₄
            
            total_evitado_vermi_kg = (df['Aterro_Acumulado'].iloc[-1] - df['Vermi_Acumulado'].iloc[-1]) * GWP_CH4
            total_evitado_vermi_tco2eq = total_evitado_vermi_kg / 1000
            
            total_evitado_compost_kg = (df['Aterro_Acumulado'].iloc[-1] - df['Compost_Acumulado'].iloc[-1]) * GWP_CH4
            total_evitado_compost_tco2eq = total_evitado_compost_kg / 1000
            
            # Calcular valor em Reais
            preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
            
            valor_vermi_brl = total_evitado_vermi_tco2eq * preco_carbono_reais
            valor_compost_brl = total_evitado_compost_tco2eq * preco_carbono_reais
            
            # Exibir métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Vermicompostagem",
                    f"{formatar_br(total_evitado_vermi_tco2eq)} tCO₂eq",
                    f"R$ {formatar_br(valor_vermi_brl)}",
                    delta_color="off"
                )
            
            with col2:
                st.metric(
                    "Compostagem",
                    f"{formatar_br(total_evitado_compost_tco2eq)} tCO₂eq",
                    f"R$ {formatar_br(valor_compost_brl)}",
                    delta_color="off"
                )
            
            # Resumo final
            st.success(f"""
            **🎯 RESUMO FINAL PARA LOTE DE {residuos_kg} kg:**
            
            **Aterro:** Emite **{formatar_br(df['Aterro_Acumulado'].iloc[-1])} kg CH₄** em **{dias_simulacao} dias** ({formatar_br(fracao_emitida*100)}% do potencial total)
            
            **Vermicompostagem:** Emite **{formatar_br(df['Vermi_Acumulado'].iloc[-1])} kg CH₄** em **apenas 50 dias** ({formatar_br((1 - df['Vermi_Acumulado'].iloc[-1]/df['Aterro_Acumulado'].iloc[-1])*100)}% de redução)
            
            **Compostagem:** Emite **{formatar_br(df['Compost_Acumulado'].iloc[-1])} kg CH₄** em **apenas 50 dias** ({formatar_br((1 - df['Compost_Acumulado'].iloc[-1]/df['Aterro_Acumulado'].iloc[-1])*100)}% de redução)
            
            **DIFERENÇA TEMPORAL CRÍTICA:** O aterro emite por **{int(anos_aterro_95)} anos**, enquanto compostagem emite por **50 dias**!
            """)
    else:
        st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Calcular Potencial de Metano' para ver os resultados.")

# =============================================================================
# ABA 2: ENTRADA CONTÍNUA (kg/dia) - SEÇÃO ORIGINAL
# =============================================================================
with tab2:
    # Título da aba 2
    st.header("Análise para Entrada Contínua (kg/dia)")
    st.markdown("""
    **Simulação Completa: Comparação de Emissões em Longo Prazo**
    
    Esta ferramenta projeta os Créditos de Carbono ao calcular as emissões de gases de efeito estufa para dois contextos de gestão de resíduos
    """)
    
    # Seção original de parâmetros
    with st.sidebar:
        if st.session_state.get('aba_atual') != 2:
            st.session_state.aba_atual = 2
            
        st.header("⚙️ Parâmetros de Entrada - Entrada Contínua")
        
        # Entrada principal de resíduos - ATUALIZAR VARIÁVEL GLOBAL
        residuos_kg_dia = st.slider("Quantidade de resíduos (kg/dia)", 
                                   min_value=10, max_value=1000, value=100, step=10,
                                   help="Quantidade diária de resíduos orgânicos gerados",
                                   key="residuos_cont")
        
        st.subheader("📊 Parâmetros Operacionais")
        
        # Umidade com formatação brasileira (0,85 em vez de 0.85)
        umidade_valor = st.slider("Umidade do resíduo (%)", 50, 95, 85, 1,
                                 help="Percentual de umidade dos resíduos orgânicos",
                                 key="umidade_cont")
        umidade = umidade_valor / 100.0
        st.write(f"**Umidade selecionada:** {formatar_br(umidade_valor)}%")
        
        # ATUALIZAR VARIÁVEIS GLOBAIS
        massa_exposta_kg = st.slider("Massa exposta na frente de trabalho (kg)", 50, 200, 100, 10,
                                    help="Massa de resíduos exposta diariamente para tratamento",
                                    key="massa_cont")
        h_exposta = st.slider("Horas expostas por dia", 4, 24, 8, 1,
                             help="Horas diárias de exposição dos resíduos",
                             key="horas_cont")
        
        st.subheader("🎯 Configuração de Simulação")
        anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5,
                                  help="Período total da simulação em anos",
                                  key="anos_cont")
        n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50,
                                 help="Número de iterações para análise de incerteza",
                                 key="n_sim_cont")
        n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16,
                             help="Número de amostras para análise de sensibilidade",
                             key="n_samples_cont")
        
        if st.button("🚀 Executar Simulação", type="primary", key="btn_cont"):
            st.session_state.run_simulation = True

    # Executar simulação quando solicitado
    if st.session_state.get('run_simulation', False):
        with st.spinner('Executando simulação...'):
            # Calcular dias e datas localmente
            dias = anos_simulacao * 365
            datas = pd.date_range(start=data_inicio, periods=dias, freq='D')
            
            # Executar modelo base
            params_base = [umidade, T, DOC]

            ch4_aterro_dia, n2o_aterro_dia = calcular_emissoes_aterro(params_base, dias, residuos_kg_dia, massa_exposta_kg, h_exposta)
            ch4_vermi_dia, n2o_vermi_dia = calcular_emissoes_vermi(params_base, dias, residuos_kg_dia)
            
            # Construir DataFrame
            df = pd.DataFrame({
                'Data': datas,
                'CH4_Aterro_kg_dia': ch4_aterro_dia,
                'N2O_Aterro_kg_dia': n2o_aterro_dia,
                'CH4_Vermi_kg_dia': ch4_vermi_dia,
                'N2O_Vermi_kg_dia': n2o_vermi_dia,
            })

            for gas in ['CH4_Aterro', 'N2O_Aterro', 'CH4_Vermi', 'N2O_Vermi']:
                df[f'{gas}_tCO2eq'] = df[f'{gas}_kg_dia'] * (GWP_CH4_20 if 'CH4' in gas else GWP_N2O_20) / 1000

            df['Total_Aterro_tCO2eq_dia'] = df['CH4_Aterro_tCO2eq'] + df['N2O_Aterro_tCO2eq']
            df['Total_Vermi_tCO2eq_dia'] = df['CH4_Vermi_tCO2eq'] + df['N2O_Vermi_tCO2eq']

            df['Total_Aterro_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_dia'].cumsum()
            df['Total_Vermi_tCO2eq_acum'] = df['Total_Vermi_tCO2eq_dia'].cumsum()
            df['Reducao_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Vermi_tCO2eq_acum']

            # Resumo anual
            df['Year'] = df['Data'].dt.year
            df_anual_revisado = df.groupby('Year').agg({
                'Total_Aterro_tCO2eq_dia': 'sum',
                'Total_Vermi_tCO2eq_dia': 'sum',
            }).reset_index()

            df_anual_revisado['Emission reductions (t CO₂eq)'] = df_anual_revisado['Total_Aterro_tCO2eq_dia'] - df_anual_revisado['Total_Vermi_tCO2eq_dia']
            df_anual_revisado['Cumulative reduction (t CO₂eq)'] = df_anual_revisado['Emission reductions (t CO₂eq)'].cumsum()

            df_anual_revisado.rename(columns={
                'Total_Aterro_tCO2eq_dia': 'Baseline emissions (t CO₂eq)',
                'Total_Vermi_tCO2eq_dia': 'Project emissions (t CO₂eq)',
            }, inplace=True)

            # Cenário UNFCCC
            ch4_compost_UNFCCC, n2o_compost_UNFCCC = calcular_emissoes_compostagem(
                params_base, dias, residuos_kg_dia, dias_compostagem=50
            )
            ch4_compost_unfccc_tco2eq = ch4_compost_UNFCCC * GWP_CH4_20 / 1000
            n2o_compost_unfccc_tco2eq = n2o_compost_UNFCCC * GWP_N2O_20 / 1000
            total_compost_unfccc_tco2eq_dia = ch4_compost_unfccc_tco2eq + n2o_compost_unfccc_tco2eq

            df_comp_unfccc_dia = pd.DataFrame({
                'Data': datas,
                'Total_Compost_tCO2eq_dia': total_compost_unfccc_tco2eq_dia
            })
            df_comp_unfccc_dia['Year'] = df_comp_unfccc_dia['Data'].dt.year

            df_comp_anual_revisado = df_comp_unfccc_dia.groupby('Year').agg({
                'Total_Compost_tCO2eq_dia': 'sum'
            }).reset_index()

            df_comp_anual_revisado = pd.merge(df_comp_anual_revisado,
                                              df_anual_revisado[['Year', 'Baseline emissions (t CO₂eq)']],
                                              on='Year', how='left')

            df_comp_anual_revisado['Emission reductions (t CO₂eq)'] = df_comp_anual_revisado['Baseline emissions (t CO₂eq)'] - df_comp_anual_revisado['Total_Compost_tCO2eq_dia']
            df_comp_anual_revisado['Cumulative reduction (t CO₂eq)'] = df_comp_anual_revisado['Emission reductions (t CO₂eq)'].cumsum()
            df_comp_anual_revisado.rename(columns={'Total_Compost_tCO2eq_dia': 'Project emissions (t CO₂eq)'}, inplace=True)

            # =============================================================================
            # EXIBIÇÃO DOS RESULTADOS COM COTAÇÃO DO CARBONO E REAL
            # =============================================================================

            # Exibir resultados
            st.header("📈 Resultados da Simulação")
            
            # Obter valores totais
            total_evitado_tese = df['Reducao_tCO2eq_acum'].iloc[-1]
            total_evitado_unfccc = df_comp_anual_revisado['Cumulative reduction (t CO₂eq)'].iloc[-1]
            
            # Obter preço do carbono e taxa de câmbio da session state
            preco_carbono = st.session_state.preco_carbono
            moeda = st.session_state.moeda_carbono
            taxa_cambio = st.session_state.taxa_cambio
            fonte_cotacao = st.session_state.fonte_cotacao
            
            # Calcular valores financeiros em Euros
            valor_tese_eur = calcular_valor_creditos(total_evitado_tese, preco_carbono, moeda)
            valor_unfccc_eur = calcular_valor_creditos(total_evitado_unfccc, preco_carbono, moeda)
            
            # Calcular valores financeiros em Reais
            valor_tese_brl = calcular_valor_creditos(total_evitado_tese, preco_carbono, "R$", taxa_cambio)
            valor_unfccc_brl = calcular_valor_creditos(total_evitado_unfccc, preco_carbono, "R$", taxa_cambio)
            
            # NOVA SEÇÃO: VALOR FINANCEIRO DAS EMISSÕES EVITADAS
            st.subheader("💰 Valor Financeiro das Emissões Evitadas")
            
            # Primeira linha: Euros
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"Preço Carbono (Euro)", 
                    f"{moeda} {formatar_br(preco_carbono)}/tCO₂eq",
                    help=f"Fonte: {fonte_cotacao}"
                )
            with col2:
                st.metric(
                    "Valor Tese (Euro)", 
                    f"{moeda} {formatar_br(valor_tese_eur)}",
                    help=f"Baseado em {formatar_br(total_evitado_tese)} tCO₂eq evitadas"
                )
            with col3:
                st.metric(
                    "Valor UNFCCC (Euro)", 
                    f"{moeda} {formatar_br(valor_unfccc_eur)}",
                    help=f"Baseado em {formatar_br(total_evitado_unfccc)} tCO₂eq evitadas"
                )
            
            # Segunda linha: Reais
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"Preço Carbono (R$)", 
                    f"R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq",
                    help="Preço do carbono convertido para Reais"
                )
            with col2:
                st.metric(
                    "Valor Tese (R$)", 
                    f"R$ {formatar_br(valor_tese_brl)}",
                    help=f"Baseado em {formatar_br(total_evitado_tese)} tCO₂eq evitadas"
                )
            with col3:
                st.metric(
                    "Valor UNFCCC (R$)", 
                    f"R$ {formatar_br(valor_unfccc_brl)}",
                    help=f"Baseado em {formatar_br(total_evitado_unfccc)} tCO₂eq evitadas"
                )
            
            # Explicação sobre compra e venda
            with st.expander("💡 Como funciona a comercialização no mercado de carbono?"):
                st.markdown(f"""
                **📊 Informações de Mercado:**
                - **Preço em Euro:** {moeda} {formatar_br(preco_carbono)}/tCO₂eq
                - **Preço em Real:** R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq
                - **Taxa de câmbio:** 1 Euro = R$ {formatar_br(taxa_cambio)}
                - **Fonte:** {fonte_cotacao}
                
                **💶 Comprar créditos (compensação):**
                - Custo em Euro: **{moeda} {formatar_br(valor_tese_eur)}**
                - Custo em Real: **R$ {formatar_br(valor_tese_brl)}**
                
                **💵 Vender créditos (comercialização):**  
                - Receita em Euro: **{moeda} {formatar_br(valor_tese_eur)}**
                - Receita em Real: **R$ {formatar_br(valor_tese_brl)}**
                
                **🌍 Mercado de Referência:**
                - European Union Allowances (EUA)
                - European Emissions Trading System (EU ETS)
                - Contratos futuros de carbono
                - Preços em tempo real do mercado regulado
                """)
            
            # =============================================================================
            # SEÇÃO ATUALIZADA: RESUMO DAS EMISSÕES EVITADAS COM MÉTRICAS ANUAIS REORGANIZADAS
            # =============================================================================
            
            # Métricas de emissões evitadas - layout reorganizado
            st.subheader("📊 Resumo das Emissões Evitadas")
            
            # Calcular médias anuais
            media_anual_tese = total_evitado_tese / anos_simulacao
            media_anual_unfccc = total_evitado_unfccc / anos_simulacao
            
            # Layout com duas colunas principais
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📋 Metodologia da Tese")
                st.metric(
                    "Total de emissões evitadas", 
                    f"{formatar_br(total_evitado_tese)} tCO₂eq",
                    help=f"Total acumulado em {anos_simulacao} anos"
                )
                st.metric(
                    "Média anual", 
                    f"{formatar_br(media_anual_tese)} tCO₂eq/ano",
                    help=f"Emissões evitadas por ano em média"
                )

            with col2:
                st.markdown("#### 📋 Metodologia UNFCCC")
                st.metric(
                    "Total de emissões evitadas", 
                    f"{formatar_br(total_evitado_unfccc)} tCO₂eq",
                    help=f"Total acumulado em {anos_simulacao} anos"
                )
                st.metric(
                    "Média anual", 
                    f"{formatar_br(media_anual_unfccc)} tCO₂eq/ano",
                    help=f"Emissões evitadas por ano em média"
                )

            # Adicionar explicação sobre as métricas anuais
            with st.expander("💡 Entenda as métricas anuais"):
                st.markdown(f"""
                **📊 Como interpretar as métricas anuais:**
                
                **Metodologia da Tese:**
                - **Total em {anos_simulacao} anos:** {formatar_br(total_evitado_tese)} tCO₂eq
                - **Média anual:** {formatar_br(media_anual_tese)} tCO₂eq/ano
                - Equivale a aproximadamente **{formatar_br(media_anual_tese / 365)} tCO₂eq/dia**
                
                **Metodologia UNFCCC:**
                - **Total em {anos_simulacao} anos:** {formatar_br(total_evitado_unfccc)} tCO₂eq
                - **Média anual:** {formatar_br(media_anual_unfccc)} tCO₂eq/ano
                - Equivale a aproximadamente **{formatar_br(media_anual_unfccc / 365)} tCO₂eq/dia**
                
                **💡 Significado prático:**
                - As métricas anuais ajudam a planejar projetos de longo prazo
                - Permitem comparar com metas anuais de redução de emissões
                - Facilitam o cálculo de retorno financeiro anual
                - A média anual representa o desempenho constante do projeto
                """)

            # Gráfico comparativo
            st.subheader("📊 Comparação Anual das Emissões Evitadas")
            df_evitadas_anual = pd.DataFrame({
                'Year': df_anual_revisado['Year'],
                'Proposta da Tese': df_anual_revisado['Emission reductions (t CO₂eq)'],
                'UNFCCC (2012)': df_comp_anual_revisado['Emission reductions (t CO₂eq)']
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            br_formatter = FuncFormatter(br_format)
            x = np.arange(len(df_evitadas_anual['Year']))
            bar_width = 0.35

            ax.bar(x - bar_width/2, df_evitadas_anual['Proposta da Tese'], width=bar_width,
                    label='Proposta da Tese', edgecolor='black')
            ax.bar(x + bar_width/2, df_evitadas_anual['UNFCCC (2012)'], width=bar_width,
                    label='UNFCCC (2012)', edgecolor='black', hatch='//')

            # Adicionar valores formatados em cima das barras
            for i, (v1, v2) in enumerate(zip(df_evitadas_anual['Proposta da Tese'], 
                                             df_evitadas_anual['UNFCCC (2012)'])):
                ax.text(i - bar_width/2, v1 + max(v1, v2)*0.01, 
                        formatar_br(v1), ha='center', fontsize=9, fontweight='bold')
                ax.text(i + bar_width/2, v2 + max(v1, v2)*0.01, 
                        formatar_br(v2), ha='center', fontsize=9, fontweight='bold')

            ax.set_xlabel('Ano')
            ax.set_ylabel('Emissões Evitadas (t CO₂eq)')
            ax.set_title('Comparação Anual das Emissões Evitadas: Proposta da Tese vs UNFCCC (2012)')
            
            # Ajustar o eixo x para ser igual ao do gráfico de redução acumulada
            ax.set_xticks(x)
            ax.set_xticklabels(df_anual_revisado['Year'], fontsize=8)

            ax.legend(title='Metodologia')
            ax.yaxis.set_major_formatter(br_formatter)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

            # Gráfico de redução acumulada
            st.subheader("📉 Redução de Emissões Acumulada")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Data'], df['Total_Aterro_tCO2eq_acum'], 'r-', label='Cenário Base (Aterro Sanitário)', linewidth=2)
            ax.plot(df['Data'], df['Total_Vermi_tCO2eq_acum'], 'g-', label='Projeto (Compostagem em reatores com minhocas)', linewidth=2)
            ax.fill_between(df['Data'], df['Total_Vermi_tCO2eq_acum'], df['Total_Aterro_tCO2eq_acum'],
                            color='skyblue', alpha=0.5, label='Emissões Evitadas')
            ax.set_title('Redução de Emissões em {} Anos'.format(anos_simulacao))
            ax.set_xlabel('Ano')
            ax.set_ylabel('tCO₂eq Acumulado')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(br_formatter)

            st.pyplot(fig)

            # Análise de Sensibilidade Global (Sobol) - PROPOSTA DA TESE
            st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Proposta da Tese")
            br_formatter_sobol = FuncFormatter(br_format)

            np.random.seed(50)  
            
            problem_tese = {
                'num_vars': 3,
                'names': ['umidade', 'T', 'DOC'],
                'bounds': [
                    [0.5, 0.85],         # umidade
                    [25.0, 45.0],       # temperatura
                    [0.15, 0.50],       # doc
                ]
            }

            param_values_tese = sample(problem_tese, n_samples)
            results_tese = Parallel(n_jobs=-1)(delayed(executar_simulacao_completa)(params, dias, residuos_kg_dia, massa_exposta_kg, h_exposta) for params in param_values_tese)
            Si_tese = analyze(problem_tese, np.array(results_tese), print_to_console=False)
            
            sensibilidade_df_tese = pd.DataFrame({
                'Parâmetro': problem_tese['names'],
                'S1': Si_tese['S1'],
                'ST': Si_tese['ST']
            }).sort_values('ST', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_tese, palette='viridis', ax=ax)
            ax.set_title('Sensibilidade Global dos Parâmetros (Índice Sobol Total) - Proposta da Tese')
            ax.set_xlabel('Índice ST')
            ax.set_ylabel('')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.xaxis.set_major_formatter(br_formatter_sobol) # Adiciona formatação ao eixo x
            st.pyplot(fig)

            # Análise de Sensibilidade Global (Sobol) - CENÁRIO UNFCCC
            st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Cenário UNFCCC")

            np.random.seed(50)
            
            problem_unfccc = {
                'num_vars': 3,
                'names': ['umidade', 'T', 'DOC'],
                'bounds': [
                    [0.5, 0.85],  # Umidade
                    [25, 45],     # Temperatura
                    [0.15, 0.50], # DOC
                ]
            }

            param_values_unfccc = sample(problem_unfccc, n_samples)
            results_unfccc = Parallel(n_jobs=-1)(delayed(executar_simulacao_unfccc)(params, dias, residuos_kg_dia, massa_exposta_kg, h_exposta) for params in param_values_unfccc)
            Si_unfccc = analyze(problem_unfccc, np.array(results_unfccc), print_to_console=False)
            
            sensibilidade_df_unfccc = pd.DataFrame({
                'Parâmetro': problem_unfccc['names'],
                'S1': Si_unfccc['S1'],
                'ST': Si_unfccc['ST']
            }).sort_values('ST', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_unfccc, palette='viridis', ax=ax)
            ax.set_title('Sensibilidade Global dos Parâmetros (Índice Sobol Total) - Cenário UNFCCC')
            ax.set_xlabel('Índice ST')
            ax.set_ylabel('')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.xaxis.set_major_formatter(br_formatter_sobol) # Adiciona formatação ao eixo x
            st.pyplot(fig)

            # Análise de Incerteza (Monte Carlo) - PROPOSTA DA TESE
            st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Proposta da Tese")

            
            def gerar_parametros_mc_tese(n):
                np.random.seed(50)
                umidade_vals = np.random.uniform(0.75, 0.90, n)
                temp_vals = np.random.normal(25, 3, n)
                doc_vals = np.random.triangular(0.12, 0.15, 0.18, n)
                
                return umidade_vals, temp_vals, doc_vals

            umidade_vals, temp_vals, doc_vals = gerar_parametros_mc_tese(n_simulations)
            
            results_mc_tese = []
            for i in range(n_simulations):
                params_tese = [umidade_vals[i], temp_vals[i], doc_vals[i]]
                results_mc_tese.append(executar_simulacao_completa(params_tese, dias, residuos_kg_dia, massa_exposta_kg, h_exposta))

            results_array_tese = np.array(results_mc_tese)
            media_tese = np.mean(results_array_tese)
            intervalo_95_tese = np.percentile(results_array_tese, [2.5, 97.5])

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(results_array_tese, kde=True, bins=30, color='skyblue', ax=ax)
            ax.axvline(media_tese, color='red', linestyle='--', label=f'Média: {formatar_br(media_tese)} tCO₂eq')
            ax.axvline(intervalo_95_tese[0], color='green', linestyle=':', label='IC 95%')
            ax.axvline(intervalo_95_tese[1], color='green', linestyle=':')
            ax.set_title('Distribuição das Emissões Evitadas (Simulação Monte Carlo) - Proposta da Tese')
            ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
            ax.set_ylabel('Frequência')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(br_formatter)
            st.pyplot(fig)

            # Análise de Incerteza (Monte Carlo) - CENÁRIO UNFCCC
            st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Cenário UNFCCC")
            
            def gerar_parametros_mc_unfccc(n):
                np.random.seed(50)
                umidade_vals = np.random.uniform(0.75, 0.90, n)
                temp_vals = np.random.normal(25, 3, n)
                doc_vals = np.random.triangular(0.12, 0.15, 0.18, n)
                
                return umidade_vals, temp_vals, doc_vals

            umidade_vals, temp_vals, doc_vals = gerar_parametros_mc_unfccc(n_simulations)
            
            results_mc_unfccc = []
            for i in range(n_simulations):
                params_unfccc = [umidade_vals[i], temp_vals[i], doc_vals[i]]
                results_mc_unfccc.append(executar_simulacao_unfccc(params_unfccc, dias, residuos_kg_dia, massa_exposta_kg, h_exposta))

            results_array_unfccc = np.array(results_mc_unfccc)
            media_unfccc = np.mean(results_array_unfccc)
            intervalo_95_unfccc = np.percentile(results_array_unfccc, [2.5, 97.5])

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(results_array_unfccc, kde=True, bins=30, color='coral', ax=ax)
            ax.axvline(media_unfccc, color='red', linestyle='--', label=f'Média: {formatar_br(media_unfccc)} tCO₂eq')
            ax.axvline(intervalo_95_unfccc[0], color='green', linestyle=':', label='IC 95%')
            ax.axvline(intervalo_95_unfccc[1], color='green', linestyle=':')
            ax.set_title('Distribuição das Emissões Evitadas (Simulação Monte Carlo) - Cenário UNFCCC')
            ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
            ax.set_ylabel('Frequência')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(br_formatter)
            st.pyplot(fig)

            # Análise Estatística de Comparação
            st.subheader("📊 Análise Estatística de Comparação")
            
            # Teste de normalidade para as diferenças
            diferencas = results_array_tese - results_array_unfccc
            _, p_valor_normalidade_diff = stats.normaltest(diferencas)
            st.write(f"Teste de normalidade das diferenças (p-value): **{formatar_br_dec(p_valor_normalidade_diff, 5)}**")

            # Teste T pareado
            ttest_pareado, p_ttest_pareado = stats.ttest_rel(results_array_tese, results_array_unfccc)
            st.write(f"Teste T pareado: Estatística t = **{formatar_br_dec(ttest_pareado, 5)}**, P-valor = **{formatar_br_dec(p_ttest_pareado, 5)}**")

            # Teste de Wilcoxon para amostras pareadas
            wilcoxon_stat, p_wilcoxon = stats.wilcoxon(results_array_tese, results_array_unfccc)
            st.write(f"Teste de Wilcoxon (pareado): Estatística = **{formatar_br_dec(wilcoxon_stat, 5)}**, P-valor = **{formatar_br_dec(p_wilcoxon, 5)}**")

            # Tabela de resultados anuais - Proposta da Tese
            st.subheader("📋 Resultados Anuais - Proposta da Tese")

            # Criar uma cópia para formatação
            df_anual_formatado = df_anual_revisado.copy()
            for col in df_anual_formatado.columns:
                if col != 'Year':
                    df_anual_formatado[col] = df_anual_formatado[col].apply(formatar_br)

            st.dataframe(df_anual_formatado)

            # Tabela de resultados anuais - Metodologia UNFCCC
            st.subheader("📋 Resultados Anuais - Metodologia UNFCCC")

            # Criar uma cópia para formatação
            df_comp_formatado = df_comp_anual_revisado.copy()
            for col in df_comp_formatado.columns:
                if col != 'Year':
                    df_comp_formatado[col] = df_comp_formatado[col].apply(formatar_br)

            st.dataframe(df_comp_formatado)

    else:
        st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação' para ver os resultados.")

# Rodapé
st.markdown("---")
st.markdown("""

**📚 Referências por Cenário:**

**Cenário de Baseline (Aterro Sanitário):**
- Metano: IPCC (2006), UNFCCC (2016) e Wang et al. (2023) 
- Óxido Nitroso: Wang et al. (2017)
- Metano e Óxido Nitroso no pré-descarte: Feng et al. (2020)

**Proposta da Tese (Compostagem em reatores com minhocas):**
- Metano e Óxido Nitroso: Yang et al. (2017)

**Cenário UNFCCC (Compostagem sem minhocas a céu aberto):**
- Protocolo AMS-III.F: UNFCCC (2016)
- Fatores de emissões: Yang et al. (2017)
""")
