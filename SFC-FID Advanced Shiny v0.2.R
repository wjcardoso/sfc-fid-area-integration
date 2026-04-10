library(shiny)
library(shinydashboard)
library(tidyverse)
library(MassSpecWavelet) # Opcional agora, mas mantido caso queira reverter depois
library(pracma)
library(minpack.lm)
library(plotly)
library(DT)
library(rhandsontable)

# ==============================================================================
# 1. FUNÇÕES AUXILIARES (BACKEND)
# ==============================================================================

# --- FUNÇÃO PARA LER O ARQUIVO .asc ---
ler_arquivo_dat <- function(caminho) {
  df <- read.delim(caminho, header = FALSE, skip = 19) %>% #Pula o cabeçalho de 19 linhas
    rename(Intensidade = V1) #Renomeia a coluna V1 para Intensidade
  df$Tempo <- seq(1:length(df$Intensidade)) * 0.0016667 #Multiplica cada índice pela taxa de amostragem do detector
  return(df) 
}

# --- FUNÇÃO DE LIMITES APRIMORADA (TREND STOP) ---
get_peak_limits <- function(y_sinal, centro_idx, baseline_thresh, tol_rise = 0.1, trend_window = 5) {
  n <- length(y_sinal)
  
  # --- ESQUERDA ---
  start_idx <- centro_idx
  while(start_idx > 1) {
    curr <- y_sinal[start_idx]
    prev <- y_sinal[start_idx - 1]
    if(curr <= baseline_thresh) break #Para se tocar na linha de base 
    if(prev > curr + (curr * tol_rise)) break #Para se subiu repentinamente (outro pico)
    if(start_idx - trend_window >= 1) {
      is_trend_up <- TRUE
      for(k in 1:trend_window) {
        val_k <- y_sinal[start_idx - k] 
        val_prev_k <- y_sinal[start_idx - k + 1]
        if(val_k <= val_prev_k) { is_trend_up <- FALSE; break } #Para se tiver tendência de X pontos
      }
      if(is_trend_up) break 
    }
    start_idx <- start_idx - 1
  }
  
  # --- DIREITA ---
  end_idx <- centro_idx
  while(end_idx < n) {
    curr <- y_sinal[end_idx]
    next_val <- y_sinal[end_idx + 1]
    if(curr <= baseline_thresh) break
    if(next_val > curr + (curr * tol_rise)) break
    if(end_idx + trend_window <= n) {
      is_trend_up <- TRUE
      for(k in 1:trend_window) {
        val_k <- y_sinal[end_idx + k]
        val_prev_k <- y_sinal[end_idx + k - 1]
        if(val_k <= val_prev_k) { is_trend_up <- FALSE; break }
      }
      if(is_trend_up) break
    }
    end_idx <- end_idx + 1
  }
  return(c(Start = start_idx, End = end_idx))
}

# --- FUNÇÃO PARA REFINAR A DETECÇÃO DO PICO ---
refinar_centros <- function(y_sinal, indices_cwt, janela) {
  indices_corrigidos <- numeric(length(indices_cwt))
  for(i in seq_along(indices_cwt)) {
    centro_atual <- indices_cwt[i]
    inicio <- max(1, centro_atual - janela)
    fim <- min(length(y_sinal), centro_atual + janela)
    segmento <- y_sinal[inicio:fim]
    if(length(segmento) == 0) next
    idx_max_local <- which.max(segmento)
    indices_corrigidos[i] <- inicio + idx_max_local - 1
  }
  return(indices_corrigidos)
}

# --- FÓRMULA MATEMÁTICA PARA O MODELO GAUSSIANO MODIFICADO ---
emg_model <- function(x, Area, mu, sigma, tau) {
  if(tau < 1e-6) tau <- 1e-6
  term1 <- Area / (2 * tau)
  term2 <- exp((sigma^2 / (2 * tau^2)) - ((x - mu) / tau))
  arg_erfc <- (sigma / (sqrt(2) * tau)) - ((x - mu) / (sqrt(2) * sigma))
  return(term1 * term2 * erfc(arg_erfc))
}

# --- REMOVER INCLINAÇÃO DE LINHA DE BASE ---
corrigir_baseline_linear <- function(tempo, intensidade) {
  n <- length(tempo)
  if(n < 2) return(list(Sinal = intensidade, Baseline = rep(0, n)))
  y_start <- mean(intensidade[1:min(3, n)])
  y_end   <- mean(intensidade[max(1, n-2):n])
  slope <- (y_end - y_start) / (tempo[n] - tempo[1])
  intercept <- y_start - slope * tempo[1]
  baseline_line <- slope * tempo + intercept
  y_corrigido <- intensidade - baseline_line
  y_corrigido[y_corrigido < 0] <- 0
  return(list(Sinal = y_corrigido, Baseline = baseline_line))
}

# --- FUNÇÃO PARA CALCULAR AS ÁREAS ---
processar_cromatograma <- function(df_amostra, tabela_regioes) {
  resultados <- list()
  base_global <- min(df_amostra$Intensidade, na.rm=TRUE)
  df_amostra$Int_Global_Corr <- df_amostra$Intensidade - base_global
  
  for(i in 1:nrow(tabela_regioes)) {
    grupo <- as.character(tabela_regioes$Grupo[i])
    if(is.null(grupo) || is.na(grupo) || grupo == "") next
    
    t_inicio <- as.numeric(tabela_regioes$Inicio_min[i])
    t_fim <- as.numeric(tabela_regioes$Fim_min[i])
    metodo <- as.character(tabela_regioes$Metodo_Integracao[i])
    
    dados_roi <- df_amostra %>% filter(Tempo >= t_inicio & Tempo <= t_fim)
    
    if(nrow(dados_roi) < 5) {
      resultados[[grupo]] <- list(Info = data.frame(Grupo = grupo, Area = 0, Metodo = "Sem Dados"), Dados = dados_roi, Fitted = NULL)
      next
    }
    
    area_calculada <- 0
    ajuste_plot <- NULL
    
    #Soma a área geométrica abaixo da curva
    if(metodo == "Soma") {
      base_local <- min(dados_roi$Intensidade)
      y_corrigido <- dados_roi$Intensidade - base_local
      area_calculada <- trapz(dados_roi$Tempo, y_corrigido)
      
    #Ajusta o modelo EMG aos dados
    } else if (metodo == "Fitting") {
      tryCatch({
        res_base <- corrigir_baseline_linear(dados_roi$Tempo, dados_roi$Intensidade)
        dados_roi$Int_Corr <- res_base$Sinal
        
        #Chutes de parâmetros iniciais
        idx_max <- which.max(dados_roi$Int_Corr)
        mu_init <- dados_roi$Tempo[idx_max]
        h_max <- dados_roi$Int_Corr[idx_max]
        sigma_init <- (t_fim - t_inicio) / 20 
        tau_init <- sigma_init * 2 
        area_init <- h_max * sigma_init * 4
        
        #Algoritmo de Levenberg-Marquardt para ajuste dos parâmetros do EMG
        fit <- nlsLM(Int_Corr ~ emg_model(Tempo, Area, mu, sigma, tau),
                     data = dados_roi,
                     start = list(Area = area_init, mu = mu_init, sigma = sigma_init, tau = tau_init),
                     control = nls.lm.control(maxiter = 100),
                     lower = c(0, t_inicio, 0, 0)) 
        
        coefs <- coef(fit)
        area_calculada <- coefs["Area"]
        ajuste_plot <- predict(fit) 
        
      }, error = function(e) {
        area_calculada <- trapz(dados_roi$Tempo, dados_roi$Intensidade - min(dados_roi$Intensidade)) 
      })
    }
    
    resultados[[grupo]] <- list(
      Info = data.frame(Grupo = grupo, Area = area_calculada, Metodo = metodo),
      Dados = dados_roi,
      Fitted = ajuste_plot
    )
  }
  return(resultados)
}

# ==============================================================================
# 2. UI
# ==============================================================================

ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "SFC-FID Advanced"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("1. Análise do Padrão", tabName = "tab_padrao", icon = icon("flask")),
      menuItem("2. Análise da Amostra", tabName = "tab_amostra", icon = icon("chart-area"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # --- ABA 1: PADRÃO ---
      tabItem(tabName = "tab_padrao",
              fluidRow(
                box(title = "1. Detecção & Limites", width = 3, status = "primary",
                    fileInput("file_padrao", "Arquivo Padrão (.asc)", accept = c(".asc", ".txt")),
                    hr(),
                    h4("Parâmetros FindPeaks"),
                    numericInput("peak_nups", "Subida Mínima:", 1, min=1),
                    numericInput("peak_ndowns", "Descida Mínima:", 1, min=1),
                    numericInput("peak_dist", "Distância Mínima:", 100, min=1),
                    numericInput("peak_minheight", "Altura Mín. Absoluta:", 150000, min=0),
                    numericInput("refine_window", "Janela Refino:", 10),
                    hr(),
                    h4("Critérios de Corte (Limites)"),
                    sliderInput("threshold_perc", "Intensidade Mínima (% do Máximo)", 0.001, 0.01, 0.001, step=0.001),
                    sliderInput("rise_tolerance", "Tolerância de Subida (%)", 0.01, 0.5, 0.1, step=0.01),
                    numericInput("trend_window", "Pontos de Tendência (Stop):", 5, min=1, max=20),
                    actionButton("btn_process_padrao", "Detectar Picos", class = "btn-primary")
                ),
                box(title = "Cromatograma do Padrão", width = 9, status = "primary",
                    plotlyOutput("plot_padrao", height = "700px")
                )
              ),
              fluidRow(
                column(width = 10,
                       box(title = "Ref: Picos Detectados", width = NULL, status = "info",
                           DTOutput("table_ref_padrao")
                       )
                ),
                column(width = 10,
                       box(title = "2. Definição de Regiões (Padrão)", width = NULL, status = "warning",
                           fluidRow(
                             column(5, actionButton("btn_preencher_padrao", "Copiar Tempos Detectados", icon("arrow-down"), class="btn-info btn-xs", style="margin-top:5px;"))
                           ),
                           br(),
                           rHandsontableOutput("table_regioes_padrao"),
                           hr(),
                           actionButton("btn_integrar_padrao", "Integrar Padrão (QA)", class="btn-success")
                       ),
                       box(title = "Resultados QA (Padrão)", width = NULL,
                           DTOutput("tbl_resultado_padrao")
                       )
                )
              )
      ),
      
      # --- ABA 2: AMOSTRA ---
      tabItem(tabName = "tab_amostra",
              fluidRow(
                box(title = "1. Carregar e Detectar", width = 3, status = "success",
                    fileInput("file_amostra", "Arquivo Amostra (.asc)", accept = c(".asc")),
                    hr(),
                    h4("Parâmetros FindPeaks"),
                    numericInput("peak_nups_samp", "Subida Mínima:", 1, min=1),
                    numericInput("peak_ndowns_samp", "Descida Mínima:", 5, min=1),
                    numericInput("peak_dist_samp", "Distância Mínima:", 100, min=1),
                    numericInput("peak_minheight_samp", "Altura Mín. Absoluta:", 30000, min=0),
                    numericInput("refine_window_samp", "Janela Refino:", 10),
                    hr(),
                    h4("Critérios de Corte"),
                    sliderInput("threshold_perc_samp", "Intensidade Mínima (% do Máximo)", 0.01, 0.1, 0.05, step=0.01),
                    sliderInput("rise_tolerance_samp", "Tolerância de Subida (%)", 0.01, 0.5, 0.1, step=0.01),
                    numericInput("trend_window_samp", "Pontos de Tendência (Stop):", 5, min=1, max=20),
                    actionButton("btn_detect_amostra", "Detectar Picos", class="btn-primary")
                ),
                box(title = "Inspeção Visual", width = 9, status = "success",
                    plotlyOutput("plot_overlay", height = "700px"),
                    helpText("Linha Preta: Padrão | Linha Azul: Amostra | Pontos Vermelhos/Números: Picos Amostra")
                )
              ),
              fluidRow(
                column(width = 9,
                       box(title = "Ref: Picos Detectados (Amostra)", width = NULL, status = "info",
                           DTOutput("table_ref_amostra")
                       )
                ),
                column(width = 9,
                       box(title = "2. Definição de Regiões (Amostra)", width = NULL, status = "warning",
                           fluidRow(
                             column(6, selectInput("metodo_astm", "Método ASTM:", c("Diesel", "Querosene"))),
                             column(6, style="margin-top: 25px;", actionButton("btn_auto_regioes", "Gerar Regiões (Auto)", icon("magic"), class="btn-warning"))
                           ),
                           helpText("Usa os limites (Start/End) reais dos picos da amostra para definir os cortes."),
                           br(),
                           rHandsontableOutput("table_regioes_amostra"),
                           hr(),
                           actionButton("btn_integrar_final", "Integrar Amostra", class="btn-danger")
                       ),
                       box(title = "Resultados Finais", width = NULL,
                           DTOutput("tbl_resultado_final")
                       )
                )
              ),
              fluidRow(
                box(title = "Visualização da Integração", width = 12, collapsible = TRUE,
                    plotlyOutput("plot_final_integrado", height = "400px")
                )
              )
      )
    )
  )
)

# ==============================================================================
# 3. SERVER
# ==============================================================================

server <- function(input, output, session) {
  
  rv <- reactiveValues(
    df_padrao = NULL, picos_padrao = NULL,
    regioes_padrao = tribble(
      ~Grupo,        ~Inicio_min, ~Fim_min, ~Metodo_Integracao,
      "Hexadecano",      0,           0,           "Soma",
      "Tolueno",         0,           0,           "Soma",
      "THN",             0,           0,           "Soma",
      "Naftaleno",       0,           0,           "Soma",
      "Dibenzotiofeno",  0,           0,           "Soma"
    ),
    df_amostra = NULL, picos_amostra = NULL,
    regioes_amostra = tribble(
      ~Grupo,         ~Inicio_min, ~Fim_min, ~Metodo_Integracao,
      "Saturados",         0,           0,           "Soma",
      "Mono-aromáticos",   0,           0,           "Soma",
      "Di-aromáticos",     0,           0,           "Soma",
      "Poli-aromáticos",   0,           0,           "Soma",
      "Saturados Pesados", 0,           0,           "Soma", 
      "Olefinas",          0,           0,           "Fitting"
    ),
    resultados_finais = NULL
  )
  
  # --- PADRÃO ---
  observeEvent(input$btn_process_padrao, {
    req(input$file_padrao)
    raw <- ler_arquivo_dat(input$file_padrao$datapath)
    raw$Intensidade <- raw$Intensidade - min(raw$Intensidade)
    rv$df_padrao <- raw
    
    # Substituição para FindPeaks
    picos_matriz <- pracma::findpeaks(raw$Intensidade, 
                                      nups = input$peak_nups, 
                                      ndowns = input$peak_ndowns, 
                                      minpeakdistance = input$peak_dist, 
                                      minpeakheight = input$peak_minheight)
    
    if(!is.null(picos_matriz)) {
      idx_raw <- picos_matriz[, 2]
    } else {
      idx_raw <- integer(0)
    }
    
    if(length(idx_raw) > 0) {
      idx_real <- refinar_centros(raw$Intensidade, idx_raw, janela = input$refine_window)
      idx_real <- unique(idx_real)
      
      tab_ref <- data.frame(ID = 1:length(idx_real), Tempo_Retencao = raw$Tempo[idx_real],
                            Tempo_Inicio = NA, Tempo_Fim = NA, Intensidade_Max = raw$Intensidade[idx_real])
      
      for(i in 1:nrow(tab_ref)) {
        cid <- idx_real[i]
        h <- raw$Intensidade[cid]
        limiar <- (input$threshold_perc/100) * h
        lims <- get_peak_limits(raw$Intensidade, cid, limiar, 
                                tol_rise = input$rise_tolerance, 
                                trend_window = input$trend_window)
        tab_ref$Tempo_Inicio[i] <- raw$Tempo[lims[1]]
        tab_ref$Tempo_Fim[i] <- raw$Tempo[lims[2]]
      }
      rv$picos_padrao <- tab_ref
    } else {
      rv$picos_padrao <- data.frame(ID = integer(), Tempo_Retencao = numeric(), Tempo_Inicio = numeric(), Tempo_Fim = numeric(), Intensidade_Max = numeric())
      showNotification("Nenhum pico detectado. Ajuste os parâmetros de FindPeaks.", type="warning")
    }
  })
  
  # --- COPIAR TEMPOS DETECTADOS PARA A TABELA (PADRÃO) ---
  observeEvent(input$btn_preencher_padrao, {
    req(rv$picos_padrao)
    n_picos <- nrow(rv$picos_padrao)
    n_rows <- nrow(rv$regioes_padrao)
    
    if (n_picos == 0) {
      showNotification("Detecte os picos primeiro!", type = "error")
      return()
    }
    
    nova_tabela <- rv$regioes_padrao
    limite <- min(n_picos, n_rows)
    
    for (i in 1:limite) {
      nova_tabela$Inicio_min[i] <- rv$picos_padrao$Tempo_Inicio[i]
      nova_tabela$Fim_min[i] <- rv$picos_padrao$Tempo_Fim[i]
    }
    
    rv$regioes_padrao <- nova_tabela
    
    if (n_picos < n_rows) {
      showNotification(paste("Apenas", n_picos, "picos foram detectados. Tabela parcialmente preenchida."), type = "warning")
    } else {
      showNotification("Tempos copiados com sucesso! Verifique a tabela.", type = "message")
    }
  })
  
  output$table_regioes_padrao <- renderRHandsontable({
    rhandsontable(rv$regioes_padrao, stretchH = "all") %>%
      hot_col("Metodo_Integracao", type = "dropdown", source = c("Soma", "Fitting"))
  })
  
  observeEvent(input$btn_integrar_padrao, {
    req(rv$df_padrao, input$table_regioes_padrao)
    rv$regioes_padrao <- hot_to_r(input$table_regioes_padrao) 
    res <- processar_cromatograma(rv$df_padrao, rv$regioes_padrao)
    df_res <- bind_rows(lapply(res, function(x) x$Info))
    total <- sum(df_res$Area)
    df_res$Porcentagem <- (df_res$Area / total) * 100
    output$tbl_resultado_padrao <- renderDT({
      datatable(df_res, options=list(dom='t', pageLength=10)) %>% formatRound(c("Area", "Porcentagem"), 2)
    })
  })
  
  output$plot_padrao <- renderPlotly({
    req(rv$df_padrao)
    p <- ggplot(rv$df_padrao, aes(x=Tempo, y=Intensidade)) + geom_line(size=0.3) + theme_minimal()
    if(!is.null(rv$picos_padrao) && nrow(rv$picos_padrao) > 0) {
      p <- p + geom_point(data=rv$picos_padrao, aes(x=Tempo_Retencao, y=Intensidade_Max), color="red")
    }
    pl <- ggplotly(p)
    if(!is.null(rv$picos_padrao) && nrow(rv$picos_padrao) > 0) {
      pl <- pl %>% add_text(data=rv$picos_padrao, x=~Tempo_Retencao, y=~Intensidade_Max, text=~ID, textposition = "top center", showlegend=F)
    }
    pl %>% config(toImageButtonOptions = list(format = "png", filename = "padrao_sfc", width = 1200, height = 700, scale = 3))
  })
  
  output$table_ref_padrao <- renderDT({
    req(rv$picos_padrao)
    datatable(rv$picos_padrao, options = list(pageLength = 5, dom='tip')) %>% formatRound(2:4, 2)
  })
  
  # --- AMOSTRA ---
  observeEvent(input$btn_detect_amostra, {
    req(input$file_amostra)
    raw <- ler_arquivo_dat(input$file_amostra$datapath)
    raw$Intensidade <- raw$Intensidade - min(raw$Intensidade)
    rv$df_amostra <- raw
    
    # Substituição para FindPeaks
    picos_matriz <- pracma::findpeaks(raw$Intensidade, 
                                      nups = input$peak_nups_samp, 
                                      ndowns = input$peak_ndowns_samp, 
                                      minpeakdistance = input$peak_dist_samp, 
                                      minpeakheight = input$peak_minheight_samp)
    
    if(!is.null(picos_matriz)) {
      idx_raw <- picos_matriz[, 2]
    } else {
      idx_raw <- integer(0)
    }
    
    if(length(idx_raw) > 0) {
      idx_real <- refinar_centros(raw$Intensidade, idx_raw, janela = input$refine_window_samp)
      idx_real <- unique(idx_real)
      
      tab_ref <- data.frame(ID = 1:length(idx_real), Tempo_Retencao = raw$Tempo[idx_real],
                            Tempo_Inicio = NA, Tempo_Fim = NA, Intensidade_Max = raw$Intensidade[idx_real])
      
      for(i in 1:nrow(tab_ref)) {
        cid <- idx_real[i]
        h <- raw$Intensidade[cid]
        limiar <- (input$threshold_perc_samp/100) * h
        lims <- get_peak_limits(raw$Intensidade, cid, limiar, 
                                tol_rise = input$rise_tolerance_samp, 
                                trend_window = input$trend_window_samp)
        tab_ref$Tempo_Inicio[i] <- raw$Tempo[lims[1]]
        tab_ref$Tempo_Fim[i] <- raw$Tempo[lims[2]]
      }
      rv$picos_amostra <- tab_ref
    } else {
      rv$picos_amostra <- data.frame(ID = integer(), Tempo_Retencao = numeric(), Tempo_Inicio = numeric(), Tempo_Fim = numeric(), Intensidade_Max = numeric())
      showNotification("Nenhum pico detectado na amostra. Ajuste os parâmetros de FindPeaks.", type="warning")
    }
  })
  
  # === LÓGICA DE REGIÕES AUTOMÁTICAS (DIESEL/QUEROSENE) ===
  observeEvent(input$btn_auto_regioes, {
    req(rv$picos_amostra, input$table_regioes_padrao)
    
    reg_std <- hot_to_r(input$table_regioes_padrao)
    
    get_sample_peak_limits <- function(marker_name) {
      row_std <- reg_std %>% filter(str_detect(Grupo, fixed(marker_name, ignore_case=TRUE)))
      if(nrow(row_std) == 0) return(NULL)
      
      t_target <- row_std$Inicio_min[1]
      if(t_target == 0) return(NULL)
      
      if(nrow(rv$picos_amostra) == 0) return(NULL)
      
      pico_prox <- rv$picos_amostra %>% 
        mutate(dist = c(abs(Tempo_Fim - t_target))) %>%
        arrange(dist) %>% slice(1)
      
      return(pico_prox) 
    }
    
    p_hex <- get_sample_peak_limits("Hexadecano")
    p_tol <- get_sample_peak_limits("Tolueno")
    p_thn <- get_sample_peak_limits("THN")
    p_naf <- get_sample_peak_limits("Naftaleno")
    p_dib <- get_sample_peak_limits("Dibenzotiofeno")
    
    get_std_start <- function(peak_name) {
      row_naft <- reg_std %>% filter(str_detect(Grupo, fixed(peak_name, ignore_case=TRUE)))
      if(nrow(row_naft) == 0) return(NULL)
      
      t_naft_start <- row_naft$Inicio_min[1]
      return(t_naft_start)
    }
    
    p_naf_std_start <- get_std_start("Naftaleno")
    p_dib_std_start <- get_std_start("Dibenzotiofeno")
    
    if(is.null(p_hex) || is.null(p_tol) || is.null(p_naf)) {
      showNotification("Não foi possível encontrar Hexadecano, Tolueno ou Naftaleno definidos no Padrão/Amostra.", type="error")
      return()
    }
    
    cut_sat_end <- p_tol$Tempo_Inicio
    metodo <- input$metodo_astm
    cut_mono_end <- 0
    
    if(metodo == "Diesel") {
      cut_mono_end <- p_naf_std_start
    } else {
      if(is.null(p_thn)) {
        showNotification("Para Querosene, defina a região do THN no Padrão.", type="error")
        return()
      }
      cut_mono_end <- p_naf$Tempo_Fim
    }
    
    cut_di_end <- p_dib_std_start
    
    nova_tabela <- tribble(
      ~Grupo,               ~Inicio_min,                      ~Fim_min,                 ~Metodo_Integracao,
      "Saturados",           rv$picos_amostra$Tempo_Inicio[1], cut_sat_end,              "Soma", 
      "Mono-aromáticos",     cut_sat_end,                      cut_mono_end,             "Soma",
      "Di-aromáticos",       cut_mono_end,                     cut_di_end,               "Soma",
      "Poli-aromáticos",     cut_di_end,                       max(rv$df_amostra$Tempo), "Soma",
      "Saturados Pesados",   0,                                0,                        "Soma",
      "Olefinas",            0,                                0,                        "Fitting"
    )
    
    rv$regioes_amostra <- nova_tabela
    showNotification(paste("Regiões geradas (Baseadas em Start/End):", metodo), type="message")
  })
  
  output$table_regioes_amostra <- renderRHandsontable({
    rhandsontable(rv$regioes_amostra, stretchH = "all") %>%
      hot_col("Metodo_Integracao", type = "dropdown", source = c("Soma", "Fitting"))
  })
  
  output$plot_overlay <- renderPlotly({
    req(rv$df_amostra)
    p <- plot_ly() %>% 
      add_lines(data = rv$df_amostra, x = ~Tempo, y = ~Intensidade, name = "Amostra", line = list(color = 'blue', width = 1)) %>%
      layout(title = "Sobreposição", xaxis = list(title = "Tempo"), yaxis = list(title = "Intensidade"))
    
    if(!is.null(rv$df_padrao)) {
      p <- p %>% add_lines(data = rv$df_padrao, x = ~Tempo, y = ~Intensidade, name = "Padrão", line = list(color = 'black', width = 1, dash='dot'))
    }
    
    if(!is.null(rv$picos_amostra) && nrow(rv$picos_amostra) > 0) {
      p <- p %>% 
        add_markers(data = rv$picos_amostra, x = ~Tempo_Retencao, y = ~Intensidade_Max, name = "Picos Amostra", marker = list(color='red', size=8)) %>%
        add_text(data = rv$picos_amostra, x = ~Tempo_Retencao, y = ~Intensidade_Max, text = ~ID, textposition = "top center", showlegend=FALSE)
    }
    p %>% config(toImageButtonOptions = list(format = "png", filename = "sobreposicao_amostra", width = 1200, height = 700, scale = 3))
  })
  
  output$table_ref_amostra <- renderDT({
    req(rv$picos_amostra)
    datatable(rv$picos_amostra, options = list(pageLength = 5, dom='tip')) %>% formatRound(2:4, 2)
  })
  
  observeEvent(input$btn_integrar_final, {
    req(rv$df_amostra, input$table_regioes_amostra)
    rv$regioes_amostra <- hot_to_r(input$table_regioes_amostra)
    res <- processar_cromatograma(rv$df_amostra, rv$regioes_amostra)
    rv$resultados_finais <- res
    df_res <- bind_rows(lapply(res, function(x) x$Info))
    total <- sum(df_res$Area)
    df_res$Porcentagem <- (df_res$Area / total) * 100
    output$tbl_resultado_final <- renderDT({
      datatable(df_res, extensions = 'Buttons', options = list(dom = 'Bfrtip', buttons = c('copy', 'excel'))) %>% formatRound(c("Area", "Porcentagem"), 2)
    })
  })
  
  output$plot_final_integrado <- renderPlotly({
    req(rv$resultados_finais, rv$df_amostra)
    p <- ggplot(rv$df_amostra, aes(x=Tempo, y=Intensidade)) + geom_line(color="gray40", size=0.3) + theme_minimal()
    cores_fixas <- c("Saturados" = "lightblue", "Mono-aromáticos" = "orange",
                     "Di-aromáticos" = "lightgreen", "Poli-aromáticos" = "darkgreen",
                     "Saturados Pesados" = "purple", "Olefinas" = "red")
    for(grp in names(rv$resultados_finais)) {
      item <- rv$resultados_finais[[grp]]
      if(nrow(item$Dados) > 0) {
        if(item$Info$Metodo == "Fitting" && !is.null(item$Fitted)) {
          p <- p + geom_line(data=item$Dados, aes(y=item$Fitted), color="red", size=0.5) +
            geom_area(data = item$Dados, aes(y = item$Fitted), fill = "red", alpha = 0.5)
        } else {
          cor <- ifelse(grp %in% names(cores_fixas), cores_fixas[grp], "gray")
          p <- p + geom_area(data=item$Dados, aes(y=Intensidade), fill=cor, alpha=0.5)
        }
      }
    }
    ggplotly(p) %>% 
      config(toImageButtonOptions = list(format = "png", filename = "integracao_final", width = 1200, height = 700, scale = 3))
  })
}

shinyApp(ui, server)

