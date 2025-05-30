/**
 * Script para controle da barra de progresso e animações do painel de espera
 * Este arquivo contém a lógica para simular e controlar o progresso da análise de imagem
 * e pode ser integrado com a API de backend para atualização em tempo real
 */

// Configurações iniciais
document.addEventListener('DOMContentLoaded', function() {
    // Referências aos elementos DOM
    const progressFill = document.getElementById('progressFill');
    const progressText = document.querySelector('.progress-text');
    const urlParams = new URLSearchParams(window.location.search);
    const processingId = urlParams.get('id'); // Obtém o ID de processamento da URL

    // Instancia a classe ProcessingAPI
    const processingAPI = new ProcessingAPI();

    /**
     * Função para atualizar a interface do usuário com o progresso
     * @param {number} progressValue - Valor do progresso (0-100)
     * @param {string} statusMessage - Mensagem de status para exibir
     */
    function updateProgressUI(progressValue, statusMessage) {
        progressFill.style.width = `${progressValue}%`;
        progressText.textContent = statusMessage || `Analisando o conteúdo do vídeo... (${progressValue}%)`;

        // Você pode adicionar lógica para mudar a mensagem de status em diferentes estágios
        if (progressValue < 10) {
            progressText.textContent = 'Iniciando processamento...';
        } else if (progressValue < 30) {
            progressText.textContent = 'Fazendo upload do vídeo...';
        } else if (progressValue < 60) {
            progressText.textContent = 'Processando dados da imagem...';
        } else if (progressValue < 90) {
            progressText.textContent = 'Aplicando algoritmos de reconhecimento...';
        } else if (progressValue < 100) {
            progressText.textContent = 'Finalizando análise...';
        } else {
            progressText.textContent = 'Análise concluída!';
        }
    }

    /**
     * Callback para quando o processamento é concluído.
     * Redireciona para a página de resultados.
     * @param {string} resultId - ID do resultado do processamento
     */
    function onProcessingComplete(resultId) {
        updateProgressUI(100, 'Análise concluída!');
        processingAPI.stopMonitoring(); // Para de monitorar
        // Redireciona para a página de resultados com o ID do resultado
        processingAPI.navigateToResults(resultId);
    }

    // Verifica se um ID de processamento foi fornecido
    if (processingId) {
        console.log(`Iniciando monitoramento para o ID: ${processingId}`);
        // Inicia o monitoramento do processamento real via API
        processingAPI.startMonitoring(processingId, updateProgressUI, onProcessingComplete);
    } else {
        // Se não houver ID, exibe uma mensagem de erro ou um fallback
        progressText.textContent = 'Nenhum ID de processamento encontrado. Redirecionando para o upload...';
        setTimeout(() => {
            window.location.href = '/painel_upload';
        }, 3000); // Redireciona após 3 segundos
    }
});