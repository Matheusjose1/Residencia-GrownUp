// static/script_espera.js

document.addEventListener('DOMContentLoaded', async function() {
    // 1. CAPTURA RIGOROSA DOS ELEMENTOS
    // Usamos querySelector para classes e getElementById para IDs conforme o teu HTML
    const loadingSpinner = document.querySelector('.loading-spinner');
    const statusMessage = document.getElementById('statusMessage');
    const instructionsMessage = document.getElementById('instructionsMessage');
    const progressBarContainer = document.querySelector('.progress-container');
    const batchSummaryContainer = document.querySelector('.batch-summary');
    const imageListContainer = document.querySelector('.image-list-container');
    const infoBox = document.querySelector('.info-box');
    
    const batchProgressFill = document.getElementById('batchProgressFill');
    const batchProgressText = document.getElementById('batchProgressText');
    const totalImagesSpan = document.getElementById('totalImages');
    const processedImagesSpan = document.getElementById('processedImages');
    const completedImagesSpan = document.getElementById('completedImages');
    const failedImagesSpan = document.getElementById('failedImages');
    const imageList = document.getElementById('imageList');

    let statusCheckInterval;
    let currentBatchId = new URLSearchParams(window.location.search).get('batch_id');
    let hasRedirected = false;

    // 2. FUNÇÃO DE ATUALIZAÇÃO (Polling)
    async function updateBatchStatus() {
    if (!currentBatchId || hasRedirected) return;

    try {
        const response = await fetch(`/api/batch-status/${currentBatchId}`);
        if (!response.ok) throw new Error('Erro ao consultar status');
        
        const batchData = await response.json();
        console.log("Dados recebidos do server:", batchData);

        // 1. MAPEAMENTO CORRETO DOS CAMPOS (Conforme seu JSON de resposta)
        const status = batchData.overall_status || "";
        const progress = batchData.overall_progress || 0;
        const total = batchData.total_images || 0;
        const processed = batchData.processed_images || 0;
        const completed = batchData.completed_images || 0;
        const failed = batchData.failed_images || 0;

        // 2. ATUALIZAÇÃO DA INTERFACE
        if (batchProgressFill) batchProgressFill.style.width = `${progress}%`;
        if (batchProgressText) batchProgressText.textContent = `${Math.round(progress)}%`;
        
        if (totalImagesSpan) totalImagesSpan.textContent = total;
        if (processedImagesSpan) processedImagesSpan.textContent = processed;
        if (completedImagesSpan) completedImagesSpan.textContent = completed;
        if (failedImagesSpan) failedImagesSpan.textContent = failed;

        // 3. LOGICA DE PARADA E REDIRECIONAMENTO
        if (status.toLowerCase() === 'completed' && !hasRedirected) {
            console.log("Sucesso! Parando loop e redirecionando para resultados...");
            hasRedirected = true;
            clearInterval(statusCheckInterval);
            
            // Redireciona para a página de resultados
            window.location.href = `/painel_resultado?batch_id=${currentBatchId}`;
        }
    } catch (error) {
        console.error('Erro crítico no polling:', error);
    }
}

    // 3. INICIALIZAÇÃO
    if (currentBatchId) {
        console.log('Batch ID encontrado:', currentBatchId);
        
        // Exibir elementos de carregamento
        if (instructionsMessage) instructionsMessage.style.display = 'none';
        if (loadingSpinner) loadingSpinner.style.display = 'block';
        if (statusMessage) statusMessage.style.display = 'block';
        if (progressBarContainer) progressBarContainer.style.display = 'block';
        if (batchSummaryContainer) batchSummaryContainer.style.display = 'block';
        if (imageListContainer) imageListContainer.style.display = 'block';
        if (infoBox) infoBox.style.display = 'block';

        // Iniciar ciclo de verificação
        updateBatchStatus();
        statusCheckInterval = setInterval(updateBatchStatus, 2000);
    } else {
        // Se não hai ID, mostra mensagem de erro no painel
        if (instructionsMessage) {
            instructionsMessage.style.display = 'block';
            instructionsMessage.innerHTML = "Nenhum lote encontrado. <a href='/static/painel_upload.html'>Voltar ao Upload</a>";
        }
    }
});