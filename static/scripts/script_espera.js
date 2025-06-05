// static/script_espera.js

document.addEventListener('DOMContentLoaded', async function() {
    // Instancia a classe ProcessingAPI
    const processingAPI = new ProcessingAPI(); // Certifique-se que esta classe está definida em api.js

    // Elementos DOM para o status do lote
    const batchStatusMessage = document.getElementById('batchStatusMessage');
    const batchProgressFill = document.getElementById('batchProgressFill');
    const batchProgressText = document.getElementById('batchProgressText');
    const totalImagesSpan = document.getElementById('totalImages');
    const processedImagesSpan = document.getElementById('processedImages');
    const completedImagesSpan = document.getElementById('completedImages');
    const failedImagesSpan = document.getElementById('failedImages');
    const imageList = document.getElementById('imageList'); // Para listar o status de cada imagem
    const mainElement = document.querySelector('main'); // Usado para adicionar a mensagem inicial

    // Elementos que podem precisar ser escondidos/mostrados
    const loadingSpinner = document.querySelector('.loading-spinner');
    const progressBarContainer = document.querySelector('.progress-bar-container');
    const processingMessage = document.querySelector('.processing-message');
    const batchSummaryContainer = document.querySelector('.batch-summary');
    const imageListContainer = document.querySelector('.image-list-container');
    const infoBox = document.querySelector('.info-box');

    let statusCheckInterval; // Variável para armazenar o ID do setInterval
    const pollInterval = 3000; // Intervalo de 3 segundos para consultar a API

    const urlParams = new URLSearchParams(window.location.search);
    const batchId = urlParams.get('batch_id');

    // Função para mostrar a interface de processamento e esconder a mensagem de instrução
    function showProcessingUI() {
        if (loadingSpinner) loadingSpinner.style.display = 'block';
        if (progressBarContainer) progressBarContainer.style.display = 'block';
        if (processingMessage) processingMessage.style.display = 'block';
        if (batchSummaryContainer) batchSummaryContainer.style.display = 'block';
        if (imageListContainer) imageListContainer.style.display = 'block';
        if (infoBox) infoBox.style.display = 'block';

        const instructionsDiv = document.getElementById('instructionsMessage');
        if (instructionsDiv) instructionsDiv.style.display = 'none';
    }

    // Função para mostrar a mensagem de instrução e esconder a interface de processamento
    function showInstructionsUI() {
        if (loadingSpinner) loadingSpinner.style.display = 'none';
        if (progressBarContainer) progressBarContainer.style.display = 'none';
        if (processingMessage) processingMessage.style.display = 'none';
        if (batchSummaryContainer) batchSummaryContainer.style.display = 'none';
        if (imageListContainer) imageListContainer.style.display = 'none';
        if (infoBox) infoBox.style.display = 'none';

        let instructionsDiv = document.getElementById('instructionsMessage');
        if (!instructionsDiv) {
            instructionsDiv = document.createElement('div');
            instructionsDiv.id = 'instructionsMessage';
            instructionsDiv.className = 'message info';
            instructionsDiv.style.marginTop = '20px';
            const processingPanel = document.querySelector('.processing-panel');
            if (processingPanel) {
                processingPanel.appendChild(instructionsDiv);
            } else {
                mainElement.appendChild(instructionsDiv);
            }
        }
        instructionsDiv.innerHTML = `
            <h3>Bem-vindo ao Painel de Espera!</h3>
            <p>Para monitorar o progresso do processamento de suas imagens, por favor, inicie um novo lote na página de upload.</p>
            <p><a href="/painel_upload" class="button">Fazer Novo Upload</a></p>
            <p style="font-size: 0.9em; color: #888;">Você será redirecionado para esta página automaticamente após enviar suas imagens.</p>
        `;
        instructionsDiv.style.display = 'block';
    }

    // Função para atualizar a UI com o status do lote
    function updateBatchStatusUI(data) {
        if (!data) return;

        // Atualizar barra de progresso
        if (batchProgressFill && batchProgressText) {
            batchProgressFill.style.width = `${data.progress}%`;
            batchProgressText.textContent = `${data.progress.toFixed(0)}%`;
        }

        // Atualizar resumo do lote
        if (batchStatusMessage) batchStatusMessage.textContent = data.message;
        if (totalImagesSpan) totalImagesSpan.textContent = data.total_images;
        if (processedImagesSpan) processedImagesSpan.textContent = data.processed_images;
        if (completedImagesSpan) completedImagesSpan.textContent = data.completed_images;
        if (failedImagesSpan) failedImagesSpan.textContent = data.failed_images;

        // Atualizar lista de imagens individuais
        if (imageList) {
            imageList.innerHTML = ''; // Limpa a lista existente
            if (data.image_statuses && data.image_statuses.length > 0) {
                data.image_statuses.forEach(img => {
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <span>${img.original_filename}</span>
                        <span class="status ${img.status.toLowerCase().replace(/ /g, '-') || 'pending'}">${img.status}</span>
                    `;
                    imageList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'Nenhuma imagem no lote ainda ou dados indisponíveis.';
                imageList.appendChild(li);
            }
        }
    }

    // Função principal de polling
    async function checkProcessingStatus() {
        if (!batchId) {
            showInstructionsUI();
            clearInterval(statusCheckInterval);
            return;
        }

        try {
            showProcessingUI();
            const statusData = await processingAPI.getBatchProcessingStatus(batchId);
            console.log("Status do lote:", statusData); // Para depuração

            updateBatchStatusUI(statusData);

            if (statusData.status === 'completed') {
                clearInterval(statusCheckInterval); // Para o polling
                batchStatusMessage.textContent = "Lote de processamento concluído com sucesso!";
                
                // Opcional: Adicionar um botão para ir para a página de resultados geral
                const actionButton = document.createElement('a');
                actionButton.href = "/painel_resultados"; // Link para o painel de resultados geral
                actionButton.classList.add('button', 'success-button');
                actionButton.textContent = "Ver Resultados";
                
                // Adiciona o botão em algum lugar, por exemplo, abaixo da mensagem de status
                const messageContainer = document.querySelector('.processing-message');
                if (messageContainer && !messageContainer.querySelector('.success-button')) { // Evita duplicar
                    messageContainer.appendChild(document.createElement('br'));
                    messageContainer.appendChild(actionButton);
                }

            } else if (statusData.status === 'failed') {
                clearInterval(statusCheckInterval); // Para o polling
                batchStatusMessage.textContent = `Lote de processamento falhou: ${statusData.message}`;

                // Opcional: Adicionar um botão para tentar novamente ou voltar ao upload
                const actionButton = document.createElement('a');
                actionButton.href = "/painel_upload";
                actionButton.classList.add('button', 'error-button');
                actionButton.textContent = "Tentar Novo Upload";

                const messageContainer = document.querySelector('.processing-message');
                if (messageContainer && !messageContainer.querySelector('.error-button')) { // Evita duplicar
                    messageContainer.appendChild(document.createElement('br'));
                    messageContainer.appendChild(actionButton);
                }

            }
        } catch (error) {
            console.error('Erro ao verificar status do processamento:', error);
            batchStatusMessage.textContent = `Erro ao carregar status: ${error.message}`;
            clearInterval(statusCheckInterval); // Para o polling em caso de erro
            showInstructionsUI(); // Exibe as instruções em caso de erro crítico
        }
    }

    // Inicia o polling apenas se um batch_id estiver presente
    if (batchId) {
        showProcessingUI();
        checkProcessingStatus(); // Chama imediatamente para obter o status inicial
        statusCheckInterval = setInterval(checkProcessingStatus, pollInterval);
    } else {
        showInstructionsUI(); // Mostra as instruções se não há batch_id na URL
    }
});