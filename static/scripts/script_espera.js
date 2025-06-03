// static/script_espera.js

document.addEventListener('DOMContentLoaded', async function() {
    // Elementos DOM para o status do lote
    const batchStatusMessage = document.getElementById('batchStatusMessage');
    const batchProgressFill = document.getElementById('batchProgressFill');
    const batchProgressText = document.getElementById('batchProgressText');
    const totalImagesSpan = document.getElementById('totalImages');
    const processedImagesSpan = document.getElementById('processedImages');
    const completedImagesSpan = document.getElementById('completedImages');
    const failedImagesSpan = document.getElementById('failedImages');
    const imageList = document.getElementById('imageList');

    // Elementos que podem precisar ser escondidos/mostrados
    const loadingSpinner = document.querySelector('.loading-spinner');
    const progressBarContainer = document.querySelector('.progress-bar-container');
    const processingMessage = document.querySelector('.processing-message'); // Mensagem "Nossa inteligência artificial..."
    const batchSummaryContainer = document.querySelector('.batch-summary'); // Container do resumo de imagens
    const imageListContainer = document.querySelector('.image-list-container'); // Container da lista de imagens
    const infoBox = document.querySelector('.info-box'); // A caixa de informação "Não feche esta janela"

    let statusCheckInterval; // Variável para armazenar o ID do setInterval
    let currentBatchId = null; // Para armazenar o ID do lote sendo monitorado

    // Função para obter o parâmetro 'batch_id' da URL
    function getBatchIdFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('batch_id');
    }

    // Função para atualizar a interface da barra de progresso do LOTE
    function updateBatchProgressUI(progressValue, message = "Processando Lote...") {
        const displayProgress = Math.min(Math.max(0, progressValue), 100);
        batchProgressFill.style.width = `${displayProgress}%`;
        batchProgressText.textContent = `${Math.round(displayProgress)}%`;
        batchStatusMessage.textContent = message;

        if (displayProgress === 100) {
            batchProgressFill.classList.add('finished');
        } else {
            batchProgressFill.classList.remove('finished');
        }
    }

    // Função para atualizar o resumo do lote
    function updateBatchSummary(total, processed, completed, failed) {
        totalImagesSpan.textContent = total;
        processedImagesSpan.textContent = processed;
        completedImagesSpan.textContent = completed;
        failedImagesSpan.textContent = failed;
    }

    // Função para renderizar a lista de imagens
    function renderImageList(images) {
        imageList.innerHTML = ''; // Limpa a lista existente
        if (images && images.length > 0) {
            images.forEach(image => {
                const listItem = document.createElement('li');
                // Adiciona classe para estilização de status (ex: status-pending, status-in_progress, etc.)
                listItem.classList.add(`status-${image.status}`); 
                listItem.innerHTML = `
                    <span>${image.original_filename}</span>
                    <span>Status: <strong>${image.status}</strong></span>
                    <span>Progresso: ${image.progress}%</span>
                    ${image.status === 'completed' && image.result && image.result.id // Verifica se existe um resultado e se tem ID
                        ? `<a href="/painel_resultados?id=${image.result.id}" target="_blank" class="view-result-link">(Ver Resultado)</a>`
                        : ''
                    }
                    ${image.status === 'failed' && image.message // Se falhou e tem mensagem
                        ? `<span class="error-message">${image.message}</span>`
                        : ''
                    }
                `;
                imageList.appendChild(listItem);
            });
        } else {
            // Se não houver imagens para renderizar ou o array for vazio
            const noImagesMessage = document.createElement('li');
            noImagesMessage.textContent = "Nenhuma imagem para exibir no lote.";
            imageList.appendChild(noImagesMessage);
        }
    }

    // Função principal para verificar o status do lote e imagens
    async function checkBatchStatus() {
        if (!currentBatchId) { // Se não há ID de lote, não tenta buscar status
            console.warn("checkBatchStatus chamado sem um currentBatchId definido. Ignorando.");
            // Não deve mais causar erro aqui, pois o bloco 'else' inicial lida com isso.
            return;
        }

        try {
            // 1. Obter status geral do lote
            const batchStatus = await getBatchStatus(currentBatchId); 
            console.log("Status do Lote recebido do backend:", batchStatus);

            if (!batchStatus) {
                // Se a API retornar null/undefined, isso pode ser um erro silencioso do backend.
                // Tratar como um erro de comunicação.
                throw new Error("Não foi possível obter o status do lote.");
            }

            // Exibe os elementos de UI normais
            if (loadingSpinner) loadingSpinner.style.display = 'flex'; // ou 'block'
            if (progressBarContainer) progressBarContainer.style.display = 'block';
            if (processingMessage) processingMessage.style.display = 'block';
            if (batchSummaryContainer) batchSummaryContainer.style.display = 'block';
            if (imageListContainer) imageListContainer.style.display = 'block';
            if (infoBox) infoBox.style.display = 'flex'; // ou 'block'

            updateBatchProgressUI(batchStatus.overall_progress, batchStatus.message);
            updateBatchSummary(
                batchStatus.total_images,
                batchStatus.processed_images,
                batchStatus.completed_images,
                batchStatus.failed_images
            );

            // 2. Obter status de cada imagem dentro do lote
            const imagesInBatch = await getAllImagesForBatch(currentBatchId); 
            console.log("Imagens no Lote recebidas do backend:", imagesInBatch);

            if (imagesInBatch) {
                renderImageList(imagesInBatch);
            }

            // 3. Verificar o status final do lote para parar o monitoramento
            if (batchStatus.overall_status === 'completed' || batchStatus.overall_status === 'failed' || batchStatus.overall_status === 'partially_completed') {
                clearInterval(statusCheckInterval); // Para de verificar o status
                statusCheckInterval = null; // Limpa a variável do intervalo
                console.log("Processamento do lote concluído. Você pode verificar os resultados individuais.");
                
                if (loadingSpinner) loadingSpinner.style.display = 'none'; // Esconde o spinner ao finalizar

                if (batchStatus.overall_status === 'completed') {
                    batchStatusMessage.textContent = 'Lote Processado: Concluído!';
                } else if (batchStatus.overall_status === 'partially_completed') {
                    batchStatusMessage.textContent = 'Lote Processado: Concluído com Falhas.';
                } else { // 'failed'
                    batchStatusMessage.textContent = 'Lote Processado: Falhou Completamente.';
                }
            }
        } catch (error) {
            console.error('Erro ao verificar status do lote:', error);
            // Em caso de erro na comunicação com a API, exibe mensagem e para o polling
            updateBatchProgressUI(0, 'Erro de comunicação com o servidor.');
            batchStatusMessage.textContent = 'Erro ao carregar dados do lote. Por favor, recarregue a página ou tente novamente.';
            
            // Esconder elementos de progresso
            if (loadingSpinner) loadingSpinner.style.display = 'none';
            if (progressBarContainer) progressBarContainer.style.display = 'none';
            if (processingMessage) processingMessage.style.display = 'none';
            
            clearInterval(statusCheckInterval);
            statusCheckInterval = null; // Limpa a variável do intervalo
            // *** REMOVIDO: NENHUM REDIRECIONAMENTO AQUI ***
        }
    }

    // --- Lógica principal na carga da página ---
    const initialBatchId = getBatchIdFromUrl();

    if (initialBatchId) {
        currentBatchId = initialBatchId;
        console.log(`Página de espera de lote carregada. Iniciando monitoramento para o Lote ID: ${initialBatchId}`);
        
        // Esconder a mensagem de instrução se ela existir (para quando um lote é válido)
        const instructionsDiv = document.getElementById('instructionsMessage');
        if (instructionsDiv) {
            instructionsDiv.style.display = 'none';
        }

        // Exibe os elementos de UI normais
        if (loadingSpinner) loadingSpinner.style.display = 'flex';
        if (progressBarContainer) progressBarContainer.style.display = 'block';
        if (processingMessage) processingMessage.style.display = 'block';
        if (batchSummaryContainer) batchSummaryContainer.style.display = 'block';
        if (imageListContainer) imageListContainer.style.display = 'block';
        if (infoBox) infoBox.style.display = 'flex';


        updateBatchProgressUI(0, 'Conectando ao servidor para monitorar o lote...');
        await checkBatchStatus(); // Chama a função uma vez imediatamente para obter o status inicial
        statusCheckInterval = setInterval(checkBatchStatus, 2000); // Configura o intervalo
    } else {
        // --- NOVO COMPORTAMENTO: Se não houver batch_id ---
        console.log("Nenhum ID de lote encontrado na URL. Exibindo instruções.");
        
        // Esconder todos os elementos de UI relacionados ao monitoramento de um lote ativo
        if (loadingSpinner) loadingSpinner.style.display = 'none';
        if (progressBarContainer) progressBarContainer.style.display = 'none';
        if (processingMessage) processingMessage.style.display = 'none';
        if (batchSummaryContainer) batchSummaryContainer.style.display = 'none';
        if (imageListContainer) imageListContainer.style.display = 'none';
        if (infoBox) infoBox.style.display = 'none'; // Esconde a caixa de informação padrão
        
        updateBatchProgressUI(0, 'Nenhum ID de lote para monitorar.'); // Limpa a barra de progresso
        batchStatusMessage.textContent = 'Nenhum Lote Selecionado.'; // Mensagem principal
        
        // Limpa os resumos para N/A
        totalImagesSpan.textContent = 'N/A';
        processedImagesSpan.textContent = 'N/A';
        completedImagesSpan.textContent = 'N/A';
        failedImagesSpan.textContent = 'N/A';
        imageList.innerHTML = ''; // Limpa a lista de imagens

        // Adiciona/exibe a mensagem instrutiva
        const mainElement = document.querySelector('main');
        let instructionsDiv = document.getElementById('instructionsMessage');
        if (!instructionsDiv) {
            instructionsDiv = document.createElement('div');
            instructionsDiv.id = 'instructionsMessage';
            instructionsDiv.className = 'message info'; // Classe para estilização (do global.css)
            instructionsDiv.style.marginTop = '20px'; 
            // Encontre o painel de processamento para adicionar a mensagem dentro dele
            const processingPanel = document.querySelector('.processing-panel');
            if (processingPanel) {
                processingPanel.appendChild(instructionsDiv);
            } else {
                // Fallback caso .processing-panel não exista ou não seja encontrado
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

        // Garante que o intervalo de verificação não está ativo
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
        }
    }
});