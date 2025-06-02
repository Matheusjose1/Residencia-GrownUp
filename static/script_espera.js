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

    let statusCheckInterval; // Variável para armazenar o ID do setInterval

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
        images.forEach(image => {
            const listItem = document.createElement('li');
            listItem.innerHTML = `
                <span>${image.original_filename}</span>
                <span>Status: <strong>${image.status}</strong></span>
                <span>Progresso: ${image.progress}%</span>
                ${image.status === 'completed' && image.result // Verifica se existe um resultado e se tem ID
                    ? `<a href="/painel_resultados?id=${image.result.id}" target="_blank">(Ver Resultado)</a>`
                    : ''
                }
                ${image.status === 'failed'
                    ? `<span class="error-message">${image.message}</span>`
                    : ''
                }
            `;
            // Adiciona classe para estilização de status (opcional, requer CSS para .status-pending, .status-in_progress, etc.)
            listItem.classList.add(`status-${image.status}`);
            imageList.appendChild(listItem);
        });
    }

    // Função principal para verificar o status do lote e imagens
    async function checkBatchStatus() {
        const batchId = getBatchIdFromUrl();

        if (!batchId) {
            console.error("Erro: ID do lote não encontrado na URL. Redirecionando para upload.");
            updateBatchProgressUI(0, 'Erro: Nenhum ID de lote. Redirecionando...');
            clearInterval(statusCheckInterval);
            setTimeout(() => {
                window.location.href = '/painel_upload';
            }, 3000);
            return;
        }

        try {
            // 1. Obter status geral do lote
            // 'getBatchStatus' é uma função definida em 'api-integration.js'
            const batchStatus = await getBatchStatus(batchId); 
            console.log("Status do Lote recebido do backend:", batchStatus);

            if (!batchStatus) {
                throw new Error("Não foi possível obter o status do lote.");
            }

            updateBatchProgressUI(batchStatus.overall_progress, batchStatus.message);
            updateBatchSummary(
                batchStatus.total_images,
                batchStatus.processed_images,
                batchStatus.completed_images,
                batchStatus.failed_images
            );

            // 2. Obter status de cada imagem dentro do lote
            // 'getAllImagesForBatch' é uma função definida em 'api-integration.js'
            const imagesInBatch = await getAllImagesForBatch(batchId); 
            console.log("Imagens no Lote recebidas do backend:", imagesInBatch);

            if (imagesInBatch) {
                renderImageList(imagesInBatch);
            }

            // 3. Verificar o status final do lote para parar o monitoramento
            if (batchStatus.overall_status === 'completed' || batchStatus.overall_status === 'failed' || batchStatus.overall_status === 'partially_completed') {
                clearInterval(statusCheckInterval); // Para de verificar o status
                console.log("Processamento do lote concluído. Você pode verificar os resultados individuais.");
                // Pode adicionar um feedback visual final aqui
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
            updateBatchProgressUI(0, 'Erro de comunicação. Redirecionando...');
            clearInterval(statusCheckInterval);
            alert("Ocorreu um erro de comunicação com o servidor. Por favor, tente novamente.");
            setTimeout(() => {
                window.location.href = '/painel_upload';
            }, 3000);
        }
    }

    // --- Início do monitoramento quando a página é carregada ---
    const initialBatchId = getBatchIdFromUrl();
    if (initialBatchId) {
        console.log(`Página de espera de lote carregada. Iniciando monitoramento para o Lote ID: ${initialBatchId}`);
        
        // Chama a função uma vez imediatamente para obter o status inicial
        // Usar await aqui garante que o primeiro fetch seja feito antes de iniciar o intervalo
        await checkBatchStatus(); 
        
        // Configura o intervalo para verificar o status a cada 2 segundos (2000 ms)
        statusCheckInterval = setInterval(checkBatchStatus, 2000); 
    } else {
        updateBatchProgressUI(0, 'Nenhum ID de lote encontrado. Redirecionando para o upload...');
        setTimeout(() => {
            window.location.href = '/painel_upload';
        }, 3000);
    }
});