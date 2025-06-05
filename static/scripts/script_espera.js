// static/script_espera.js

document.addEventListener('DOMContentLoaded', async function() {
    // ... (Seus elementos DOM e variáveis existentes) ...

    let statusCheckInterval; // Variável para armazenar o ID do setInterval
    let currentBatchId = new URLSearchParams(window.location.search).get('batch_id');
    let hasRedirected = false; // Flag para evitar múltiplos redirecionamentos

    // Função para mostrar mensagens de erro/sucesso (se você tiver uma)
    function displayMessage(message, type = 'info') {
        const messageDiv = document.getElementById('statusMessage') || document.createElement('div');
        messageDiv.id = 'statusMessage';
        messageDiv.className = `message ${type}`;
        messageDiv.innerHTML = message;
        if (!messageDiv.parentNode) {
            document.querySelector('.processing-panel').prepend(messageDiv);
        }
        messageDiv.style.display = 'block';
    }

    // Função principal para verificar o status do lote
    async function updateBatchStatus() {
        if (!currentBatchId || hasRedirected) {
            // Se não tem batch_id ou já redirecionou, para a verificação
            showNoBatchInstructions();
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
            }
            return;
        }

        try {
            const response = await fetch(`/api/batch-status/${currentBatchId}`);
            if (!response.ok) {
                throw new Error(`Erro HTTP: ${response.status}`);
            }
            const batchData = await response.json();
            console.log('Dados do Lote recebidos:', batchData);

            // Atualiza os elementos visuais
            totalImagesSpan.textContent = batchData.total_images;
            processedImagesSpan.textContent = batchData.processed_images;
            completedImagesSpan.textContent = batchData.completed_images;
            failedImagesSpan.textContent = batchData.failed_images;

            const progress = (batchData.processed_images / batchData.total_images) * 100;
            batchProgressFill.style.width = `${progress}%`;
            batchProgressText.textContent = `${Math.round(progress)}%`;

            // Mostra os containers relevantes
            loadingSpinner.style.display = 'none';
            progressBarContainer.style.display = 'block';
            processingMessage.style.display = 'block';
            batchSummaryContainer.style.display = 'block';
            imageListContainer.style.display = 'block';
            infoBox.style.display = 'block';

            // Atualiza a lista de imagens
            imageList.innerHTML = ''; // Limpa a lista existente
            if (batchData.images && Array.isArray(batchData.images)) {
                batchData.images.forEach(image => {
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <span>${image.original_filename}</span>
                        <span class="status ${image.status}">${image.status}</span>
                        ${image.status === 'completed' && image.processed_image_url ?
                            `<a href="${image.processed_image_url}" target="_blank" class="view-link" title="Ver Imagem Processada">
                                <i class="fas fa-eye"></i>
                             </a>` : ''
                        }
                        ${image.status === 'completed' && image.excel_report_url ?
                            `<a href="${image.excel_report_url}" target="_blank" class="download-link" title="Baixar Relatório Excel">
                                <i class="fas fa-file-excel"></i>
                             </a>` : ''
                        }
                    `;
                    imageList.appendChild(li);
                });
            } else {
                console.warn("batchData.images não é um array ou está vazio:", batchData.images);
            }

            // Lógica de REDIRECIONAMENTO para a página de resultados
            if (batchData.status === 'completed_batch' && !hasRedirected) { // O status 'completed_batch' indica que o lote inteiro foi concluído
                console.log('Lote concluído. Redirecionando para a página de resultados...');
                hasRedirected = true; // Define a flag para true
                clearInterval(statusCheckInterval); // Para de verificar
                // Adiciona um pequeno atraso antes de redirecionar para o usuário ver o 100%
                setTimeout(() => {
                    window.location.href = `/painel_resultados?batch_id=${currentBatchId}`; // Redireciona com o ID do lote
                }, 1500); // 1.5 segundos de atraso
            } else if (batchData.status === 'failed_batch' && !hasRedirected) {
                console.error('Lote falhou. Exibindo mensagem de erro.');
                hasRedirected = true;
                clearInterval(statusCheckInterval);
                displayMessage('O processamento do lote falhou. Por favor, tente novamente.', 'error');
                // Opcional: redirecionar para uma página de erro ou upload
            }

        } catch (error) {
            console.error('Erro ao buscar status do lote:', error);
            // Se o lote não for encontrado (ex: 404), mostre as instruções de não-lote
            if (error.message.includes('404')) {
                showNoBatchInstructions();
            }
            clearInterval(statusCheckInterval); // Para a verificação em caso de erro
        }
    }

    // Função para mostrar instruções quando não há batch_id
    function showNoBatchInstructions() {
        const mainElement = document.querySelector('main');
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

        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
        }
    }

    // Inicialização: Verifica se há um batch_id na URL
    if (currentBatchId) {
        console.log('Batch ID encontrado na URL:', currentBatchId);
        // Esconde as instruções iniciais e mostra o spinner/progresso
        document.getElementById('instructionsMessage').style.display = 'none';
        loadingSpinner.style.display = 'block';
        processingMessage.style.display = 'block'; // Mostra a mensagem "Nossa inteligência artificial..."
        progressBarContainer.style.display = 'block'; // Mostra a barra de progresso (vazia no início)
        batchSummaryContainer.style.display = 'block'; // Mostra o resumo
        imageListContainer.style.display = 'block'; // Mostra a lista
        infoBox.style.display = 'block'; // Mostra a caixa de info

        // Inicia a primeira verificação imediatamente
        updateBatchStatus();
        // Configura a verificação a cada 2 segundos
        statusCheckInterval = setInterval(updateBatchStatus, 2000);
    } else {
        console.log('Nenhum Batch ID encontrado na URL. Mostrando instruções.');
        showNoBatchInstructions();
    }
});