// script.js (REVISADO)
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageUploadInput = document.getElementById('imageUpload');
    const processingStatusDiv = document.getElementById('processingStatus');
    const resultsDisplayDiv = document.getElementById('resultsDisplay');
    const detectionList = document.getElementById('detectionList');
    const reportLinksDiv = document.getElementById('reportLinks');
    const progressContainer = document.querySelector('.progress-container');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    const api = new ProcessingAPI(); // Instancia a classe de integração da API

    // Esconde os elementos de status e resultados inicialmente
    processingStatusDiv.style.display = 'none';
    resultsDisplayDiv.style.display = 'none';
    progressContainer.style.display = 'none';

    // Event listener para o formulário de upload
    uploadForm.addEventListener('submit', async function(event) {
        event.preventDefault(); // Impede o comportamento padrão de envio do formulário

        const files = imageUploadInput.files;
        if (files.length === 0) {
            alert("Por favor, selecione pelo menos uma imagem para upload.");
            return;
        }

        // Exibir status de processamento e barra de progresso
        processingStatusDiv.style.display = 'block';
        resultsDisplayDiv.style.display = 'none'; // Esconde resultados antigos
        progressContainer.style.display = 'block';
        updateProgressUI(0, 'Iniciando upload...');
        uploadForm.style.display = 'none'; // Esconde o formulário durante o upload

        try {
            // Acompanhamento básico de upload (não é o progresso do YOLO)
            // Isso requer uma abordagem diferente (XMLHttpRequest com evento onprogress)
            // Para simplicidade, vamos apenas simular um "progresso de upload" rápido
            let uploadProgress = 0;
            const uploadInterval = setInterval(() => {
                uploadProgress = Math.min(uploadProgress + 10, 90);
                updateProgressUI(uploadProgress, 'Enviando imagens...');
                if (uploadProgress >= 90) clearInterval(uploadInterval);
            }, 100);

            const apiResponse = await api.processImages(files);

            clearInterval(uploadInterval); // Para a simulação de upload
            updateProgressUI(100, 'Processamento concluído!');

            // Exibir os resultados
            processingStatusDiv.style.display = 'none'; // Esconde a mensagem de processamento
            resultsDisplayDiv.style.display = 'block'; // Exibe a área de resultados
            uploadForm.style.display = 'block'; // Mostra o formulário novamente para um novo upload
            
            displayResults(apiResponse);

        } catch (error) {
            console.error("Erro no processamento:", error);
            processingStatusDiv.textContent = `Erro: ${error.message || "Falha ao processar imagens."}`;
            processingStatusDiv.style.color = 'red';
            uploadForm.style.display = 'block'; // Mostra o formulário novamente
            progressContainer.style.display = 'none'; // Esconde a barra de progresso em caso de erro
        }
    });

    /**
     * Função para atualizar a interface com o progresso atual
     * @param {number} value - Valor do progresso (0-100)
     * @param {string} [status] - Mensagem de status opcional
     */
    function updateProgressUI(value, status) {
        progressFill.style.width = `${value}%`;
        progressText.textContent = status;
    }

    /**
     * Função para exibir os resultados da API na UI
     * @param {object} data - Os dados retornados pela API
     */
    function displayResults(data) {
        detectionList.innerHTML = ''; // Limpa resultados anteriores
        reportLinksDiv.innerHTML = ''; // Limpa links anteriores

        if (data.results && data.results.length > 0) {
            data.results.forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <strong>Arquivo:</strong> ${item.filename}<br>
                    <strong>ID Imagem:</strong> ${item.image_id}<br>
                    <strong>Detecções:</strong> ${item.detections.join(', ')} (${item.confidences.map(c => c.toFixed(2)).join(', ')})<br>
                    ${item.processed_image_url ? `<a href="${item.processed_image_url}" target="_blank">Ver Imagem Processada</a>` : ''}
                `;
                detectionList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = "Nenhuma detecção encontrada ou erro no processamento.";
            detectionList.appendChild(li);
        }

        if (data.report_url) {
            const reportLink = document.createElement('a');
            reportLink.href = data.report_url;
            reportLink.target = '_blank';
            reportLink.textContent = 'Baixar Relatório XLSX';
            reportLink.classList.add('report-link'); // Adicionar uma classe para estilo
            reportLinksDiv.appendChild(reportLink);
        }
    }
});