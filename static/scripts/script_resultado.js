// /static/scripts/script_resultado.js

// Classe ProcessingAPI: Gerencia as interações com a API FastAPI
// Esta classe é incorporada diretamente aqui para simplificar a organização.
// Em projetos maiores, ela poderia estar em um arquivo JS separado (ex: processing_api.js)
// e ser importada usando módulos (import { ProcessingAPI } from './processing_api.js';)
class ProcessingAPI {
    constructor() {
        // Base URL para a API FastAPI. Ajuste se sua API tiver um prefixo diferente (ex: '/api/v1')
        this.baseUrl = '/api'; 
    }

    /**
     * Obtém os resultados de processamento de uma imagem específica pelo ID.
     * @param {string} resultId - O ID do resultado de processamento.
     * @returns {Promise<Object>} - Os dados de resultado do processamento.
     * @throws {Error} Se a requisição falhar.
     */
    async getProcessingResult(resultId) {
        try {
            // Busca a lista de imagens do lote
            const response = await fetch(`/api/batch-images/${resultId}`);
            const images = await response.json(); // Isso agora é um ARRAY []

            console.log("Dados recebidos do lote:", images);

            if (images && images.length > 0) {
                // PEGUE A PRIMEIRA IMAGEM DA LISTA PARA EXIBIR
                const firstImage = images[0];

                // Preencha o cabeçalho usando firstImage em vez de data
                processingIdSpan.textContent = firstImage.id.substring(0, 8) || 'N/A';
                processingDateSpan.textContent = new Date().toLocaleDateString('pt-BR');

                // Renderiza o carrossel com todas as imagens
                carrosselImagesUl.innerHTML = ''; 
                images.forEach((img, index) => {
                    const li = document.createElement('li');
                    li.className = `carrossel-item ${index === 0 ? 'active' : ''}`;
                    
                    // Corrige o caminho da imagem (adicionando / se faltar)
                    let path = img.processed_image_path;
                    if (path && !path.startsWith('/')) path = '/' + path;
                    
                    li.innerHTML = `<img src="${path}" alt="Processado">`;
                    carrosselImagesUl.appendChild(li);
                });

                // Se a função renderDetectionsTable não existir, use:
                if (firstImage.detections) {
                    mainObjectTypeSpan.textContent = firstImage.detections.length > 0 
                        ? firstImage.detections[0].class_name 
                        : 'Nenhum';
                    // Chame sua função de renderizar a tabela aqui
                }

            } else {
                resultsContent.innerHTML = '<p>Nenhum resultado encontrado para este lote.</p>';
            }
        } catch (error) {
            console.error('Erro ao carregar resultados:', error);
            resultsContent.innerHTML = `<p class="error-message">Erro: ${error.message}</p>`;
        }
    }

    /**
     * Constrói a URL para download de um relatório Excel.
     * @param {string} fileName - O nome do arquivo Excel.
     * @returns {string} - A URL completa para download.
     */
    getExcelDownloadUrl(fileName) {
        return `${this.baseUrl}/download-excel/${fileName}`;
    }

    /**
     * Constrói a URL para uma imagem estática (processada, original, etc.).
     * @param {string} fileName - O nome do arquivo da imagem.
     * @returns {string} - A URL completa para a imagem.
     */
    getProcessedImageUrl(fileName) {
        return `/static/processed_images/${fileName}`; // Assumindo que imagens processadas são servidas por /static/processed_images/
    }

    // Adicione aqui outros métodos para interagir com a API, se precisar
}


// Event listener para garantir que o DOM esteja completamente carregado antes de executar o script
document.addEventListener('DOMContentLoaded', async function() {
    const urlParams = new URLSearchParams(window.location.search);
    const batchId = urlParams.get('batch_id');
    
    // Elementos da Interface
    const processingIdSpan = document.getElementById('processingId');
    const mainObjectTypeSpan = document.getElementById('mainObjectType');
    const processingDateSpan = document.getElementById('processingDate');
    const carrosselImagesUl = document.getElementById('carrosselImages');

    if (!batchId) {
        console.error("Batch ID não encontrado na URL");
        return;
    }

    try {
        const response = await fetch(`/api/batch-images/${batchId}`);
        const images = await response.json();

        console.log("Dados recebidos do lote:", images);

        if (images && images.length > 0) {
            const firstImage = images[0];

            // 1. Preencher Cabeçalho (Evita o erro de undefined)
            if (processingIdSpan) processingIdSpan.textContent = firstImage.id.substring(0, 8);
            if (processingDateSpan) processingDateSpan.textContent = new Date().toLocaleDateString('pt-BR');
            if (mainObjectTypeSpan) {
                mainObjectTypeSpan.textContent = (firstImage.detections && firstImage.detections.length > 0) 
                    ? firstImage.detections[0].class_name 
                    : "Nenhum";
            }

            // 2. Renderizar Imagens no Carrossel
            carrosselImagesUl.innerHTML = '';
            images.forEach((img, index) => {
            // 1. Em vez de usar o caminho do disco (que causa o erro de 'startsWith'), 
            // usamos o endpoint de download que o FastAPI já providencia.
            const imageUrl = `/api/download-processed-image/${img.id}`;

            console.log("A carregar imagem através da API:", imageUrl);

            const li = document.createElement('li');
            li.className = `carrossel-item ${index === 0 ? 'active' : ''}`;
            
            // 2. O src aponta para o endpoint, que retornará o ficheiro real
            li.innerHTML = `<img src="${imageUrl}" alt="Imagem Processada ${index + 1}" style="max-width:100%; display:block;">`;
            carrosselImagesUl.appendChild(li);
        });

        } else {
            console.warn("Nenhuma imagem encontrada para este lote.");
        }
    } catch (error) {
        console.error("Erro ao renderizar tela de resultados:", error);
    }
});

async function loadResults() {
    const urlParams = new URLSearchParams(window.location.search);
    const batchId = urlParams.get('batch_id');

    if (!batchId) return;

    try {
        const response = await fetch(`/api/batch-images/${batchId}`);
        const images = await response.json();

        if (images && images.length > 0) {
            const firstImageData = images[0];
            
            // Preenchimento de IDs e Datas
            document.getElementById('processingId').textContent = firstImageData.id.substring(0, 8);
            document.getElementById('processingDate').textContent = new Date().toLocaleDateString('pt-BR');
            
            if (firstImageData.detections && firstImageData.detections.length > 0) {
                document.getElementById('mainObjectType').textContent = firstImageData.detections[0].class_name;
            }

            const carrosselUl = document.getElementById('carrosselImages');
            carrosselUl.innerHTML = ''; 

            images.forEach((img, index) => {
                const li = document.createElement('li');
                li.className = `carrossel-item ${index === 0 ? 'active' : ''}`;
                
                // SOLUÇÃO: Usar o endpoint da API em vez do caminho do disco
                const imageUrl = `/api/download-processed-image/${img.id}`;
                console.log("Carregando imagem via API:", imageUrl);

                li.innerHTML = `<img src="${imageUrl}" alt="Processado" style="max-width:100%; display:block;">`;
                carrosselUl.appendChild(li);
            });

            // Removida a chamada ao updateCarousel() para evitar o ReferenceError
            console.log("Imagens carregadas com sucesso.");
            
        } else {
            document.getElementById('resultsContent').innerHTML = "<p>Nenhum dado encontrado.</p>";
        }
    } catch (error) {
        console.error("Erro ao renderizar tela de resultados:", error);
    }
}

loadResults();