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
            const response = await fetch(`${this.baseUrl}/results/${resultId}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Erro HTTP: ${response.status}`);
            }
            return response.json();
        } catch (error) {
            console.error("Erro na chamada da API getProcessingResult:", error);
            throw error; // Re-lança o erro para ser tratado pelo chamador
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
    // Obtém o ID do resultado da URL (ex: painel_resultados.html?id=SEU_ID_AQUI)
    const urlParams = new URLSearchParams(window.location.search);
    const resultId = urlParams.get('id');

    // Seleciona os elementos HTML onde os dados serão exibidos
    const resultsContent = document.getElementById('resultsContent');
    const processingIdSpan = document.getElementById('processingId');
    const mainObjectTypeSpan = document.getElementById('mainObjectType');
    const processingDateSpan = document.getElementById('processingDate');
    const carrosselImagesUl = document.getElementById('carrosselImages');

    // Instancia a classe da API
    const processingAPI = new ProcessingAPI();

    // Exibe mensagem de erro se o ID do resultado não for fornecido
    if (!resultId) {
        resultsContent.innerHTML = '<p class="error-message">ID do resultado de processamento não fornecido na URL.</p>';
        return; // Sai da função, pois não há ID para buscar
    }

    try {
        // Faz a chamada à API para obter os resultados do processamento
        const data = await processingAPI.getProcessingResult(resultId);
        console.log("Dados de resultados recebidos:", data); // Para depuração

        // --- Preencher Dados Principais (ID, Tipo Principal, Data) ---
        processingIdSpan.textContent = data.processing_id || 'N/A';
        
        // Lógica para determinar o "Tipo Principal":
        // Por simplicidade, pega a classe do primeiro objeto detectado.
        // Você pode ajustar isso para o objeto com maior confiança, ou o mais frequente.
        if (data.detection_data && data.detection_data.length > 0) {
            mainObjectTypeSpan.textContent = data.detection_data[0].class_name || 'N/A';
        } else {
            mainObjectTypeSpan.textContent = 'Nenhum Objeto Detectado';
        }

        // Para a data, o backend deve retornar um campo 'processing_date' (ou similar)
        // Se o backend não retornar, pode-se usar a data atual como fallback.
        if (data.processing_date) {
            const date = new Date(data.processing_date);
            processingDateSpan.textContent = date.toLocaleDateString('pt-BR'); // Formato DD/MM/YYYY
        } else {
            processingDateSpan.textContent = new Date().toLocaleDateString('pt-BR'); // Exemplo: data atual do cliente
        }
        
        // --- Carrossel de Imagens ---
        // Limpa o carrossel antes de adicionar novas imagens
        carrosselImagesUl.innerHTML = ''; 
        if (data.processed_image_url) {
            // Cria um item para a imagem processada.
            // Se houver mais imagens (ex: original_image_url), crie mais <li> aqui
            const processedImageItem = document.createElement('li');
            processedImageItem.classList.add('carrossel-item', 'active'); // 'active' para o primeiro item
            processedImageItem.dataset.index = 0; // Para a lógica do carrossel
            processedImageItem.innerHTML = `<img src="${data.processed_image_url}" alt="Imagem Processada">`;
            carrosselImagesUl.appendChild(processedImageItem);

            // Se você quiser adicionar a imagem original também (assumindo que 'original_image_url' seja retornado pelo backend)
            // if (data.original_image_url) {
            //     const originalImageItem = document.createElement('li');
            //     originalImageItem.classList.add('carrossel-item');
            //     originalImageItem.dataset.index = 1;
            //     originalImageItem.innerHTML = `<img src="${data.original_image_url}" alt="Imagem Original">`;
            //     carrosselImagesUl.appendChild(originalImageItem);
            // }

        } else {
            const noImageItem = document.createElement('li');
            noImageItem.classList.add('carrossel-item');
            noImageItem.innerHTML = '<p>Nenhuma imagem processada disponível.</p>';
            carrosselImagesUl.appendChild(noImageItem);
        }


        // --- Conteúdo da Seção de Resultados (Tabela de Detecções, Botão Excel) ---
        let resultsHtml = `
            <h3>Arquivo Original: ${data.original_filename || 'N/A'}</h3>
            <div class="results-content-wrapper">
        `;

        // Tabela de objetos detectados
        if (data.detection_data && data.detection_data.length > 0) {
            resultsHtml += `
                <h4>Objetos Detectados:</h4>
                <table class="detections-table">
                    <thead>
                        <tr>
                            <th>Objeto</th>
                            <th>Confiança</th>
                            <th>Coordenadas (X1, Y1, X2, Y2)</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            data.detection_data.forEach(detection => {
                // Formata as coordenadas da caixa para exibição
                const boxCoords = detection.box_coords ? `[${detection.box_coords.map(c => c.toFixed(0)).join(', ')}]` : 'N/A';
                resultsHtml += `
                        <tr>
                            <td>${detection.class_name || 'N/A'}</td>
                            <td>${(detection.confidence * 100).toFixed(2)}%</td>
                            <td>${boxCoords}</td>
                        </tr>
                `;
            });
            resultsHtml += `
                    </tbody>
                </table>
            `;
        } else {
            resultsHtml += `
                <p>Nenhum objeto detectado nesta imagem.</p>
            `;
        }

        // Botão de download do relatório Excel
        if (data.excel_report_url) {
            resultsHtml += `
                <a href="${data.excel_report_url}" class="download-button" download>Baixar Relatório Excel</a>
            `;
        } else {
            resultsHtml += `
                <p>Relatório Excel não disponível para este processamento.</p>
            `;
        }

        resultsHtml += `</div>`; // Fecha results-content-wrapper
        resultsContent.innerHTML = resultsHtml;

    } catch (error) {
        // Captura e exibe erros de API ou de processamento
        console.error('Erro ao carregar resultados:', error);
        resultsContent.innerHTML = `<p class="error-message">Erro ao carregar os resultados: ${error.message}. Verifique o ID e a conexão com a API.</p>`;
    }

    // --- Lógica do Carrossel (navegação) ---
    // Esta parte deve ser executada APÓS o carrosselImagesUl ter sido preenchido
    // Assegura que o carrossel funciona mesmo com um único item.
    const carrosselContainer = document.getElementById('carrosselContainer');
    const prevBtn = carrosselContainer.querySelector('.prev-btn');
    const nextBtn = carrosselContainer.querySelector('.next-btn');
    let currentIndex = 0;

    function updateCarousel() {
        const items = carrosselImagesUl.querySelectorAll('.carrossel-item');
        if (items.length === 0) {
            prevBtn.style.display = 'none'; // Esconde botões se não há imagens
            nextBtn.style.display = 'none';
            return;
        }

        // Garante que currentIndex esteja dentro dos limites
        currentIndex = (currentIndex + items.length) % items.length;

        items.forEach((item, index) => {
            if (index === currentIndex) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // Mostra/Esconde botões de navegação se houver mais de uma imagem
        if (items.length > 1) {
            prevBtn.style.display = 'block';
            nextBtn.style.display = 'block';
        } else {
            prevBtn.style.display = 'none';
            nextBtn.style.display = 'none';
        }
    }

    prevBtn.addEventListener('click', () => {
        currentIndex--;
        updateCarousel();
    });

    nextBtn.addEventListener('click', () => {
        currentIndex++;
        updateCarousel();
    });

    // Chama updateCarousel inicialmente para configurar o estado inicial (primeiro item visível)
    // E também para esconder os botões se houver apenas uma imagem.
    updateCarousel(); 
});