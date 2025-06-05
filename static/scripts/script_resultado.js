// script_resultado.js

class ProcessingAPI {
    /**
     * @param {string} baseUrl - A URL base para a API (ex: 'http://localhost:8000').
     */
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    /**
     * Obtém os resultados completos de um processamento específico.
     * @param {string} resultId - O ID do resultado do processamento.
     * @returns {Promise<Object>} - Os dados do resultado do processamento.
     * @throws {Error} - Se a requisição falhar.
     */
    async getProcessingResult(resultId) {
        const response = await fetch(`${this.baseUrl}/api/processing-result/${resultId}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Erro ao buscar dados de resultado.');
        }
        return await response.json();
    }

    /**
     * Obtém as URLs das imagens originais associadas a um processamento.
     * Esta função depende da existência da rota /api/original-images/{result_id} no FastAPI.
     * @param {string} resultId - O ID do resultado do processamento.
     * @returns {Promise<string[]>} - Um array de URLs para as imagens originais.
     * @throws {Error} - Se a requisição falhar.
     */
    async getOriginalImages(resultId) {
        const response = await fetch(`${this.baseUrl}/api/original-images/${resultId}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Erro ao buscar imagens originais.');
        }
        return await response.json(); // Espera-se um array de URLs de imagem
    }
}

document.addEventListener('DOMContentLoaded', async function() {
    // Obter o ID do resultado da URL
    const urlParams = new URLSearchParams(window.location.search);
    const resultId = urlParams.get('id');

    // Referências aos elementos HTML
    const originalImagesCarrossel = document.getElementById('originalImagesCarrossel');
    const resultIdDisplay = document.getElementById('resultIdDisplay');
    const resultTypeDisplay = document.getElementById('resultTypeDisplay');
    const resultDateDisplay = document.getElementById('resultDateDisplay');
    const modelAccuracyDisplay = document.getElementById('modelAccuracyDisplay');
    const downloadZipButton = document.getElementById('downloadZipButton');

    // Instanciar a classe de API
    const processingAPI = new ProcessingAPI(); 

    // Verificar se o ID do resultado foi fornecido
    if (!resultId) {
        if (originalImagesCarrossel) {
            originalImagesCarrossel.innerHTML = '<p class="error-message">ID do resultado não fornecido na URL. Por favor, volte para a página de upload e envie uma imagem.</p>';
        }
        // Desabilitar botões se não houver ID
        if (downloadZipButton) downloadZipButton.style.display = 'none';
        return;
    }

    try {
        // 1. Obter os dados de resultado do processamento (imagem processada, dados de detecção, etc.)
        const resultData = await processingAPI.getProcessingResult(resultId);
        console.log("Dados de resultados recebidos:", resultData);

        // Atualizar os detalhes do diagnóstico no HTML
        if (resultIdDisplay) resultIdDisplay.textContent = `#${resultData.id || resultId}`; // Usa o ID da URL se não vier da API
        if (resultTypeDisplay) resultTypeDisplay.textContent = resultData.type || 'Não disponível';
        if (resultDateDisplay) resultDateDisplay.textContent = resultData.date || 'Não disponível';
        if (modelAccuracyDisplay) modelAccuracyDisplay.textContent = `${(resultData.model_accuracy * 100).toFixed(2)}%` || 'Não disponível';

        // 2. Obter as URLs das imagens originais para o carrossel
        const originalImageUrls = await processingAPI.getOriginalImages(resultId);
        console.log("URLs de imagens originais recebidas:", originalImageUrls);

        if (originalImageUrls && originalImageUrls.length > 0) {
            // Limpa o conteúdo existente do carrossel
            if (originalImagesCarrossel) originalImagesCarrossel.innerHTML = ''; 
            
            // Limita a exibição às 10 primeiras imagens de entrada
            const imagesToShow = originalImageUrls.slice(0, 10); 
            
            imagesToShow.forEach((url, index) => {
                const listItem = document.createElement('li');
                listItem.classList.add('carrossel-item');
                // Ativa o primeiro item do carrossel por padrão
                if (index === 0) { 
                    listItem.classList.add('active'); 
                }
                listItem.dataset.index = index; // Adiciona o índice para controle do carrossel
                const img = document.createElement('img');
                img.src = url;
                img.alt = `Imagem Original ${index + 1}`;
                listItem.appendChild(img);
                if (originalImagesCarrossel) originalImagesCarrossel.appendChild(listItem);
            });

            // Inicializar o carrossel após as imagens serem carregadas
            // Isso assume que a lógica de carrossel está em uma função global ou script separado
            if (typeof initializeCarrossel === 'function') { 
                initializeCarrossel(); 
            } else {
                console.warn("Função 'initializeCarrossel' não encontrada. O carrossel pode não funcionar corretamente.");
            }
        } else {
            if (originalImagesCarrossel) {
                originalImagesCarrossel.innerHTML = '<p>Nenhuma imagem original encontrada para este resultado.</p>';
            }
        }

        // Configurar o botão de download ZIP
        if (downloadZipButton) {
            // Verifica se a URL para download do ZIP está disponível nos dados de resultado
            if (resultData.zip_download_url) {
                downloadZipButton.onclick = () => {
                    window.location.href = resultData.zip_download_url;
                };
                downloadZipButton.style.display = 'inline-block'; // Garante que o botão seja visível
            } else {
                downloadZipButton.style.display = 'none'; // Esconde o botão se não houver URL
            }
        }

    } catch (error) {
        console.error('Erro ao carregar resultados ou imagens originais:', error);
        if (originalImagesCarrossel) {
            originalImagesCarrossel.innerHTML = `<p class="error-message">Erro ao carregar os resultados: ${error.message}. Por favor, tente novamente.</p>`;
        }
        if (downloadZipButton) downloadZipButton.style.display = 'none';
    }
});

/**
 * Função para inicializar a lógica de navegação do carrossel.
 * Esta função idealmente estaria em um arquivo 'carrossel.js' separado.
 */
function initializeCarrossel() {
    const carrosselContainer = document.querySelector('.carrossel-container');
    if (!carrosselContainer) {
        console.warn("Carrossel container não encontrado. A lógica do carrossel não será inicializada.");
        return;
    }
    const carrossel = carrosselContainer.querySelector('.carrossel');
    const prevBtn = carrosselContainer.querySelector('.prev-btn');
    const nextBtn = carrosselContainer.querySelector('.next-btn');

    let currentIndex = 0;

    function updateCarrossel() {
        const items = carrossel.querySelectorAll('.carrossel-item');
        if (items.length === 0) return;

        // Calcula a largura total dos itens para centralizar ou mover corretamente
        // Assumindo que os itens têm largura fixa ou usam flexbox/grid com espaçamento
        const itemWidth = items[0].offsetWidth + (parseFloat(getComputedStyle(items[0]).marginRight) || 0); // Considera margem

        // Move o carrossel com base no índice atual
        carrossel.style.transform = `translateX(${-currentIndex * itemWidth}px)`;

        // Atualiza a classe 'active' para o item visível
        items.forEach((item, idx) => {
            if (idx === currentIndex) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    // Adiciona event listeners aos botões de navegação
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            const items = carrossel.querySelectorAll('.carrossel-item');
            currentIndex = (currentIndex > 0) ? currentIndex - 1 : items.length - 1; // Loop de volta para o final
            updateCarrossel();
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            const items = carrossel.querySelectorAll('.carrossel-item');
            currentIndex = (currentIndex < items.length - 1) ? currentIndex + 1 : 0; // Loop de volta para o início
            updateCarrossel();
        });
    }

    // Inicializa a posição do carrossel
    updateCarrossel();
}