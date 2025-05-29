// api-integration.js (REVISADO)
/**
 * API Integration Module
 * Este arquivo contém funções específicas para integração com a API de backend
 * para processamento síncrono de imagens.
 */

// Configuração da API
const API_CONFIG = {
    baseUrl: '/api',
    endpoints: {
        processImages: '/process_images/' // Seu endpoint atual
    }
};

/**
 * Classe para gerenciar a integração com a API de processamento de imagem
 */
class ProcessingAPI {
    constructor() {
        // Não precisamos de state para monitoramento assíncrono aqui
    }

    /**
     * Envia as imagens para a API de processamento.
     * Assume um processamento síncrono e retorna os resultados diretamente.
     * @param {FileList} files - Objeto FileList de um input type="file"
     * @returns {Promise<Object>} - Promessa que resolve com os dados da resposta da API
     */
    async processImages(files) {
        if (!files || files.length === 0) {
            throw new Error("Nenhuma imagem selecionada para upload.");
        }

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        try {
            const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.processImages}`, {
                method: 'POST',
                body: formData // FormData para envio de arquivos
            });

            if (response.ok) {
                const data = await response.json();
                console.log("Processamento concluído com sucesso:", data);
                return data; // Retorna os resultados da API
            } else {
                const errorData = await response.json();
                console.error('Erro ao processar imagens:', errorData.detail || response.statusText);
                throw new Error(errorData.detail || "Erro desconhecido ao processar imagens.");
            }
        } catch (error) {
            console.error('Falha na comunicação com a API:', error);
            throw error; // Propaga o erro para o chamador
        }
    }

    /**
     * Redireciona para uma página de resultados, se necessário.
     * Por enquanto, vamos retornar os dados diretamente na mesma página.
     * @param {Object} results - Os resultados retornados da API
     */
    navigateToResults(results) {
        // Implementar lógica para exibir os resultados ou redirecionar
        // Por exemplo, você pode passar os resultados como um objeto JSON stringificado
        // para a próxima página via localStorage ou queryString (se pequenos).
        // Por agora, apenas logs.
        console.log("Resultados para exibição:", results);
        // window.location.href = `/painel-resultado?data=${encodeURIComponent(JSON.stringify(results))}`;
        // Esta função será adaptada no script.js
    }
}

// Exporta a classe para uso no script principal
window.ProcessingAPI = ProcessingAPI;