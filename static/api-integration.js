// app/static/api-integration.js (VERSÃO FINAL E CORRIGIDA, SEM MARCADORES DE FORMATAÇÃO)
/**
 * API Integration Module
 * Este arquivo contém funções específicas para integração com a API de backend
 * para upload e monitoramento de processamento de imagens.
 */

// Configuração da API
const API_CONFIG = {
    baseUrl: '/api',
    endpoints: {
        // O endpoint para upload inicia o processamento assíncrono
        uploadImage: '/upload-image',
        // Endpoint para verificar o status do processamento
        getProcessingStatus: '/processing-status/',
        // Endpoint para obter os resultados finais
        getProcessingResult: '/processing-result/',
        // Endpoint para download do Excel
        downloadExcel: '/download-excel/'
    }
};

/**
 * Classe para gerenciar a integração com a API de processamento de imagem
 */
class ProcessingAPI {
    constructor() {
        // Não precisamos de estado interno para esta classe, pois será usada por um script externo
    }

    /**
     * Envia um arquivo de imagem para a API para iniciar o processamento assíncrono.
     * @param {File} file - O objeto File da imagem selecionada.
     * @returns {Promise<Object>} - Promessa que resolve com o ID de processamento.
     */
    async uploadImage(file) {
        if (!file) {
            throw new Error("Nenhuma imagem selecionada para upload.");
        }

        const formData = new FormData();
        formData.append('file', file); // O backend espera 'file', não 'files'

        try {
            // CORRIGIDO: Removido formatação LaTeX/Markdown
            const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.uploadImage}`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                console.log("Upload bem-sucedido. ID de processamento:", data.processing_id);
                return data; // Deve retornar { "processing_id": "seu-uuid", "message": "..." }
            } else {
                const errorData = await response.json();
                console.error('Erro ao fazer upload da imagem:', errorData.detail || response.statusText);
                throw new Error(errorData.detail || "Erro desconhecido ao fazer upload.");
            }
        } catch (error) {
            console.error('Falha na comunicação com a API (upload):', error);
            throw error;
        }
    }

    /**
     * Consulta o status de um processamento específico.
     * @param {string} processingId - O ID do processamento a ser verificado.
     * @returns {Promise<Object>} - Promessa que resolve com o objeto de status.
     */
    async getProcessingStatus(processingId) {
        try {
            // CORRIGIDO: Removido formatação LaTeX/Markdown
            const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.getProcessingStatus}${processingId}`);
            if (response.ok) {
                const data = await response.json();
                return data; // Deve retornar { "progress": N, "status": "...", "message": "...", "result_id": N }
            } else if (response.status === 404) {
                throw new Error("ID de processamento não encontrado.");
            } else {
                const errorData = await response.json();
                console.error('Erro ao obter status:', errorData.detail || response.statusText);
                throw new Error(errorData.detail || "Erro desconhecido ao obter status.");
            }
        } catch (error) {
            console.error('Falha na comunicação com a API (status):', error);
            throw error;
        }
    }

    /**
     * Obtém os resultados finais de um processamento concluído.
     * @param {number} resultId - O ID do resultado do processamento (retornado pelo status).
     * @returns {Promise<Object>} - Promessa que resolve com os dados completos do resultado.
     */
    async getProcessingResult(resultId) {
        try {
            // CORRIGIDO: Removido formatação LaTeX/Markdown
            const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.getProcessingResult}${resultId}`);
            if (response.ok) {
                const data = await response.json();
                return data; // Deve retornar os detalhes completos do resultado
            } else if (response.status === 404) {
                throw new Error("Resultado não encontrado no banco de dados.");
            } else {
                const errorData = await response.json();
                console.error('Erro ao obter resultado:', errorData.detail || response.statusText);
                throw new Error(errorData.detail || "Erro desconhecido ao obter resultado.");
            }
        } catch (error) {
            console.error('Falha na comunicação com a API (resultado):', error);
            throw error;
        }
    }

    /**
     * Constrói a URL para download do arquivo Excel.
     * @param {string} filename - O nome do arquivo Excel.
     * @returns {string} - A URL completa para download.
     */
    getExcelDownloadUrl(filename) {
        // CORRIGIDO: Removido formatação LaTeX/Markdown
        return `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.downloadExcel}${filename}`;
    }
}

// Exporta a classe para uso no script principal (script.js)
window.ProcessingAPI = ProcessingAPI;