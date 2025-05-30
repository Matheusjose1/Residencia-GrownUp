/**
 * API Integration Module
 * Este arquivo contém funções específicas para integração com a API de backend
 * Utilize este arquivo para implementar as chamadas reais à API de processamento de imagem
 */

// Configuração da API
const API_CONFIG = {
    baseUrl: '/api', // Substitua pelo URL base da sua API
    endpoints: {
        upload: '/upload-image', // <--- NOVO ENDPOINT DE UPLOAD
        status: '/processing-status',
        result: '/processing-result'
    },
    requestInterval: 2000 // Intervalo em ms para verificar o status do processamento
};

/**
 * Classe para gerenciar a integração com a API de processamento de imagem
 */
class ProcessingAPI {
    constructor() {
        this.processingId = null;
        this.statusCheckInterval = null;
        this.onProgressUpdate = null;
        this.onProcessingComplete = null;
    }

    /**
     * Envia um arquivo para a API para iniciar o processamento
     * @param {File} file - O objeto File a ser enviado
     * @returns {Promise<string>} - Promise que resolve com o ID de processamento
     */
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.upload}`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                this.processingId = data.processing_id;
                return this.processingId;
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Erro desconhecido no upload.');
            }
        } catch (error) {
            console.error('Falha na comunicação com a API (upload):', error);
            throw error;
        }
    }

    /**
     * Inicia o monitoramento do processamento de uma imagem
     * @param {string} processingId - ID do processamento retornado pela API de upload
     * @param {Function} progressCallback - Função chamada quando o progresso é atualizado
     * @param {Function} completeCallback - Função chamada quando o processamento é concluído
     */
    startMonitoring(processingId, progressCallback, completeCallback) {
        this.processingId = processingId;
        this.onProgressUpdate = progressCallback;
        this.onProcessingComplete = completeCallback;

        // Inicia a verificação periódica do status
        this.checkStatus(); // Chama imediatamente
        this.statusCheckInterval = setInterval(() => this.checkStatus(), API_CONFIG.requestInterval);
    }

    /**
     * Verifica o status do processamento na API
     * @private
     */
    async checkStatus() {
        if (!this.processingId) {
            console.warn('Processing ID não definido para verificação de status.');
            return;
        }

        try {
            const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.status}/${this.processingId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const data = await response.json();
                
                // Atualiza o progresso na UI
                if (this.onProgressUpdate) {
                    this.onProgressUpdate(data.progress, data.message); // Use 'message' em vez de 'statusMessage'
                }

                // Verifica se o processamento foi concluído
                if (data.progress >= 100 && data.status === 'completed') { // Adicionado '&& data.status === 'completed''
                    this.stopMonitoring();
                    
                    if (this.onProcessingComplete) {
                        this.onProcessingComplete(data.result_id); // Use 'result_id' que vem do backend
                    }
                } else if (data.status === 'failed') { // Adicionado tratamento para falha
                    this.stopMonitoring();
                    if (this.onProcessingComplete) {
                        this.onProcessingComplete(null, data.message); // Passa null para result_id e a mensagem de erro
                    }
                }
            } else {
                console.error('Erro ao verificar status do processamento:', await response.text());
                this.stopMonitoring(); // Parar de monitorar em caso de erro na API
                if (this.onProgressUpdate) {
                     this.onProgressUpdate(0, 'Erro: Não foi possível obter o status.');
                }
            }
        } catch (error) {
            console.error('Falha na comunicação com a API:', error);
            this.stopMonitoring(); // Parar de monitorar em caso de erro de rede
            if (this.onProgressUpdate) {
                 this.onProgressUpdate(0, 'Erro de rede: Verifique sua conexão.');
            }
        }
    }

    /**
     * Para o monitoramento do processamento
     */
    stopMonitoring() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }
    }

    /**
     * Redireciona para a página de resultados
     * @param {string} resultId - ID do resultado do processamento
     */
    navigateToResults(resultId) {
        if (resultId) {
            window.location.href = `/painel_resultados?id=${resultId}`; // Ajustado para /results
        } else {
            console.error("Não foi possível redirecionar: resultId é nulo ou inválido.");
            // Opcional: redirecionar para uma página de erro ou mostrar uma mensagem
            alert("O processamento falhou. Por favor, tente novamente.");
            window.location.href = '/painel_upload'; // Volta para a página de upload
        }
    }
}

// Exporta a classe para uso no script principal
window.ProcessingAPI = ProcessingAPI;