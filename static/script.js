
document.addEventListener('DOMContentLoaded', function() {
    // Elementos DOM
    const progressFill = document.getElementById('progressFill');
    const progressText = document.querySelector('.progress-text');
    

    const processingAPI = new ProcessingAPI(); 

    let statusCheckInterval; // Variável para armazenar o ID do setInterval

    // Função para obter o parâmetro 'id' da URL (o processing_id)
    function getProcessingIdFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('id');
    }

    // Função para atualizar a interface da barra de progresso
    function updateProgressUI(progressValue, message = "Processando...") {
        // Garante que o progresso não ultrapasse 100%
        const displayProgress = Math.min(Math.max(0, progressValue), 100); 
        progressFill.style.width = `${displayProgress}%`;
        progressText.textContent = `${message} (${Math.round(displayProgress)}%)`;

        if (displayProgress === 100) {
            progressText.textContent = 'Análise concluída!';
            // Adiciona uma classe para estilos de conclusão, se você tiver no CSS
            progressFill.classList.add('finished'); 
        }
    }

    // Função principal para verificar o status do processamento e redirecionar
    async function checkProcessingStatusAndRedirect() {
        const processingId = getProcessingIdFromUrl();

        // Se não houver ID na URL, é um erro ou acesso direto à página de espera
        if (!processingId) {
            console.error("Erro: ID de processamento não encontrado na URL. Redirecionando para upload.");
            updateProgressUI(0, 'Erro: Nenhum ID de processamento. Redirecionando...');
            // Para o intervalo se ele estiver rodando (caso haja um erro de lógica)
            clearInterval(statusCheckInterval); 
            setTimeout(() => {
                window.location.href = '/painel_upload'; // Redireciona para a página de upload
            }, 3000);
            return; // Sai da função
        }

        try {
            // Chama a API para obter o status atual do processamento
            const statusData = await processingAPI.getProcessingStatus(processingId);
            console.log("Status recebido do backend:", statusData);

            // Atualiza a barra de progresso e o texto com os dados reais da API
            updateProgressUI(statusData.progress, statusData.message);

            // Verifica o status final do processamento
            if (statusData.status === 'completed') {
                clearInterval(statusCheckInterval); // Para de verificar o status
                console.log("Processamento concluído com sucesso. Redirecionando...");

                // Se o backend retornou um result_id, redireciona para a página de resultados
                if (statusData.result_id) {
                    window.location.href = `/painel_resultados?id=${statusData.result_id}`;
                } else {
                    // Caso o processamento tenha completado mas sem result_id (erro inesperado)
                    console.error("Processamento concluído, mas result_id é nulo ou inválido.");
                    updateProgressUI(100, 'Erro: Resultado não encontrado. Redirecionando...');
                    alert("O processamento falhou. Por favor, tente novamente.");
                    setTimeout(() => {
                        window.location.href = '/painel_upload';
                    }, 3000);
                }
            } else if (statusData.status === 'failed') {
                clearInterval(statusCheckInterval); // Para de verificar o status
                console.error("Processamento falhou no backend:", statusData.message);
                updateProgressUI(0, `Falha: ${statusData.message}`);
                alert("O processamento falhou. Por favor, tente novamente.");
                setTimeout(() => {
                    window.location.href = '/painel_upload'; // Volta para a página de upload
                }, 3000);
            }
            // Se o status for 'in_progress', a função simplesmente retornará e o setInterval a chamará novamente
        } catch (error) {
            // Erro na comunicação com a API (rede, servidor fora, etc.)
            console.error('Erro ao verificar status do processamento:', error);
            updateProgressUI(0, 'Erro de comunicação. Redirecionando...');
            clearInterval(statusCheckInterval); // Para de verificar o status
            alert("Ocorreu um erro de comunicação com o servidor. Por favor, tente novamente.");
            setTimeout(() => {
                window.location.href = '/painel_upload';
            }, 3000);
        }
    }

    // --- Início do monitoramento quando a página é carregada ---
    const initialProcessingId = getProcessingIdFromUrl();
    if (initialProcessingId) {
        console.log(`Página de espera carregada. Iniciando monitoramento para o ID: ${initialProcessingId}`);
        
        // Chama a função uma vez imediatamente para obter o status inicial
        checkProcessingStatusAndRedirect(); 
        
        // Configura o intervalo para verificar o status a cada 1 segundo (1000 ms)
        statusCheckInterval = setInterval(checkProcessingStatusAndRedirect, 1000);
    } else {
        // Caso a página seja acessada diretamente sem um ID de processamento
        updateProgressUI(0, 'Nenhum ID de processamento encontrado. Redirecionando para o upload...');
        setTimeout(() => {
            window.location.href = '/painel_upload';
        }, 3000);
    }
});