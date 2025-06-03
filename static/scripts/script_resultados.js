document.addEventListener('DOMContentLoaded', async function() {
    const urlParams = new URLSearchParams(window.location.search);
    const resultId = urlParams.get('id');
    const resultsContent = document.getElementById('resultsContent');
    const processingAPI = new ProcessingAPI(); // Instancia a classe

    if (!resultId) {
        resultsContent.innerHTML = '<p class="error-message">ID do resultado não fornecido na URL.</p>';
        return;
    }

    try {
        // Usar a função da classe ProcessingAPI para obter os resultados
        const data = await processingAPI.getProcessingResult(resultId);
        console.log("Dados de resultados recebidos:", data);

        let htmlContent = `
            <h3>Arquivo Original: ${data.original_filename}</h3>
            <div class="results-content-wrapper">
        `;

        // Adiciona a imagem processada, se disponível
        if (data.processed_image_url) {
            htmlContent += `
                <h4>Imagem Processada:</h4>
                <img src="${data.processed_image_url}" alt="Imagem Processada" class="result-image">
            `;
        } else {
            htmlContent += `
                <p>Imagem processada não disponível.</p>
            `;
        }

        // Adiciona a tabela de detecções, se houver dados
        if (data.detection_data && data.detection_data.length > 0) {
            htmlContent += `
                <h4>Objetos Detectados:</h4>
                <table class="detection-table">
                    <thead>
                        <tr>
                            <th>Classe</th>
                            <th>Confiança</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            data.detection_data.forEach(detection => {
                htmlContent += `
                        <tr>
                            <td>${detection.class_name}</td>
                            <td>${(detection.confidence * 100).toFixed(2)}%</td>
                        </tr>
                `;
            });
            htmlContent += `
                    </tbody>
                </table>
            `;
        } else {
            htmlContent += `
                <p>Nenhum objeto detectado.</p>
            `;
        }

        // Adiciona o botão de download do Excel, se disponível
        if (data.excel_report_url) {
            // O backend já deve retornar a URL completa para download,
            // então podemos usá-la diretamente.
            // Caso o backend retorne apenas o nome do arquivo, usar:
            // const excelDownloadUrl = processingAPI.getExcelDownloadUrl(data.excel_report_url.split('/').pop());
            // Mas assumindo que data.excel_report_url já é a URL completa para download:
            htmlContent += `
                <a href="${data.excel_report_url}" class="download-button" download>Baixar Relatório Excel</a>
            `;
        } else {
            htmlContent += `
                <p>Relatório Excel não disponível.</p>
            `;
        }

        htmlContent += `</div>`; // Fecha results-content-wrapper
        resultsContent.innerHTML = htmlContent;

    } catch (error) {
        console.error('Erro ao carregar resultados:', error);
        resultsContent.innerHTML = `<p class="error-message">Erro ao carregar os resultados: ${error.message}. Por favor, tente novamente.</p>`;
    }
});