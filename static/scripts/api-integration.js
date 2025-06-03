
async function uploadImagesBatch(files) {
    const formData = new FormData();
    files.forEach(file => {
        formData.append('files', file); // 'images' deve corresponder ao nome do parâmetro no seu endpoint FastAPI (List[UploadFile] = File(...))
    });

    try {
        const response = await fetch('/api/upload-image', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Erro no servidor: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        return result.batch_id; // Retorna o ID do lote
    } catch (error) {
        console.error('Erro ao enviar imagens em lote:', error);
        alert(`Erro ao processar as imagens: ${error.message}`);
        return null;
    }
}

// Função para obter o status de um lote específico
async function getBatchStatus(batchId) {
    try {
        const response = await fetch(`/api/batch-status/${batchId}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Erro no servidor: ${response.status} - ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Erro ao obter status do lote:', error);
        return null;
    }
}

// Função para obter o status de processamento de uma imagem individual (mantida para compatibilidade, mas menos usada com o fluxo de lote)
async function getImageProcessingStatus(processingId) {
    try {
        const response = await fetch(`/api/image-status/${processingId}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Erro no servidor: ${response.status} - ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Erro ao obter status da imagem:', error);
        return null;
    }
}

// Função para obter os detalhes de um resultado de processamento
async function getResultDetails(resultId) {
    try {
        const response = await fetch(`/api/results/${resultId}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Erro no servidor: ${response.status} - ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Erro ao obter detalhes do resultado:', error);
        return null;
    }
}

// Função para obter todos os resultados de imagens para um lote específico
async function getAllImagesForBatch(batchId) {
    try {
        const response = await fetch(`/api/batch-images/${batchId}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Erro no servidor: ${response.status} - ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Erro ao obter imagens do lote:', error);
        return null;
    }
}