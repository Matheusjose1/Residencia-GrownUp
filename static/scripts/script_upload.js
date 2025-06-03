// static/script_upload.js

document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('imageInput');
    const selectImagesButton = document.getElementById('selectImagesButton');
    const processImagesButton = document.getElementById('processImagesButton');
    const selectedImagesCount = document.getElementById('selectedImagesCount');
    const messageDiv = document.getElementById('message');

    let selectedFiles = []; // Armazenará os arquivos selecionados

    // Abre o seletor de arquivos quando o botão "Selecionar Imagens" é clicado
    selectImagesButton.addEventListener('click', () => {
        imageInput.click();
    });

    // Lida com a mudança no input de arquivo (quando arquivos são selecionados)
    imageInput.addEventListener('change', (event) => {
        selectedFiles = Array.from(event.target.files); // Converte FileList para Array
        updateSelectedFilesDisplay();
        validateFiles();
    });

    // Lida com o clique no botão "Processar Imagens"
    processImagesButton.addEventListener('click', async () => {
        if (selectedFiles.length > 0) {
            processImagesButton.disabled = true; // Desabilita para evitar múltiplos cliques
            selectImagesButton.disabled = true; // Desabilita o botão de seleção
            showMessage("Enviando imagens para processamento...", "info");
            
            try {
                // 'uploadImagesBatch' é uma função definida em 'api-integration.js'
                const batchId = await uploadImagesBatch(selectedFiles); 
                if (batchId) {
                    // Redireciona para o painel de espera com o batch_id
                    window.location.href = `/painel_espera?batch_id=${batchId}`;
                } else {
                    showMessage("Erro ao iniciar o processamento em lote.", "error");
                    processImagesButton.disabled = false;
                    selectImagesButton.disabled = false;
                }
            } catch (error) {
                console.error("Erro no upload do lote:", error);
                showMessage("Erro inesperado ao processar imagens. Tente novamente.", "error");
                processImagesButton.disabled = false;
                selectImagesButton.disabled = false;
            }
        } else {
            showMessage("Por favor, selecione pelo menos uma imagem para processar.", "error");
        }
    });

    // Atualiza o texto exibindo a contagem de arquivos selecionados
    function updateSelectedFilesDisplay() {
        if (selectedFiles.length === 0) {
            selectedImagesCount.textContent = "Nenhuma imagem selecionada.";
        } else if (selectedFiles.length === 1) {
            selectedImagesCount.textContent = `1 imagem selecionada: ${selectedFiles[0].name}`;
        } else {
            selectedImagesCount.textContent = `${selectedFiles.length} imagem(ns) selecionada(s).`;
        }
    }

    // Valida se há arquivos selecionados para habilitar o botão de processamento
    function validateFiles() {
        if (selectedFiles.length > 0) {
            processImagesButton.disabled = false;
        } else {
            processImagesButton.disabled = true;
        }
    }

    // Exibe mensagens para o usuário
    function showMessage(text, type) {
        messageDiv.textContent = text;
        messageDiv.className = `message ${type}`; // Adiciona classe para estilização (info, error)
        messageDiv.style.display = 'block';
    }
});