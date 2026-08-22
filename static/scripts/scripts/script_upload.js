
document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('file-input'); // input real do HTML
    const processButton = document.getElementById('process-button'); // botão real do HTML
    const fileNameDisplay = document.getElementById('file-name'); // div que mostra nome dos arquivos

    let selectedFiles = [];

    imageInput.addEventListener('change', (event) => {
        selectedFiles = Array.from(event.target.files);
        updateSelectedFilesDisplay();
        validateFiles();
    });

    processButton.addEventListener('click', async () => {
        if (selectedFiles.length > 0) {
            processButton.disabled = true;
            fileNameDisplay.textContent = "Enviando imagens para o servidor...";

            try {
                const batchId = await uploadImagesBatch(selectedFiles); // Chama a função da api-integration.js
                if (batchId) {
                    window.location.href = `/painel_espera?batch_id=${batchId}`;
                } else {
                    fileNameDisplay.textContent = "Erro ao iniciar o processamento.";
                    processButton.disabled = false;
                }
            } catch (error) {
                console.error(error);
                fileNameDisplay.textContent = "Erro ao enviar as imagens. Tente novamente.";
                processButton.disabled = false;
            }
        }
    });

    function updateSelectedFilesDisplay() {
        if (selectedFiles.length === 0) {
            fileNameDisplay.textContent = "Nenhuma imagem selecionada.";
        } else if (selectedFiles.length === 1) {
            fileNameDisplay.textContent = `1 imagem selecionada: ${selectedFiles[0].name}`;
        } else {
            fileNameDisplay.textContent = `${selectedFiles.length} imagens selecionadas.`;
        }
    }

    function validateFiles() {
        processButton.disabled = selectedFiles.length === 0;
    }
});