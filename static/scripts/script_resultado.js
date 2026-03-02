let globalImagesData = [];

async function loadResults() {
    const urlParams = new URLSearchParams(window.location.search);
    const batchId = urlParams.get('batch_id');

    if (!batchId) {
        console.error("ID do lote não encontrado na URL.");
        return;
    }

    try {
        const response = await fetch(`/api/batch-images/${batchId}`);
        if (!response.ok) throw new Error("Erro ao buscar imagens");
        
        globalImagesData = await response.json();
        console.log("Dados carregados:", globalImagesData);

        if (globalImagesData && globalImagesData.length > 0) {
            const carrosselUl = document.getElementById('carrosselImages');
            carrosselUl.innerHTML = ''; 

            globalImagesData.forEach((img, index) => {
                const li = document.createElement('li');
                li.className = `carrossel-item ${index === 0 ? 'active' : ''}`;
                const imageUrl = `/api/download-processed-image/${img.id}`;
                li.innerHTML = `<img src="${imageUrl}" alt="Imagem ${index + 1}">`;
                carrosselUl.appendChild(li);
            });

            updateImageInfo(0);

            setupCarouselLogic();
            
        } else {
            document.getElementById('resultsContent').innerHTML = "<p>Nenhum dado encontrado.</p>";
        }
    } catch (error) {
        console.error("Erro ao carregar resultados:", error);
    }
}

/**
 * Atualiza os campos de texto ID, Tipo e Data com base no índice da imagem atual
 */
function updateImageInfo(index) {
    const imageData = globalImagesData[index];
    if (!imageData) return;

    const idElement = document.getElementById('processingId');
    if (idElement) idElement.textContent = imageData.id.substring(0, 8);

    const dateElement = document.getElementById('processingDate');
    if (dateElement) dateElement.textContent = new Date().toLocaleDateString('pt-BR');

    const typeElement = document.getElementById('mainObjectType');
    if (typeElement) {
        if (imageData.detections && imageData.detections.length > 0) {
            typeElement.textContent = imageData.detections[0].class_name;
        } else {
            typeElement.textContent = "Nenhuma detecção";
        }
    }
}

function setupCarouselLogic() {
    const items = document.querySelectorAll('.carrossel-item');
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');

    if (items.length <= 1) {
        if (prevBtn) prevBtn.style.display = 'none';
        if (nextBtn) nextBtn.style.display = 'none';
        return;
    }

    let currentIndex = 0;

    function changeSlide(newIndex) {
        items.forEach((item, i) => {
            item.classList.toggle('active', i === newIndex);
        });
        
        updateImageInfo(newIndex);
    }

    if (nextBtn) {
        nextBtn.onclick = (e) => {
            e.preventDefault();
            currentIndex = (currentIndex + 1) % items.length;
            changeSlide(currentIndex);
        };
    }

    if (prevBtn) {
        prevBtn.onclick = (e) => {
            e.preventDefault();
            currentIndex = (currentIndex - 1 + items.length) % items.length;
            changeSlide(currentIndex);
        };
    }
}

document.addEventListener('DOMContentLoaded', loadResults);