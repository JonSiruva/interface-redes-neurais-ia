// Referências
document.addEventListener("DOMContentLoaded", () => {
  const filePreview = document.getElementById("file");
  const btnPrever = document.getElementById("btn");
  const preview = document.getElementById("imgPreview");
  const out = document.getElementById("out");
  const dropArea = document.getElementById('drop-area');
  const fileInput = document.getElementById('file-input');
  const loadingMessage = document.getElementById('loading-message');

  // Carregar modelo
  let model;
  const CLASS_MAP = [
      "Barroco",
      "Cubismo",
      "Expressionismo",
      "Impressionismo",
      "Minimalismo",
      "Pós-Impressionismo",
      "Realismo",
      "Romantismo",
      "Simbolismo"
  ];

async function loadModel() {
  // display da mensagem de loading
  loadingMessage.style.display = 'block';

  try {
    // tenta carregar o modelo
    model = await tf.loadGraphModel("model_js_graph/model.json");
    console.log("Modelo carregado com sucesso!");

    // mensagem de sucesso
    loadingMessage.innerText = '✅ Modelo carregado!';

  } catch (error) {
    // erro se o carregamento falhar
    console.error("Falha ao carregar o modelo:", error);
    loadingMessage.innerText = '❌ Falha ao carregar o modelo.';

  } finally {
    // O 'finally' executa SEMPRE, com sucesso ou com erro.
    // Damos 2 segundos para o usuário poder ler a mensagem final.
    setTimeout(() => {
      loadingMessage.style.display = 'none';
    }, 2000);
  }
}

// inicia o carregamento do modelo
loadModel();

  /*****************************************/

  // não abre a imagem no navegador
  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }
  
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

// pra deixar bonito
["dragenter", "dragover"].forEach(eventName => {
  dropArea.addEventListener(eventName, () => {
    dropArea.style.backgroundColor = "#f0f0f0";
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropArea.addEventListener(eventName, () => {
    dropArea.style.backgroundColor = "transparent";
  });
});

// quando solta o arquivo na área
dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  processFile(files[0]);
}

dropArea.addEventListener('click', () => {
  fileInput.click();
})

fileInput.addEventListener("change", e => {
  processFile(e.target.files[0]);
});

function processFile(file) {
  if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
      preview.src = e.target.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(file);
  } else {
    preview.src = "";
    preview.style.display = "none";
  }
}

/********************************/

// Rodar a predição no clique do botão
btnPrever.addEventListener("click", async () => {
    if (!model) {
        alert("O modelo ainda não carregou, aguarde.");
        return;
    }
    if (!preview.src || preview.style.display === "none") {
        alert("Selecione uma imagem primeiro.");
        return;
    }

    // Converte a imagem para tensor
    const originalTensor = tf.browser.fromPixels(preview);
    const [height, width] = originalTensor.shape;

    // Calcula os parâmetros para o corte (crop) quadrado aleatório
    let cropSize = Math.min(height, width);
    // Calcula o ponto de início para centralizar o crop
    let startY = Math.floor((height - cropSize) / 2);
    let startX = Math.floor((width - cropSize) / 2);

    // Corta o tensor para criar um quadrado
    const cropped = tf.slice(originalTensor, [startY, startX, 0], [cropSize, cropSize, 3]);

    // Redimensiona o quadrado cortado para o tamanho esperado pelo modelo (255x255)
    const resized = tf.image.resizeNearestNeighbor(cropped, [128, 128]);

    // Normaliza os pixels e adiciona a dimensão do batch
    const normalized = resized.toFloat().div(255.0).expandDims(0);

    // Predição
    const preds = model.predict(normalized);
    const probs = await preds.data();

    
    // Pega top 3
    const pairs = Array.from(probs)
        .map((p, i) => ({ i, p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3);

    // Mostrar resultados
    out.innerHTML =
        "<h2>Resultado da classificação</h2>" +
        pairs
            .map(
                (p) =>
                    `<p style=font-size:large; ><b>${CLASS_MAP[p.i]}</b> — ${(p.p * 100).toFixed(2)}%</p>`
            )
            .join("");

    const topResult = CLASS_MAP[pairs[0].i];
    const topProb = (pairs[0].p * 100).toFixed(2);

    document.getElementById("modalImg").src = preview.src;
    document.getElementById("modalResult").textContent =
        `${topResult} — ${topProb}%`;

    document.getElementById("resultModal").style.display = "flex";

    // liberar memória (importante adicionar os novos tensores)
    originalTensor.dispose();
    cropped.dispose();
    resized.dispose();
    normalized.dispose();
    preds.dispose();
});

// fecha modal
document.querySelector(".close").addEventListener("click", () => {
  document.getElementById("resultModal").style.display = "none";
});

// fecha modal
window.addEventListener("click", (e) => {
  if (e.target.id === "resultModal") {
    document.getElementById("resultModal").style.display = "none";
  }
});

});