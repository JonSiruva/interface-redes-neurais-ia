// Referências dos elementos HTML que serão manipulados
document.addEventListener("DOMContentLoaded", () => {
  const btnPrever = document.getElementById("btn"); 
  const preview = document.getElementById("imgPreview");
  const out = document.getElementById("out");
  const dropArea = document.getElementById('drop-area');
  const fileInput = document.getElementById('file-input');
  const loadingMessage = document.getElementById('loading-message');

  // Armazena o modelo carregado
  let model;
  // Mapeamento das classes do modelo, na ordem exata do treinamento
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

// Função para carregar o modelo TensorFlow.js
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
    // finally sempre executa, com sucesso ou com erro.
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
  // eventos para a área de drop
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

// muda o estilo da área de drop quando o arquivo está sobre ela
["dragenter", "dragover"].forEach(eventName => {
  dropArea.addEventListener(eventName, () => {
    dropArea.style.backgroundColor = "#f0f0f0";
  });
});

// restaura o estilo quando o arquivo sai da área de drop
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

// abre o seletor de arquivos ao clicar na área de drop
dropArea.addEventListener('click', () => {
  fileInput.click();
})

// quando seleciona o arquivo no seletor
fileInput.addEventListener("change", e => {
  processFile(e.target.files[0]);
});

// processa o arquivo selecionado ou dropado
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

    // Corta a imagem para um quadrado central
    let cropSize = Math.min(height, width);
    let startY = Math.floor((height - cropSize) / 2);
    let startX = Math.floor((width - cropSize) / 2);

    // Corta o tensor para criar um quadrado
    const cropped = tf.slice(originalTensor, [startY, startX, 0], [cropSize, cropSize, 3]);

    // Redimensiona o quadrado cortado para o tamanho esperado pelo modelo
    const resized = tf.image.resizeNearestNeighbor(cropped, [128, 128]);

    // Normaliza os pixels e adiciona a dimensão do batch
    const normalized = resized.toFloat().div(255.0).expandDims(0);

    // Predição
    const preds = model.predict(normalized);
    const probs = await preds.data();

    
    // Pega top 3
    const pairs = Array.from(probs) // Converte o resultado em um array JS.
        .map((p, i) => ({ i, p })) // Mapeia para objetos {índice, probabilidade}.
        .sort((a, b) => b.p - a.p) // Ordena em ordem decrescente de probabilidade.
        .slice(0, 3); // Pega apenas os 3 primeiros (top 3).

    // Mostra resultados
    out.innerHTML =
        "<h2>Resultado da classificação</h2>" +
        pairs
            .map(
                (p) =>
                    `<p style=font-size:large; ><b>${CLASS_MAP[p.i]}</b> — ${(p.p * 100).toFixed(2)}%</p>`
            )
            .join("");

    // mostra o resultado principal no modal
    const topResult = CLASS_MAP[pairs[0].i];
    const topProb = (pairs[0].p * 100).toFixed(2);

    document.getElementById("modalImg").src = preview.src;
    document.getElementById("modalResult").textContent =`${topResult} — ${topProb}%`;
    document.getElementById("resultModal").style.display = "flex";

    // libera a memória alocada para os tensores
    originalTensor.dispose();
    cropped.dispose();
    resized.dispose();
    normalized.dispose();
    preds.dispose();
});

// fecha modal pelo botão de fechar
document.querySelector(".close").addEventListener("click", () => {
  document.getElementById("resultModal").style.display = "none";
});

// fecha modal clicando fora do conteúdo
window.addEventListener("click", (e) => {
  if (e.target.id === "resultModal") {
    document.getElementById("resultModal").style.display = "none";
  }
});

});