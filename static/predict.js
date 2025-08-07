
// select 要素を動的に生成
window.addEventListener("DOMContentLoaded", () => {
  const yearSel = document.getElementById("year");
  const monthSel = document.getElementById("month");
  const daySel = document.getElementById("day");

  for (let y = 2025; y <= 2029; y++) {
    yearSel.append(new Option(y, y));
  }

  for (let m = 1; m <= 12; m++) {
    monthSel.append(new Option(m, m));
  }

  for (let d = 1; d <= 31; d++) {
    daySel.append(new Option(d, d));
  }
});


function uploadCSV() {
  const fileInput = document.getElementById("csvFile");
  const file = fileInput.files[0];
  if (!file) {
    alert("CSVファイルを選んでください");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  fetch("http://localhost:5000/upload-csv", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById("uploadStatus").textContent = data.message;
      document.getElementById("r2_score").textContent = "r2:"+data.r2_score;
      document.getElementById("mse").textContent = "mse:"+data.mse;
      document.getElementById("mae").textContent = "mae:"+data.mae;
      document.getElementById("best").textContent = JSON.stringify(data.best_params, null, 2);

    })
    .catch(err => {
      document.getElementById("uploadStatus").textContent = "アップロード失敗";
    });
}


function sendData() {
  const year = document.getElementById("year").value;
  const month = document.getElementById("month").value;
  const day = document.getElementById("day").value;
  const weather = document.getElementById("weather").value;

  const data = {
    year: parseInt(year),
    month: parseInt(month),
    day: parseInt(day),
    weather: parseInt(weather)
  };

  fetch("http://localhost:5000/sendData", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById("result").textContent = `予測売上：¥${data.predicted_sales}`;
    })
    .catch(err => {
      document.getElementById("result").textContent = "予測に失敗しました";
    });
}

