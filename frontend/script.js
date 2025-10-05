document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("csvFile");
    const columnMappingDiv = document.getElementById("columnMapping");
    const stackSelect = document.getElementById("stackSelect");
    const submitBtn = document.getElementById("submitBtn");
    const predOutput = document.getElementById("predOutput");
    const resultsDiv = document.getElementById("results");

    let csvData = [];
    let headers = [];

    const baseFeatures = [
        'insol','insol_err1','insol_err2',
        'period','period_err1','period_err2',
        'prad','prad_err1','prad_err2',
        'steff','steff_err1','steff_err2',
        'srad','srad_err1','srad_err2'
    ];

    // --- Read CSV and populate selects ---
    fileInput.addEventListener("change", (e) => {
        console.log("[DEBUG] File input triggered.");
        const file = e.target.files[0];
        if (!file) {
            console.warn("[DEBUG] No file selected.");
            return;
        }
        console.log("[DEBUG] File selected:", file.name);

        const reader = new FileReader();
        reader.onload = (event) => {
            console.log("[DEBUG] FileReader finished loading.");
            const text = event.target.result.trim();
            console.log("[DEBUG] Raw CSV text length:", text.length);

            const rows = text.split("\n").filter(r => r.trim().length > 0);
            headers = rows[0].split(",");
            console.log("[DEBUG] Parsed headers:", headers);

            csvData = rows.slice(1).map(r => r.split(","));
            console.log("[DEBUG] CSV rows count:", csvData.length);

            // Clear previous mapping
            columnMappingDiv.innerHTML = "<h3>Map CSV columns to model features:</h3>";

            // Create selects dynamically
            baseFeatures.forEach(feature => {
                const label = document.createElement("label");
                label.textContent = feature + ": ";

                const select = document.createElement("select");
                select.dataset.feature = feature;

                headers.forEach(h => {
                    const option = document.createElement("option");
                    option.value = h;
                    option.textContent = h;
                    select.appendChild(option);
                });

                label.appendChild(select);
                columnMappingDiv.appendChild(label);
            });

            console.log("[DEBUG] Mapping dropdowns created.");
            columnMappingDiv.classList.remove("hidden");
        };

        reader.onerror = (err) => {
            console.error("[DEBUG] Error reading CSV:", err);
            alert("Failed to read CSV file.");
        };

        reader.readAsText(file);
    });

    // --- Submit CSV for prediction ---
    submitBtn.addEventListener("click", async () => {
        console.log("[DEBUG] Submit button clicked.");

        if (!fileInput.files[0]) {
            alert("Please upload a CSV first!");
            console.warn("[DEBUG] Submit blocked: no CSV file uploaded.");
            return;
        }

        if (!stackSelect.value) {
            alert("Please select a model stack!");
            console.warn("[DEBUG] Submit blocked: no model stack selected.");
            return;
        }

        // Build mapping: CSV column → base feature
        const mapping = {};
        let allMapped = true;

        columnMappingDiv.querySelectorAll("select").forEach(select => {
            if (!select.value) {
                allMapped = false;
                console.warn("[DEBUG] Unmapped feature:", select.dataset.feature);
            } else {
                mapping[select.value] = select.dataset.feature;
            }
        });

        if (!allMapped) {
            alert("Please map all required features!");
            return;
        }

        console.log("[DEBUG] Final mapping:", mapping);

        const formData = new FormData();
        formData.append("stack", stackSelect.value);
        formData.append("mapping", JSON.stringify(mapping));
        formData.append("csv", fileInput.files[0]);

        try {
            console.log("[DEBUG] Sending request to backend with stack:", stackSelect.value);

            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData
            });

            console.log("[DEBUG] Response status:", response.status);

            const contentType = response.headers.get("content-type");
            console.log("[DEBUG] Response Content-Type:", contentType);

            if (contentType && contentType.includes("application/json")) {
                const result = await response.json();
                console.error("[DEBUG] Backend returned JSON (likely error):", result);

                resultsDiv.classList.remove("hidden");
                predOutput.textContent = JSON.stringify(result, null, 2);
                alert("Backend error: " + JSON.stringify(result));
                return;
            }

            // ✅ get the blob if no error JSON
            const blob = await response.blob();
            console.log("[DEBUG] Blob received, size:", blob.size);

            if (blob.size === 0) {
                alert("Received empty file. Something went wrong on the server.");
                console.error("[DEBUG] Blob is empty.");
                return;
            }

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `all_predictions_${stackSelect.value}.zip`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);

            console.log("[DEBUG] File download triggered.");

        } catch (err) {
            console.error("[DEBUG] Fetch error:", err);
            alert("Something went wrong while predictings.");
        }
    });
});
