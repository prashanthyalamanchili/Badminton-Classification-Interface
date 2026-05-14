async function uploadVideo() {
    const fileInput = document.getElementById("videoInput");
    const status = document.getElementById("status");
    const outputVideo = document.getElementById("outputVideo");
    const analyseBtn = document.getElementById("analyseBtn");

    if (fileInput.files.length === 0) {
        alert("Please select a video first.");
        return;
    }

    const formData = new FormData();
    formData.append("video", fileInput.files[0]);

    status.innerText = "Analysing video... please wait.";
    outputVideo.style.display = "none";
    analyseBtn.disabled = true;
    analyseBtn.innerText = "Analysing...";

    try {
        const response = await fetch("/analyse", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            status.innerText = "Analysis failed. Please try again.";
            analyseBtn.disabled = false;
            analyseBtn.innerText = "Analyse Video";
            return;
        }

        status.innerText = "Done. Output video saved.";

        outputVideo.src = result.output_video;
        outputVideo.style.display = "block";

    } catch (error) {
        console.error(error);
        status.innerText = "Something went wrong during analysis.";
    }

    analyseBtn.disabled = false;
    analyseBtn.innerText = "Analyse Video";
}
