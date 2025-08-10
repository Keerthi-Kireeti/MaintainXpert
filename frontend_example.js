async function getPrediction() {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            machine_type: 'CNC',
            issue_type: 'Calibration',
            severity: 3,
            parts_required: 1,
            technician_experience: 5,
            days_already_in_progress: 2,
            last_service_date: '2025-08-01'
        })
    });
    const data = await response.json();
    console.log(data);
}
getPrediction();