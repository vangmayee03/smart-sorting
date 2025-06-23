 const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const resultDisplay = document.getElementById('resultDisplay');
        const resultImage = document.getElementById('resultImage');
        const fruitType = document.getElementById('fruitType');
        const resultDescription = document.getElementById('resultDescription');
        const confidenceScore = document.getElementById('confidenceScore');
        // Handle file selection
        fileInput.addEventListener('change', function(e) {
            if (fileInput.files.length) {
                const file = fileInput.files[0];
                const reader = new FileReader();
                
                reader.onload = function(event) {
                    // Display the uploaded image
                    resultImage.src = event.target.result;
                    
                    // For demo purposes, pick a random fruit result
                    const randomFruit = fruitExamples[Math.floor(Math.random() * fruitExamples.length)];
                    
                    // Update result display
                    fruitType.textContent = randomFruit.type;
                    resultDescription.textContent = randomFruit.description;
                    confidenceScore.textContent = randomFruit.confidence + '%';
                    confidenceScore.className = randomFruit.status;
                    
                    // Show results
                    resultDisplay.style.display = 'block';
                };
                
                reader.readAsDataURL(file);
            }
        });

        // Drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            uploadArea.style.borderColor = '#4CAF50';
            uploadArea.style.backgroundColor = '#f0f0f0';
        }

        function unhighlight() {
            uploadArea.style.borderColor = '#ccc';
            uploadArea.style.backgroundColor = '#f9f9f9';
        }

        uploadArea.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }

        // Form submission
        document.getElementById('contactForm').addEventListener('submit', function(e) {
            e.preventDefault();
            alert('Thank you for your message! We will get back to you soon.');
            this.reset();
        });