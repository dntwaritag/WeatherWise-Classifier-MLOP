document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const filterButtons = document.querySelectorAll('.filter-btn');
    const visualizationCards = document.querySelectorAll('.visualization-card');
    const downloadButtons = document.querySelectorAll('.download-btn');
    const fullscreenViz = document.getElementById('fullscreen-viz');
    const fullscreenImage = document.getElementById('fullscreen-image');
    const fullscreenCaption = document.getElementById('fullscreen-caption');
    const closeFullscreen = document.querySelector('.close-fullscreen');

    // Filter visualizations
    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Update active button
            filterButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            const filter = this.dataset.filter;
            
            // Show/hide cards based on filter
            visualizationCards.forEach(card => {
                if (filter === 'all' || card.dataset.category.includes(filter)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });

    // Handle download buttons
    downloadButtons.forEach(button => {
        button.addEventListener('click', function() {
            const vizName = this.dataset.viz;
            alert(`In a real application, this would download the ${vizName} visualization.`);
            // In a real app, you would trigger a download here
            // window.location.href = `/download/${vizName}`;
        });
    });

    // Handle fullscreen view
    visualizationCards.forEach(card => {
        const img = card.querySelector('.visualization-image');
        const title = card.querySelector('.card-header h2').textContent;
        const description = card.querySelector('.card-description p').textContent;
        
        img.addEventListener('click', function() {
            fullscreenImage.src = this.src;
            fullscreenCaption.textContent = `${title}: ${description}`;
            fullscreenViz.style.display = 'flex';
        });
    });

    // Close fullscreen view
    closeFullscreen.addEventListener('click', function() {
        fullscreenViz.style.display = 'none';
    });

    // Close when clicking outside image
    fullscreenViz.addEventListener('click', function(e) {
        if (e.target === this) {
            this.style.display = 'none';
        }
    });

    // Close with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && fullscreenViz.style.display === 'flex') {
            fullscreenViz.style.display = 'none';
        }
    });
});