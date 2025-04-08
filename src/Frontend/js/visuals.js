document.addEventListener('DOMContentLoaded', function() {
    // Image modal functionality
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImage");
    const captionText = document.getElementById("caption");
    
    // Set up all zoom buttons
    document.querySelectorAll('.zoom-button').forEach(button => {
        button.addEventListener('click', function() {
            modal.style.display = "block";
            modalImg.src = this.getAttribute('data-image');
            captionText.innerHTML = this.parentElement.parentElement.querySelector('.card-header h2').textContent;
        });
    });
    
    // Close modal
    document.querySelector('.close-modal').addEventListener('click', function() {
        modal.style.display = "none";
    });
    
    // Close when clicking outside image
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.style.display = "none";
        }
    });
    
    // Refresh visualizations
    document.getElementById('refreshVisualizations').addEventListener('click', function() {
        // Show loading state
        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
        
        // Simulate refresh (in a real app, this would fetch new data)
        setTimeout(() => {
            this.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
            showToast('Visualizations refreshed successfully', 'success');
        }, 1500);
    });
    
    // Time range filter
    document.getElementById('timeRange').addEventListener('change', function() {
        showToast(`Showing data for: ${this.options[this.selectedIndex].text}`, 'info');
        // In a real app, this would filter the data and update visualizations
    });
    
    // Card animation on scroll
    const cards = document.querySelectorAll('.visualization-card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, 150 * index);
            }
        });
    }, { threshold: 0.1 });
    
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = `opacity 0.5s ease ${index * 0.1}s, transform 0.5s ease ${index * 0.1}s`;
        observer.observe(card);
    });
    
    // Toast notification function
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 
                              type === 'success' ? 'fa-check-circle' : 
                              'fa-info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('show');
        }, 10);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 5000);
    }
});