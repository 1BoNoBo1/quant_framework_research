/**
 * QFrame Documentation - Scripts JavaScript personnalisés
 * Fonctionnalités interactives et améliorations UX
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 QFrame Documentation loaded');

    // =============================================================================
    // Amélioration de la recherche
    // =============================================================================

    // Recherche intelligente avec raccourcis
    document.addEventListener('keydown', function(e) {
        // Ctrl+K ou Cmd+K pour ouvrir la recherche
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('.md-search__input');
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        }

        // Escape pour fermer la recherche
        if (e.key === 'Escape') {
            const searchInput = document.querySelector('.md-search__input');
            if (searchInput && document.activeElement === searchInput) {
                searchInput.blur();
            }
        }
    });

    // =============================================================================
    // Navigation améliorée
    // =============================================================================

    // Scroll smooth pour les liens d'ancrage
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Indicateur de progression de lecture
    function updateReadingProgress() {
        const article = document.querySelector('.md-content__inner');
        if (!article) return;

        const articleHeight = article.offsetHeight;
        const windowHeight = window.innerHeight;
        const scrollTop = window.pageYOffset;
        const progress = scrollTop / (articleHeight - windowHeight);
        const progressPercent = Math.min(100, Math.max(0, progress * 100));

        // Créer ou mettre à jour la barre de progression
        let progressBar = document.querySelector('.reading-progress');
        if (!progressBar) {
            progressBar = document.createElement('div');
            progressBar.className = 'reading-progress';
            progressBar.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: ${progressPercent}%;
                height: 3px;
                background: linear-gradient(90deg, #7c4dff, #9c27b0);
                z-index: 1000;
                transition: width 0.2s ease;
            `;
            document.body.appendChild(progressBar);
        } else {
            progressBar.style.width = `${progressPercent}%`;
        }
    }

    window.addEventListener('scroll', updateReadingProgress);
    window.addEventListener('resize', updateReadingProgress);

    // =============================================================================
    // Amélioration des tables
    // =============================================================================

    // Rendre les tables responsive
    document.querySelectorAll('table').forEach(table => {
        if (!table.closest('.md-typeset')) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'table-responsive';
        wrapper.style.cssText = `
            overflow-x: auto;
            margin: 1em 0;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(124, 77, 255, 0.2);
        `;

        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);

        // Ajouter des tooltips pour les colonnes tronquées
        table.querySelectorAll('td, th').forEach(cell => {
            if (cell.scrollWidth > cell.clientWidth) {
                cell.title = cell.textContent.trim();
            }
        });
    });

    // =============================================================================
    // Code highlighting et interactions
    // =============================================================================

    // Bouton de copie pour les blocs de code
    document.querySelectorAll('pre code').forEach(codeBlock => {
        const pre = codeBlock.parentNode;
        if (pre.querySelector('.copy-button')) return; // Éviter les doublons

        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '📋';
        copyButton.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(124, 77, 255, 0.8);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 5px 8px;
            cursor: pointer;
            font-size: 12px;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;

        pre.style.position = 'relative';
        pre.appendChild(copyButton);

        // Afficher le bouton au survol
        pre.addEventListener('mouseenter', () => {
            copyButton.style.opacity = '1';
        });

        pre.addEventListener('mouseleave', () => {
            copyButton.style.opacity = '0';
        });

        // Fonction de copie
        copyButton.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(codeBlock.textContent);
                copyButton.innerHTML = '✅';
                setTimeout(() => {
                    copyButton.innerHTML = '📋';
                }, 2000);
            } catch (err) {
                console.error('Erreur de copie:', err);
                copyButton.innerHTML = '❌';
                setTimeout(() => {
                    copyButton.innerHTML = '📋';
                }, 2000);
            }
        });
    });

    // =============================================================================
    // Métriques et animations
    // =============================================================================

    // Animation des métriques au scroll
    function animateMetrics() {
        const metricCards = document.querySelectorAll('.metric-card');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animation = 'slideInUp 0.6s ease-out';
                    entry.target.style.opacity = '1';
                }
            });
        }, { threshold: 0.1 });

        metricCards.forEach(card => {
            card.style.opacity = '0';
            observer.observe(card);
        });
    }

    // Ajouter les styles d'animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInUp {
            from {
                transform: translateY(30px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);

    animateMetrics();

    // =============================================================================
    // Thème et préférences utilisateur
    // =============================================================================

    // Détection du thème système
    function updateThemeBasedOnSystem() {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const palette = document.querySelector('[data-md-color-scheme]');

        if (palette && !localStorage.getItem('data-md-color-scheme')) {
            palette.setAttribute('data-md-color-scheme', prefersDark ? 'slate' : 'default');
        }
    }

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateThemeBasedOnSystem);
    updateThemeBasedOnSystem();

    // =============================================================================
    // Performance et analytics
    // =============================================================================

    // Mesure du temps de chargement de la page
    window.addEventListener('load', function() {
        const loadTime = performance.now();
        console.log(`📊 Page loaded in ${Math.round(loadTime)}ms`);

        // Optionnel : Envoyer à un service d'analytics
        // analytics.track('page_load_time', { duration: loadTime });
    });

    // Tracking des interactions utilisateur (anonyme)
    let interactions = 0;
    document.addEventListener('click', function(e) {
        interactions++;
        if (interactions % 10 === 0) {
            console.log(`👆 ${interactions} interactions on this page`);
        }
    });

    // =============================================================================
    // Amélioration de l'accessibilité
    // =============================================================================

    // Focus visible pour navigation clavier
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });

    document.addEventListener('mousedown', function() {
        document.body.classList.remove('keyboard-navigation');
    });

    // Ajouter les styles d'accessibilité
    const accessibilityStyle = document.createElement('style');
    accessibilityStyle.textContent = `
        .keyboard-navigation *:focus {
            outline: 2px solid #7c4dff !important;
            outline-offset: 2px !important;
        }

        /* Amélioration du contraste pour les liens */
        .md-content a {
            text-decoration: underline;
            text-decoration-color: rgba(124, 77, 255, 0.5);
        }

        .md-content a:hover {
            text-decoration-color: #7c4dff;
        }
    `;
    document.head.appendChild(accessibilityStyle);

    // =============================================================================
    // Features spéciales QFrame
    // =============================================================================

    // Badge de statut du framework
    function addFrameworkStatus() {
        const header = document.querySelector('.md-header');
        if (!header || header.querySelector('.framework-status')) return;

        const statusBadge = document.createElement('div');
        statusBadge.className = 'framework-status';
        statusBadge.innerHTML = '🟢 100% Opérationnel';
        statusBadge.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(76, 175, 80, 0.9);
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            z-index: 1001;
        `;

        header.style.position = 'relative';
        header.appendChild(statusBadge);
    }

    setTimeout(addFrameworkStatus, 1000);

    console.log('✨ QFrame Documentation enhancements loaded successfully');
});