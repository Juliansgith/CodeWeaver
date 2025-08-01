document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('nav a');
    const pages = document.querySelectorAll('.page');
    let fileBrowserTargetInput = null;
    let embeddingProvidersStatus = {};
    let costUpdateInterval = null;

    // --- Navigation ---
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);

            pages.forEach(page => page.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');

            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    // --- WebSocket for Real-time Updates ---
    let ws;
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

        ws.onopen = () => addLogMessage('Connected to CodeWeaver server.');
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            switch (msg.type) {
                case 'log':
                    addLogMessage(msg.data);
                    break;
                case 'digest_complete':
                    addLogMessage(`✅ Digest complete! Path: ${msg.data.path}`);
                    break;
                case 'mcp_status_update':
                    updateMcpStatusUI(msg.data);
                    break;
                case 'cost_session_started':
                    startCostTracking();
                    break;
                case 'cost_summary':
                    updateCostDisplay(msg.data);
                    break;
                case 'error':
                    addLogMessage(`❌ ERROR: ${msg.data}`, 'error');
                    break;
            }
        };
        ws.onclose = () => {
            addLogMessage('Connection lost. Reconnecting...', 'error');
            setTimeout(connectWebSocket, 3000);
        };
    }
    connectWebSocket();

    // --- Log Panel ---
    const logOutput = document.getElementById('log-output');
    function addLogMessage(message, level = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.innerHTML = `<span>${timestamp}</span> - ${message}`;
        if (level === 'error') {
            logEntry.style.color = '#f56565';
        }
        logOutput.appendChild(logEntry);
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    // --- Digest Form ---
    document.getElementById('digest-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const path = document.getElementById('digest-path').value;
        const purpose = document.getElementById('digest-purpose').value;
        const budget = document.getElementById('digest-budget').value;

        if (!path || !purpose) {
            alert('Project Path and Purpose are required.');
            return;
        }
        
        await addRecentProject(path);

        try {
            const response = await fetch('/api/digest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path, purpose, budget: parseInt(budget) }),
            });
            const data = await response.json();
            if (response.ok) {
                addLogMessage('Digest generation started...');
            } else {
                throw new Error(data.error || 'Failed to start digest generation.');
            }
        } catch (error) {
            addLogMessage(error.message, 'error');
        }
    });

    // --- File Browser ---
    const modal = document.getElementById('file-browser-modal');
    document.querySelectorAll('.browse-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            fileBrowserTargetInput = document.getElementById(btn.dataset.target);
            openFileBrowser();
            modal.style.display = 'block';
        });
    });

    document.querySelector('.close-button').onclick = () => modal.style.display = 'none';
    window.onclick = (event) => {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    };

    async function openFileBrowser(path = null) {
        try {
            const response = await fetch('/api/browse', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path }),
            });
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            document.getElementById('browser-path').textContent = data.current_path;
            const listEl = document.getElementById('browser-list');
            listEl.innerHTML = '';

            if (data.parent_path) {
                const upEl = document.createElement('li');
                upEl.textContent = '⬆️ ..';
                upEl.onclick = () => openFileBrowser(data.parent_path);
                listEl.appendChild(upEl);
            }

            data.items.forEach(item => {
                const itemEl = document.createElement('li');
                itemEl.textContent = `📁 ${item.name}`;
                itemEl.onclick = () => openFileBrowser(item.path);
                listEl.appendChild(itemEl);
            });
        } catch (error) {
            addLogMessage(`File browser error: ${error.message}`, 'error');
        }
    }

    document.getElementById('select-path-btn').addEventListener('click', () => {
        if (fileBrowserTargetInput) {
            const selectedPath = document.getElementById('browser-path').textContent;
            fileBrowserTargetInput.value = selectedPath;
            modal.style.display = 'none';
        }
    });

    // --- Recent Projects ---
    const recentProjectsList = document.getElementById('recent-projects-list');
    async function loadRecentProjects() {
        try {
            const response = await fetch('/api/recent-projects');
            const projects = await response.json();
            recentProjectsList.innerHTML = '';
            if (projects.length === 0) {
                recentProjectsList.innerHTML = '<li>No recent projects.</li>';
            } else {
                projects.forEach(p => {
                    const li = document.createElement('li');
                    li.innerHTML = `<strong>${p.name}</strong><small>${p.path}</small>`;
                    li.onclick = () => {
                        document.querySelectorAll('.path-input').forEach(input => input.value = p.path);
                        addLogMessage(`Loaded recent project: ${p.name}`);
                    };
                    recentProjectsList.appendChild(li);
                });
            }
        } catch (error) {
            recentProjectsList.innerHTML = '<li>Error loading projects.</li>';
        }
    }
    
    async function addRecentProject(path) {
        try {
            await fetch('/api/recent-projects', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path }),
            });
            loadRecentProjects();
        } catch (error) {
            console.error('Failed to add recent project:', error);
        }
    }

    // --- AI Services ---
    const embeddingStatusEl = document.getElementById('embedding-status');
    const embeddingForm = document.getElementById('embedding-form');
    const mcpStatusIndicator = document.querySelector('#mcp-status .status-indicator');
    const mcpProjectPathEl = document.getElementById('mcp-project-path');
    const mcpStartBtn = document.getElementById('mcp-start-btn');
    const mcpStopBtn = document.getElementById('mcp-stop-btn');
    const mcpPathInput = document.getElementById('mcp-path');

    async function updateEmbeddingStatus() {
        try {
            const response = await fetch('/api/embeddings/status');
            const data = await response.json();
            embeddingProvidersStatus = data; // Store for later checks
            let html = '<ul>';
            for (const [provider, status] of Object.entries(data)) {
                html += `<li><b>${provider.toUpperCase()}:</b> ${status.configured ? `✅ Configured (Model: ${status.model})` : '❌ Not Configured'}</li>`;
            }
            html += '</ul>';
            embeddingStatusEl.innerHTML = html;
        } catch (error) {
            embeddingStatusEl.textContent = 'Failed to load status.';
        }
    }

    embeddingForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const provider = document.getElementById('embedding-provider').value;
        const apiKey = document.getElementById('embedding-apikey').value;

        if (!apiKey) {
            alert('API Key is required.');
            return;
        }

        try {
            await fetch('/api/embeddings/configure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, apiKey }),
            });
            addLogMessage(`API Key for ${provider} configured.`);
            document.getElementById('embedding-apikey').value = '';
            updateEmbeddingStatus();
        } catch (error) {
            addLogMessage(`Failed to configure API Key: ${error.message}`, 'error');
        }
    });

    function updateMcpStatusUI(data) {
        mcpStatusIndicator.className = `status-indicator ${data.running ? 'running' : 'stopped'}`;
        mcpProjectPathEl.textContent = data.running ? `Project: ${data.project_path}` : 'Server is stopped.';
        mcpStartBtn.disabled = data.running;
        mcpStopBtn.disabled = !data.running;
        mcpPathInput.disabled = data.running;
    }
    
    async function fetchMcpStatus() {
        try {
            const response = await fetch('/api/mcp/status');
            const data = await response.json();
            updateMcpStatusUI(data);
        } catch (error) {
            mcpStatusIndicator.className = 'status-indicator unknown';
            mcpProjectPathEl.textContent = 'Failed to get status.';
        }
    }

    mcpStartBtn.addEventListener('click', async () => {
        // Check if any embedding provider is configured
        const isAnyConfigured = Object.values(embeddingProvidersStatus).some(s => s.configured);
        if (!isAnyConfigured) {
            alert('MCP Server requires at least one Embedding Provider to be configured. Please add an API key in the section above.');
            return;
        }

        const path = mcpPathInput.value;
        if (!path) {
            alert('Project path is required to start the MCP server.');
            return;
        }
        await addRecentProject(path);
        try {
            await fetch('/api/mcp/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path }),
            });
        } catch (error) {
            addLogMessage(`Failed to start MCP server: ${error.message}`, 'error');
        }
    });

    mcpStopBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/mcp/stop', { method: 'POST' });
        } catch (error) {
            addLogMessage(`Failed to stop MCP server: ${error.message}`, 'error');
        }
    });
    
    // Initial data load
    loadRecentProjects();
    updateEmbeddingStatus();
    fetchMcpStatus();
    
    // Initialize cost tracking
    initializeCostTracker();
});

// --- Cost Tracking Functions ---
function initializeCostTracker() {
    updateDailyCosts();
    // Update daily costs every 30 seconds
    setInterval(updateDailyCosts, 30000);
}

function startCostTracking() {
    // Start polling for current session costs
    if (costUpdateInterval) {
        clearInterval(costUpdateInterval);
    }
    
    costUpdateInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/costs/current');
            if (response.ok) {
                const data = await response.json();
                updateSessionCosts(data);
            }
        } catch (error) {
            console.error('Failed to fetch current costs:', error);
        }
    }, 2000); // Update every 2 seconds during active session
}

function stopCostTracking() {
    if (costUpdateInterval) {
        clearInterval(costUpdateInterval);
        costUpdateInterval = null;
    }
}

function updateSessionCosts(data) {
    const sessionCostEl = document.getElementById('session-cost');
    const tokenCountEl = document.getElementById('token-count');
    
    if (data.active) {
        const cost = data.total_cost || 0;
        const tokens = data.total_tokens || 0;
        
        sessionCostEl.textContent = `$${cost.toFixed(4)}`;
        tokenCountEl.textContent = tokens.toLocaleString();
        
        // Color coding based on cost
        sessionCostEl.className = 'cost-value';
        if (cost > 1.0) {
            sessionCostEl.classList.add('very-high-cost');
        } else if (cost > 0.1) {
            sessionCostEl.classList.add('high-cost');
        }
        
        // Add recent operations to log if available
        if (data.recent_operations && data.recent_operations.length > 0) {
            const recent = data.recent_operations[data.recent_operations.length - 1];
            if (recent && recent.cost > 0.001) {
                addLogMessage(`💰 ${recent.provider} ${recent.operation}: $${recent.cost.toFixed(4)} (${recent.tokens} tokens)`);
            }
        }
    } else {
        sessionCostEl.textContent = '$0.0000';
        tokenCountEl.textContent = '0';
        sessionCostEl.className = 'cost-value';
        stopCostTracking();
    }
}

async function updateDailyCosts() {
    try {
        const response = await fetch('/api/costs/summary?hours=24');
        if (response.ok) {
            const data = await response.json();
            const dailyCostEl = document.getElementById('daily-cost');
            const cost = data.total_cost || 0;
            
            dailyCostEl.textContent = `$${cost.toFixed(4)}`;
            dailyCostEl.className = 'cost-value';
            
            if (cost > 5.0) {
                dailyCostEl.classList.add('very-high-cost');
            } else if (cost > 1.0) {
                dailyCostEl.classList.add('high-cost');
            }
        }
    } catch (error) {
        console.error('Failed to fetch daily costs:', error);
    }
}

function updateCostDisplay(costData) {
    // This is called when we receive final cost summary via WebSocket
    if (costData.total_cost > 0) {
        addLogMessage(`💰 Session completed! Total cost: $${costData.total_cost.toFixed(4)} (${costData.total_tokens} tokens)`);
        updateDailyCosts(); // Refresh daily costs
    }
    stopCostTracking();
}