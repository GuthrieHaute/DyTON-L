document.addEventListener('DOMContentLoaded', function () {
    const eventList = document.getElementById('event-list');
    const connectionStatus = document.getElementById('connection-status');
    const eventCountSpan = document.getElementById('event-count');
    const clearLogBtn = document.getElementById('clear-log-btn');
    let eventCount = 0;
    let socket;

    function updateConnectionStatus(status, message) {
        connectionStatus.textContent = message;
        connectionStatus.className = 'status-' + status;
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        updateConnectionStatus('connecting', 'Connecting...');
        socket = new WebSocket(wsUrl);

        socket.onopen = function () {
            console.log('WebSocket connected');
            updateConnectionStatus('connected', 'Connected');
        };

        socket.onmessage = function (event) {
            try {
                const eventData = JSON.parse(event.data);
                eventCount++;
                eventCountSpan.textContent = eventCount;

                const listItem = document.createElement('li');
                
                const metaSpan = document.createElement('span');
                metaSpan.className = 'event-meta';
                const now = new Date();
                metaSpan.textContent = `Received at: ${now.toLocaleTimeString()} | Timestamp: ${eventData.timestamp || 'N/A'} | Type: ${eventData.input_type || 'N/A'}`;
                
                const dataPre = document.createElement('pre');
                dataPre.className = 'event-data';
                dataPre.textContent = JSON.stringify(eventData, null, 2);

                listItem.appendChild(metaSpan);
                listItem.appendChild(dataPre);
                
                // Add to top of list for most recent first, or bottom for chronological
                // eventList.appendChild(listItem); // Chronological
                eventList.insertBefore(listItem, eventList.firstChild); // Most recent first

                // Optional: Auto-scroll to bottom if showing chronological and scrolled to bottom
                // if (eventList.scrollTop + eventList.clientHeight >= eventList.scrollHeight - listItem.offsetHeight) {
                //    eventList.scrollTop = eventList.scrollHeight;
                // }

            } catch (e) {
                console.error('Error processing event data:', e);
                // Display raw data if parsing fails
                const listItem = document.createElement('li');
                listItem.textContent = `Error processing: ${event.data}`;
                eventList.appendChild(listItem);
            }
        };

        socket.onclose = function (event) {
            console.log('WebSocket disconnected');
            updateConnectionStatus('disconnected', 'Disconnected');
            // Attempt to reconnect after a delay
            setTimeout(connectWebSocket, 5000); 
        };

        socket.onerror = function (error) {
            console.error('WebSocket error:', error);
            updateConnectionStatus('disconnected', 'Error');
            // socket.close(); // Error might lead to close, then onclose will handle reconnect
        };
    }

    clearLogBtn.addEventListener('click', function() {
        eventList.innerHTML = '';
        eventCount = 0;
        eventCountSpan.textContent = eventCount;
        console.log('Event log cleared');
    });

    // Initial connection attempt
    connectWebSocket();
});
