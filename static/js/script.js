document.addEventListener('DOMContentLoaded', () => {
    const chatbotInputField = document.getElementById('chatbot-input-field');
    const chatbotSendBtn = document.getElementById('chatbot-send-btn');
    const chatbotMessages = document.getElementById('chatbot-messages');
    const closeChatbotBtn = document.getElementById('close-chatbot-btn'); // Close button
    const chatbot = document.getElementById('chatbot'); // Chatbot element    
    const enlarge = document.getElementById('enlarge-chatbot-btn');
    let msg = JSON.parse(localStorage.getItem('chatbot-messages')) || []; // Retrieve messages from local storage
    const botImagePath = document.getElementById('bot-image-path').value; // Get the bot image path
    const userImagePath = document.getElementById('user-image-path').value; // Get the bot image path
    const typingIndicator = document.getElementById('typing-indicator');

    const imageTag = `<img src="${botImagePath}" alt="robo" style="width: 30px; height: 30px; margin-right: 10px;">`;
    const userImageTag = `<img src="${userImagePath}" alt="user" style="width: 30px; height: 30px; margin-right: 10px; color:white;">`;

    function getGreeting() {
        const now = new Date();
        const hours = now.getHours();
        let greeting;

        if (hours < 12) {
            greeting = "Good Morning";
        } else if (hours < 18) {
            greeting = "Good Afternoon";
        } else {
            greeting = "Good Evening";
        }

        return `Hi, ${greeting}, how can I help you today?`;
    }

    greetingSentence = getGreeting();

    // Display the initial bot message if there are no messages in local storage
    if (msg.length === 0) {
        appendMessage(`${greetingSentence}`, 'bot-message', false);
        
    }
    else {
        // Display messages from local storage
        msg.forEach((message) => {
            appendMessage(message.message, message.className, false);
        });
    }

    // Clean local storage every 2 minutes
    setInterval(() => {
        localStorage.removeItem('chatbot-messages');
        msg = [];
    }, 5 * 60 * 1000);

    chatbotSendBtn.addEventListener('click', () => {
        const userMessage = chatbotInputField.value;
        if (userMessage.trim() !== '') {
            appendMessage(`${userMessage}`, 'user-message', true);

            chatbotInputField.value = '';

            // Simulate bot response with typing indicator
            showTypingIndicator();

            setTimeout(() => {
                hideTypingIndicator();
                sendUserMessage(userMessage);
            }, 1000);
        }
    });

    function showTypingIndicator() {
        typingIndicator.style.display = 'flex';
    }

    function hideTypingIndicator() {
        typingIndicator.style.display = 'none';
    }

    chatbotInputField.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            chatbotSendBtn.click();
        }
    });

    const chatbotButton = document.querySelector('.chatbot-button');
    const mainDiv = document.getElementById('main-div');

    chatbotButton.addEventListener('click', () => {
        if (mainDiv.style.display === 'none' || mainDiv.style.display === '') {
            mainDiv.style.display = 'block';
        } else {
            mainDiv.style.display = 'none';
        }
    });

    function appendMessage(message, className, store) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${className}`;
        messageElement.innerHTML = message;
        chatbotMessages.appendChild(messageElement);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;

        if (store) {
            // Add message to msg array
            msg.push({ message, className });

            // Store the messages in local storage
            localStorage.setItem('chatbot-messages', JSON.stringify(msg));
        }
    }

    function sendUserMessage(message) {
 
            // Fetch response for other messages
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: message })
            })
            .then(response => response.json())
            .then(data => {
                appendMessage(data.response, 'bot-message', true);
            })
            .catch(error => {
                console.error('Error:', error);
                appendMessage(` Sorry, there was an error. Please try again later.`, 'bot-message', true);
            });
        }
    

    // Event listener for close button
    closeChatbotBtn.addEventListener('click', () => {
        mainDiv.style.display = 'none';
        // Ensure side chatbot button remains visible
        const botBtn = document.querySelector('.bot-btn');
        botBtn.style.display = 'block';
    });

    let isEnlarged = false; // Track if chatbot is enlarged 

    function toggleEnlarged() {
        const chatbot = document.getElementById('chatbot');
        const chatbotMessages = document.getElementById('chatbot-messages');
        const chatbotInput = document.getElementById('chatbot-input');
        const botBtn = document.querySelector('.bot-btn');

        if (!isEnlarged) {
            // Enlarge chatbot
            chatbot.classList.add('enlarged');
            chatbotMessages.classList.add('enlarge-chatbot-messages');
            chatbotInput.style.marginLeft = '-1rem';
            botBtn.style.display = 'none';
            // Change the width to 100%
            chatbot.style.width = '100%';
            // Change the max-width
            chatbot.style.maxWidth = '100vw';
            chatbot.style.position = 'static';
            
            chatbotMessages.style.height = '62vh'; 

            chatbotInput.style.position = 'fixed';
            chatbotInput.style.bottom = "63px";

        } else {
            // Restore chatbot to normal size
            chatbot.classList.remove('enlarged');
            chatbotMessages.classList.remove('enlarge-chatbot-messages');
            chatbotMessages.style.height = '30vh';
            
            // Change the width to 100%
            chatbot.style.width = '75%';
            // Change the max-width
            chatbot.style.maxWidth = '31.25rem';
            chatbot.style.position = 'fixed';

            chatbotInput.style.position = 'static   ';
            chatbotInput.style.bottom = "0px";
            chatbotInput.style.marginLeft = "0rem";

            botBtn.style.display = "block";
        }

        isEnlarged = !isEnlarged; // Toggle enlarged state
    }

    // Event listener for enlarge button
    enlarge.addEventListener('click', toggleEnlarged);
});
