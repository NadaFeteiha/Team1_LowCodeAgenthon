import { useState, useRef, useEffect } from 'react'
import WelcomeScreen from '../components/WelcomeScreen'
import MessageList from '../components/MessageList'
import ChatInput from '../components/ChatInput'
import './Chatbot.css'

function Chatbot() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your SupplySoul assistant. Ask me about stock levels, item details, or search for specific medications.",
      sender: 'bot',
      timestamp: new Date()
    }
  ])
  const [inputText, setInputText] = useState('')
  const [isListening, setIsListening] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [agentError, setAgentError] = useState(null)
  const messagesEndRef = useRef(null)
  const recognitionRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // -------------------------
  // iGentic config
  // -------------------------
  const AGENT_ID = "f800f4c2-eb25-467c-942b-b81de85e2f1c"
  const IGENTIC_ENDPOINT_BASE = "https://container-hackathon-sk.salmonpebble-59bd07ab.eastus.azurecontainerapps.io/api/iGenticAutonomousAgent/Executor"
  const IGENTIC_URL = `${IGENTIC_ENDPOINT_BASE}/${AGENT_ID}`

  const IGENTIC_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_IGENTIC_TOKEN"
  }

  const sendToAgent = async (text) => {
    setIsProcessing(true)
    setAgentError(null)

    const userMessage = {
      id: messages.length + 1,
      text,
      sender: 'user',
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])

    const payload = {
      UserInput: JSON.stringify({ prompt: text }),
      sessionId: localStorage.getItem("igentic_chat_session") || "",
      executionId: crypto.randomUUID ? crypto.randomUUID() : (Date.now().toString() + Math.random().toString()),
      connectionID: "react-chatbot",
      isImage: false,
      base64string: "",
      evalId: "",
      userInputType: "text"
    }

    try {
      const res = await fetch(IGENTIC_URL, {
        method: 'POST',
        headers: IGENTIC_HEADERS,
        body: JSON.stringify(payload)
      })

      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`iGentic API error: ${res.status} ${txt}`)
      }

      const data = await res.json()
      if (data.session_id) localStorage.setItem("igentic_chat_session", data.session_id)

      const botMessage = {
        id: messages.length + 2,
        text: data.result || JSON.stringify(data, null, 2),
        sender: 'bot',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, botMessage])
    } catch (err) {
      console.error(err)
      setAgentError(err.message || String(err))
      const botMessage = {
        id: messages.length + 2,
        text: `Error: ${err.message || "Something went wrong with the agent."}`,
        sender: 'bot',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, botMessage])
    } finally {
      setIsProcessing(false)
    }
  }

  const handleSendMessage = (text = inputText) => {
    if (!text.trim() || isProcessing) return
    setInputText('')
    sendToAgent(text)
  }

  // -------------------------
  // Voice input (same as before)
  // -------------------------
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = false
      recognitionRef.current.interimResults = false
      recognitionRef.current.lang = 'en-US'

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript
        setInputText(transcript)
        setIsListening(false)
        setTimeout(() => handleSendMessage(transcript), 100)
      }

      recognitionRef.current.onerror = () => setIsListening(false)
      recognitionRef.current.onend = () => setIsListening(false)
    }
  }, [])

  const handleVoiceInput = () => {
    if (!recognitionRef.current) {
      alert('Speech recognition is not supported in your browser.')
      return
    }
    if (isListening) recognitionRef.current.stop()
    else recognitionRef.current.start()
    setIsListening(!isListening)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const quickQuestions = [
    "What items are low in stock?",
    "Show me all pain relief items",
    "How many items are out of stock?",
    "What's the total inventory count?"
  ]

  const handleQuickQuestion = (question) => {
    setInputText(question)
    setTimeout(() => handleSendMessage(question), 100)
  }

  return (
    <div className="chatbot-page-chatgpt">
      <div className="chat-container-chatgpt">
        {messages.length === 1 && (
          <WelcomeScreen
            quickQuestions={quickQuestions}
            handleQuickQuestion={handleQuickQuestion}
          />
        )}

        <MessageList
          messages={messages}
          isProcessing={isProcessing}
          messagesEndRef={messagesEndRef}
        />

        <ChatInput
          inputText={inputText}
          setInputText={setInputText}
          handleKeyPress={handleKeyPress}
          handleVoiceInput={handleVoiceInput}
          isListening={isListening}
          handleSendMessage={handleSendMessage}
          isProcessing={isProcessing}
        />

        {agentError && <div className="agent-error">Agent Error: {agentError}</div>}
      </div>
    </div>
  )
}

export default Chatbot
