import { useState, useEffect } from 'react'
import { useTheme } from '../contexts/ThemeContext'
import StatsGrid from '../components/StatsGrid'
import ChartsSection from '../components/ChartsSection'
import LowStockAlerts from '../components/LowStockAlerts'
import AddItemForm from '../components/AddItemForm'
import InventoryTable from '../components/InventoryTable'
import './Dashboard.css'

function Dashboard() {
  const { theme } = useTheme()
  const [inventory, setInventory] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedItemId, setSelectedItemId] = useState('')
  const [editingId, setEditingId] = useState(null)
  const [editForm, setEditForm] = useState({ name: '', category: '', quantity: '', threshold: '' })
  const [newItem, setNewItem] = useState({ name: '', category: '', quantity: '', threshold: '' })
  const [showAddForm, setShowAddForm] = useState(false)

  // iGentic / agent response state
  const [agentResponse, setAgentResponse] = useState(null)
  const [agentLoading, setAgentLoading] = useState(false)
  const [agentError, setAgentError] = useState(null)

  // Fetch real inventory data
  useEffect(() => {
    fetch("http://127.0.0.1:8080/api/inventory")
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          const formatted = data.items.map(item => ({
            id: item.inventory_id,
            name: item.item_name,
            category: item.item_type || 'Unknown',
            quantity: item.initial_stock || 0,
            threshold: item.minimum_required || 0,
            raw: item
          }))
          setInventory(formatted)
        }
        setLoading(false)
      })
      .catch(err => {
        console.error(err)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return <div className="loading">Loading inventory...</div>
  }

  const chartColors = {
    dark: { bg: '#1e293b', border: '#334155', text: '#f1f5f9', textSecondary: '#cbd5e1', grid: '#334155' },
    light: { bg: '#ffffff', border: '#e2e8f0', text: '#1e293b', textSecondary: '#64748b', grid: '#e2e8f0' }
  }
  const colors = chartColors[theme]

  const getStockStatus = (item) => {
    if (item.quantity === 0) return 'out-of-stock'
    if (item.quantity <= item.threshold) return 'low-stock'
    return 'in-stock'
  }

  const stats = {
    totalItems: inventory.length,
    inStock: inventory.filter(item => item.quantity > item.threshold).length,
    lowStock: inventory.filter(item => item.quantity > 0 && item.quantity <= item.threshold).length,
    outOfStock: inventory.filter(item => item.quantity === 0).length,
    totalQuantity: inventory.reduce((sum, item) => sum + item.quantity, 0)
  }

  const categoryData = inventory.reduce((acc, item) => {
    if (!acc[item.category]) acc[item.category] = { name: item.category, quantity: 0, items: 0 }
    acc[item.category].quantity += item.quantity
    acc[item.category].items += 1
    return acc
  }, {})

  const categoryChartData = Object.values(categoryData).map(cat => ({
    name: cat.name,
    quantity: cat.quantity,
    items: cat.items
  }))

  const statusData = [
    { name: 'In Stock', value: stats.inStock, color: '#10b981' },
    { name: 'Low Stock', value: stats.lowStock, color: '#f59e0b' },
    { name: 'Out of Stock', value: stats.outOfStock, color: '#ef4444' }
  ]

  const consumptionData = [
    { month: 'Jan', usage: 450 },
    { month: 'Feb', usage: 520 },
    { month: 'Mar', usage: 480 },
    { month: 'Apr', usage: 610 },
    { month: 'May', usage: 550 },
    { month: 'Jun', usage: 680 }
  ]

  const lowStockAlerts = inventory.filter(item => item.quantity <= item.threshold)

  const handleEdit = (item) => {
    setEditingId(item.id)
    setEditForm({
      name: item.name,
      category: item.category,
      quantity: item.quantity.toString(),
      threshold: item.threshold.toString()
    })
  }

  const handleSaveEdit = () => {
    setInventory(inventory.map(item =>
      item.id === editingId
        ? { ...item, name: editForm.name, category: editForm.category, quantity: parseInt(editForm.quantity) || 0, threshold: parseInt(editForm.threshold) || 0 }
        : item
    ))
    setEditingId(null)
    setEditForm({ name: '', category: '', quantity: '', threshold: '' })
  }

  const handleCancelEdit = () => {
    setEditingId(null)
    setEditForm({ name: '', category: '', quantity: '', threshold: '' })
  }

  const categories = [...new Set(inventory.map(item => item.category))]

  // -------------------------
  // iGentic integration
  // -------------------------
  const IGENTIC_ORCHESTRATOR_ID = "df6578f6-7485-4946-85d3-0c6c1fb9114e"
  const IGENTIC_ENDPOINT_BASE = "https://container-hackathon-sk.salmonpebble-59bd07ab.eastus.azurecontainerapps.io/api/iGenticAutonomousAgent/Executor"
  const IGENTIC_URL = `${IGENTIC_ENDPOINT_BASE}/${IGENTIC_ORCHESTRATOR_ID}`

  const IGENTIC_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_IGENTIC_TOKEN"
  }

async function sendToAgent(item) {
  if (!item) return
  setAgentLoading(true)
  setAgentError(null)
  setAgentResponse(null)

  try {
    const userInputPayload = {
      item_id: item.id,
      item_name: item.name,
      forecast_output: [],
      threshold_status: {
        flag_below_min: item.quantity <= item.threshold,
        reorder_level: item.threshold,
        reason: item.quantity <= item.threshold ? "Below minimum" : "Stock OK"
      },
      stock_info: {
        Closing_Stock: item.quantity,
        Min_Stock_Limit: item.threshold,
        Vendor: { vendor_name: (item.raw && item.raw.vendor_name) ? item.raw.vendor_name : "Vendor_ABC" }
      },
      prompt: `Generate a detailed forecast report for ${item.name}, including consumption trends, low-stock warnings, and recommended actions.`
    }

    const payload = {
      UserInput: JSON.stringify(userInputPayload),
      sessionId: localStorage.getItem("igentic_session") || "",
      executionId: crypto.randomUUID ? crypto.randomUUID() : (Date.now().toString() + Math.random().toString()),
      connectionID: "react-frontend",
      isImage: false,
      base64string: "",
      evalId: "",
      userInputType: ""
    }

    const res = await fetch(IGENTIC_URL, {
      method: "POST",
      headers: IGENTIC_HEADERS,
      body: JSON.stringify(payload)
    })

    if (!res.ok) {
      const txt = await res.text()
      throw new Error(`iGentic API error: ${res.status} ${txt}`)
    }

    const data = await res.json()
    if (data.session_id) localStorage.setItem("igentic_session", data.session_id)
    setAgentResponse(data)

  } catch (err) {
    console.error(err)
    setAgentError(err.message || String(err))
  } finally {
    setAgentLoading(false)
  }
}

  return (
    <div className="dashboard-page">

      {/* iGentic Response Panel at the top */}
      <div className="agent-response-panel">
        <h3 className="section-title">iGentic Agent Response</h3>
        {agentError && <div className="agent-error">Error: {agentError}</div>}
        {agentLoading && <div className="agent-loading">Waiting for agent response...</div>}
        {agentResponse && (
          <div className="agent-response-card">
            <pre>
              {agentResponse.result || JSON.stringify(agentResponse, null, 2)}
            </pre>
          </div>
        )}
      </div>

      <div className="page-header" style={{ marginBottom: '1rem' }}>
        <h1 className="page-title">Dashboard</h1>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <select
            value={selectedItemId}
            onChange={e => setSelectedItemId(e.target.value)}
            style={{ marginTop: '1rem', padding: '0.5rem' }}
          >
            <option value="">-- Select Inventory Item --</option>
            {inventory.map(item => (
              <option key={item.id} value={item.id}>{item.name}</option>
            ))}
          </select>

          <button
            onClick={() => {
              const item = inventory.find(i => i.id === selectedItemId)
              if (item) sendToAgent(item)
            }}
            style={{ padding: "0.5rem", marginTop: '1rem' }}
            disabled={!selectedItemId || agentLoading}
          >
            {agentLoading ? 'Sending...' : 'Send to iGentic'}
          </button>

          <button
            onClick={() => {
              setAgentResponse(null)
              setAgentError(null)
              localStorage.removeItem("igentic_session")
            }}
            style={{ padding: "0.5rem", marginTop: '1rem' }}
          >
            Reset iGentic Session
          </button>
        </div>
      </div>

      <StatsGrid stats={stats} />

      <ChartsSection
        categoryChartData={categoryChartData}
        statusData={statusData}
        consumptionData={consumptionData}
        colors={colors}
      />

      <LowStockAlerts lowStockAlerts={lowStockAlerts} />

      <div className="management-section">
        <div className="section-header">
          <h2 className="section-title">Inventory Management</h2>
          <button onClick={() => setShowAddForm(!showAddForm)} className="add-button">
            {showAddForm ? 'Cancel' : '+ Add New Item'}
          </button>
        </div>

        {showAddForm && (
          <AddItemForm
            newItem={newItem}
            setNewItem={setNewItem}
            categories={categories}
            handleAddItem={() => {
              const nextId = `INV${(Math.random()*100000).toFixed(0)}`
              const created = {
                id: nextId,
                name: newItem.name || 'New Item',
                category: newItem.category || 'Unknown',
                quantity: parseInt(newItem.quantity) || 0,
                threshold: parseInt(newItem.threshold) || 0
              }
              setInventory([created, ...inventory])
              setShowAddForm(false)
              setNewItem({ name: '', category: '', quantity: '', threshold: '' })
            }}
          />
        )}

        <InventoryTable
          inventory={inventory}
          editingId={editingId}
          editForm={editForm}
          setEditForm={setEditForm}
          categories={categories}
          getStockStatus={getStockStatus}
          handleEdit={handleEdit}
          handleSaveEdit={handleSaveEdit}
          handleCancelEdit={handleCancelEdit}
          handleDelete={id => setInventory(inventory.filter(it => it.id !== id))}
        />
      </div>
    </div>
  )
}

export default Dashboard


